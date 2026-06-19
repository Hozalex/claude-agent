"""
MCP server — infrastructure vector search.

Exposes search_infrastructure as a native tool for the Claude agent.
Runs as a subprocess (stdio transport) managed by Claude Code.
"""
import asyncio
import os
import re

import asyncpg
import httpx
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Works both as a package module (`python -m bot.mcp_infra`) and as a bare script
# (`python /app/bot/mcp_infra.py`), which is how the MCP server is launched.
try:
    from bot.redact import redact
except ImportError:  # bare-script launch puts /app/bot (not /app) on sys.path
    from redact import redact

server = Server("infra")

# kubectl identifiers: DNS-1123 / context names. Reject anything else to keep the
# subprocess argv free of injection and stray flags.
_IDENT_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._\-]{0,252}$")
_LOGS_MAX_TAIL = 200

_pool: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=3)
    return _pool


async def _embed(query: str) -> list[float]:
    url = os.environ.get("EMBEDDINGS_URL", "http://embeddings-api/embed")
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json={"input": query})
        resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


async def _search(
    query: str,
    cluster: str | None,
    kind: str | None,
    limit: int,
) -> str:
    embedding = await _embed(query)

    filters: list[str] = []
    params: list = [str(embedding), limit]

    if cluster:
        params.append(cluster)
        filters.append(f"cluster = ${len(params)}")
    if kind:
        params.append(kind)
        filters.append(f"kind = ${len(params)}")

    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    pool = await _get_pool()
    rows = await pool.fetch(
        f"""
        SELECT cluster, kind, name, namespace, content, enriched,
               1 - (embedding <=> $1::vector) AS similarity
        FROM   infrastructure
        {where}
        ORDER  BY embedding <=> $1::vector
        LIMIT  $2
        """,
        *params,
    )

    if not rows:
        return "No results found."

    lines: list[str] = []
    for row in rows:
        ns  = f"/{row['namespace']}" if row['namespace'] else ""
        tag = "[enriched]" if row["enriched"] else "[template]"
        lines.append(
            f"[{row['cluster']}] {row['kind']} {row['name']}{ns}"
            f"  sim={row['similarity']:.2f}  {tag}"
        )
        lines.append(row["content"])
        lines.append("")

    return "\n".join(lines)


def _valid_ident(value: str) -> bool:
    return bool(_IDENT_RE.match(value))


async def _get_logs(
    cluster: str,
    namespace: str,
    pod: str,
    container: str | None,
    tail: int,
    previous: bool,
) -> str:
    """Run `kubectl logs` read-only and return PII-redacted output.

    This is the ONLY sanctioned path to pod logs: raw `kubectl logs` via Bash is
    blocked because its output bypasses our code and cannot be redacted. Here the
    output is held in-process and scrubbed before it is returned to the model.
    """
    for label, value in (("cluster", cluster), ("namespace", namespace), ("pod", pod)):
        if not value or not _valid_ident(value):
            return f"Invalid {label}: {value!r}"
    if container and not _valid_ident(container):
        return f"Invalid container: {container!r}"

    tail = max(1, min(int(tail), _LOGS_MAX_TAIL))

    argv = [
        "kubectl", "--context", cluster,
        "logs", pod, "-n", namespace,
        f"--tail={tail}",
    ]
    if container:
        argv += ["-c", container]
    if previous:
        argv.append("--previous")

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=30.0)
    except asyncio.TimeoutError:
        return f"kubectl logs timed out for {namespace}/{pod} on {cluster}."
    except Exception as exc:  # noqa: BLE001 — surface kubectl/exec failures to the agent
        return f"Failed to run kubectl logs: {exc}"

    if proc.returncode != 0:
        return f"kubectl logs failed ({namespace}/{pod} on {cluster}): {err.decode(errors='replace').strip()}"

    raw = out.decode(errors="replace")
    if not raw.strip():
        return f"No log output for {namespace}/{pod} on {cluster} (tail={tail})."
    return redact(raw)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_infrastructure",
            description=(
                "Search the Kubernetes infrastructure knowledge base using semantic similarity. "
                "Returns matching resources (Deployments, Services, HTTPRoutes, RabbitmqClusters, etc.) "
                "across all clusters. Use this to understand what a service does, find its dependencies, "
                "or assess the blast radius of an incident. "
                "Results marked [enriched] contain LLM-generated descriptions; "
                "[template] contains raw spec data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "cluster": {
                        "type": "string",
                        "description": "Filter by cluster name, e.g. production (optional)",
                    },
                    "kind": {
                        "type": "string",
                        "description": "Filter by resource kind, e.g. Deployment, Service (optional)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_logs",
            description=(
                "Read recent logs from a single pod (read-only `kubectl logs`). "
                "This is the ONLY way to read pod logs — raw `kubectl logs` in Bash is blocked. "
                "Output is automatically scrubbed of personal data (emails, phone numbers, "
                "national IDs, payment cards, tokens, names) before being returned. "
                "Always scope to the exact namespace/pod from the alert or the resource under investigation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster": {
                        "type": "string",
                        "description": "Exact kubectl context name, e.g. production-cluster (resolve aliases first).",
                    },
                    "namespace": {"type": "string", "description": "Pod namespace."},
                    "pod": {"type": "string", "description": "Pod name."},
                    "container": {
                        "type": "string",
                        "description": "Container name (optional; needed for multi-container pods).",
                    },
                    "tail": {
                        "type": "integer",
                        "description": "Number of trailing lines (default 100, max 200).",
                        "default": 100,
                    },
                    "previous": {
                        "type": "boolean",
                        "description": "Read logs from the previous terminated container instance (optional).",
                        "default": False,
                    },
                },
                "required": ["cluster", "namespace", "pod"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "search_infrastructure":
        result = await _search(
            query=arguments["query"],
            cluster=arguments.get("cluster"),
            kind=arguments.get("kind"),
            limit=arguments.get("limit", 5),
        )
        return [types.TextContent(type="text", text=result)]
    if name == "get_logs":
        result = await _get_logs(
            cluster=arguments["cluster"],
            namespace=arguments["namespace"],
            pod=arguments["pod"],
            container=arguments.get("container"),
            tail=arguments.get("tail", 100),
            previous=arguments.get("previous", False),
        )
        return [types.TextContent(type="text", text=result)]
    raise ValueError(f"Unknown tool: {name}")


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
