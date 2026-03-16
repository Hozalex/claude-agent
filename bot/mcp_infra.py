"""
MCP server — infrastructure vector search.

Exposes search_infrastructure as a native tool for the Claude agent.
Runs as a subprocess (stdio transport) managed by Claude Code.
"""
import asyncio
import os

import asyncpg
import httpx
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("infra")


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

    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=1)
    try:
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
    finally:
        await pool.close()

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
        )
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
