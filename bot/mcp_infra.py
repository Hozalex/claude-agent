"""
MCP server — infrastructure tools for the Claude agent.

Exposes search_infrastructure (vector search), get_logs (PII-redacted pod logs)
and query_metrics (Prometheus) as native tools.
Runs as a subprocess (stdio transport) managed by Claude Code.
"""
import asyncio
import os
import re
import time
from datetime import datetime, timezone

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

# ── Prometheus ─────────────────────────────────────────────────────────────────
#
# Each cluster runs its own kube-prometheus-stack (externalLabels.cluster), so
# there is one Prometheus URL per cluster. Configured via PROMETHEUS_URLS as
#   "cluster=url,cluster=url"
# e.g. "production-cluster=http://prometheus.prod.local,infra-cluster=http://..."
# Keeping it in config (not code) means a moved endpoint is a manifest change.
_PROM_TIMEOUT = 20.0
_PROM_MAX_SERIES = 25     # cap the reply so a broad query can't flood the context
_PROM_MAX_POINTS = 500    # per-series points fetched before we summarise them
_DURATION_RE = re.compile(r"^(\d+)([smhdw])$")
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def _parse_prom_urls(raw: str) -> dict[str, str]:
    urls: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        cluster, _, url = entry.partition("=")
        cluster, url = cluster.strip(), url.strip()
        if cluster and url:
            urls[cluster] = url.rstrip("/")
    return urls


_PROM_URLS = _parse_prom_urls(os.environ.get("PROMETHEUS_URLS", ""))

# ── Alertmanager ───────────────────────────────────────────────────────────────
#
# A single Alertmanager (infra cluster) receives alerts from every cluster, so
# alerts are told apart by their `cluster` label. That label carries the SHORT
# name from each Prometheus's externalLabels (prod / stage / dev / infra), not
# the kubectl context name — hence the mapping below.
_ALERTMANAGER_URL = os.environ.get("ALERTMANAGER_URL", "").rstrip("/")
_ALERTS_MAX = 40
_SEVERITY_ICONS = {"critical": "🔴", "warning": "🟠", "info": "🟡"}
_DEFAULT_CLUSTER_LABELS = {"development": "dev", "production": "prod"}


def _parse_cluster_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        ctx, _, label = entry.partition("=")
        if ctx.strip() and label.strip():
            labels[ctx.strip()] = label.strip()
    return labels


_CLUSTER_LABELS = _parse_cluster_labels(os.environ.get("CLUSTER_LABELS", ""))


def _cluster_label(cluster: str) -> str:
    """kubectl context name -> the `cluster` label value used in alerts."""
    if cluster in _CLUSTER_LABELS:
        return _CLUSTER_LABELS[cluster]
    base = cluster[: -len("-cluster")] if cluster.endswith("-cluster") else cluster
    return _DEFAULT_CLUSTER_LABELS.get(base, base)


def _alert_age(starts_at: str) -> str:
    try:
        started = datetime.fromisoformat(starts_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return "?"
    seconds = int((datetime.now(timezone.utc) - started).total_seconds())
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d{(seconds % 86400) // 3600:02d}h"


def _fmt_alert(alert: dict) -> str:
    labels = alert.get("labels", {})
    annotations = alert.get("annotations", {})
    status = alert.get("status", {})

    severity = labels.get("severity", "")
    icon = _SEVERITY_ICONS.get(severity, "⚪")
    name = labels.get("alertname", "?")
    scope = "/".join(x for x in (labels.get("cluster"), labels.get("namespace")) if x)
    state = status.get("state", "")
    suffix = " [SILENCED]" if status.get("silencedBy") else ""
    suffix += " [INHIBITED]" if status.get("inhibitedBy") else ""

    target = "  ".join(
        f"{k}={labels[k]}" for k in ("pod", "instance", "job", "service") if labels.get(k)
    )
    summary = annotations.get("summary") or annotations.get("description") or ""

    line = f"{icon} {name}  [{scope}]  {state} for {_alert_age(alert.get('startsAt', ''))}{suffix}"
    if target:
        line += f"\n   {target}"
    if summary:
        line += f"\n   {summary.strip()[:300]}"
    return line


async def _get_alerts(
    cluster: str | None,
    namespace: str | None,
    alertname: str | None,
    severity: str | None,
    include_silenced: bool,
) -> str:
    """List alerts currently known to Alertmanager (all clusters, one instance)."""
    if not _ALERTMANAGER_URL:
        return (
            "Alertmanager is not configured for this bot (ALERTMANAGER_URL is empty). "
            "Current alert state is unavailable — say so instead of guessing."
        )

    filters: list[tuple[str, str]] = []
    if cluster:
        filters.append(("filter", f'cluster="{_cluster_label(cluster)}"'))
    if namespace:
        filters.append(("filter", f'namespace="{namespace}"'))
    if alertname:
        filters.append(("filter", f'alertname="{alertname}"'))
    if severity:
        filters.append(("filter", f'severity="{severity}"'))

    params = filters + [
        ("active", "true"),
        ("silenced", "true" if include_silenced else "false"),
        ("inhibited", "true" if include_silenced else "false"),
    ]

    try:
        async with httpx.AsyncClient(timeout=_PROM_TIMEOUT) as client:
            resp = await client.get(_ALERTMANAGER_URL + "/api/v2/alerts", params=params)
    except Exception as exc:  # noqa: BLE001 — surface connectivity failures to the agent
        return f"Alertmanager request failed: {exc}"

    if resp.status_code != 200:
        return f"Alertmanager returned HTTP {resp.status_code}: {resp.text[:300]}"

    try:
        alerts = resp.json()
    except Exception:
        return "Alertmanager returned a non-JSON response."

    if not alerts:
        scope = ", ".join(
            f"{k}={v}"
            for k, v in (
                ("cluster", cluster), ("namespace", namespace),
                ("alertname", alertname), ("severity", severity),
            )
            if v
        )
        return f"No firing alerts{' for ' + scope if scope else ''}."

    # Most severe and most recent first.
    order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(
        key=lambda a: (
            order.get(a.get("labels", {}).get("severity", ""), 3),
            a.get("startsAt", ""),
        )
    )

    lines = [f"{len(alerts)} alert(s) firing:"]
    lines += [_fmt_alert(a) for a in alerts[:_ALERTS_MAX]]
    if len(alerts) > _ALERTS_MAX:
        lines.append(f"… and {len(alerts) - _ALERTS_MAX} more (narrow with cluster/namespace/severity)")
    return "\n".join(lines)


def _parse_duration(value: str) -> int | None:
    """'15m' / '6h' / '2d' -> seconds. None if malformed."""
    m = _DURATION_RE.match(value.strip())
    if not m:
        return None
    return int(m.group(1)) * _DURATION_UNITS[m.group(2)]


def _pick_step(span_seconds: int) -> str:
    """Step that keeps a range query under _PROM_MAX_POINTS samples per series."""
    step = max(15, -(-span_seconds // _PROM_MAX_POINTS))  # ceil division
    return f"{step}s"


def _fmt_labels(metric: dict) -> str:
    name = metric.get("__name__", "")
    labels = ",".join(f"{k}={v}" for k, v in sorted(metric.items()) if k != "__name__")
    return f"{name}{{{labels}}}" if labels else name or "{}"


def _fmt_instant(result: list) -> str:
    lines = []
    for item in result[:_PROM_MAX_SERIES]:
        value = item.get("value", [None, "?"])[1]
        lines.append(f"{_fmt_labels(item.get('metric', {}))} = {value}")
    if len(result) > _PROM_MAX_SERIES:
        lines.append(f"… and {len(result) - _PROM_MAX_SERIES} more series (narrow the query)")
    return "\n".join(lines)


def _fmt_range(result: list) -> str:
    """Summarise each series instead of dumping every point.

    A raw range response is thousands of numbers; min/max/avg/last plus the time
    of the peak is what actually answers 'was there a problem in this window'.
    """
    lines = []
    for item in result[:_PROM_MAX_SERIES]:
        values = item.get("values", [])
        nums = []
        for ts, raw in values:
            try:
                nums.append((float(ts), float(raw)))
            except (TypeError, ValueError):
                continue
        if not nums:
            continue
        peak_ts, peak = max(nums, key=lambda p: p[1])
        low = min(n for _, n in nums)
        avg = sum(n for _, n in nums) / len(nums)
        last = nums[-1][1]
        peak_at = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(peak_ts))
        lines.append(
            f"{_fmt_labels(item.get('metric', {}))}\n"
            f"  min={low:.4g}  avg={avg:.4g}  max={peak:.4g} (at {peak_at})  last={last:.4g}"
            f"  [{len(nums)} points]"
        )
    if len(result) > _PROM_MAX_SERIES:
        lines.append(f"… and {len(result) - _PROM_MAX_SERIES} more series (narrow the query)")
    return "\n".join(lines)


async def _query_metrics(
    cluster: str,
    query: str,
    lookback: str | None,
    step: str | None,
    end: float | None,
) -> str:
    """Run a PromQL query (instant, or range when `lookback` is given)."""
    if not _PROM_URLS:
        return (
            "Prometheus is not configured for this bot (PROMETHEUS_URLS is empty). "
            "Metrics are unavailable — say so instead of guessing."
        )
    base = _PROM_URLS.get(cluster)
    if not base:
        return (
            f"No Prometheus endpoint for cluster {cluster!r}. "
            f"Configured clusters: {', '.join(sorted(_PROM_URLS)) or 'none'}."
        )
    if not query.strip():
        return "Empty PromQL query."

    end_ts = end if end else time.time()

    if lookback:
        span = _parse_duration(lookback)
        if span is None:
            return f"Invalid lookback {lookback!r}. Use forms like 30m, 6h, 2d."
        params = {
            "query": query,
            "start": f"{end_ts - span:.0f}",
            "end": f"{end_ts:.0f}",
            "step": step or _pick_step(span),
        }
        path = "/api/v1/query_range"
    else:
        params = {"query": query, "time": f"{end_ts:.0f}"}
        path = "/api/v1/query"

    try:
        async with httpx.AsyncClient(timeout=_PROM_TIMEOUT) as client:
            resp = await client.get(base + path, params=params)
    except Exception as exc:  # noqa: BLE001 — surface connectivity failures to the agent
        return f"Prometheus request to {cluster} failed: {exc}"

    if resp.status_code != 200:
        return f"Prometheus {cluster} returned HTTP {resp.status_code}: {resp.text[:300]}"

    try:
        payload = resp.json()
    except Exception:
        return f"Prometheus {cluster} returned a non-JSON response."

    if payload.get("status") != "success":
        return f"Prometheus {cluster} error: {payload.get('error', 'unknown')}"

    data = payload.get("data", {})
    result = data.get("result", [])
    if not result:
        return f"[{cluster}] no data for: {query}"

    header = (
        f"[{cluster}] {'range' if lookback else 'instant'} query: {query}"
        + (f"  (last {lookback}, step {params['step']})" if lookback else "")
    )
    body = _fmt_range(result) if lookback else _fmt_instant(result)
    return f"{header}\n{body}"

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
        types.Tool(
            name="query_metrics",
            description=(
                "Run a PromQL query against the cluster's Prometheus. This is the ONLY way to see "
                "metrics and, crucially, HISTORY — use it for any question about the past "
                "('was there a problem yesterday', 'did it recover', 'how long did it last'). "
                "Omit 'lookback' for the current value; set it (e.g. 6h, 2d) for a time range — "
                "range results are summarised per series as min/avg/max (with the time of the peak) "
                "and last value. Retention is limited (about 10 days), so older windows return no data. "
                "Alert labels map directly onto PromQL selectors: use the alert's namespace/pod/instance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster": {
                        "type": "string",
                        "description": "Cluster name, e.g. production-cluster (resolve aliases first).",
                    },
                    "query": {
                        "type": "string",
                        "description": "PromQL expression, e.g. sum(rate(http_requests_total{namespace=\"x\"}[5m])) by (status).",
                    },
                    "lookback": {
                        "type": "string",
                        "description": "Time window back from 'end' for a range query: 30m, 6h, 2d, 1w. Omit for an instant query.",
                    },
                    "step": {
                        "type": "string",
                        "description": "Range resolution, e.g. 60s or 5m. Optional — chosen automatically from the window.",
                    },
                    "end": {
                        "type": "number",
                        "description": "Unix timestamp for the end of the window (default: now). Use it to look at when an alert fired.",
                    },
                },
                "required": ["cluster", "query"],
            },
        ),
        types.Tool(
            name="get_alerts",
            description=(
                "List alerts currently firing in Alertmanager. One Alertmanager (infra cluster) "
                "receives alerts from ALL clusters, so filter by cluster to scope it. "
                "Use this to check whether an alert is still firing or already resolved, to see what "
                "else is firing alongside it (correlated incidents), and to confirm severity — "
                "instead of asking the user. Silenced and inhibited alerts are hidden unless "
                "include_silenced is set. This shows the CURRENT state only; for history use query_metrics."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster": {
                        "type": "string",
                        "description": "Cluster to filter by, e.g. production-cluster (mapped to the short alert label automatically). Omit for all clusters.",
                    },
                    "namespace": {"type": "string", "description": "Filter by namespace (optional)."},
                    "alertname": {
                        "type": "string",
                        "description": "Filter by exact alert name, e.g. PgActiveQueryDurationCritical (optional).",
                    },
                    "severity": {
                        "type": "string",
                        "description": "Filter by severity: critical, warning or info (optional).",
                    },
                    "include_silenced": {
                        "type": "boolean",
                        "description": "Also include silenced and inhibited alerts (default false).",
                        "default": False,
                    },
                },
                "required": [],
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
    if name == "query_metrics":
        result = await _query_metrics(
            cluster=arguments["cluster"],
            query=arguments["query"],
            lookback=arguments.get("lookback"),
            step=arguments.get("step"),
            end=arguments.get("end"),
        )
        return [types.TextContent(type="text", text=result)]
    if name == "get_alerts":
        result = await _get_alerts(
            cluster=arguments.get("cluster"),
            namespace=arguments.get("namespace"),
            alertname=arguments.get("alertname"),
            severity=arguments.get("severity"),
            include_silenced=arguments.get("include_silenced", False),
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
