"""
/status command — checks cluster connectivity, DB, and Anthropic balance.
All checks run concurrently with short timeouts.
"""
import asyncio
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


_KUBECTL_TIMEOUT = 8  # seconds per cluster check


def _get_contexts() -> list[str]:
    """Return all context names from the active kubeconfig."""
    try:
        result = subprocess.run(
            ["kubectl", "config", "get-contexts", "-o", "name"],
            capture_output=True, text=True, timeout=5,
        )
        return [c.strip() for c in result.stdout.splitlines() if c.strip()]
    except Exception:
        return []


async def _check_cluster(context: str) -> tuple[str, bool, str]:
    """Returns (context, ok, detail)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "--context", context, "cluster-info",
            "--request-timeout=5s",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_KUBECTL_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            return context, False, "timeout"

        if proc.returncode == 0:
            # Extract control plane URL from first line
            first = stdout.decode().splitlines()[0] if stdout else ""
            detail = first.replace("Kubernetes control plane is running at ", "").strip()
            return context, True, detail
        else:
            err = stderr.decode().strip().splitlines()[-1] if stderr else "error"
            return context, False, err[:80]
    except Exception as e:
        return context, False, str(e)[:80]


async def _check_db() -> tuple[bool, str]:
    try:
        import asyncpg
        dsn = os.environ.get("DATABASE_URL", "")
        if not dsn:
            logger.error("status.db: DATABASE_URL is not set")
            return False, "DATABASE_URL not set"
        logger.debug("status.db: connecting to DB")
        conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=5.0)
        row = await conn.fetchrow("SELECT count(*) AS n FROM infrastructure")
        await conn.close()
        logger.info("status.db: OK, %s resources indexed", row['n'])
        return True, f"{row['n']} resources indexed"
    except asyncio.TimeoutError:
        logger.error("status.db: connection timeout")
        return False, "timeout"
    except Exception as e:
        logger.exception("status.db: failed")
        return False, str(e)


async def _check_embeddings() -> tuple[bool, str]:
    try:
        import httpx
        url = os.environ.get("EMBEDDINGS_URL", "http://embeddings-api/embed")
        health = url.rsplit("/", 1)[0] + "/health"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(health)
            if resp.status_code == 200:
                data = resp.json()
                model = data.get("model", "?")
                return True, model
            return False, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)[:80]


async def _check_anthropic() -> tuple[bool, str]:
    """Check Anthropic API key validity and list available models."""
    try:
        import httpx
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return False, "ANTHROPIC_API_KEY not set"
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            )
            if resp.status_code == 200:
                data = resp.json()
                models = [m["id"] for m in data.get("data", [])]
                return True, ", ".join(models) if models else "API key valid (no models)"
            elif resp.status_code in (401, 403):
                return False, "invalid API key"
            else:
                return False, f"HTTP {resp.status_code}"
    except Exception as e:
        logger.exception("status.anthropic: failed")
        return False, str(e)[:80]


async def get_status() -> str:
    contexts = _get_contexts()
    cluster_tasks = [_check_cluster(c) for c in contexts]
    results = await asyncio.gather(
        *cluster_tasks,
        _check_db(),
        _check_embeddings(),
        _check_anthropic(),
    )

    n = len(contexts)
    cluster_results = results[:n]
    db_ok, db_detail = results[n]
    emb_ok, emb_detail = results[n + 1]
    ant_ok, ant_detail = results[n + 2]

    lines: list[str] = ["Status\n"]

    lines.append("Clusters:" if contexts else "Clusters: none found in kubeconfig")
    for ctx, ok, detail in cluster_results:
        icon = "OK" if ok else "FAIL"
        lines.append(f"  {icon}  {ctx}")
        if detail:
            lines.append(f"       {detail}")

    lines.append("")
    lines.append("Services:")
    lines.append(f"  {'OK' if db_ok else 'FAIL'}  Database — {db_detail}")
    lines.append(f"  {'OK' if emb_ok else 'FAIL'}  Embeddings — {emb_detail}")
    lines.append(f"  {'OK' if ant_ok else 'FAIL'}  Anthropic — {ant_detail}")

    return "\n".join(lines)
