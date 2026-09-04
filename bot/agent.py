import dataclasses
import logging
import os
import re
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition, HookMatcher
from claude_agent_sdk.types import (
    AssistantMessage,
    TextBlock,
    ResultMessage,
    HookContext,
)

logger = logging.getLogger(__name__)

# ── Errors ────────────────────────────────────────────────────────────────────

_CLAUDE_ERROR_MESSAGES: dict[str, str] = {
    "billing_error": (
        "💳 Insufficient API credits.\n"
        "Top up at: https://console.anthropic.com/settings/billing"
    ),
    "rate_limit_error": "⏳ Claude API rate limit reached. Please wait a moment and try again.",
    "authentication_error": "🔑 Anthropic API key is invalid or missing. Check the ANTHROPIC_API_KEY env var.",
    "overloaded_error": "🔥 Claude is overloaded right now. Please try again in a few seconds.",
    "invalid_request_error": "❌ Invalid request sent to Claude. Check the bot configuration.",
}

_DEFAULT_ERROR_MESSAGE = "❌ Claude API error: {code}"


class ClaudeAPIError(Exception):
    """Raised when Claude returns a known API error."""

    def __init__(self, code: str, user_message: str) -> None:
        super().__init__(code)
        self.code = code
        self.user_message = user_message


def _make_api_error(code: str) -> ClaudeAPIError:
    msg = _CLAUDE_ERROR_MESSAGES.get(code, _DEFAULT_ERROR_MESSAGE.format(code=code))
    return ClaudeAPIError(code=code, user_message=msg)


# ── Safety ────────────────────────────────────────────────────────────────────

# Bash commands that are never allowed, regardless of skill or user request.
#
# Matching is word- and position-aware, NOT plain substring: a naive "rm " pattern
# also matched inside "smsfin-platform 2>&1" and blocked ordinary read-only kubectl
# calls. Dangerous binaries are therefore only matched at the start of a command,
# and kubectl verbs are matched as whole words within one command segment (which
# also catches indirection like `$KUBECTL port-forward`).

# kubectl invoked directly or through a shell variable.
_KUBECTL = r"(?:kubectl|\$\{?KUBECTL\}?)"
# Rest of the same command segment — stops at a pipe/semicolon/newline.
_SEG = r"[^;|&\n]*?"
# A dangerous binary only counts when it starts a command.
_CMD_START = r"(?:\A|[\n;&|(`]|\$\()\s*"

BLOCKED_BASH_RULES: list[tuple[str, re.Pattern[str]]] = [
    # Logs — raw kubectl logs is blocked: its output bypasses our code and cannot
    # be redacted of PII. Use the get_logs MCP tool, which scrubs PII before return.
    ("kubectl logs", re.compile(rf"{_KUBECTL}{_SEG}\blogs\b", re.IGNORECASE)),
    # Kubernetes — destructive mutations
    ("kubectl mutation", re.compile(
        rf"{_KUBECTL}{_SEG}\b(?:delete|apply|create|patch|edit|replace|scale|drain|"
        rf"cordon|uncordon|taint)\b", re.IGNORECASE)),
    ("kubectl rollout restart/undo", re.compile(
        rf"{_KUBECTL}{_SEG}\brollout\s+(?:restart|undo)\b", re.IGNORECASE)),
    # Kubernetes — pod access and network exposure
    ("kubectl pod access", re.compile(
        rf"{_KUBECTL}{_SEG}\b(?:exec|cp|attach|debug)\b", re.IGNORECASE)),
    ("kubectl network exposure", re.compile(
        rf"{_KUBECTL}{_SEG}\b(?:proxy|port-forward)\b", re.IGNORECASE)),
    # Kubernetes — secrets
    ("kubectl secrets", re.compile(
        rf"{_KUBECTL}{_SEG}\b(?:get|describe)\b{_SEG}\bsecrets?\b", re.IGNORECASE)),
    # Shell — destructive / privilege escalation / env dumping, at command start only
    ("dangerous command", re.compile(
        _CMD_START + r"(?:rm|rmdir|shred|sudo|su|chmod|chown|printenv|env|tee)\b",
        re.IGNORECASE)),
    ("write to absolute path", re.compile(r">\s*/")),
    # Secrets — sensitive paths and variables (specific enough to match literally)
    ("sensitive path or variable", re.compile(
        r"/proc/self/(?:environ|mem)|/var/run/secrets/|~/\.(?:ssh|aws|kube)/"
        r"|/app/bot/|/app/\.env|DATABASE_URL|ANTHROPIC_API_KEY|BOT_TOKEN",
        re.IGNORECASE)),
]


def find_blocked_rule(command: str) -> tuple[str, str] | None:
    """Return (rule name, matched text) for the first rule the command trips."""
    for rule, pattern in BLOCKED_BASH_RULES:
        m = pattern.search(command)
        if m:
            return rule, m.group().strip()
    return None


async def pre_tool_use_guard(
    input_data: dict,
    tool_use_id: str | None,
    context: HookContext,
) -> dict:
    """Block dangerous bash commands before execution.

    Registered as a PreToolUse hook rather than as `can_use_tool`: with
    permission_mode="bypassPermissions" the SDK auto-approves every tool call and
    never consults can_use_tool (it warns about this as CanUseToolShadowedWarning),
    so that callback silently enforced nothing. Hooks still run, and they also fire
    for tool calls made by subagents — which is where the skills run kubectl.
    """
    if input_data.get("tool_name") != "Bash":
        return {}

    command = (input_data.get("tool_input") or {}).get("command", "")
    hit = find_blocked_rule(command)
    if hit is None:
        return {}

    rule, matched = hit
    logger.warning("Blocked command [%s on %r]: %s", rule, matched, command[:120])
    if rule == "kubectl logs":
        reason = (
            "Raw 'kubectl logs' is blocked. Use the get_logs tool instead "
            "(args: cluster, namespace, pod, optional container/tail) — it returns "
            "the same logs with personal data redacted."
        )
    elif rule == "kubectl network exposure":
        reason = (
            "port-forward and proxy are blocked. To read metrics use the query_metrics "
            "tool (args: cluster, query, optional lookback) — it reaches Prometheus for you. "
            "Do not try to tunnel to it yourself."
        )
    else:
        reason = (
            f"Blocked by rule '{rule}' (matched {matched!r}). "
            "Only read-only operations are permitted."
        )
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


# ── Skill routing ──────────────────────────────────────────────────────────────

_SKILLS_DIR = Path(__file__).parent.parent / ".claude" / "skills"


def _parse_frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return {}
    result: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip()
    return result


def _build_routing_rules() -> str:
    """Read all skill files and build mandatory subagent routing rules."""
    if not _SKILLS_DIR.exists():
        return ""
    rules: list[str] = []
    for path in sorted(_SKILLS_DIR.glob("*.md")):
        fm = _parse_frontmatter(path.read_text())
        agent = fm.get("agent")
        desc = fm.get("description", "").strip()
        if agent and desc:
            rules.append(f'• {desc} → Agent tool, subagent="{agent}"')
    if not rules:
        return ""
    lines = [
        "\n\nSUBAGENT ROUTING — MANDATORY:",
        "When a user request matches one of the rules below, you MUST use the Agent tool",
        "to delegate the task. Do NOT use Bash or any other tool yourself.",
    ] + rules
    return "\n".join(lines)


# ── Agent config ───────────────────────────────────────────────────────────────

# Common constraints appended to every subagent's prompt.
_SUBAGENT_PROMPT_SUFFIX = "Plain text only, no markdown headers. Read-only: never delete or modify any resources."


_BASE_SYSTEM_PROMPT = (
    "You are an expert DevOps and SRE assistant. Your goal is to help engineers diagnose incidents, "
    "monitor Kubernetes, analyze alerts, and provide actionable remediation.\n\n"
    "CONSTRAINTS & SAFETY (CRITICAL):\n"
    "1. Read-Only Mode: You are STRICTLY RESTRICTED to read-only operations.\n"
    "2. No Secrets: You are forbidden from accessing or requesting Kubernetes Secrets.\n"
    "3. Mutations: For ANY action requiring state changes (apply, delete, edit, scale, restart), "
    "you MUST NOT attempt it. Instead, write the exact manual command (e.g., kubectl) for the engineer to run.\n"
    "4. Scope: You only handle DevOps, SRE, Kubernetes, CI/CD, and platform infrastructure. "
    "If a prompt is clearly out-of-scope, reply ONLY with: 'I only assist with DevOps and infrastructure topics.' "
    "If the request is vague but might be related to an incident, ask for clarification first.\n\n"
    "TOOL USAGE: search_infrastructure\n"
    "You have access to 'search_infrastructure' (vector index of all K8s resources across clusters).\n"
    "- Parameters: query (required), cluster (optional), kind (optional), limit (default: 5).\n"
    "- Rule 1: ALWAYS use this tool FIRST before checking logs/metrics to understand service dependencies or blast radius.\n"
    "- Rule 2: ALWAYS include the 'cluster' name in your response.\n"
    "- Rule 3: If the search returns no results, DO NOT hallucinate architecture. State that the resource was not found "
    "and ask the user for the exact namespace or cluster.\n"
    "- Rule 4: If search_infrastructure tool fails or is unavailable, explicitly tell the user: "
    "'Knowledge base unavailable — answering from general knowledge only.' "
    "Do NOT silently answer as if the DB was consulted.\n\n"
    "KUBECTL USAGE:\n"
    "- ALWAYS specify --context when running kubectl. Available clusters:\n"
    "  development-cluster (aliases: dev, development, dev-cluster)\n"
    "  infra-cluster (aliases: infra, infrastructure, infra-cluster)\n"
    "- Map user's cluster references to the exact context name above before running kubectl.\n"
    "- If the cluster name in the request does NOT match any known context above — stop immediately "
    "and tell the user: 'Cluster X is not accessible. Available clusters: development-cluster, infra-cluster.'\n"
    "- If kubectl fails with connection errors, certificate errors, or timeout — explicitly tell the user "
    "which cluster is unreachable. Do NOT retry with different commands.\n"
    "- DO NOT read pod logs with 'kubectl logs' — it is blocked. Use the get_logs tool "
    "(args: cluster, namespace, pod, optional container/tail up to 200). It returns the same logs "
    "with personal data (emails, phones, national IDs, cards, tokens, names) automatically redacted. "
    "Always scope get_logs to the exact namespace/pod under investigation.\n\n"
    "GITOPS AWARENESS:\n"
    "- Before suggesting any change to a resource, check for ArgoCD annotations: "
    "argocd.argoproj.io/tracking-id or argocd.argoproj.io/managed-by.\n"
    "- If present, the resource is managed by ArgoCD. Direct kubectl mutations (patch, set resources, edit) "
    "will be overwritten by ArgoCD sync. DO NOT suggest them.\n"
    "- Instead: identify the ArgoCD app name from the annotation, tell the user to update "
    "the values.yaml or resource manifest in Git, then sync via ArgoCD.\n"
    "- Example: argocd.argoproj.io/tracking-id=dev-loki:apps/StatefulSet:loki/loki-results-cache "
    "means ArgoCD app 'dev-loki' manages this resource. Recommend changing dev-loki Helm values in Git.\n\n"
    "INVESTIGATION DISCIPLINE:\n"
    "- If during analysis you find a signal that requires further investigation "
    "(restarts > 1, OOMKilled, pending pods, high error rate) — investigate it IMMEDIATELY using available tools.\n"
    "- Do NOT list such signals as user action items or recommendations. The user expects YOU to diagnose, "
    "not to be handed a list of commands to run themselves.\n"
    "- The Recommendations section is only for remediation actions that require human decision "
    "(scaling nodes, changing Git config, enabling features). Pure diagnostics belong in your analysis.\n"
    "- STOP AND DECLARE LIMITS: If the requested information (e.g. message contents, application logs, "
    "network traffic) is not accessible through search_infrastructure or kubectl read-only commands — "
    "say so immediately. Do NOT attempt multiple alternative approaches. One attempt per tool, then conclude.\n"
    "- ALERT STATE COMES FROM ALERTMANAGER: When triaging an alert, call get_alerts (filtered by cluster "
    "and alertname) to check whether it is STILL FIRING or already resolved, and to see what else is "
    "firing in the same namespace — correlated alerts usually point at the real cause. Never ask the user "
    "whether an alert is still active. Note: one Alertmanager (infra cluster) holds alerts from every "
    "cluster, and the 'cluster' label there is the SHORT name (prod / stage / dev / infra) — the tool maps "
    "context names for you, so just pass production-cluster.\n"
    "- HISTORY LIVES IN METRICS: kubectl shows only the CURRENT state and events are short-lived, so for "
    "ANY question about the past ('was there a problem yesterday', 'check monitoring for that time', "
    "'did it recover', 'how long did it last') use the query_metrics tool with a lookback window. "
    "Never answer a question about the past from kubectl alone, and never tell the user history is "
    "unavailable without having tried query_metrics. Prometheus retention is about 10 days — if the "
    "window is older than that, say so. If query_metrics reports Prometheus is unconfigured or "
    "unreachable, state that plainly instead of guessing.\n"
    "- DO NOT INTERROGATE THE USER: Never ask for facts already present in this conversation (cluster, "
    "namespace, pod, instance and time are usually right there in the alert text). Re-read the conversation "
    "first. Ask a clarifying question ONLY when the information is genuinely absent and the task cannot "
    "proceed without it.\n\n"
    # "TOOL USAGE: Logs & Metrics\n"  # TODO: enable when log/metric tools are added
    "OUTPUT FORMATTING (TELEGRAM OPTIMIZED):\n"
    "- Use plain text only. No markdown, no headers, no bold, no backticks.\n"
    "- Be extremely concise. Get straight to the point.\n"
    "- Use short bullet points (-) for steps or lists.\n"
    "- Write commands and resource names as plain text (e.g. kubectl rollout restart deploy/foo).\n"
    "- Respond in the same language as the user's message.\n"
    "- All technical names (pod, deployment, namespace, node, service, ingress, etc.) always in English regardless of response language."
)

def _load_extra_prompt() -> str:
    """Load optional prompt extension from a mounted ConfigMap file."""
    path = os.environ.get("EXTRA_PROMPT_FILE", "/app/config/prompt_extra.txt")
    try:
        text = Path(path).read_text().strip()
        if text:
            logger.info("Loaded extra prompt from %s (%d chars)", path, len(text))
            return "\n\n" + text
    except FileNotFoundError:
        pass
    except Exception:
        logger.exception("Failed to load extra prompt from %s", path)
    return ""


SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT + _build_routing_rules() + _load_extra_prompt()
logger.debug("System prompt routing section:\n%s", SYSTEM_PROMPT[len(_BASE_SYSTEM_PROMPT):])

def _on_stderr(line: str) -> None:
    logger.error("Claude CLI stderr: %s", line.rstrip())


_MAX_TURNS = int(os.environ.get("CLAUDE_MAX_TURNS", "10"))

OPTIONS = ClaudeAgentOptions(
    system_prompt=SYSTEM_PROMPT,
    model="claude-haiku-4-5-20251001",
    permission_mode="bypassPermissions",
    allowed_tools=[
        "Bash", "Agent",
        "mcp__infra__search_infrastructure",
        "mcp__infra__get_logs",
        "mcp__infra__query_metrics",
        "mcp__infra__get_alerts",
    ],
    disallowed_tools=["Write", "Edit", "NotebookEdit"],  # never write files
    max_turns=_MAX_TURNS,
    cwd="/app",
    setting_sources=["project"],
    hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[pre_tool_use_guard])]},
    stderr=_on_stderr,
    agents={
        "haiku": AgentDefinition(
            description="Executes routine operational tasks quickly and cheaply.",
            prompt=(
                "You are an efficient SRE operations assistant. "
                "Execute tasks quickly and report findings concisely. "
                + _SUBAGENT_PROMPT_SUFFIX
            ),
            tools=[
                "Bash",
                "mcp__infra__get_logs",
                "mcp__infra__query_metrics",
                "mcp__infra__get_alerts",
            ],
            model="haiku",
        ),
        "sonnet": AgentDefinition(
            description="Performs analysis, incident triage, and problem-solving.",
            prompt=(
                "You are an SRE incident response expert. "
                "Analyze problems thoroughly and provide clear, prioritized action steps. "
                + _SUBAGENT_PROMPT_SUFFIX
            ),
            tools=[
                "Bash",
                "mcp__infra__get_logs",
                "mcp__infra__query_metrics",
                "mcp__infra__get_alerts",
            ],
            model="sonnet",
        ),
        "opus": AgentDefinition(
            description="Handles complex analysis, postmortems, and architectural decisions.",
            prompt=(
                "You are a senior SRE architect. "
                "Perform deep analysis and think through all implications carefully. "
                + _SUBAGENT_PROMPT_SUFFIX
            ),
            tools=[
                "Bash",
                "mcp__infra__get_logs",
                "mcp__infra__query_metrics",
                "mcp__infra__get_alerts",
            ],
            model="opus",
        ),
    },
)

# ── Conversation continuation ──────────────────────────────────────────────────

# Every query() call starts a fresh Claude session unless we resume an existing one.
# Without this a follow-up message ("check the monitoring for that alert") arrives
# with an empty context and the agent has to ask the user for facts it already had.
_RESUME_SUPPORTED = dataclasses.is_dataclass(ClaudeAgentOptions) and any(
    f.name == "resume" for f in dataclasses.fields(ClaudeAgentOptions)
)
if not _RESUME_SUPPORTED:
    logger.warning(
        "Installed claude-agent-sdk has no 'resume' option — "
        "follow-up messages will start a new session and lose context"
    )

# API errors that mean "retrying won't help" — never fall back to a fresh session for these.
_FATAL_API_ERRORS = {
    "billing_error", "rate_limit_error", "authentication_error", "overloaded_error",
}


def _options_for(session_id: str | None) -> ClaudeAgentOptions:
    """Base options, plus `resume` when continuing an existing conversation."""
    if session_id and _RESUME_SUPPORTED:
        return dataclasses.replace(OPTIONS, resume=session_id)
    return OPTIONS


# ── Model label helpers ────────────────────────────────────────────────────────

_MODEL_LABELS = {"haiku": "Haiku", "sonnet": "Sonnet", "opus": "Opus"}


def _short_model(raw_name: str) -> str:
    lower = raw_name.lower()
    for key, label in _MODEL_LABELS.items():
        if key in lower:
            return label
    return raw_name


# ── Main query ─────────────────────────────────────────────────────────────────

async def _as_stream(text: str):
    """Wrap a plain string as AsyncIterable — streaming input mode, required for hooks."""
    yield {"type": "user", "message": {"role": "user", "content": text}}


async def _run_query(
    prompt: str,
    session_id: str | None,
) -> tuple[str, str | None, str | None]:
    """One query() run. Returns (response_text, cost_info, session_id)."""
    response_parts: list[str] = []
    cost_info: str | None = None
    models_seen: list[str] = []  # ordered, deduped
    new_session_id: str | None = None

    async for message in query(prompt=_as_stream(prompt), options=_options_for(session_id)):
        logger.debug("SDK message: type=%s %s", type(message).__name__, vars(message))
        if isinstance(message, AssistantMessage):
            error_code = getattr(message, "error", None)
            if error_code:
                logger.error("AssistantMessage error: %s", error_code)
                raise _make_api_error(error_code)
            raw_model = getattr(message, "model", None)
            if raw_model and raw_model != "<synthetic>":
                label = _short_model(raw_model)
                if label not in models_seen:
                    models_seen.append(label)
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_parts.append(block.text)
        elif isinstance(message, ResultMessage):
            duration_s = message.duration_ms / 1000
            cost = message.total_cost_usd or 0
            turns = message.num_turns
            models_str = " + ".join(models_seen) if models_seen else "?"
            cost_info = f"{models_str} · ${cost:.4f} · {duration_s:.1f}s · {turns} turns"
            new_session_id = message.session_id
            if message.is_error:
                logger.error("Session %s error: subtype=%s", message.session_id, message.subtype)
                raise _make_api_error(message.subtype or "session_error")
            logger.info("Session %s finished: %s", message.session_id, cost_info)

    text = "".join(response_parts) or "No response from Claude."
    return text, cost_info, new_session_id


async def ask_claude(
    prompt: str,
    session_id: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Send a prompt to Claude, continuing `session_id` when given.

    Returns (response_text, cost_info, session_id) — pass the returned session id
    back on the next message to keep the conversation context.

    A stored session can be gone (pod restart wipes the CLI's session files), so a
    failed resume falls back to a fresh session instead of breaking the chat.
    """
    try:
        return await _run_query(prompt, session_id)
    except ClaudeAPIError as exc:
        if session_id and exc.code not in _FATAL_API_ERRORS:
            logger.warning(
                "Resume of session %s failed (%s) — retrying with a fresh session",
                session_id, exc.code,
            )
            return await _run_query(prompt, None)
        raise
    except Exception:
        if session_id:
            logger.exception(
                "Resume of session %s failed — retrying with a fresh session", session_id
            )
            return await _run_query(prompt, None)
        raise
