import logging
import os
import re
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition
from claude_agent_sdk.types import (
    AssistantMessage,
    TextBlock,
    ResultMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
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
# Checked as substrings of the full command (case-insensitive).
BLOCKED_BASH_PATTERNS: list[str] = [
    # Kubernetes — destructive mutations
    "kubectl delete",
    "kubectl apply",
    "kubectl create",
    "kubectl patch",
    "kubectl edit",
    "kubectl replace",
    "kubectl rollout restart",
    "kubectl rollout undo",
    "kubectl scale",
    "kubectl drain",
    "kubectl cordon",
    "kubectl uncordon",
    "kubectl taint",
    # Kubernetes — sensitive data
    "kubectl get secret",
    "kubectl get secrets",
    "kubectl describe secret",
    "kubectl exec",          # shell into pods
    "kubectl cp",            # copy files from/to pods
    "kubectl proxy",         # exposes API server
    "kubectl port-forward",  # network exposure
    # Shell — file system
    "rm ",
    "rm\t",
    "rmdir",
    "shred",
    " > /",                  # redirect writes to absolute paths
    "tee /",
    # Shell — privilege escalation
    "sudo ",
    "su ",
    "chmod ",
    "chown ",
    # Secrets — environment variables
    "printenv",
    "env ",
    "env\t",
    "env\n",
    "DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "BOT_TOKEN",
    # Secrets — sensitive filesystem paths
    "/proc/self/environ",   # env vars via procfs
    "/proc/self/mem",       # process memory
    "/var/run/secrets/",    # k8s service account token
    "~/.ssh/",              # SSH keys
    "~/.aws/",              # AWS credentials
    "~/.kube/",             # kubeconfig (use in-cluster SA instead)
    # Secrets — bot source and config
    "/app/bot/",            # bot source code (contains logic, not secrets, but unnecessary)
    "/app/.env",            # env file if present
]

# Compiled once at module load for fast case-insensitive matching in can_use_tool.
# Replaces the per-call loop that repeatedly called pattern.lower() on all 40+ entries.
_BLOCKED_BASH_REGEX: re.Pattern[str] = re.compile(
    "|".join(re.escape(p) for p in BLOCKED_BASH_PATTERNS),
    re.IGNORECASE,
)


async def can_use_tool(
    tool_name: str,
    input_data: dict,
    context: ToolPermissionContext,
) -> PermissionResultAllow | PermissionResultDeny:
    """Block dangerous bash commands before execution."""
    if tool_name == "Bash":
        command = input_data.get("command", "")
        m = _BLOCKED_BASH_REGEX.search(command)
        if m:
            pattern = m.group().lower()  # lower() recovers the original pattern (all patterns are lowercase)
            logger.warning("Blocked command: %s", command[:120])
            return PermissionResultDeny(
                message=(
                    f"Blocked: pattern '{pattern}' is not allowed. "
                    "Only read-only operations are permitted."
                ),
                interrupt=True,
            )
    return PermissionResultAllow(updated_input=input_data)


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
    "- If kubectl fails with connection errors, certificate errors, or timeout — explicitly tell the user "
    "which cluster is unreachable (e.g. 'Cannot connect to development-cluster'). "
    "Do NOT silently answer from general knowledge if live cluster data was needed.\n\n"
    # "TOOL USAGE: Logs & Metrics\n"  # TODO: enable when log/metric tools are added
    "OUTPUT FORMATTING (TELEGRAM OPTIMIZED):\n"
    "- Use plain text only. No markdown, no headers, no bold, no backticks.\n"
    "- Be extremely concise. Get straight to the point.\n"
    "- Use short bullet points (-) for steps or lists.\n"
    "- Write commands and resource names as plain text (e.g. kubectl rollout restart deploy/foo)."
)

SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT + _build_routing_rules()
logger.debug("System prompt routing section:\n%s", SYSTEM_PROMPT[len(_BASE_SYSTEM_PROMPT):])

def _on_stderr(line: str) -> None:
    logger.error("Claude CLI stderr: %s", line.rstrip())


_MAX_TURNS = int(os.environ.get("CLAUDE_MAX_TURNS", "10"))

OPTIONS = ClaudeAgentOptions(
    system_prompt=SYSTEM_PROMPT,
    model="claude-haiku-4-5-20251001",
    permission_mode="bypassPermissions",
    allowed_tools=["Bash", "Agent", "mcp__infra__search_infrastructure"],
    disallowed_tools=["Write", "Edit", "NotebookEdit"],  # never write files
    max_turns=_MAX_TURNS,
    cwd="/app",
    setting_sources=["project"],
    can_use_tool=can_use_tool,
    stderr=_on_stderr,
    agents={
        "haiku": AgentDefinition(
            description="Executes routine operational tasks quickly and cheaply.",
            prompt=(
                "You are an efficient SRE operations assistant. "
                "Execute tasks quickly and report findings concisely. "
                + _SUBAGENT_PROMPT_SUFFIX
            ),
            tools=["Bash"],
            model="haiku",
        ),
        "sonnet": AgentDefinition(
            description="Performs analysis, incident triage, and problem-solving.",
            prompt=(
                "You are an SRE incident response expert. "
                "Analyze problems thoroughly and provide clear, prioritized action steps. "
                + _SUBAGENT_PROMPT_SUFFIX
            ),
            tools=["Bash"],
            model="sonnet",
        ),
        "opus": AgentDefinition(
            description="Handles complex analysis, postmortems, and architectural decisions.",
            prompt=(
                "You are a senior SRE architect. "
                "Perform deep analysis and think through all implications carefully. "
                + _SUBAGENT_PROMPT_SUFFIX
            ),
            tools=["Bash"],
            model="opus",
        ),
    },
)

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
    """Wrap a plain string as AsyncIterable — required for can_use_tool."""
    yield {"type": "user", "message": {"role": "user", "content": text}}


async def ask_claude(prompt: str) -> tuple[str, str | None]:
    """Send a prompt to Claude. Returns (response_text, cost_info)."""
    response_parts: list[str] = []
    cost_info: str | None = None
    models_seen: list[str] = []  # ordered, deduped

    async for message in query(prompt=_as_stream(prompt), options=OPTIONS):
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
            if message.is_error:
                logger.error("Session %s error: subtype=%s", message.session_id, message.subtype)
                raise _make_api_error(message.subtype or "session_error")
            logger.info("Session %s finished: %s", message.session_id, cost_info)

    text = "".join(response_parts) or "No response from Claude."
    return text, cost_info
