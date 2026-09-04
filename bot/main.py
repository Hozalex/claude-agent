import logging
import os
import time

import anyio

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from dotenv import load_dotenv

from bot.agent import ClaudeAPIError, ask_claude
from bot.status import get_status

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

dp = Dispatcher()


def _get_message_text(message: Message) -> str | None:
    """Return the text or caption of a Telegram message, whichever is present."""
    return message.text or message.caption


# ── Conversation sessions ──────────────────────────────────────────────────────
#
# Claude starts a fresh session per query unless we resume one, so a follow-up
# ("check the monitoring for that alert") would otherwise arrive with no context.
# We keep the last session id per chat and resume it.
#
# Kept in memory on purpose: the CLI's own session files live in the pod, so a
# restart invalidates them anyway — surviving a restart here would buy nothing.
_SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))

_sessions: dict[int, tuple[str, float]] = {}  # chat_id -> (session_id, last_used)


def _get_session(chat_id: int) -> str | None:
    entry = _sessions.get(chat_id)
    if entry is None:
        return None
    session_id, last_used = entry
    if time.monotonic() - last_used > _SESSION_TTL:
        _sessions.pop(chat_id, None)
        logger.info("Session for chat_id=%s expired (TTL %ss)", chat_id, _SESSION_TTL)
        return None
    return session_id


def _store_session(chat_id: int, session_id: str | None) -> None:
    if session_id:
        _sessions[chat_id] = (session_id, time.monotonic())


def _clear_session(chat_id: int) -> None:
    _sessions.pop(chat_id, None)


@dp.message(Command("status"))
async def handle_status(message: Message) -> None:
    checking = await message.answer("Checking...")
    try:
        text = await get_status()
    except Exception:
        logger.exception("Status check failed")
        text = "Failed to collect status."
    finally:
        await checking.delete()
    await message.answer(text)


@dp.message(Command("reset"))
async def handle_reset(message: Message) -> None:
    """Drop the stored conversation so the next message starts clean."""
    _clear_session(message.chat.id)
    await message.answer("🧹 Context cleared. The next message starts a new conversation.")


@dp.message(CommandStart())
async def handle_start(message: Message) -> None:
    _clear_session(message.chat.id)
    await message.answer(
        "👋 SRE Assistant powered by Claude.\n\n"
        "I can help with:\n"
        "• Kubernetes cluster health checks\n"
        "• Alert triage and incident analysis\n"
        "• Root cause analysis and postmortems\n\n"
        "Just describe the problem or ask me to check the cluster.\n"
        "Follow-up questions keep the context. Reply to an alert to triage it.\n"
        "/reset — start a new conversation\n\n"
        "─\n"
        "💳 Billing: https://console.anthropic.com/settings/billing\n"
        "📊 Usage: https://console.anthropic.com/settings/usage"
    )


# Telegram rejects a text message over 4096 characters with "message is too long",
# which used to escape the handler and leave the user with no reply at all.
_TG_TEXT_LIMIT = 3900  # under the 4096 cap, with room for multi-byte characters


def _split_for_telegram(text: str, limit: int = _TG_TEXT_LIMIT) -> list[str]:
    """Split a reply into Telegram-sized chunks, preferring line boundaries."""
    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:  # a single line longer than the limit
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks or [""]


async def _send(message: Message, text: str) -> None:
    """Reply, splitting long text so Telegram never rejects the whole answer."""
    for chunk in _split_for_telegram(text):
        try:
            await message.answer(chunk)
        except Exception:
            logger.exception("Failed to send a reply chunk to chat_id=%s", message.chat.id)
            return


async def _run_claude(message: Message, prompt: str, *, new_session: bool = False) -> None:
    """Send prompt to Claude and reply with the result.

    `new_session=True` starts a clean conversation (used for alert triage, so two
    unrelated incidents in the same chat never share context). Otherwise the chat's
    previous session is resumed so follow-up questions keep their context.
    """
    chat_id = message.chat.id
    if new_session:
        _clear_session(chat_id)
    session_id = None if new_session else _get_session(chat_id)

    thinking = await message.answer("⏳ Thinking...")
    try:
        reply, cost_info, new_session_id = await ask_claude(prompt, session_id)
        _store_session(chat_id, new_session_id)
    except ClaudeAPIError as exc:
        logger.error("Claude API error [%s]: %s", exc.code, exc.user_message)
        await _send(message, exc.user_message)
        return
    except Exception:
        logger.exception("Error calling Claude")
        await _send(message, "❌ An error occurred. Please try again.")
        return
    finally:
        try:
            await thinking.delete()
        except Exception:
            logger.warning("Could not delete the 'Thinking...' message", exc_info=True)

    body = f"{reply}\n\n─\n🔢 {cost_info}" if cost_info else reply
    await _send(message, body)


def _build_alert_prompt(alert_text: str, user_command: str) -> str:
    """Combine alert content and user command into a structured prompt."""
    cmd = user_command.lstrip("/").strip() or "triage this alert"
    return (
        f"The following alert was received from the monitoring system:\n\n"
        f"---\n{alert_text.strip()}\n---\n\n"
        f"User request: {cmd}"
    )


# Reply to an alert message with any text to trigger alert triage
@dp.message(F.reply_to_message & (F.text | F.caption))
async def handle_alert_reply(message: Message) -> None:
    replied = message.reply_to_message
    alert_text = _get_message_text(replied)  # type: ignore[arg-type]
    if not alert_text:
        await message.answer("⚠️ The message you replied to has no text.")
        return

    user_command = _get_message_text(message) or ""
    logger.info(
        "Alert triage request from chat_id=%s, alert=%s",
        message.chat.id,
        alert_text[:80],
    )
    prompt = _build_alert_prompt(alert_text, user_command)
    # A reply to an alert is a new incident — start a clean session for it.
    await _run_claude(message, prompt, new_session=True)


# Catch both plain text and commands (e.g. "/check cluster health")
@dp.message(F.text | F.caption)
async def handle_message(message: Message) -> None:
    text = _get_message_text(message)
    assert text is not None
    logger.info("Message from chat_id=%s: %s", message.chat.id, text[:80])

    # Strip leading slash so Claude Code doesn't treat input as a slash command
    prompt = text.lstrip("/")
    await _run_claude(message, prompt)


async def main() -> None:
    bot = Bot(token=TOKEN)
    logger.info("Starting bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    anyio.run(main, backend="asyncio")
