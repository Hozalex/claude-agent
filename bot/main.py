import logging
import os

import anyio

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from bot.agent import ClaudeAPIError, ask_claude

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


@dp.message(CommandStart())
async def handle_start(message: Message) -> None:
    await message.answer(
        "👋 SRE Assistant powered by Claude.\n\n"
        "I can help with:\n"
        "• Kubernetes cluster health checks\n"
        "• Alert triage and incident analysis\n"
        "• Root cause analysis and postmortems\n\n"
        "Just describe the problem or ask me to check the cluster.\n\n"
        "─\n"
        "💳 Billing: https://console.anthropic.com/settings/billing\n"
        "📊 Usage: https://console.anthropic.com/settings/usage"
    )


async def _run_claude(message: Message, prompt: str) -> None:
    """Send prompt to Claude and reply with the result."""
    thinking = await message.answer("⏳ Thinking...")
    try:
        reply, cost_info = await ask_claude(prompt)
    except ClaudeAPIError as exc:
        logger.error("Claude API error [%s]: %s", exc.code, exc.user_message)
        await message.answer(exc.user_message)
        return
    except Exception:
        logger.exception("Error calling Claude")
        await message.answer("❌ An error occurred. Please try again.")
        return
    finally:
        await thinking.delete()

    if cost_info:
        await message.answer(f"{reply}\n\n─\n🔢 {cost_info}")
    else:
        await message.answer(reply)


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
    await _run_claude(message, prompt)


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
