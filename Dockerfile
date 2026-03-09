# ── Stage 1: download kubectl + install Python deps ───────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x kubectl

WORKDIR /deps
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/deps/packages -r requirements.txt


# ── Stage 2: final image ───────────────────────────────────────────────────────
FROM python:3.12-slim

# ca-certificates needed for HTTPS (Telegram, Anthropic APIs)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# kubectl binary only — no curl in final image
COPY --from=builder /kubectl /usr/local/bin/kubectl

# Python packages
COPY --from=builder /deps/packages /usr/local

# Non-root user — required by Claude Code CLI (bypassPermissions blocked for root)
RUN useradd -m -u 1000 bot

WORKDIR /app
COPY bot/ ./bot/
COPY .claude/ ./.claude/
RUN chown -R bot:bot /app

USER bot

CMD ["python", "-m", "bot.main"]
