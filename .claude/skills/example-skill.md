---
description: Example skill template. Replace with your own description — this is used by the agent to match incoming requests to this skill.
agent: haiku
---

# Example Skill

This is a template for creating new skills.

## Frontmatter fields

- `description` — how the agent recognizes when to use this skill (used for routing)
- `agent` — which subagent to delegate to: `haiku` (routine tasks), `sonnet` (analysis), `opus` (deep/postmortem)

## Instructions

Write clear step-by-step instructions for the subagent here.
Include exact commands, expected output format, and response structure.

## Response format

Describe the expected output format for Telegram:
- First line: status summary
- Bullet list of findings
- Max N lines
