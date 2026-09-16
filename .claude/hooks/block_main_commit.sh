#!/usr/bin/env bash
# PreToolUse hook (matcher: Bash). Blocks `git commit` while checked out on
# main/master, enforcing the "always branch before committing" rule in
# CLAUDE.md's Git workflow section -- a technical backstop, since relying on
# an instruction alone has already been observed to fail in this project.
set -euo pipefail

payload="$(cat)"
cmd="$(echo "$payload" | jq -r '.tool_input.command // empty')"

# Match `git commit` as its own command, even when chained (&&, ;, |).
if echo "$cmd" | grep -Eq '(^|[;&|]) *git +commit\b'; then
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
  if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then
    reason="Direct commits to '$branch' are blocked (see CLAUDE.md Git workflow). Create a feature branch first, e.g.: git checkout -b <name>"
    jq -n --arg reason "$reason" '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $reason}}'
    exit 0
  fi
fi

exit 0
