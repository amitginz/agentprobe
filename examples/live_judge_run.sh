#!/usr/bin/env bash
# Step 1 of calibrating the live LLM judge: run the docs-rag agent on the 62-question dataset and
# let the real judge grade it. Needs ANTHROPIC_API_KEY in the environment (never put it in a file
# in this repo) and a docs-rag-mcp checkout.
#
#   export ANTHROPIC_API_KEY=...        # in your own shell
#   DOCS_RAG_PATH=~/Downloads/docs-rag-mcp bash examples/live_judge_run.sh
#
# Then grade by hand and measure agreement:
#   agentprobe label --run <id> --n 40
#   agentprobe calibrate --run <id> --require-reliable
set -euo pipefail
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY in your environment first}"
DOCS_RAG_PATH="${DOCS_RAG_PATH:-$HOME/Downloads/docs-rag-mcp}"
PORT="${PORT:-9000}"
DB="${DATABASE_URL:-sqlite:///agentprobe.db}"
LOG="$(mktemp)"

# The agent under test runs in docs-rag's own virtualenv (it needs chromadb).
# The key is deliberately NOT forwarded to it: the agent answers extractively, so any
# LLM spend in this experiment is the judge's alone.
env -u ANTHROPIC_API_KEY DOCS_RAG_PATH="$DOCS_RAG_PATH" \
  "$DOCS_RAG_PATH/.venv/bin/python" -u examples/docs_rag_agent.py --port "$PORT" >"$LOG" 2>&1 &
AGENT_PID=$!
trap 'kill $AGENT_PID 2>/dev/null || true; rm -f "$LOG"' EXIT
for _ in $(seq 1 60); do grep -q serving "$LOG" 2>/dev/null && break; sleep 2; done

agentprobe --db "$DB" run --dataset examples/docs_rag_llm.json \
  --agent "http:http://localhost:$PORT/agent?k=3" --label "live judge calibration" \
  --concurrency 2 --budget "${JUDGE_BUDGET_USD:-1.00}"
