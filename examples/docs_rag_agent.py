"""Expose the docs-rag-mcp pipeline as an HTTP agent that AgentProbe can evaluate.

Run it with the *docs-rag-mcp* virtualenv (it needs chromadb, which AgentProbe itself does not):

    DOCS_RAG_PATH=~/Downloads/docs-rag-mcp \
      ~/Downloads/docs-rag-mcp/.venv/bin/python examples/docs_rag_agent.py --port 9000

Then evaluate it. Query-string parameters override the pipeline's settings per run, so a
configuration change is just a different agent spec:

    agentprobe run --dataset examples/docs_rag_hard.json --agent "http:http://localhost:9000/agent?k=3"
    agentprobe run --dataset examples/docs_rag_hard.json --agent "http:http://localhost:9000/agent?k=1"

The pipeline is used unmodified: this file only wraps `rag.answer.answer`. It reports the ranked
retrieval list in the output so deterministic judges can check *which document* was found, not
just that an answer was produced.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DOCS_RAG = Path(os.environ.get("DOCS_RAG_PATH", "~/Downloads/docs-rag-mcp")).expanduser()
sys.path.insert(0, str(DOCS_RAG))

from rag import answer as rag_answer
from rag.store import DocStore

LOCK = threading.Lock()  # chroma and the module-level threshold are shared state
STORE: DocStore | None = None


def handle_question(question: str, k: int, max_distance: float) -> dict:
    assert STORE is not None
    with LOCK:
        hits = STORE.search(question, k)
        previous = rag_answer.MAX_DISTANCE
        rag_answer.MAX_DISTANCE = max_distance  # the pipeline reads this when filtering hits
        try:
            result = rag_answer.answer(STORE, question, k)
        finally:
            rag_answer.MAX_DISTANCE = previous
    kept = [h for h in hits if h.distance <= max_distance]
    ranked = ", ".join(h.source for h in kept) or "(none)"
    output = f"{result.text}\nRetrieved (ranked): {ranked}\nGrounded: {result.grounded}"
    return {"output": output, "tool_calls": ["ask_docs"]}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        url = urlparse(self.path)
        params = {key: values[0] for key, values in parse_qs(url.query).items()}
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            payload = handle_question(
                body["input"], int(params.get("k", 4)), float(params.get("max_distance", 0.75))
            )
            status = 200
        except Exception as exc:  # noqa: BLE001 - report to the evaluator rather than crash
            payload, status = {"error": f"{type(exc).__name__}: {exc}"}, 500
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args) -> None:  # keep the console readable during evaluations
        pass


def main() -> None:
    global STORE
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--db-dir", default=os.environ.get("RAG_DB_DIR", ".docs_rag_chroma"))
    args = parser.parse_args()
    STORE = DocStore(args.db_dir)  # separate index dir: docs-rag-mcp's own .chroma is never touched
    count = STORE.index(DOCS_RAG / "docs")
    print(f"indexed {count} chunks from {DOCS_RAG / 'docs'}; serving on http://localhost:{args.port}/agent")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
