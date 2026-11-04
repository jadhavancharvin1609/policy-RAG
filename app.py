"""A small, local, safety-first RAG assistant with source citations.

This is an extractive RAG example: it retrieves policy passages and presents
only grounded excerpts. In a production app, replace the retrieval method and
optionally add an approved LLM with the same grounding and evaluation rules.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]{1,}")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in", "is", "it", "of", "on",
    "or", "that", "the", "this", "to", "what", "when", "where", "with", "you", "your", "do", "can", "my", "we",
}


@dataclass(frozen=True)
class Chunk:
    source: str
    heading: str
    text: str


def tokens(text: str) -> list[str]:
    return [word.lower() for word in WORD.findall(text) if word.lower() not in STOP_WORDS]


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(DOCS.glob("*.md")):
        heading = path.stem.replace("-", " ").title()
        buffer: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                if buffer:
                    chunks.append(Chunk(path.name, heading, " ".join(buffer).strip()))
                    buffer = []
                heading = line.lstrip("#").strip()
            elif line.strip():
                buffer.append(line.strip())
            elif buffer:
                chunks.append(Chunk(path.name, heading, " ".join(buffer).strip()))
                buffer = []
        if buffer:
            chunks.append(Chunk(path.name, heading, " ".join(buffer).strip()))
    if not chunks:
        raise FileNotFoundError("No Markdown policy documents were found in docs/.")
    return chunks


class LocalRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        document_frequency: Counter[str] = Counter()
        self.counts = [Counter(tokens(chunk.text + " " + chunk.heading)) for chunk in chunks]
        for count in self.counts:
            document_frequency.update(count.keys())
        self.idf = {term: math.log((1 + len(chunks)) / (1 + frequency)) + 1 for term, frequency in document_frequency.items()}
        self.vectors = [self._vector(count) for count in self.counts]

    def _vector(self, count: Counter[str]) -> dict[str, float]:
        values = {term: frequency * self.idf.get(term, 0.0) for term, frequency in count.items() if term in self.idf}
        length = math.sqrt(sum(value * value for value in values.values())) or 1.0
        return {term: value / length for term, value in values.items()}

    def search(self, question: str, limit: int = 3) -> list[tuple[Chunk, float]]:
        query = self._vector(Counter(tokens(question)))
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            score = sum(query.get(term, 0.0) * value for term, value in vector.items())
            results.append((chunk, score))
        return sorted(results, key=lambda item: item[1], reverse=True)[:limit]


def best_sentence(text: str, question: str) -> str:
    query_terms = set(tokens(question))
    candidates = SENTENCE.split(text)
    return max(candidates, key=lambda sentence: len(query_terms.intersection(tokens(sentence)))).strip()


def answer_question(retriever: LocalRetriever, question: str) -> dict[str, object]:
    question = question.strip()
    if not question:
        return {"answer": "Ask a policy question, for example: How do I submit a travel expense?", "sources": []}
    results = retriever.search(question)
    if not results or results[0][1] < 0.08:
        return {
            "answer": "I could not find enough approved policy text to answer that. Please contact the relevant team or add an approved document.",
            "sources": [],
        }
    sources = []
    excerpts = []
    for chunk, score in results:
        if score < 0.04:
            continue
        excerpt = best_sentence(chunk.text, question)
        excerpts.append(f"{chunk.heading}: {excerpt}")
        sources.append({"document": chunk.source, "section": chunk.heading, "excerpt": excerpt, "relevance": round(score, 3)})
    return {
        "answer": "Based on the approved policy passages, here is the relevant guidance:\n\n" + "\n\n".join(excerpts),
        "sources": sources,
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grounded Policy Assistant</title><style>
body{font-family:Arial,sans-serif;background:#f8fafc;color:#172033;margin:0;padding:32px;line-height:1.5}main{max-width:820px;margin:auto}.subtitle{color:#5b6472}section,.card{background:white;border-radius:12px;padding:20px;box-shadow:0 1px 4px #cbd5e155;margin:18px 0}form{display:flex;gap:10px}input{flex:1;padding:12px;border:1px solid #94a3b8;border-radius:7px;font-size:1rem}button{background:#1d4ed8;color:white;border:0;border-radius:7px;padding:12px 18px;font-weight:bold;cursor:pointer}.answer{white-space:pre-wrap}.source{border-left:3px solid #60a5fa;padding-left:12px;margin:12px 0}.notice{background:#eff6ff;border-left:4px solid #2563eb;padding:12px;border-radius:4px}small{color:#5b6472}@media(max-width:600px){body{padding:18px}form{display:block}button{width:100%;margin-top:10px}}</style></head>
<body><main><h1>Grounded Policy Assistant</h1><p class="subtitle">Local RAG starter - answers are limited to retrieved policy passages and include sources.</p>
<section class="notice"><strong>Use responsibly:</strong> check the linked source and contact the policy owner for decisions or exceptions.</section>
<section><form id="ask"><input id="question" aria-label="Policy question" value="How do I submit a travel expense?" required><button>Ask</button></form></section>
<section id="result" hidden><h2>Answer</h2><div id="answer" class="answer"></div><h3>Sources to review</h3><div id="sources"></div></section>
<script>
const form=document.querySelector('#ask'), result=document.querySelector('#result'), answer=document.querySelector('#answer'), sources=document.querySelector('#sources');
form.addEventListener('submit', async (event)=>{event.preventDefault();answer.textContent='Searching approved documents...';sources.innerHTML='';result.hidden=false;const q=document.querySelector('#question').value;const response=await fetch('/api/ask?q='+encodeURIComponent(q));const data=await response.json();answer.textContent=data.answer;for(const source of data.sources){const box=document.createElement('div');box.className='source';box.innerHTML='<strong>'+source.section+'</strong> <small>('+source.document+')</small><br>'+source.excerpt;sources.appendChild(box);}});
</script></main></body></html>"""


class PolicyHandler(BaseHTTPRequestHandler):
    retriever: LocalRetriever

    def send_json(self, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - method name is required by BaseHTTPRequestHandler
        request = urlparse(self.path)
        if request.path == "/":
            encoded = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        if request.path == "/api/ask":
            question = parse_qs(request.query).get("q", [""])[0]
            self.send_json(answer_question(self.retriever, question))
            return
        self.send_error(404, "Page not found")

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local, grounded policy RAG assistant.")
    parser.add_argument("--question", help="Answer a question in the terminal instead of starting the web UI.")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = LocalRetriever(load_chunks())
    if args.question:
        response = answer_question(retriever, args.question)
        print(response["answer"])
        for source in response["sources"]:
            print(f"- {source['section']} ({source['document']})")
        return
    PolicyHandler.retriever = retriever
    server = ThreadingHTTPServer(("127.0.0.1", args.port), PolicyHandler)
    print(f"Policy assistant running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
