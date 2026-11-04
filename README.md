# Grounded Policy RAG Assistant

A local, safety-first Retrieval-Augmented Generation (RAG) starter for company policies. It retrieves relevant passages from the `docs/` folder, answers only with those passages, and shows the source of every result.

This implementation intentionally uses no external LLM or API key. That makes it easy to run and demonstrates the core RAG workflow: document ingestion, chunking, vectorisation, retrieval, grounded response, citations, and a user interface.

## Run it

```powershell
python app.py
```

Then open <http://localhost:8000>. Stop the server with `Ctrl+C`.

To use it without the web interface:

```powershell
python app.py --question "How do I submit a travel expense?"
```

## Safety design

- It never treats document contents as program commands.
- It returns “not found” when retrieval is weak, rather than inventing an answer.
- It gives document citations for review.
- It should use access-controlled, approved documents in a real organisation.

## Portfolio upgrade ideas

- Replace keyword TF-IDF retrieval with embedding search, hybrid search, and reranking.
- Add authentication and document-level permissions.
- Add an approved LLM for answer synthesis, plus evaluation, feedback review, and hallucination checks.
- Create prompt-injection tests and audit logs.
