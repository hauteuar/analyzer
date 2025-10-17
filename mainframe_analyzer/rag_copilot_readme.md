

# ---------------------------------------------------------------------
# README.md (below) - included in this single textdoc for convenience
# ---------------------------------------------------------------------

"""
# RAG Agent (FAISS + SQLite) exposed as MCP (stdio)

This repository contains a minimal prototype RAG agent implemented in Python that:

- Parses COBOL source files (simple heuristic parser; optional tree-sitter can be integrated)
- Builds embeddings using `sentence-transformers`
- Stores metadata (file/program/section/text) in SQLite
- Stores vectors in FAISS and maps FAISS indices to SQLite rowids
- Exposes a JSON-RPC over **stdin/stdout** (MCP-friendly) with methods `index`, `search`, `info`, `rebuild`

---

## Files

- `rag_agent.py` - main MCP-compatible RAG agent server (this file).

---

## Requirements

Tested with Python 3.10+. Install dependencies:

```bash
pip install sentence-transformers faiss-cpu numpy
```

Optional (for higher-quality parsing): build tree-sitter with the COBOL grammar and modify `parse_cobol_chunks`

---

## Quickstart

1. Place your COBOL files under `./cobol_src/` (recursive). Example extensions: `.cbl`, `.cob`, `.cpy`.
2. Run the agent:

```bash
python rag_agent.py
```

3. Send JSON-RPC requests (newline-delimited) to stdin. Example using `jq`/`python` piping:

```bash
# index folder (this will parse all files and add them to SQLite + FAISS)
echo '{"id":1,"method":"index","params":{"path":"./cobol_src"}}' | python rag_agent.py

# search
echo '{"id":2,"method":"search","params":{"query":"update customer address","top_k":5}}' | python rag_agent.py

# info
echo '{"id":3,"method":"info"}' | python rag_agent.py
```

> **Note:** In practice, Copilot MCP will launch `python rag_agent.py` and talk JSON-RPC over STDIO. Configure your `settings.json` MCP server to run the command, e.g.:

```json
"mcpServers": {
  "rag-agent": {
    "command": "python",
    "args": ["/path/to/rag_agent.py"],
    "transport": "stdio"
  }
}
```

Then the Copilot/LLM can call methods `index`, `search`, `info`, `rebuild` as JSON-RPC.

---

## Example JSON-RPC method signatures

- `index` - params: `{ "path": "./cobol_src" }` -> indexes all COBOL files under path
- `search` - params: `{ "query": "update customer", "top_k": 5 }` -> performs vector search and returns hits
- `info` - params: `{}` -> returns counts
- `rebuild` - params: `{ "path": "./cobol_src" }` -> clears and rebuilds index

Each response follows JSON-RPC 2.0 and is newline-delimited.

---

## How retrieval works (high-level)

1. **Indexing**: `index` reads each COBOL file, splits it into chunks (paragraphs, EXEC SQL blocks), generates embeddings for each chunk, writes chunk metadata into SQLite, and pushes normalized embeddings to FAISS. A mapping table stores mapping from FAISS index id -> sqlite rowid.

2. **Searching**: `search` embeds the user query, searches FAISS (inner product on normalized vectors = cosine similarity), and maps the top FAISS ids back to sqlite rows to return file, program, section, snippet, and score.

3. **Expansion**: This prototype focuses on retrieval. To implement graph traversal (call graphs, dynamic resolution), build additional extractors that populate edge tables in SQLite (e.g., `calls(caller,callee)`, `assigns(var, value_location)`) and expose extra MCP methods like `resolve_dynamic_calls` which run logic to infer `TMS + CAR` style dynamic targets.

---

## Next steps / improvements

- Integrate `tree-sitter-cobol` for robust parsing of DATA/PROCEDURE DIVISION constructs, copybook expansion, and call/perform extraction.
- Add keyword + vector hybrid retrieval (SQLite FTS + FAISS) for high precision on identifiers.
- Populate call graph and IO metadata tables and expose methods so the agent can perform graph traversals on query time.
- Add authentication, access controls, logging, and rate limiting for enterprise use.

---

## Troubleshooting

- If you get an error about FAISS install, try `pip install faiss-cpu` or use an alternative like `chromadb`.
- If embeddings are large/slow, switch to a lighter (or heavier) SentenceTransformers model depending on throughput and accuracy needs.

---

## License

MIT

"""
