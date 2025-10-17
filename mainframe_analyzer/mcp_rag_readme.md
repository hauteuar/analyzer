# ================================
# File: README.md
# ================================
README = r"""
# Tree-sitter + Graph + RAG MCP Bundle (Prototype)

This bundle contains three main components:

1. `parse_and_build_graph.py` - parse COBOL, build graph, build FAISS indices (code + docs), store metadata in SQLite
2. `mcp_rag_agent.py` - run as an MCP tool (JSON-RPC over STDIO) to query code index, doc index and graph
3. `flow_generator.py` - generate Mermaid diagrams (or basic flow export) for a given node from the graph

## Quickstart

Install dependencies:

```bash
pip install sentence-transformers faiss-cpu numpy networkx sqlite-utils
```

Optional (for better parsing): install tree-sitter and build the COBOL grammar:

```bash
pip install tree_sitter
# then clone and build tree-sitter-cobol into build/my-languages.so (follow tree-sitter docs)
```

### 1) Build the graph & indexes

```bash
python parse_and_build_graph.py --src ./cobol_src --docs ./docs
```

Outputs:
- `faiss_code.index`
- `faiss_doc.index`
- `metadata.db` (sqlite with `snippets` and `docs` tables)
- `code_graph.gml` (NetworkX graph)

### 2) Run MCP Agent (for Copilot)

Run the agent (it listens on stdin/stdout for JSON-RPC newline-delimited requests):

```bash
python mcp_rag_agent.py
```

Configure Copilot MCP server (example `settings.json`):

```json
"mcpServers": {
  "rag-agent": {
    "command": "python",
    "args": ["/path/to/mcp_rag_agent.py"],
    "transport": "stdio"
  }
}
```

Then Copilot can call methods like `search_code`, `search_docs`, `graph_neighbors`, `flow_mermaid`.

### 3) Generate visual flow

```bash
python flow_generator.py --node prog:UPDATEADDR --depth 2 --out flow.txt
```

You can paste the generated Mermaid text into any Mermaid renderer (VSCode plugin, mermaid.live) or translate to draw.io XML using online tools.

## How retrieval works (concept)

1. Offline: tree-sitter (or heuristics) parse COBOL to extract code chunks. Embeddings are created with a sentence-transformers model and stored in FAISS (normalized for cosine search). A graph of program relationships is created with NetworkX.
2. Runtime: the MCP agent loads the FAISS indexes and the graph. For a query it:
   - Searches code index (semantic) and docs index.
   - Optionally expands context by traversing the graph for related nodes.
   - Returns structured hits and optionally a Mermaid flow for explanation.

## Next steps / Improvements

- Add explicit FAISS-to-SQLite id mapping and persistent idmap table for robust retrieval order mapping.
- Add copybook expansion (resolve COPYs) during parsing.
- Implement hybrid retrieval: SQLite FTS for exact identifier matches + FAISS for semantic search.
- Add a resolver method to infer dynamic calls (e.g., TMS + CAR) and populate `dynamic_call` edges.
- For large corpora, replace NetworkX with Neo4j or TigerGraph for scalable graph queries.

"""

# Write README to disk for convenience
Path('README_generated.md').write_text(README)

print('Bundle generated: parse_and_build_graph.py, mcp_rag_agent.py, flow_generator.py, README_generated.md')
