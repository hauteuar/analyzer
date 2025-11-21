# File: rag_agent.py
"""
A minimal MCP-compatible RAG agent over STDIO that:
- Indexes COBOL files (simple parser + optional tree-sitter)
- Creates embeddings (sentence-transformers)
- Stores metadata in SQLite and vectors in FAISS
- Exposes JSON-RPC methods over stdin/stdout: `index`, `search`, `info`, `rebuild`

NOTES:
- This is a prototype to plug into Copilot MCP as an external tool using stdio.
- For production, harden the JSON-RPC parsing, concurrency, incremental updates, and error handling.

Usage (quick):
  python rag_agent.py

Then send JSON-RPC requests (newline-delimited) to its stdin, e.g.:
  {"id":1,"method":"index","params":{"path":"./cobol_src"}}
  {"id":2,"method":"search","params":{"query":"update customer address","top_k":5}}

It will reply with JSON RPC responses on stdout.
"""

import os
import sys
import json
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import traceback
import re
from pathlib import Path

# -----------------------------
# Configuration / constants
# -----------------------------
DB_FILE = "cobol_index.db"
FAISS_FILE = "cobol.faiss"
IDMAP_TABLE = "id_map"  # maps faiss_idx -> sqlite rowid
EMBED_DIM = 384  # default for all-MiniLM-L6-v2 is 384
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# -----------------------------
# Utilities: simple COBOL parser
# -----------------------------

def parse_cobol_chunks(file_path):
    """Return list of chunks dict {file, program, section, text}.
    Uses simple heuristics: Program-ID, paragraphs (split by lines ending with dot), and EXEC SQL blocks.
    """
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    program_match = re.search(r'PROGRAM-ID\.?\s+(\w+)', text, flags=re.IGNORECASE)
    program = program_match.group(1) if program_match else Path(file_path).stem

    # Extract EXEC SQL blocks first
    chunks = []
    for m in re.finditer(r'EXEC\s+SQL([\s\S]*?)END-EXEC\.', text, flags=re.IGNORECASE):
        chunk_text = m.group(0).strip()
        if len(chunk_text) > 20:
            chunks.append({"file": str(file_path), "program": program, "section": "EXEC_SQL", "text": chunk_text})

    # Split paragraphs by lines that start at column and end with a dot (heuristic)
    paragraphs = re.split(r'(?<=\.)\n(?=\w)', text)
    for i, p in enumerate(paragraphs):
        p_clean = p.strip()
        if len(p_clean) < 40:
            continue
        # try to capture paragraph/section name
        mname = re.search(r'^(\w[\w-]*\.)', p_clean)
        section = mname.group(1).strip() if mname else f"para_{i}"
        chunks.append({"file": str(file_path), "program": program, "section": section, "text": p_clean})

    return chunks

# -----------------------------
# Storage: SQLite metadata + FAISS vector store
# -----------------------------
class IndexStore:
    def __init__(self, db_file=DB_FILE, faiss_file=FAISS_FILE, embed_dim=EMBED_DIM, model_name=EMBED_MODEL_NAME):
        self.db_file = db_file
        self.faiss_file = faiss_file
        self.embed_dim = embed_dim
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

        # sqlite
        self.conn = sqlite3.connect(self.db_file)
        self._ensure_tables()

        # faiss index
        self.index = None
        self._load_faiss()

    def _ensure_tables(self):
        cur = self.conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS code_index(
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                file TEXT,
                program TEXT,
                section TEXT,
                text TEXT
            );"""
        )
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {IDMAP_TABLE} (faiss_idx INTEGER PRIMARY KEY, rowid INTEGER UNIQUE);"
        )
        self.conn.commit()

    def _load_faiss(self):
        # try to load existing faiss index
        if os.path.exists(self.faiss_file):
            try:
                self.index = faiss.read_index(self.faiss_file)
                # if dimension mismatch, rebuild
                if self.index.d != self.embed_dim:
                    print(f"[WARN] FAISS dim {self.index.d} != embed_dim {self.embed_dim}, rebuilding index.", file=sys.stderr)
                    self.index = faiss.IndexFlatIP(self.embed_dim)
            except Exception as e:
                print("[WARN] Failed to read faiss index, creating new.", file=sys.stderr)
                self.index = faiss.IndexFlatIP(self.embed_dim)
        else:
            self.index = faiss.IndexFlatIP(self.embed_dim)

    def _save_faiss(self):
        faiss.write_index(self.index, self.faiss_file)

    def embed_texts(self, texts):
        # returns normalized vectors for cosine similarity using inner product
        embs = self.model.encode(texts, show_progress_bar=False)
        arr = np.array(embs).astype('float32')
        # normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr

    def add_chunks(self, chunks):
        cur = self.conn.cursor()
        texts = [c['text'] for c in chunks]
        embs = self.embed_texts(texts)

        # insert metadata first to get sqlite rowids
        rowids = []
        for c in chunks:
            cur.execute("INSERT INTO code_index(file, program, section, text) VALUES(?,?,?,?)",
                        (c['file'], c['program'], c['section'], c['text']))
            rowids.append(cur.lastrowid)
        self.conn.commit()

        # add to faiss and id_map
        n_before = int(self.index.ntotal)
        self.index.add(embs)
        # map faiss indices to sqlite rowids
        for i, rid in enumerate(rowids):
            faiss_idx = n_before + i
            cur.execute(f"INSERT OR REPLACE INTO {IDMAP_TABLE}(faiss_idx, rowid) VALUES(?,?)", (faiss_idx, rid))
        self.conn.commit()
        self._save_faiss()
        return len(rowids)

    def search(self, query, top_k=5):
        q_emb = self.embed_texts([query])
        if int(self.index.ntotal) == 0:
            return []
        D, I = self.index.search(q_emb, top_k)
        # I is shape (1, k)
        idxs = I[0].tolist()
        cur = self.conn.cursor()
        results = []
        for faiss_idx, score in zip(idxs, D[0].tolist()):
            if faiss_idx < 0:
                continue
            cur.execute(f"SELECT rowid, file, program, section, text FROM code_index WHERE rowid = (SELECT rowid FROM {IDMAP_TABLE} WHERE faiss_idx=?)", (faiss_idx,))
            r = cur.fetchone()
            if r:
                rowid, file, program, section, text = r
                results.append({
                    "rowid": rowid,
                    "file": file,
                    "program": program,
                    "section": section,
                    "text": text,
                    "score": float(score)
                })
        return results

    def count(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM code_index")
        total = cur.fetchone()[0]
        return {"total_chunks": total, "faiss_ntotal": int(self.index.ntotal)}

    def close(self):
        try:
            self._save_faiss()
        except Exception:
            pass
        self.conn.close()

# -----------------------------
# Simple JSON-RPC over stdin/stdout
# -----------------------------

def send_response(resp):
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()


def handle_request(req, store: IndexStore):
    try:
        method = req.get("method")
        params = req.get("params", {})
        req_id = req.get("id")

        if method == "index":
            path = params.get("path")
            if not path:
                raise ValueError("'path' is required for index")
            # scan folder
            all_chunks = []
            for root, _, files in os.walk(path):
                for f in files:
                    if f.lower().endswith(('.cbl', '.cob', '.cpy')):
                        try:
                            chunks = parse_cobol_chunks(os.path.join(root, f))
                            all_chunks.extend(chunks)
                        except Exception as e:
                            print(f"[WARN] parse failed for {f}: {e}", file=sys.stderr)
            added = store.add_chunks(all_chunks)
            result = {"status": "ok", "indexed_chunks": added}
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        elif method == "search":
            query = params.get("query")
            top_k = int(params.get("top_k", 5))
            if not query:
                raise ValueError("'query' is required for search")
            res = store.search(query, top_k=top_k)
            return {"jsonrpc": "2.0", "id": req_id, "result": {"hits": res}}

        elif method == "info":
            return {"jsonrpc": "2.0", "id": req_id, "result": store.count()}

        elif method == "rebuild":
            # clear tables and faiss, then index path
            path = params.get("path")
            if not path:
                raise ValueError("'path' is required for rebuild")
            # drop tables
            cur = store.conn.cursor()
            cur.execute("DELETE FROM code_index")
            cur.execute(f"DELETE FROM {IDMAP_TABLE}")
            store.conn.commit()
            # reset faiss
            store.index = faiss.IndexFlatIP(store.embed_dim)
            added = 0
            for root, _, files in os.walk(path):
                for f in files:
                    if f.lower().endswith(('.cbl', '.cob', '.cpy')):
                        chunks = parse_cobol_chunks(os.path.join(root, f))
                        added += store.add_chunks(chunks)
            return {"jsonrpc": "2.0", "id": req_id, "result": {"status": "ok", "indexed_chunks": added}}

        else:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method {method} not found"}}

    except Exception as e:
        tb = traceback.format_exc()
        return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": -32000, "message": str(e), "data": tb}}


def main():
    print("RAG Agent starting - ready to accept JSON-RPC on stdin/stdout", file=sys.stderr)
    store = IndexStore()
    # read newline-delimited JSON requests
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            send_response({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
            continue
        resp = handle_request(req, store)
        send_response(resp)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(0)


# -----------------------------
# End of rag_agent.py
# -----------------------------


