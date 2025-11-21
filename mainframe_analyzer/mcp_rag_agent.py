# ================================
# File: mcp_rag_agent.py
# ================================
"""
MCP-compatible retrieval agent that loads prebuilt FAISS indexes and the graph, and exposes
JSON-RPC methods over STDIO for Copilot to call.

Methods:
 - search_code(query, top_k=5)
 - search_docs(query, top_k=5)
 - graph_neighbors(node, depth=1)
 - flow_mermaid(node, depth=2)
 - info()

Usage:
  python mcp_rag_agent.py

This listens on stdin for newline-delimited JSON-RPC requests and responds on stdout.
"""

import sys
import json
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx

FAISS_CODE_FILE = 'faiss_code.index'
FAISS_DOC_FILE = 'faiss_doc.index'
SQLITE_FILE = 'metadata.db'
GRAPH_FILE = 'code_graph.gml'
EMBED_MODEL = 'all-MiniLM-L6-v2'

class MCPAgent:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.dim = 384
        self.code_index = None
        self.doc_index = None
        self.load_faiss()
        self.conn = sqlite3.connect(SQLITE_FILE)
        try:
            self.graph = nx.read_gml(GRAPH_FILE)
        except Exception:
            self.graph = nx.DiGraph()

    def load_faiss(self):
        if os.path.exists(FAISS_CODE_FILE := FAISS_CODE_FILE):
            try:
                self.code_index = faiss.read_index(FAISS_CODE_FILE)
            except Exception:
                self.code_index = None
        else:
            self.code_index = None
        if os.path.exists(FAISS_DOC_FILE := FAISS_DOC_FILE):
            try:
                self.doc_index = faiss.read_index(FAISS_DOC_FILE)
            except Exception:
                self.doc_index = None
        else:
            self.doc_index = None

    def embed(self, texts):
        arr = self.model.encode(texts, show_progress_bar=False)
        arr = np.array(arr).astype('float32')
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms==0]=1.0
        arr = arr / norms
        return arr

    def search_code(self, query, top_k=5):
        if self.code_index is None or self.code_index.ntotal == 0:
            return []
        qv = self.embed([query])
        D,I = self.code_index.search(qv, top_k)
        hits = []
        cur = self.conn.cursor()
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            # sqlite row mapping: snippets table rowid likely maps to index order of insertion; we stored rowids sequentially
            # For a robust mapping, production should store explicit idmap; here we return by rowid assuming same order
            rowid = idx+1
            cur.execute('SELECT rowid, source, program, section, text FROM snippets WHERE rowid=?', (rowid,))
            r = cur.fetchone()
            if r:
                hits.append({'rowid': r[0], 'source': r[1], 'program': r[2], 'section': r[3], 'text': r[4], 'score': float(score)})
        return hits

    def search_docs(self, query, top_k=5):
        if self.doc_index is None or self.doc_index.ntotal == 0:
            return []
        qv = self.embed([query])
        D,I = self.doc_index.search(qv, top_k)
        hits = []
        cur = self.conn.cursor()
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            rowid = idx+1
            cur.execute('SELECT rowid, source, path, text FROM docs WHERE rowid=?', (rowid,))
            r = cur.fetchone()
            if r:
                hits.append({'rowid': r[0], 'source': r[1], 'path': r[2], 'text': r[3], 'score': float(score)})
        return hits

    def graph_neighbors(self, node, depth=1):
        if node not in self.graph:
            return {'error': 'node not found'}
        nodes = set([node])
        current = {node}
        for _ in range(depth):
            nxt = set()
            for n in current:
                nxt.update(self.graph.successors(n))
                nxt.update(self.graph.predecessors(n))
            nodes.update(nxt)
            current = nxt
        sub = self.graph.subgraph(nodes).copy()
        # serialize
        nodes_out = [{ 'id': n, 'attrs': dict(sub.nodes[n]) } for n in sub.nodes]
        edges_out = [{ 'u': u, 'v': v, 'attrs': dict(sub.edges[u,v]) } for u,v in sub.edges]
        return {'nodes': nodes_out, 'edges': edges_out}

    def flow_mermaid(self, node, depth=2):
        out = self.graph_neighbors(node, depth)
        if 'error' in out:
            return out
        lines = ['graph LR']
        for n in out['nodes']:
            lab = n['id'].replace(':','_')
            label = n['attrs'].get('name', n['id'])
            lines.append(f"  {lab}[\"{label}\"]")
        for e in out['edges']:
            u = e['u'].replace(':','_')
            v = e['v'].replace(':','_')
            et = e['attrs'].get('type','')
            lines.append(f"  {u} -->|{et}| {v}")
        return '\n'.join(lines)

# Simple JSON-RPC on stdin/stdout

def send(resp):
    sys.stdout.write(json.dumps(resp)+"\n")
    sys.stdout.flush()


def handle(req, agent: MCPAgent):
    try:
        mid = req.get('id')
        method = req.get('method')
        params = req.get('params', {})
        if method == 'search_code':
            q = params.get('query')
            k = int(params.get('top_k',5))
            res = agent.search_code(q, top_k=k)
            return {'jsonrpc':'2.0','id':mid,'result':res}
        if method == 'search_docs':
            q = params.get('query')
            k = int(params.get('top_k',5))
            res = agent.search_docs(q, top_k=k)
            return {'jsonrpc':'2.0','id':mid,'result':res}
        if method == 'graph_neighbors':
            node = params.get('node')
            d = int(params.get('depth',1))
            res = agent.graph_neighbors(node, depth=d)
            return {'jsonrpc':'2.0','id':mid,'result':res}
        if method == 'flow_mermaid':
            node = params.get('node')
            d = int(params.get('depth',2))
            res = agent.flow_mermaid(node, depth=d)
            return {'jsonrpc':'2.0','id':mid,'result':res}
        if method == 'info':
            return {'jsonrpc':'2.0','id':mid,'result':{'has_code_index': agent.code_index is not None and agent.code_index.ntotal>0, 'graph_nodes': agent.graph.number_of_nodes()}}
        return {'jsonrpc':'2.0','id':mid,'error':{'code':-32601,'message':'method not found'}}
    except Exception as e:
        tb = traceback.format_exc()
        return {'jsonrpc':'2.0','id':req.get('id'), 'error':{'code':-32000,'message':str(e),'data':tb}}

import traceback, os

def main():
    agent = MCPAgent()
    # listen
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            send({'jsonrpc':'2.0','id':None,'error':{'code':-32700,'message':'parse error'}})
            continue
        resp = handle(req, agent)
        send(resp)

if __name__ == '__main__':
    main()

