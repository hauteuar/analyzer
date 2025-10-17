# parse_and_build_graph_updated.py
"""
Enhanced parse_and_build_graph.py (standalone updated file)

Features added/expanded:
 - Optional tree-sitter parsing (fallback to heuristics)
 - Generic dynamic call resolver (STRING, MOVE, concatenations)
 - Extraction for EXEC SQL (DB2), EXEC CICS, MQ calls, XML parse/generate
 - COPYBOOK expansion (simple include from copybook dirs)
 - JCL parsing to extract job->program edges
 - Stores snippets/docs in SQLite, builds FAISS indices, builds NetworkX graph
 - Persists inferred dynamic calls in SQLite (dynamic_calls table)

Usage:
  python parse_and_build_graph_updated.py --src ./cobol_src --copybooks ./copybooks --jcl ./jcl --docs ./docs

Notes:
 - This is a pragmatic prototype to run locally. For production, replace naive regex with full tree-sitter grammar and robust copybook resolution.
"""

import argparse
import os
import re
from pathlib import Path
import sqlite3
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import networkx as nx

# Optional tree-sitter
try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except Exception:
    TREE_SITTER_AVAILABLE = False

# ------------------
# Configuration
# ------------------
EMBED_MODEL = 'all-MiniLM-L6-v2'
EMBED_DIM = 384
FAISS_CODE_FILE = 'faiss_code.index'
FAISS_DOC_FILE = 'faiss_doc.index'
SQLITE_FILE = 'metadata.db'
GRAPH_FILE = 'code_graph.gml'

# ------------------
# Helper functions
# ------------------

def load_text(path):
    try:
        return Path(path).read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ''

# Heuristic COBOL chunking: paragraphs + EXEC SQL + COPY lines
def simple_cobol_chunks(path):
    text = load_text(path)
    prog_m = re.search(r"PROGRAM-ID\.?\s+([\w-]+)", text, flags=re.IGNORECASE)
    program = prog_m.group(1) if prog_m else Path(path).stem
    chunks = []
    # collect copy statements
    copy_matches = [m.group(1) for m in re.finditer(r"COPY\s+([\w\-']+)", text, flags=re.IGNORECASE)]
    if copy_matches:
        chunks.append({'file': str(path), 'program': program, 'section': 'COPY_STATEMENTS', 'text': 'COPY: '+', '.join(copy_matches)})

    # exec sql blocks
    for m in re.finditer(r"EXEC\s+SQL([\s\S]*?)END-EXEC\.", text, flags=re.IGNORECASE):
        s = m.group(0).strip()
        if len(s) > 30:
            chunks.append({'file': str(path), 'program': program, 'section': 'EXEC_SQL', 'text': s})

    # exec cics blocks
    for m in re.finditer(r"EXEC\s+CICS([\s\S]*?)END-EXEC\.", text, flags=re.IGNORECASE):
        s = m.group(0).strip()
        if len(s) > 20:
            chunks.append({'file': str(path), 'program': program, 'section': 'EXEC_CICS', 'text': s})

    # paragraphs
    paras = re.split(r'(?<=\.)\n(?=\w)', text)
    for i,p in enumerate(paras):
        p = p.strip()
        if len(p) < 40:
            continue
        mname = re.search(r'^(\w[\w-]*\.)', p)
        section = mname.group(1).strip() if mname else f'para_{i}'
        chunks.append({'file': str(path), 'program': program, 'section': section, 'text': p})
    return chunks

# Tree-sitter based extraction (if parser available and built)
PARSER = None
if TREE_SITTER_AVAILABLE and os.path.exists('build/my-languages.so'):
    try:
        COBOL_LANG = Language('build/my-languages.so', 'cobol')
        PARSER = Parser()
        PARSER.set_language(COBOL_LANG)
    except Exception:
        PARSER = None


def tree_sitter_chunks(path):
    if not PARSER:
        return simple_cobol_chunks(path)
    text = load_text(path)
    if not text:
        return []
    tree = PARSER.parse(bytes(text, 'utf8'))
    root = tree.root_node
    chunks = []
    def node_text(n):
        return text[n.start_byte:n.end_byte]
    # collect common nodes
    for node in root.walk():
        try:
            if node.type in ('procedure_division','data_division','exec_sql','exec_cics'):
                snippet = node_text(node).strip()
                if len(snippet) > 30:
                    chunks.append({'file': str(path), 'program': Path(path).stem, 'section': node.type, 'text': snippet})
        except Exception:
            continue
    if not chunks:
        return simple_cobol_chunks(path)
    return chunks

# ------------------
# Copybook expansion (simple inline)
# ------------------

def expand_copybooks(text, copy_dirs):
    def repl(m):
        name = m.group(1).strip().strip('.').strip("'")
        for d in copy_dirs:
            # try common copybook extensions
            for ext in ('.cpy','.cpyb','.cbl','.copy'):
                p = Path(d) / (name + ext)
                if p.exists():
                    return '\n' + load_text(p) + '\n'
        return m.group(0)
    return re.sub(r"COPY\s+([\w\-']+)\.?(\s|$)", repl, text, flags=re.IGNORECASE)

# ------------------
# JCL parsing (simple)
# ------------------

def parse_jcl_jobs(jcl_path):
    text = load_text(jcl_path)
    jobs = []
    for m in re.finditer(r"//(\w+)\s+JOB\b(.*?\n)([\s\S]*?)(?=//\w+\s+JOB|$)", text, flags=re.IGNORECASE):
        jobname = m.group(1)
        body = m.group(3)
        jobs.append({'job': jobname, 'body': body})
    steps = []
    for job in jobs:
        for m in re.finditer(r"//(\w+)\s+EXEC\s+PGM=([\w,']+)", job['body'], flags=re.IGNORECASE):
            step = m.group(1)
            pgm = m.group(2).strip().strip("'")
            steps.append({'job': job['job'], 'step': step, 'program': pgm})
    return steps

# ------------------
# Dynamic Call Resolver (generic heuristics)
# ------------------

def resolve_dynamic_calls_in_text(text):
    statements = re.split(r'\.|\n', text)
    assignments = {}
    string_ops = []
    for stmt in statements:
        s = stmt.strip()
        if not s:
            continue
        m = re.search(r"MOVE\s+'([^']+)'\s+TO\s+([\w\-]+)", s, flags=re.IGNORECASE)
        if m:
            val, var = m.group(1), m.group(2)
            assignments.setdefault(var, set()).add(val)
            continue
        m2 = re.search(r"MOVE\s+([\w\-]+)\s+TO\s+([\w\-]+)", s, flags=re.IGNORECASE)
        if m2:
            src, dst = m2.group(1), m2.group(2)
            if src in assignments:
                for v in assignments[src]:
                    assignments.setdefault(dst, set()).add(v)
            continue
        m3 = re.search(r"STRING\s+(.+)\s+INTO\s+([\w\-]+)", s, flags=re.IGNORECASE)
        if m3:
            parts = m3.group(1)
            tgt = m3.group(2)
            parts_list = [p.strip().strip("',") for p in re.split(r'\s+', parts) if p.upper()!='DELIMITED' and p.upper()!='BY' and p.upper()!='SIZE']
            string_ops.append({'target': tgt, 'parts': parts_list})
            continue
    inferred = []
    for op in string_ops:
        tgt = op['target']
        parts = op['parts']
        candidates = ['']
        trace = []
        for p in parts:
            new_cands = []
            if re.match(r"^'([^']+)'$", p) or re.match(r"^[A-Z0-9]+$", p):
                token = p.strip("'")
                for c in candidates:
                    new_cands.append(c+token)
                trace.append(p)
            else:
                vals = assignments.get(p, None)
                if vals:
                    for v in vals:
                        for c in candidates:
                            new_cands.append(c+v)
                    trace.append(p)
                else:
                    for c in candidates:
                        new_cands.append(c+'<'+p+'>')
                    trace.append(p)
            candidates = new_cands
        inferred.append({'target_var': tgt, 'inferred_values': list(set(candidates)), 'trace': trace, 'confidence': 0.6 + 0.2*min(1, sum(1 for t in trace if not t.startswith('<'))/max(1,len(trace)))})
    return inferred

# ------------------
# Builder class
# ------------------
class Builder:
    def __init__(self, model_name=EMBED_MODEL, embed_dim=EMBED_DIM, copy_dirs=None):
        self.model = SentenceTransformer(model_name)
        self.embed_dim = embed_dim
        self.copy_dirs = copy_dirs or []
        self.code_vectors = []
        self.code_meta = []
        self.doc_vectors = []
        self.doc_meta = []
        self.graph = nx.DiGraph()
        self.conn = sqlite3.connect(SQLITE_FILE)
        self._ensure_tables()

    def _ensure_tables(self):
        cur = self.conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS snippets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            program TEXT,
            section TEXT,
            text TEXT
        );''')
        cur.execute('''CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            path TEXT,
            text TEXT
        );''')
        cur.execute('''CREATE TABLE IF NOT EXISTS dynamic_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_program TEXT,
            target_text TEXT,
            target_var TEXT,
            trace TEXT,
            confidence REAL
        );''')
        self.conn.commit()

    def add_code_chunks(self, chunks, source_program=None):
        cur = self.conn.cursor()
        texts = [c['text'] for c in chunks]
        if not texts:
            return 0
        processed_texts = []
        for c in chunks:
            txt = c['text']
            if 'COPY' in txt.upper() and self.copy_dirs:
                txt = expand_copybooks(txt, self.copy_dirs)
            processed_texts.append(txt)
        embs = self.model.encode(processed_texts, show_progress_bar=False)
        embs = np.array(embs).astype('float32')
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms==0]=1.0
        embs = embs / norms
        for i,c in enumerate(chunks):
            cur.execute('INSERT INTO snippets(source,program,section,text) VALUES(?,?,?,?)',
                        (c['file'], c.get('program', source_program or Path(c['file']).stem), c['section'], processed_texts[i]))
            self.code_meta.append({'rowid': cur.lastrowid, 'file': c['file'], 'program': c.get('program', source_program or Path(c['file']).stem), 'section': c['section']})
        self.conn.commit()
        for v in embs:
            self.code_vectors.append(v)
        return len(texts)

    def add_docs(self, docs):
        cur = self.conn.cursor()
        texts = [d['text'] for d in docs]
        if not texts:
            return 0
        embs = self.model.encode(texts, show_progress_bar=False)
        embs = np.array(embs).astype('float32')
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms==0]=1.0
        embs = embs / norms
        for i,d in enumerate(docs):
            cur.execute('INSERT INTO docs(source,path,text) VALUES(?,?,?)', (d.get('source','doc'), d['path'], d['text']))
            self.doc_meta.append({'rowid': cur.lastrowid, 'path': d['path']})
        self.conn.commit()
        for v in embs:
            self.doc_vectors.append(v)
        return len(texts)

    def build_faiss(self, out_file, vectors):
        if not vectors:
            print(f'[WARN] No vectors to build for {out_file}')
            return
        arr = np.array(vectors).astype('float32')
        d = arr.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(arr)
        faiss.write_index(index, out_file)
        print(f'[INFO] wrote faiss index {out_file} with {index.ntotal} vectors')

    
    def extract_graph_edges(self, chunks, current_program=None):
        for c in chunks:
            text = c['text']
            prog = c.get('program', current_program or Path(c['file']).stem)
            pid = f'prog:{prog}'
            if not self.graph.has_node(pid):
                self.graph.add_node(pid, type='program', name=prog, file=c['file'])

            # Static CALLs
            for m in re.finditer(r"CALL\s+'?([A-Z0-9_-]+)'?", text, flags=re.IGNORECASE):
                tgt = m.group(1)
                tid = f'prog:{tgt}'
                self.graph.add_edge(pid, tid, type='call', confidence=1.0)

            # CICS EXEC blocks
            for m in re.finditer(r"EXEC\s+CICS\s+LINK\s+PROGRAM\('([A-Z0-9_-]+)'\)", text, flags=re.IGNORECASE):
                tgt = m.group(1)
                self.graph.add_edge(pid, f'prog:{tgt}', type='cics_link', confidence=0.9)

            for m in re.finditer(r"EXEC\s+CICS\s+(SEND|RECEIVE)\s+MAP\('([A-Z0-9_-]+)'\)", text, flags=re.IGNORECASE):
                action, tgt = m.groups()
                self.graph.add_edge(pid, f'map:{tgt}', type=f'cics_{action.lower()}', confidence=0.8)

            # DB2 EXEC SQL table access
            for m in re.finditer(r"EXEC\s+SQL\s+(SELECT|UPDATE|INSERT|DELETE)\s+.*?FROM\s+([A-Z0-9_-]+)", text, flags=re.IGNORECASE|re.DOTALL):
                verb, table = m.groups()
                self.graph.add_edge(pid, f'db2:{table}', type=f'sql_{verb.lower()}', confidence=0.9)

            # MQ Calls
            for m in re.finditer(r"CALL\s+'?(MQPUT|MQGET|MQOPEN|MQCLOSE)'?", text, flags=re.IGNORECASE):
                mqverb = m.group(1).upper()
                self.graph.add_edge(pid, f'mq:{mqverb}', type='mq_call', confidence=0.8)

            # XML statements
            for m in re.finditer(r"XML\s+(PARSE|GENERATE)\s+([A-Z0-9_-]+)", text, flags=re.IGNORECASE):
                action, var = m.groups()
                self.graph.add_edge(pid, f'xml:{var}', type=f'xml_{action.lower()}', confidence=0.8)

            # Dynamic Calls (heuristics)
            inferred = resolve_dynamic_calls_in_text(text)
            cur = self.conn.cursor()
            for inf in inferred:
                for val in inf['inferred_values']:
                    tid = f'prog:{val}'
                    self.graph.add_edge(pid, tid, type='dynamic_call', confidence=inf['confidence'])
                    cur.execute('INSERT INTO dynamic_calls(source_program,target_text,target_var,trace,confidence) VALUES(?,?,?,?,?)',
                                (prog, val, inf['target_var'], json.dumps(inf['trace']), inf['confidence']))
            self.conn.commit()

    def save_graph(self, path=GRAPH_FILE):
        nx.write_gml(self.graph, path)
        print(f'[INFO] wrote graph with {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges to {path}')

# ------------------
# Main Execution
# ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='COBOL source directory')
    ap.add_argument('--copybooks', default=None, help='Copybook directory')
    ap.add_argument('--jcl', default=None, help='JCL directory')
    ap.add_argument('--docs', default=None, help='Docs directory')
    args = ap.parse_args()

    copy_dirs = [args.copybooks] if args.copybooks else []
    b = Builder(copy_dirs=copy_dirs)

    # COBOL sources
    for root,_,files in os.walk(args.src):
        for f in files:
            if not f.lower().endswith(('.cbl','.cob','.cpy')):
                continue
            path = os.path.join(root,f)
            chunks = tree_sitter_chunks(path)
            b.add_code_chunks(chunks)
            b.extract_graph_edges(chunks)

    # JCL jobs
    if args.jcl and os.path.exists(args.jcl):
        for root,_,files in os.walk(args.jcl):
            for f in files:
                if not f.lower().endswith(('.jcl','.cntl')):
                    continue
                steps = parse_jcl_jobs(os.path.join(root,f))
                for s in steps:
                    b.graph.add_edge(f'job:{s["job"]}', f'prog:{s["program"]}', type='jcl_step', confidence=1.0)

    # Docs
    docs = []
    if args.docs and os.path.exists(args.docs):
        for root,_,files in os.walk(args.docs):
            for f in files:
                if f.lower().endswith(('.md','.txt','.docx')):
                    text = load_text(os.path.join(root,f))
                    if len(text)>40:
                        docs.append({'path': os.path.join(root,f), 'text': text})
        if docs:
            b.add_docs(docs)

    # Build FAISS indexes
    b.build_faiss(FAISS_CODE_FILE, b.code_vectors)
    b.build_faiss(FAISS_DOC_FILE, b.doc_vectors)

    # Save Graph
    b.save_graph()

if __name__ == '__main__':
    main()
