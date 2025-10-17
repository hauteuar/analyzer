
# ================================
# File: flow_generator.py
# ================================
"""
Utility CLI to generate a Mermaid diagram or basic Draw.io-like XML for a given node from the graph.

Usage:
  python flow_generator.py --node prog:UPDATEADDR --depth 2 --out mermaid.txt

This script relies on code_graph.gml produced by parse_and_build_graph.py
"""

import argparse
import networkx as nx

GRAPH_FILE = 'code_graph.gml'


def make_mermaid_subgraph(graph, node, depth=2):
    if node not in graph:
        raise ValueError('node not found')
    nodes = set([node])
    curr = {node}
    for _ in range(depth):
        nxt = set()
        for n in curr:
            nxt.update(graph.successors(n))
            nxt.update(graph.predecessors(n))
        nodes.update(nxt)
        curr = nxt
    sub = graph.subgraph(nodes).copy()
    lines = ['graph LR']
    for n in sub.nodes:
        label = sub.nodes[n].get('name', n)
        lab = n.replace(':','_')
        lines.append(f'  {lab}["{label}"]')
    for u,v in sub.edges:
        et = sub.edges[u,v].get('type','')
        lines.append(f'  {u.replace(":","_")} -->|{et}| {v.replace(":","_")}')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--node', required=True)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    g = nx.read_gml(GRAPH_FILE)
    m = make_mermaid_subgraph(g, args.node, args.depth)
    if args.out:
        Path(args.out).write_text(m)
        print(f'Wrote {args.out}')
    else:
        print(m)


if __name__ == '__main__':
    main()