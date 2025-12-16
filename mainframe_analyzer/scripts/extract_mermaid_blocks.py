from pathlib import Path
import re
p = Path(r"c:\Users\Admin\analyzer\mainframe_analyzer\work_flow_automation.html")
s = p.read_text(encoding='utf-8')
blocks = re.findall(r"<pre class=\"mermaid\">(.*?)</pre>", s, flags=re.S)
out = Path(r"c:\Users\Admin\analyzer\mainframe_analyzer\scripts\mermaid_blocks")
out.mkdir(exist_ok=True)
for i, b in enumerate(blocks, start=1):
    fn = out / f"block_{i}.mmd"
    fn.write_text(b.strip() + "\n", encoding='utf-8')
print('wrote', len(blocks), 'blocks to', out)