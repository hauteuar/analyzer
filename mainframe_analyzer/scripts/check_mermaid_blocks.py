from pathlib import Path
import re

p = Path(r"c:\Users\Admin\analyzer\mainframe_analyzer\work_flow_automation.html")
s = p.read_text(encoding='utf-8')
blocks = re.findall(r"<pre class=\"mermaid\">(.*?)</pre>", s, flags=re.S)

issues = []
for i, b in enumerate(blocks, start=1):
    text = b.strip('\n')
    # remove leading/trailing spaces on each line
    lines = [ln.rstrip() for ln in text.splitlines()]
    # check bracket balance
    pairs = {'(':')','[':']','{':'}'}
    stack = []
    for ln_no, ln in enumerate(lines, start=1):
        for ch in ln:
            if ch in pairs:
                stack.append((ch, ln_no, ln))
            elif ch in pairs.values():
                if not stack:
                    issues.append((i, 'unmatched_close', ch, ln_no, ln))
                else:
                    open_ch, o_ln_no, o_ln = stack.pop()
                    if pairs[open_ch] != ch:
                        issues.append((i, 'mismatch', open_ch+ch, ln_no, ln))
    if stack:
        for open_ch, o_ln_no, o_ln in stack[::-1]:
            issues.append((i, 'unmatched_open', open_ch, o_ln_no, o_ln))
    # check for HTML tags inside block
    if re.search(r"<[^>]+>", text):
        issues.append((i, 'html_tag_in_block', None, None, None))
    # check for triple backticks
    if '```' in text:
        issues.append((i, 'backticks', None, None, None))

print('found_blocks:', len(blocks))
for i, b in enumerate(blocks, start=1):
    print('\n--- block', i, 'preview ---')
    lines = b.strip().splitlines()
    preview = '\n'.join(lines[:8])
    print(preview)
    print('\nlines containing < or >:')
    for ln_no, ln in enumerate(lines, start=1):
        if '<' in ln or '>' in ln:
            print(ln_no, ln)

print('\nISSUES FOUND:')
for it in issues:
    print(it)

if len(issues) == 0:
    print('NO_ISSUES')
else:
    print('ISSUE_COUNT', len(issues))
