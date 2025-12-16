import subprocess
from pathlib import Path
import sys

blocks_dir = Path(r"c:\Users\Admin\analyzer\mainframe_analyzer\scripts\mermaid_blocks")
out_dir = Path(r"c:\Users\Admin\analyzer\mainframe_analyzer\scripts\mermaid_outputs")
out_dir.mkdir(exist_ok=True)

for fp in sorted(blocks_dir.glob('block_*.mmd')):
    out_fp = out_dir / (fp.stem + '.svg')
    print('\n--- Validating', fp.name, '---')
    try:
        # use npx to invoke the CLI if 'mmdc' not in PATH
        mmdc_cmd = r"C:\\Program Files\\nodejs\\mmdc.cmd"
        cmd = [mmdc_cmd, '-i', str(fp), '-o', str(out_fp), '-w', '800', '-H', '600']
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as e:
        print('ERROR running mmdc:', e)
        continue
    print('returncode', res.returncode)
    if res.stdout:
        print('STDOUT:', res.stdout)
    if res.stderr:
        print('STDERR:', res.stderr)
    if res.returncode != 0:
        print('FAILED:', fp.name)
    else:
        print('OK:', fp.name)
