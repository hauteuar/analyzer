"""
Enhanced COBOL Parser and Component Extractor with LLM-driven naming and proper column handling

This module provides:
- COBOLParser: parsing divisions, components, file operations, CICS, MQ, XML/JSON, DB2 SQL blocks,
  data movements (MOVE/COMPUTE/ADD), and record layouts (01/05/15 levels).
- CobolField and RecordLayout dataclasses used by the ComponentExtractor.

The parser carefully extracts the program area (columns 8-72) to avoid sequence numbers and comments.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CobolField:
    name: str
    level: int
    picture: str = ""
    usage: str = ""
    occurs: int = 0
    redefines: str = ""
    value: str = ""
    line_number: int = 0
    friendly_name: str = ""
    source_references: List[Dict] = field(default_factory=list)


@dataclass
class RecordLayout:
    name: str
    level: int
    fields: List[CobolField] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0
    source_code: str = ""
    section: str = "WORKING-STORAGE"
    friendly_name: str = ""
    # any derived annotations
    annotations: Dict = field(default_factory=dict)


class COBOLParser:
    def __init__(self):
        # Division/section/paragraph patterns
        self.division_pattern = re.compile(r'^\s*(IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION', re.IGNORECASE)
        self.section_pattern = re.compile(r'^\s*(\w+)\s+SECTION\s*\.', re.IGNORECASE)
        self.paragraph_pattern = re.compile(r'^\s*([A-Z0-9\-]+)\s*\.', re.IGNORECASE)

        # Data item
        self.data_item_pattern = re.compile(r'^\s*(\d+)\s+([A-Z0-9\-]+)(?:\s+(.*))?$', re.IGNORECASE)

        # File / program / copy patterns (operate on program area only)
        self.fd_pattern = re.compile(r'FD\s+([A-Z][A-Z0-9\-]{2,})', re.IGNORECASE)
        self.select_pattern = re.compile(r'SELECT\s+([A-Z][A-Z0-9\-]{2,})\s+ASSIGN\s+TO\s+([A-Z0-9\-\.]+)', re.IGNORECASE)
        self.file_op_pattern = re.compile(r'\b(READ|WRITE|OPEN|CLOSE|REWRITE|DELETE)\s+([A-Z][A-Z0-9\-]{2,})', re.IGNORECASE)
        self.call_pattern = re.compile(r'CALL\s+[\'\"]([A-Z0-9\-]{3,})[\'\"]', re.IGNORECASE)
        self.copy_pattern = re.compile(r'COPY\s+([A-Z0-9\-]{3,})(?:\s+|\.)', re.IGNORECASE)

        # CICS patterns
        self.cics_simple_pattern = re.compile(r'EXEC\s+CICS\s+(READ|WRITE|REWRITE|DELETE)\b', re.IGNORECASE)

    def extract_program_area_only(self, line: str) -> str:
        """Return columns 8-72 of a COBOL line (program area) or empty if comment/blank."""
        if not line or len(line) < 8:
            return ""
        # column 7 is index 6
        indicator = line[6] if len(line) > 6 else ' '
        if indicator in ['*', '/', 'C', 'c', 'D', 'd']:
            return ""
        # program area columns 8-72 -> indices 7..71
        if len(line) <= 72:
            return line[7:].rstrip('\n')
        return line[7:72]

    def _is_valid_cobol_filename(self, name: str) -> bool:
        if not name or len(name) < 3:
            return False
        name_upper = name.upper().strip()
        if not re.match(r'^[A-Z]', name_upper):
            return False
        if not re.match(r'^[A-Z0-9\-\.]+$', name_upper):
            return False
        if name_upper in {'PIC', 'PICTURE', 'VALUE', 'USAGE', 'OCCURS', 'REDEFINES'}:
            return False
        return True

    def extract_file_operations(self, lines: List[str]) -> List[Dict]:
        ops = []
        seen = set()
        for i, raw in enumerate(lines):
            pa = self.extract_program_area_only(raw)
            if not pa:
                continue
            up = pa.upper()
            for m in self.fd_pattern.findall(up):
                if self._is_valid_cobol_filename(m):
                    key = f'FD::{m}'
                    if key not in seen:
                        seen.add(key)
                        ops.append({'operation': 'FD', 'file_name': m, 'line_number': i+1, 'line_content': pa.strip(), 'file_type': 'FD_FILE', 'io_direction': 'DECLARATION'})
            for logical, phys in self.select_pattern.findall(up):
                if self._is_valid_cobol_filename(logical):
                    key = f'SELECT::{logical}'
                    if key not in seen:
                        seen.add(key)
                        ops.append({'operation': 'SELECT', 'file_name': logical, 'physical_name': phys, 'line_number': i+1, 'line_content': pa.strip(), 'file_type': 'SELECT_FILE', 'io_direction': 'DECLARATION'})
            # procedural file ops
            for op, fname in re.findall(r'\b(READ|WRITE|OPEN|CLOSE|REWRITE|DELETE)\s+([A-Z][A-Z0-9\-]{2,})', up):
                if self._is_valid_cobol_filename(fname):
                    key = f'{op}::{fname}::{i}'
                    if key not in seen:
                        seen.add(key)
                        ops.append({'operation': op, 'file_name': fname, 'line_number': i+1, 'line_content': pa.strip(), 'file_type': 'PROCEDURAL_FILE'})
        return ops

    def extract_program_calls(self, lines: List[str]) -> List[Dict]:
        calls = []
        seen = set()
        for i, raw in enumerate(lines):
            pa = self.extract_program_area_only(raw)
            if not pa:
                continue
            up = pa.upper()
            for prog in self.call_pattern.findall(up):
                if self._is_valid_cobol_filename(prog):
                    key = f'CALL::{prog}'
                    if key not in seen:
                        seen.add(key)
                        calls.append({'operation': 'CALL', 'program_name': prog, 'line_number': i+1, 'line_content': pa.strip(), 'relationship_type': 'PROGRAM_CALL'})
            # CICS LINK/XCTL
            for m in re.findall(r'EXEC\s+CICS\s+LINK\s+PROGRAM\s*\(\s*[\'\"]?([A-Z0-9\-]{3,})[\'\"]?\s*\)', up):
                if self._is_valid_cobol_filename(m):
                    key = f'CICS_LINK::{m}'
                    if key not in seen:
                        seen.add(key)
                        calls.append({'operation': 'CICS_LINK', 'program_name': m, 'line_number': i+1, 'line_content': pa.strip(), 'relationship_type': 'CICS_LINK'})
        return calls

    def _extract_complete_cics_command(self, lines: List[str], start: int) -> str:
        # collect a few lines until END-EXEC or a short window
        buf = []
        for j in range(start, min(len(lines), start+8)):
            pa = self.extract_program_area_only(lines[j])
            if not pa:
                continue
            buf.append(pa)
            if 'END-EXEC' in pa.upper():
                break
        return ' '.join(buf)

    def _parse_cics_command(self, cics_command: str, line_number: int) -> List[Dict]:
        ops = []
        up = cics_command.upper()
        patterns = [r'EXEC\s+CICS\s+READ\s+.*?(?:FILE|DATASET)\s*\(\s*[\'\"]?([A-Z0-9\-]{3,})[\'\"]?\s*\)',
                    r'EXEC\s+CICS\s+WRITE\s+.*?(?:FILE|DATASET)\s*\(\s*[\'\"]?([A-Z0-9\-]{3,})[\'\"]?\s*\)']
        for pat in patterns:
            for m in re.findall(pat, up):
                if self._is_valid_cobol_filename(m):
                    ops.append({'operation': 'CICS_FILE_OP', 'file_name': m, 'line_number': line_number, 'line_content': cics_command[:200], 'file_type': 'CICS_FILE'})
        return ops

    def extract_cics_operations(self, lines: List[str]) -> List[Dict]:
        """Extract CICS operations with proper file name extraction"""
        operations = []
        seen_operations = set()
        
        for i, line in enumerate(lines):
            program_area = self.extract_program_area_only(line)
            if not program_area:
                continue
            
            program_upper = program_area.upper()
            
            # Multi-line CICS command handling
            if 'EXEC CICS' in program_upper:
                cics_command = self._extract_complete_cics_command(lines, i)
                if cics_command:
                    cics_ops = self._parse_cics_command_enhanced(cics_command, i + 1)
                    for cics_op in cics_ops:
                        op_key = f"CICS_{cics_op['operation']}_{cics_op.get('file_name', 'NOFILE')}_{i}"
                        if op_key not in seen_operations:
                            seen_operations.add(op_key)
                            operations.append(cics_op)
        
        return operations

    def _parse_cics_command_enhanced(self, cics_command: str, line_number: int) -> List[Dict]:
        """Enhanced CICS parsing to detect files from DATASET parameter"""
        operations = []
        cics_upper = cics_command.upper()
        
        # Pattern from your code: EXEC CICS READ DATASET ('TMS92ASO')
        cics_patterns = [
            # Primary DATASET patterns - these should catch TMS92ASO
            (r'EXEC\s+CICS\s+READ\s+.*?DATASET\s*\(\s*[\'"]?([A-Z0-9]{3,})[\'"]?\s*\)', 'CICS READ', 'INPUT'),
            (r'EXEC\s+CICS\s+WRITE\s+.*?DATASET\s*\(\s*[\'"]?([A-Z0-9]{3,})[\'"]?\s*\)', 'CICS WRITE', 'OUTPUT'),
            (r'EXEC\s+CICS\s+REWRITE\s+.*?DATASET\s*\(\s*[\'"]?([A-Z0-9]{3,})[\'"]?\s*\)', 'CICS REWRITE', 'OUTPUT'),
            (r'EXEC\s+CICS\s+DELETE\s+.*?DATASET\s*\(\s*[\'"]?([A-Z0-9]{3,})[\'"]?\s*\)', 'CICS DELETE', 'OUTPUT'),
            
            # Alternative FILE patterns
            (r'EXEC\s+CICS\s+READ\s+.*?FILE\s*\(\s*[\'"]?([A-Z0-9]{3,})[\'"]?\s*\)', 'CICS READ', 'INPUT'),
            (r'EXEC\s+CICS\s+WRITE\s+.*?FILE\s*\(\s*[\'"]?([A-Z0-9]{3,})[\'"]?\s*\)', 'CICS WRITE', 'OUTPUT'),
            
            # Specific patterns for files ending in ASO, DAO, DCO (common CICS file suffixes)
            (r'EXEC\s+CICS\s+READ.*?([A-Z0-9]{3,}(?:ASO|DAO|DCO)).*?(?:END-EXEC|\))', 'CICS READ', 'INPUT'),
            (r'EXEC\s+CICS\s+WRITE.*?([A-Z0-9]{3,}(?:ASO|DAO|DCO)).*?(?:END-EXEC|\))', 'CICS WRITE', 'OUTPUT'),
            
            # Catch any valid file name in CICS operations (fallback)
            (r'EXEC\s+CICS\s+READ.*?\(\s*[\'"]?([A-Z][A-Z0-9]{5,12})[\'"]?\s*\)', 'CICS READ', 'INPUT'),
            (r'EXEC\s+CICS\s+WRITE.*?\(\s*[\'"]?([A-Z][A-Z0-9]{5,12})[\'"]?\s*\)', 'CICS WRITE', 'OUTPUT'),
        ]
        
        for pattern, operation, io_direction in cics_patterns:
            matches = re.findall(pattern, cics_upper, re.DOTALL)
            for file_name in matches:
                # Enhanced validation specifically for CICS files
                if self._is_valid_cics_filename_enhanced(file_name):
                    operations.append({
                        'operation': operation,
                        'file_name': file_name,
                        'friendly_name': self.generate_friendly_name(file_name, 'CICS File'),
                        'line_number': line_number,
                        'line_content': cics_command[:100],
                        'file_type': 'CICS_FILE',
                        'io_direction': io_direction,
                        'confidence_score': 0.95
                    })
        
        return operations

    def _is_valid_cics_filename_enhanced(self, name: str) -> bool:
        """Enhanced validation specifically for CICS file names like TMS92ASO"""
        if not name or len(name) < 6:
            return False
        
        name_upper = name.upper()
        
        # Must start with letter
        if not re.match(r'^[A-Z]', name_upper):
            return False
        
        # Must be valid CICS file name pattern (6-30 chars, alphanumeric + limited special chars)
        if not re.match(r'^[A-Z][A-Z0-9]{5,29}', name_upper):
            return False
        
        # EXCLUDE obvious comment tags and sequence numbers
        if re.match(r'^[A-Z]{2}\d{6}', name_upper):  # Like SR000182
            return False
        
        # EXCLUDE working storage variables
        if re.match(r'^(WS|LS|WK|TMP)-', name_upper):
            return False
        
        # EXCLUDE COBOL keywords
        cobol_keywords = {
            'PICTURE', 'REDEFINES', 'OCCURS', 'VALUE', 'USAGE', 'COMP', 'BINARY',
            'DISPLAY', 'PACKED', 'PERFORM', 'SECTION', 'PARAGRAPH', 'DIVISION'
        }
        if name_upper in cobol_keywords:
            return False
        
        # INCLUDE common CICS file patterns
        # Files ending in ASO, DAO, DCO, etc. are typically CICS files
        if re.match(r'^[A-Z0-9]{3,}(?:ASO|DAO|DCO|FILE|TBL|IDX)', name_upper):
            return True
        
        # General pattern for CICS files (at least 6 chars, starts with letter)
        if len(name_upper) >= 6 and re.match(r'^[A-Z][A-Z0-9]{5,}', name_upper):
            return True
        
        return False


    def extract_mq_operations(self, lines: List[str]) -> List[Dict]:
        ops = []
        for i, raw in enumerate(lines):
            pa = self.extract_program_area_only(raw)
            if not pa:
                continue
            if re.search(r'CALL\s+[\'\"]MQ(OPEN|GET|PUT|PUT1|CLOSE)[\'\"]', pa, re.IGNORECASE):
                ops.append({'operation': 'MQ_CALL', 'line_number': i+1, 'line_content': pa.strip()})
        return ops

    def extract_xml_operations(self, lines: List[str]) -> List[Dict]:
        ops = []
        for i, raw in enumerate(lines):
            pa = self.extract_program_area_only(raw)
            if not pa:
                continue
            if re.search(r'XML\s+PARSE|XML\s+GENERATE|JSON\s+PARSE|JSON\s+GENERATE', pa, re.IGNORECASE):
                ops.append({'operation': 'XML_JSON', 'line_number': i+1, 'line_content': pa.strip()})
        return ops

    def extract_db2_operations(self, lines: List[str]) -> List[Dict]:
        ops = []
        in_sql = False
        buf = []
        start = 0
        for i, raw in enumerate(lines):
            pa = self.extract_program_area_only(raw)
            if not pa:
                continue
            if re.search(r'EXEC\s+SQL', pa, re.IGNORECASE):
                in_sql = True
                buf = [pa]
                start = i+1
                if re.search(r'END-EXEC', pa, re.IGNORECASE):
                    ops.append({'sql': ' '.join(buf), 'line_number': start})
                    in_sql = False
                    buf = []
            elif in_sql:
                buf.append(pa)
                if re.search(r'END-EXEC', pa, re.IGNORECASE):
                    ops.append({'sql': ' '.join(buf), 'line_number': start})
                    in_sql = False
                    buf = []
            else:
                if re.search(r'\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b', pa, re.IGNORECASE):
                    ops.append({'sql': pa.strip(), 'line_number': i+1})
        return ops

    def extract_divisions(self, lines: List[str]) -> List[Dict]:
        divisions = []
        cur = None
        for i, raw in enumerate(lines):
            pa = self.extract_program_area_only(raw)
            if not pa:
                continue
            m = self.division_pattern.match(pa)
            if m:
                if cur:
                    cur['line_end'] = i
                    divisions.append(cur)
                cur = {'name': m.group(1).upper(), 'line_start': i+1, 'line_end': len(lines)}
        if cur:
            divisions.append(cur)
        return divisions

    def extract_components(self, lines: List[str]) -> List[Dict]:
        components = []
        cur = None
        for i, raw in enumerate(lines):
            pa = self.extract_program_area_only(raw)
            if not pa:
                continue
            m = self.paragraph_pattern.match(pa)
            if m:
                if cur:
                    cur['line_end'] = i
                    components.append(cur)
                cur = {'name': m.group(1), 'line_start': i+1, 'line_end': len(lines)}
        if cur:
            components.append(cur)
        return components

    def parse_cobol_field(self, line: str, line_number: int, level: int, name: str, definition: str) -> CobolField:
        f = CobolField(name=name, level=level, line_number=line_number)
        pic = re.search(r'PIC(?:TURE)?\s+([X9SVP\(\)\.,\+\-]+)', definition, re.IGNORECASE)
        if pic:
            f.picture = pic.group(1)
        usage = re.search(r'USAGE\s+(COMP|COMP-3|DISPLAY|BINARY|PACKED-DECIMAL)', definition, re.IGNORECASE)
        if usage:
            f.usage = usage.group(1)
        occ = re.search(r'OCCURS\s+(\d+)', definition, re.IGNORECASE)
        if occ:
            f.occurs = int(occ.group(1))
        ref = re.search(r'REDEFINES\s+([A-Z0-9\-]+)', definition, re.IGNORECASE)
        if ref:
            f.redefines = ref.group(1)
        val = re.search(r'VALUE\s+(["\'].*?["\']|\S+)', definition, re.IGNORECASE)
        if val:
            f.value = val.group(1).strip('\"\'')
        return f

    def extract_record_layouts(self, lines: List[str]) -> List[RecordLayout]:
        layouts = []
        cur = None
        fields = []
        in_data = False
        current_section = None
        for i, raw in enumerate(lines):
            pa = self.extract_program_area_only(raw)
            if pa is None:
                continue
            if self.division_pattern.match(pa):
                in_data = 'DATA' in pa.upper()
                continue
            if not in_data:
                continue
            if pa.strip().upper().endswith('SECTION.'):
                current_section = pa.strip().upper().replace('SECTION.', '').strip()
                if cur:
                    cur.line_end = i
                    cur.fields = fields
                    cur.source_code = '\n'.join([self.extract_program_area_only(l) or '' for l in lines[cur.line_start-1:i]])
                    cur.section = current_section
                    layouts.append(cur)
                    cur = None
                    fields = []
                continue
            m = self.data_item_pattern.match(pa.strip())
            if m:
                level = int(m.group(1))
                name = m.group(2)
                rest = m.group(3) or ''
                if level == 1:
                    if cur:
                        cur.line_end = i
                        cur.fields = fields
                        cur.source_code = '\n'.join([self.extract_program_area_only(l) or '' for l in lines[cur.line_start-1:i]])
                        layouts.append(cur)
                        cur = None
                        fields = []
                    if name.upper() != 'FILLER':
                        cur = RecordLayout(name=name, level=level, line_start=i+1)
                        cur.section = current_section or 'WORKING-STORAGE'
                        fields = []
                else:
                    if cur and name.upper() != 'FILLER':
                        f = self.parse_cobol_field(pa.strip(), i+1, level, name, rest)
                        fields.append(f)
        if cur:
            cur.line_end = len(lines)
            cur.fields = fields
            cur.source_code = '\n'.join([self.extract_program_area_only(l) or '' for l in lines[cur.line_start-1:cur.line_end]])
            layouts.append(cur)
        return layouts

    def extract_data_movements(self, lines: List[str]) -> List[Dict]:
        movements = []
        seen = set()
        move_re = re.compile(r'MOVE\s+([A-Z0-9\-\(\)\'\"]+)\s+TO\s+([A-Z0-9\-\(\)]+)', re.IGNORECASE)
        compute_re = re.compile(r'COMPUTE\s+([A-Z0-9\-]+)\s*=\s*(.+)', re.IGNORECASE)
        add_re = re.compile(r'ADD\s+(.+?)\s+TO\s+([A-Z0-9\-\(\)]+)', re.IGNORECASE)
        for i, raw in enumerate(lines):
            pa = self.extract_program_area_only(raw)
            if not pa:
                continue
            up = pa.upper()
            for s,t in move_re.findall(up):
                key = f'MOVE::{s}::{t}::{i}'
                if key in seen:
                    continue
                seen.add(key)
                movements.append({'operation':'MOVE','source_field':s.strip(),'target_field':t.strip(),'line_number':i+1,'line_content':pa.strip()})
            for tgt,expr in compute_re.findall(up):
                key = f'COMPUTE::{tgt}::{i}'
                if key in seen:
                    continue
                seen.add(key)
                movements.append({'operation':'COMPUTE','target_field':tgt.strip(),'expression':expr.strip(),'line_number':i+1,'line_content':pa.strip()})
            for src,tgt in add_re.findall(up):
                key = f'ADD::{src}::{tgt}::{i}'
                if key in seen:
                    continue
                seen.add(key)
                movements.append({'operation':'ADD_TO','source_expression':src.strip(),'target_field':tgt.strip(),'line_number':i+1,'line_content':pa.strip()})
        return movements

    def parse_cobol_file(self, content: str, filename: str) -> Dict:
        raw_lines = content.split('\n')
        # compute a program-friendly name from filename
        program_base = filename.split('/')[-1].split('\\')[-1]
        program_base = program_base.rsplit('.', 1)[0] if '.' in program_base else program_base

        parsed = {
            'friendly_name': self.generate_friendly_name(program_base, 'Program'),
            'filename': filename,
            'divisions': self.extract_divisions(raw_lines),
            'components': self.extract_components(raw_lines),
            'record_layouts': self.extract_record_layouts(raw_lines),
            'file_operations': self.extract_file_operations(raw_lines),
            'program_calls': self.extract_program_calls(raw_lines),
            'copybooks': [ { 'copybook_name': m } for m in self.copy_pattern.findall(content) ],
            'cics_operations': self.extract_cics_operations(raw_lines),
            'mq_operations': self.extract_mq_operations(raw_lines),
            'xml_operations': self.extract_xml_operations(raw_lines),
            'db2_operations': self.extract_db2_operations(raw_lines),
            'data_movements': self.extract_data_movements(raw_lines),
            'total_lines': len(raw_lines),
            'executable_lines': sum(1 for l in raw_lines if self.extract_program_area_only(l)),
            'comment_lines': sum(1 for l in raw_lines if len(l)>6 and l[6] in ['*','/','C','c','D','d']),
            'business_comments': [l.strip() for l in raw_lines if len(l)>6 and l[6] in ['*','/','C','c','D','d'] and len(l.strip())>30][:20]
        }
        return parsed

    def generate_friendly_name(self, technical_name: str, context: str = '') -> str:
        """Create a simple business-friendly name from a technical identifier.

        This is lightweight and deterministic: removes common prefixes like WS-, FD-,
        replaces hyphens/underscores with spaces and title-cases the result. Used as a
        fallback when LLM-based naming is not available.
        """
        if not technical_name:
            return context.title() if context else 'Unknown'

        name = str(technical_name).upper().strip()

        # Remove common technical prefixes
        for prefix in ('WS-', 'LS-', 'WK-', 'FD-', 'FD_', 'TB-', 'TB_', 'SRV-', 'PRG-'):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        # Remove file extensions or path-like fragments if present
        name = name.split('/')[-1].split('\\')[-1]

        # Replace separators and multiple spaces
        name = re.sub(r'[_\-\.]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()

        # Title case looks nicer for a friendly name
        friendly = name.title()

        # If empty after cleaning, fall back to context
        if not friendly:
            return (context.title() if context else technical_name)

        return friendly


class ComponentExtractor:
    def __init__(self, llm_client, token_manager, db_manager):
        self.llm_client = llm_client
        self.token_manager = token_manager
        self.db_manager = db_manager
        self.cobol_parser = COBOLParser()

    # ...existing methods from previous file remain (not duplicated here to keep patch small)
"""
Enhanced COBOL Parser and Component Extractor with LLM-driven naming and proper column handling
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class COBOLParser:
    def __init__(self):
        # Keep existing patterns but add proper column handling
        self.division_pattern = re.compile(r'^\s*(IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION', re.IGNORECASE)
        self.section_pattern = re.compile(r'^\s*(\w+)\s+SECTION\s*\.', re.IGNORECASE)
        self.data_item_pattern = re.compile(r'^\s*(\d+)\s+([A-Z0-9\-]+)(?:\s+(.*))?$', re.IGNORECASE)
        
        # Enhanced file operation patterns - only match within program area
        self.file_op_pattern = re.compile(r'\b(READ|WRITE|OPEN|CLOSE|REWRITE|DELETE)\s+([A-Z][A-Z0-9\-]{2,})', re.IGNORECASE)
        self.select_pattern = re.compile(r'SELECT\s+([A-Z][A-Z0-9\-]{2,})\s+ASSIGN\s+TO\s+([A-Z0-9\-\.]+)', re.IGNORECASE)
        self.fd_pattern = re.compile(r'FD\s+([A-Z][A-Z0-9\-]{2,})', re.IGNORECASE)
        self.cics_file_pattern = re.compile(r'EXEC\s+CICS\s+(READ|WRITE|REWRITE|DELETE)\s+(?:FILE|DATASET)\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', re.IGNORECASE)
        self.call_pattern = re.compile(r'CALL\s+[\'"]([A-Z0-9\-]{3,})[\'"]', re.IGNORECASE)
        self.copy_pattern = re.compile(r'COPY\s+([A-Z0-9\-]{3,})(?:\s+|\.)', re.IGNORECASE)


@dataclass
class CobolField:
    name: str
    level: int
    picture: str = ""
    usage: str = ""
    occurs: int = 0
    redefines: str = ""
    value: str = ""
    line_number: int = 0


@dataclass
class RecordLayout:
    name: str
    level: int
    fields: List[CobolField] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0
    source_code: str = ""
    section: str = "WORKING-STORAGE"

    def extract_program_area_only(self, line: str) -> str:
        """Extract COBOL program area (columns 8-72) properly, ignoring sequence and comment areas"""
        if len(line) < 8:
            return ""
        
        # Skip if it's a comment line (asterisk in column 7)
        if len(line) > 6 and line[6] in ['*', '/', 'C', 'c', 'D', 'd']:
            return ""
        
        # Extract columns 8-72 (COBOL program area)
        if len(line) <= 72:
            program_area = line[7:]  # From column 8 to end
        else:
            program_area = line[7:72]  # Columns 8-72 only
        
        return program_area.strip()

    def extract_file_operations(self, lines: List[str]) -> List[Dict]:
        """Extract file operations with proper column handling to avoid junk characters"""
        operations = []
        seen_operations = set()
        
        for i, line in enumerate(lines):
            # Extract only the COBOL program area and skip comments
            program_area = self.extract_program_area_only(line)
            if not program_area:
                continue
            
            program_upper = program_area.upper()
            
            # FD entries
            fd_matches = self.fd_pattern.findall(program_upper)
            for file_name in fd_matches:
                if self._is_valid_cobol_filename(file_name):
                    op_key = f"FD_{file_name}"
                    if op_key not in seen_operations:
                        seen_operations.add(op_key)
                        operations.append({
                            'operation': 'FD',
                            'file_name': file_name,
                            'line_number': i + 1,
                            'line_content': program_area,
                            'file_type': 'FD_FILE',
                            'io_direction': 'DECLARATION'
                        })
            
            # SELECT statements
            select_matches = self.select_pattern.findall(program_upper)
            for logical_name, physical_name in select_matches:
                if self._is_valid_cobol_filename(logical_name):
                    op_key = f"SELECT_{logical_name}"
                    if op_key not in seen_operations:
                        seen_operations.add(op_key)
                        operations.append({
                            'operation': 'SELECT',
                            'file_name': logical_name,
                            'physical_name': physical_name,
                            'line_number': i + 1,
                            'line_content': program_area,
                            'file_type': 'SELECT_FILE',
                            'io_direction': 'DECLARATION'
                        })
            
            # File operations with I/O direction
            file_ops = self._extract_file_operations_with_direction(program_upper)
            for operation, file_name, io_direction in file_ops:
                if self._is_valid_cobol_filename(file_name):
                    op_key = f"{operation}_{file_name}"
                    if op_key not in seen_operations:
                        seen_operations.add(op_key)
                        operations.append({
                            'operation': operation.upper(),
                            'file_name': file_name,
                            'line_number': i + 1,
                            'line_content': program_area,
                            'file_type': 'PROCEDURAL_FILE',
                            'io_direction': io_direction
                        })
        
        return operations

    def extract_program_calls(self, lines: List[str]) -> List[Dict]:
        """Extract program calls with proper column handling"""
        operations = []
        seen_operations = set()
        
        for i, line in enumerate(lines):
            program_area = self.extract_program_area_only(line)
            if not program_area:
                continue
            
            program_upper = program_area.upper()
            
            # Static CALL statements
            call_matches = self.call_pattern.findall(program_upper)
            for program_name in call_matches:
                if self._is_valid_cobol_filename(program_name):
                    op_key = f"CALL_{program_name}"
                    if op_key not in seen_operations:
                        seen_operations.add(op_key)
                        operations.append({
                            'operation': 'CALL',
                            'program_name': program_name,
                            'call_type': 'STATIC',
                            'line_number': i + 1,
                            'line_content': program_area,
                            'relationship_type': 'PROGRAM_CALL'
                        })
            
            # CICS program operations
            cics_program_ops = [
                (r'EXEC\s+CICS\s+LINK\s+PROGRAM\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', 'CICS_LINK'),
                (r'EXEC\s+CICS\s+XCTL\s+PROGRAM\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', 'CICS_XCTL'),
            ]
            
            for pattern, op_type in cics_program_ops:
                matches = re.findall(pattern, program_upper)
                for program_name in matches:
                    if self._is_valid_cobol_filename(program_name):
                        op_key = f"{op_type}_{program_name}"
                        if op_key not in seen_operations:
                            seen_operations.add(op_key)
                            operations.append({
                                'operation': op_type,
                                'program_name': program_name,
                                'call_type': 'CICS',
                                'line_number': i + 1,
                                'line_content': program_area,
                                'relationship_type': op_type
                            })
        
        return operations

    def extract_cics_operations(self, lines: List[str]) -> List[Dict]:
        """Extract CICS operations with proper column handling"""
        operations = []
        seen_operations = set()
        
        for i, line in enumerate(lines):
            program_area = self.extract_program_area_only(line)
            if not program_area:
                continue
            
            program_upper = program_area.upper()
            
            # Multi-line CICS command handling
            if 'EXEC CICS' in program_upper:
                cics_command = self._extract_complete_cics_command(lines, i)
                if cics_command:
                    cics_ops = self._parse_cics_command(cics_command, i + 1)
                    for cics_op in cics_ops:
                        op_key = f"CICS_{cics_op['operation']}_{cics_op.get('file_name', 'NOFILE')}_{i}"
                        if op_key not in seen_operations:
                            seen_operations.add(op_key)
                            operations.append(cics_op)
        
        return operations

    def extract_copybooks(self, lines: List[str]) -> List[Dict]:
        """Extract copybooks with proper column handling"""
        copybooks = []
        seen_copybooks = set()
        
        for i, line in enumerate(lines):
            program_area = self.extract_program_area_only(line)
            if not program_area:
                continue
            
            matches = self.copy_pattern.findall(program_area)
            for copybook_name in matches:
                if self._is_valid_cobol_filename(copybook_name):
                    if copybook_name not in seen_copybooks:
                        seen_copybooks.add(copybook_name)
                        copybooks.append({
                            'copybook_name': copybook_name,
                            'line_number': i + 1,
                            'line_content': program_area
                        })
        
        return copybooks

    def _extract_complete_cics_command(self, lines: List[str], start_line: int) -> str:
        """Extract complete CICS command with proper column handling"""
        cics_command = ""
        
        for i in range(start_line, min(len(lines), start_line + 10)):
            program_area = self.extract_program_area_only(lines[i])
            if not program_area:
                continue
            
            cics_command += " " + program_area
            
            if 'END-EXEC' in cics_command.upper():
                break
        
        return cics_command.strip()

    def _is_valid_cobol_filename(self, name: str) -> bool:
        """Enhanced validation for COBOL file names"""
        if not name or len(name) < 3:
            return False
        
        name_upper = name.upper().strip()
        
        # Must start with letter
        if not re.match(r'^[A-Z]', name_upper):
            return False
        
        # Must be reasonable length (3-30 characters)
        if len(name_upper) < 3 or len(name_upper) > 30:
            return False
        
        # Must contain only valid COBOL identifier characters
        if not re.match(r'^[A-Z0-9\-]+$', name_upper):
            return False
        
        # Cannot start or end with hyphen
        if name_upper.startswith('-') or name_upper.endswith('-'):
            return False
        
        # Exclude COBOL keywords and constructs
        excluded_keywords = {
            'PIC', 'PICTURE', 'VALUE', 'USAGE', 'OCCURS', 'REDEFINES',
            'COMP', 'COMP-3', 'BINARY', 'DISPLAY', 'PACKED-DECIMAL',
            'IF', 'ELSE', 'MOVE', 'COMPUTE', 'PERFORM', 'UNTIL', 'VARYING',
            'THRU', 'THROUGH', 'TIMES', 'GIVING', 'TO', 'FROM', 'INTO', 'BY',
            'WHEN', 'EVALUATE', 'STRING', 'UNSTRING', 'ACCEPT', 'STOP', 'RUN',
            'X', 'XX', 'XXX', '9', '99', '999', 'S9', 'V9'
        }
        
        if name_upper in excluded_keywords:
            return False
        
        # Check for PIC clause patterns
        if re.match(r'^[X9SV\(\),\.]+$', name_upper):
            return False
        
        return True

    def _extract_file_operations_with_direction(self, program_line: str) -> List[tuple]:
        """Extract file operations and determine I/O direction"""
        operations = []
        
        patterns = [
            (r'OPEN\s+INPUT\s+([A-Z][A-Z0-9\-]{2,})', 'OPEN', 'INPUT'),
            (r'OPEN\s+OUTPUT\s+([A-Z][A-Z0-9\-]{2,})', 'OPEN', 'OUTPUT'),
            (r'OPEN\s+I-O\s+([A-Z][A-Z0-9\-]{2,})', 'OPEN', 'INPUT_OUTPUT'),
            (r'OPEN\s+EXTEND\s+([A-Z][A-Z0-9\-]{2,})', 'OPEN', 'OUTPUT'),
            (r'READ\s+([A-Z][A-Z0-9\-]{2,})', 'READ', 'INPUT'),
            (r'WRITE\s+([A-Z][A-Z0-9\-]{2,})', 'WRITE', 'OUTPUT'),
            (r'REWRITE\s+([A-Z][A-Z0-9\-]{2,})', 'REWRITE', 'OUTPUT'),
            (r'DELETE\s+([A-Z][A-Z0-9\-]{2,})', 'DELETE', 'OUTPUT'),
            (r'CLOSE\s+([A-Z][A-Z0-9\-]{2,})', 'CLOSE', 'NEUTRAL')
        ]
        
        for pattern, operation, direction in patterns:
            matches = re.findall(pattern, program_line)
            for file_name in matches:
                if self._is_valid_cobol_filename(file_name):
                    operations.append((operation, file_name, direction))
        
        return operations

    def _parse_cics_command(self, cics_command: str, line_number: int) -> List[Dict]:
        """Parse complete CICS command"""
        operations = []
        cics_upper = cics_command.upper()
        
        cics_file_patterns = [
            (r'EXEC\s+CICS\s+READ\s+.*?(?:FILE|DATASET)\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', 'CICS READ', 'INPUT'),
            (r'EXEC\s+CICS\s+WRITE\s+.*?(?:FILE|DATASET)\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', 'CICS WRITE', 'OUTPUT'),
            (r'EXEC\s+CICS\s+REWRITE\s+.*?(?:FILE|DATASET)\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', 'CICS REWRITE', 'OUTPUT'),
            (r'EXEC\s+CICS\s+DELETE\s+.*?(?:FILE|DATASET)\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', 'CICS DELETE', 'OUTPUT'),
        ]
        
        for pattern, operation, io_direction in cics_file_patterns:
            matches = re.findall(pattern, cics_upper)
            for file_name in matches:
                if self._is_valid_cobol_filename(file_name):
                    operations.append({
                        'operation': operation,
                        'file_name': file_name,
                        'line_number': line_number,
                        'line_content': cics_command[:100],
                        'file_type': 'CICS_FILE',
                        'io_direction': io_direction,
                        'confidence_score': 0.95
                    })
        
        return operations

    def extract_mq_operations(self, lines: List[str]) -> List[Dict]:
        operations = []
        mq_patterns = [r'CALL\s+[\'\"]MQOPEN[\'\"]', r'CALL\s+[\'\"]MQGET[\'\"]', r'CALL\s+[\'\"]MQPUT[\'\"]']
        for i, raw_line in enumerate(lines):
            program_area = self.extract_program_area_only(raw_line)
            if not program_area:
                continue
            for p in mq_patterns:
                if re.search(p, program_area, re.IGNORECASE):
                    operations.append({'operation': re.findall(r'CALL\s+[\'\"](MQ[ A-Z0-9_]*)[\'\"]', program_area, re.IGNORECASE), 'line_number': i+1, 'line_content': program_area[:120]})
        return operations

    def extract_xml_operations(self, lines: List[str]) -> List[Dict]:
        operations = []
        patterns = [r'XML\s+PARSE', r'XML\s+GENERATE', r'JSON\s+PARSE', r'JSON\s+GENERATE']
        for i, raw_line in enumerate(lines):
            program_area = self.extract_program_area_only(raw_line)
            if not program_area:
                continue
            for p in patterns:
                if re.search(p, program_area, re.IGNORECASE):
                    operations.append({'operation': p.replace('\\s+', ' '), 'line_number': i+1, 'line_content': program_area[:120]})
        return operations

    def extract_db2_operations(self, lines: List[str]) -> List[Dict]:
        operations = []
        in_exec_sql = False
        buffer = []
        start_line = 0
        for i, raw_line in enumerate(lines):
            program_area = self.extract_program_area_only(raw_line)
            if not program_area:
                continue
            if re.search(r'EXEC\s+SQL', program_area, re.IGNORECASE):
                in_exec_sql = True
                buffer = [program_area]
                start_line = i+1
                if re.search(r'END-EXEC', program_area, re.IGNORECASE):
                    in_exec_sql = False
                    operations.append({'sql': ' '.join(buffer), 'line_number': start_line})
                    buffer = []
            elif in_exec_sql:
                buffer.append(program_area)
                if re.search(r'END-EXEC', program_area, re.IGNORECASE):
                    in_exec_sql = False
                    operations.append({'sql': ' '.join(buffer), 'line_number': start_line})
                    buffer = []
            else:
                # look for inline SQL statements
                if re.search(r'\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b', program_area, re.IGNORECASE):
                    operations.append({'sql': program_area.strip(), 'line_number': i+1})

        return operations

    def extract_divisions(self, lines: List[str]) -> List[Dict]:
        divisions = []
        current = None
        for i, raw_line in enumerate(lines):
            program_area = self.extract_program_area_only(raw_line)
            if not program_area:
                continue
            m = self.division_pattern.match(program_area)
            if m:
                if current:
                    current['line_end'] = i
                    divisions.append(current)
                current = {'name': m.group(1).upper(), 'line_start': i+1, 'line_end': len(lines)}
        if current:
            divisions.append(current)
        return divisions

    def extract_components(self, lines: List[str]) -> List[Dict]:
        components = []
        para_re = re.compile(r'^\s*([A-Z0-9\-]+)\s*\.')
        current = None
        for i, raw_line in enumerate(lines):
            program_area = self.extract_program_area_only(raw_line)
            if not program_area:
                continue
            m = para_re.match(program_area)
            if m:
                name = m.group(1)
                if current:
                    current['line_end'] = i
                    components.append(current)
                current = {'name': name, 'line_start': i+1, 'line_end': len(lines)}
        if current:
            components.append(current)
        return components

    def parse_cobol_field(self, line: str, line_number: int, level: int, name: str, definition: str) -> CobolField:
        field = CobolField(name=name, level=level, line_number=line_number)
        # PIC
        pic_m = re.search(r'PIC(?:TURE)?\s+([X9SVP\(\)\.,\+\-]+)', definition, re.IGNORECASE)
        if pic_m:
            field.picture = pic_m.group(1)
        # USAGE
        u_m = re.search(r'USAGE\s+(COMP|COMP-3|DISPLAY|BINARY|PACKED-DECIMAL)', definition, re.IGNORECASE)
        if u_m:
            field.usage = u_m.group(1)
        # OCCURS
        o_m = re.search(r'OCCURS\s+(\d+)', definition, re.IGNORECASE)
        if o_m:
            field.occurs = int(o_m.group(1))
        # REDEFINES
        r_m = re.search(r'REDEFINES\s+([A-Z0-9\-]+)', definition, re.IGNORECASE)
        if r_m:
            field.redefines = r_m.group(1)
        # VALUE
        v_m = re.search(r'VALUE\s+(["\'].*?["\']|\S+)', definition, re.IGNORECASE)
        if v_m:
            field.value = v_m.group(1).strip('"\'')
        return field

    def extract_record_layouts(self, lines: List[str]) -> List[RecordLayout]:
        layouts = []
        current = None
        fields = []
        in_data = False
        current_section = None
        for i, raw_line in enumerate(lines):
            program_area = self.extract_program_area_only(raw_line)
            if program_area is None:
                continue
            pa = program_area
            # Division start
            d = self.division_pattern.match(pa)
            if d:
                in_data = d.group(1).upper() == 'DATA'
                continue
            if not in_data:
                continue
            # Section
            if pa.strip().upper().endswith('SECTION.'):
                current_section = pa.strip().upper().replace('SECTION.', '').strip()
                # finalize
                if current:
                    current.line_end = i
                    current.fields = fields
                    current.source_code = '\n'.join([self.extract_program_area_only(l) or '' for l in lines[current.line_start-1:i]])
                    current.section = current_section
                    layouts.append(current)
                    current = None
                    fields = []
                continue
            # data items
            m = self.data_item_pattern.match(pa.strip())
            if m:
                level = int(m.group(1))
                name = m.group(2)
                rest = m.group(3) or ''
                if level == 1:
                    if current:
                        current.line_end = i
                        current.fields = fields
                        current.source_code = '\n'.join([self.extract_program_area_only(l) or '' for l in lines[current.line_start-1:i]])
                        layouts.append(current)
                        current = None
                        fields = []
                    if name.upper() != 'FILLER':
                        current = RecordLayout(name=name, level=level, line_start=i+1)
                        current.section = current_section or 'WORKING-STORAGE'
                        fields = []
                else:
                    if current and name.upper() != 'FILLER':
                        f = self.parse_cobol_field(pa.strip(), i+1, level, name, rest)
                        fields.append(f)
        if current:
            current.line_end = len(lines)
            current.fields = fields
            current.source_code = '\n'.join([self.extract_program_area_only(l) or '' for l in lines[current.line_start-1:current.line_end]])
            layouts.append(current)
        return layouts

    def parse_cobol_file(self, content: str, filename: str) -> Dict:
        raw_lines = content.split('\n')
        # process lines to keep raw formatting for storage but use program area for parsing
        # derive a simple program base name
        program_base = filename.split('/')[-1].split('\\')[-1]
        program_base = program_base.rsplit('.', 1)[0] if '.' in program_base else program_base

        parsed = {
            'friendly_name': self.generate_friendly_name(program_base, 'Program'),
            'filename': filename,
            'divisions': self.extract_divisions(raw_lines),
            'components': self.extract_components(raw_lines),
            'record_layouts': self.extract_record_layouts(raw_lines),
            'file_operations': self.extract_file_operations(raw_lines),
            'program_calls': self.extract_program_calls(raw_lines),
            'copybooks': self.extract_copybooks(raw_lines),
            'cics_operations': self.extract_cics_operations(raw_lines),
            'mq_operations': self.extract_mq_operations(raw_lines),
            'xml_operations': self.extract_xml_operations(raw_lines),
            'db2_operations': self.extract_db2_operations(raw_lines),
            'data_movements': self.extract_data_movements(raw_lines),
            'total_lines': len(raw_lines),
            'executable_lines': sum(1 for l in raw_lines if self.extract_program_area_only(l)),
            'comment_lines': sum(1 for l in raw_lines if len(l)>6 and l[6] in ['*','/','C','c','D','d'])
        }
        return parsed

    def generate_friendly_name(self, technical_name: str, context: str = '') -> str:
        """Create a simple business-friendly name from a technical identifier.

        Lightweight fallback for UI/metadata: strips common prefixes and title-cases.
        """
        if not technical_name:
            return context.title() if context else 'Unknown'

        name = str(technical_name).upper().strip()

        for prefix in ('WS-', 'LS-', 'WK-', 'FD-', 'FD_', 'TB-', 'TB_', 'SRV-', 'PRG-'):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        name = name.split('/')[-1].split('\\')[-1]
        name = re.sub(r'[_\-\.]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        friendly = name.title()
        if not friendly:
            return (context.title() if context else technical_name)
        return friendly

    def extract_data_movements(self, lines: List[str]) -> List[Dict]:
        """Extract data movement operations (MOVE, COMPUTE, ADD ... TO) from COBOL program area

        Uses `extract_program_area_only` to avoid sequence/comment columns. Returns a list of
        dicts with operation, source_field, target_field (where applicable), line_number and
        a short line_content for evidence. Results are deduplicated.
        """
        movements = []
        seen = set()

        move_pattern = re.compile(r'MOVE\s+([A-Z0-9\-\(\)\'\"]+)\s+TO\s+([A-Z0-9\-\(\)]+)', re.IGNORECASE)
        compute_pattern = re.compile(r'COMPUTE\s+([A-Z0-9\-]+)\s*=\s*(.+)', re.IGNORECASE)
        add_to_pattern = re.compile(r'ADD\s+([^TO]+?)\s+TO\s+([A-Z0-9\-\(\)]+)', re.IGNORECASE)

        for idx, raw_line in enumerate(lines):
            program_area = self.extract_program_area_only(raw_line)
            if not program_area:
                continue

            program_upper = program_area.upper()

            # MOVE matches
            for m in move_pattern.findall(program_upper):
                source, target = m[0].strip(), m[1].strip()
                key = f"MOVE::{source}::{target}::{idx}"
                if key in seen:
                    continue
                seen.add(key)
                movements.append({
                    'operation': 'MOVE',
                    'source_field': source,
                    'target_field': target,
                    'line_number': idx + 1,
                    'line_content': program_area.strip()
                })

            # COMPUTE matches
            for c in compute_pattern.findall(program_upper):
                target, expr = c[0].strip(), c[1].strip()
                key = f"COMPUTE::{target}::{idx}"
                if key in seen:
                    continue
                seen.add(key)
                movements.append({
                    'operation': 'COMPUTE',
                    'target_field': target,
                    'expression': expr,
                    'line_number': idx + 1,
                    'line_content': program_area.strip()
                })

            # ADD ... TO matches (treat as MOVE_TARGET with expression)
            for a in add_to_pattern.findall(program_upper):
                source_expr, target = a[0].strip(), a[1].strip()
                key = f"ADD::{source_expr}::{target}::{idx}"
                if key in seen:
                    continue
                seen.add(key)
                movements.append({
                    'operation': 'ADD_TO',
                    'source_expression': source_expr,
                    'target_field': target,
                    'line_number': idx + 1,
                    'line_content': program_area.strip()
                })

        return movements


class ComponentExtractor:
    def __init__(self, llm_client, token_manager, db_manager):
        self.llm_client = llm_client
        self.token_manager = token_manager
        self.db_manager = db_manager
        self.cobol_parser = COBOLParser()

    def generate_friendly_names_with_llm(self, session_id: str, items: List[Dict], context: str) -> Dict[str, str]:
        """Generate friendly names using LLM for better business context"""
        try:
            # Prepare items for LLM analysis
            item_list = []
            for item in items[:20]:  # Limit to avoid token overflow
                if isinstance(item, dict):
                    name = item.get('name', item.get('copybook_name', item.get('file_name', item.get('program_name', 'UNKNOWN'))))
                    item_type = item.get('type', context)
                else:
                    name = str(item)
                    item_type = context
                
                if name and name != 'UNKNOWN':
                    item_list.append(f"- {name} ({item_type})")
            
            if not item_list:
                return {}
            
            prompt = f"""
Generate business-friendly names for these COBOL {context} items in a wealth management system.
Convert technical COBOL names to meaningful business names that describe their purpose.

Items to name:
{chr(10).join(item_list)}

Return JSON format:
{{
    "TECHNICAL_NAME": "Business Friendly Name",
    "ANOTHER_NAME": "Another Business Name"
}}

Focus on:
- Clear, descriptive business names
- Remove technical prefixes like WS-, FD-, etc.
- Use proper capitalization
- Keep names concise but meaningful
"""
            
            response = self.llm_client.call_llm(prompt, max_tokens=800, temperature=0.3)
            
            # Log LLM call
            self.db_manager.log_llm_call(
                session_id, 'friendly_naming', 1, 1,
                response.prompt_tokens, response.response_tokens, response.processing_time_ms,
                response.success, response.error_message
            )
            
            if response.success:
                friendly_names = self.llm_client.extract_json_from_response(response.content)
                if isinstance(friendly_names, dict):
                    logger.info(f"Generated {len(friendly_names)} friendly names via LLM")
                    return friendly_names
            
            logger.warning(f"LLM friendly name generation failed, using fallback")
            return {}
            
        except Exception as e:
            logger.error(f"Error generating friendly names with LLM: {str(e)}")
            return {}

    def analyze_record_layout_with_llm(self, session_id: str, layout, program_context: str) -> Dict:
        """Analyze record layout using LLM for better business understanding"""
        try:
            field_names = [f.name for f in layout.fields[:15]]  # First 15 fields
            field_info = []
            
            for field in layout.fields[:10]:  # Detail for first 10
                field_info.append(f"- {field.name} (Level {field.level}, PIC {field.picture})")
            
            prompt = f"""
Analyze this COBOL record layout for a wealth management system and provide business context.

Record Layout: {layout.name}
Program: {program_context}
Total Fields: {len(layout.fields)}
Section: {getattr(layout, 'section', 'WORKING-STORAGE')}

Field Details:
{chr(10).join(field_info)}

All Field Names: {', '.join(field_names)}

Analyze and return JSON:
{{
    "friendly_name": "Business-friendly name for this record",
    "business_purpose": "What this record represents in business terms",
    "usage_classification": "INPUT|OUTPUT|INPUT_OUTPUT|STATIC",
    "business_domain": "CUSTOMER|ACCOUNT|TRANSACTION|PORTFOLIO|GENERAL",
    "key_insights": ["insight1", "insight2"],
    "field_groupings": {{
        "customer_info": ["field1", "field2"],
        "amounts": ["field3", "field4"]
    }}
}}
"""
            
            response = self.llm_client.call_llm(prompt, max_tokens=600, temperature=0.3)
            
            # Log LLM call
            self.db_manager.log_llm_call(
                session_id, 'record_analysis', 1, 1,
                response.prompt_tokens, response.response_tokens, response.processing_time_ms,
                response.success, response.error_message
            )
            
            if response.success:
                analysis = self.llm_client.extract_json_from_response(response.content)
                if isinstance(analysis, dict):
                    return analysis
                    
            # Fallback analysis
            return {
                "friendly_name": layout.name.replace('-', ' ').title(),
                "business_purpose": f"Data structure with {len(layout.fields)} fields",
                "usage_classification": "STATIC",
                "business_domain": "GENERAL",
                "key_insights": [],
                "field_groupings": {}
            }
            
        except Exception as e:
            logger.error(f"Error analyzing record layout with LLM: {str(e)}")
            return {
                "friendly_name": layout.name,
                "business_purpose": "Analysis failed",
                "usage_classification": "STATIC",
                "business_domain": "GENERAL",
                "key_insights": [],
                "field_groupings": {}
            }

    def _extract_cobol_components(self, session_id: str, content: str, filename: str) -> List[Dict]:
        """Enhanced COBOL component extraction with LLM-driven analysis"""
        logger.info(f"Starting enhanced COBOL analysis for {filename}")
        
        try:
            parsed_data = self.cobol_parser.parse_cobol_file(content, filename)
            
            program_name = filename.replace('.cbl', '').replace('.CBL', '').replace('.cob', '').replace('.COB', '')
            
            # Generate friendly names for various components using LLM
            file_ops = parsed_data.get('file_operations', [])
            prog_calls = parsed_data.get('program_calls', [])
            copybooks = parsed_data.get('copybooks', [])
            record_layouts = parsed_data.get('record_layouts', [])
            
            # Get LLM-generated friendly names
            if file_ops:
                file_friendly_names = self.generate_friendly_names_with_llm(session_id, file_ops, "FILES")
                for file_op in file_ops:
                    file_name = file_op.get('file_name')
                    if file_name in file_friendly_names:
                        file_op['friendly_name'] = file_friendly_names[file_name]
            
            if prog_calls:
                prog_friendly_names = self.generate_friendly_names_with_llm(session_id, prog_calls, "PROGRAMS")
                for prog_call in prog_calls:
                    prog_name = prog_call.get('program_name')
                    if prog_name in prog_friendly_names:
                        prog_call['friendly_name'] = prog_friendly_names[prog_name]
            
            if copybooks:
                copybook_friendly_names = self.generate_friendly_names_with_llm(session_id, copybooks, "COPYBOOKS")
                for copybook in copybooks:
                    cb_name = copybook.get('copybook_name')
                    if cb_name in copybook_friendly_names:
                        copybook['friendly_name'] = copybook_friendly_names[cb_name]
            
            # Generate program summary with LLM
            program_summary = self._generate_component_summary(session_id, parsed_data, 'PROGRAM')
            
            # Create main program component
            program_component = {
                'name': program_name,
                'friendly_name': program_summary.get('business_purpose', f"Program {program_name}"),
                'type': 'PROGRAM',
                'file_path': filename,
                'content': content,
                'total_lines': parsed_data['total_lines'],
                'executable_lines': parsed_data.get('executable_lines', 0),
                'comment_lines': parsed_data.get('comment_lines', 0),
                'llm_summary': program_summary,
                'business_purpose': program_summary.get('business_purpose', ''),
                'complexity_score': program_summary.get('complexity_score', 0.5),
                'divisions': parsed_data['divisions'],
                'file_operations': file_ops,
                'program_calls': prog_calls,
                'copybooks': copybooks,
                'cics_operations': parsed_data.get('cics_operations', []),
                'derived_components': [],
                'record_layouts': [],
                'fields': []
            }
            
            components = [program_component]
            layout_components = []
            
            # Process each record layout with LLM analysis
            for layout in record_layouts:
                layout_name = layout.name
                
                # Analyze layout with LLM
                layout_analysis = self.analyze_record_layout_with_llm(session_id, layout, program_name)
                
                # Create layout component
                layout_component = {
                    'name': f"{program_name}_{layout_name}",
                    'friendly_name': layout_analysis.get('friendly_name', layout_name),
                    'type': 'RECORD_LAYOUT',
                    'parent_component': program_name,
                    'file_path': filename,
                    'content': layout.source_code,
                    'total_lines': layout.line_end - layout.line_start + 1,
                    'line_start': layout.line_start,
                    'line_end': layout.line_end,
                    'level': layout.level,
                    'section': getattr(layout, 'section', 'WORKING-STORAGE'),
                    'business_purpose': layout_analysis.get('business_purpose', ''),
                    'record_classification': layout_analysis.get('usage_classification', 'STATIC'),
                    'record_usage_description': layout_analysis.get('business_purpose', ''),
                    'business_domain': layout_analysis.get('business_domain', 'GENERAL'),
                    'key_insights': layout_analysis.get('key_insights', []),
                    'fields': [],
                    'total_fields': len(layout.fields)
                }
                
                # Process fields with enhanced analysis
                enhanced_fields = []
                for field in layout.fields:
                    try:
                        field_analysis = self._complete_field_source_analysis(field.name, content, program_name)
                        
                        enhanced_field = {
                            'name': field.name,
                            'friendly_name': field.name.replace('-', ' ').title(),  # Will be enhanced by LLM later
                            'level': field.level,
                            'picture': field.picture,
                            'usage': field.usage,
                            'line_number': field.line_number,
                            'usage_type': field_analysis['primary_usage'],
                            'business_purpose': field_analysis['business_purpose'],
                            'total_program_references': len(field_analysis['all_references']),
                            'source_field': field_analysis.get('primary_source_field', ''),
                            'confidence_score': 0.9,
                            'record_classification': layout_analysis.get('usage_classification', 'STATIC'),
                            'parent_layout': layout_name,
                            'inherited_from_record': field_analysis['primary_usage'] == 'STATIC',
                            'effective_classification': field_analysis['primary_usage']
                        }
                        
                        enhanced_fields.append(enhanced_field)
                        
                    except Exception as field_error:
                        logger.error(f"Error analyzing field {field.name}: {str(field_error)}")
                        continue
                
                layout_component['fields'] = enhanced_fields
                layout_components.append(layout_component)
                
                # Update main program component
                program_component['derived_components'].append(layout_name)
                program_component['record_layouts'].append({
                    'name': layout_name,
                    'friendly_name': layout_analysis.get('friendly_name', layout_name),
                    'business_purpose': layout_analysis.get('business_purpose', ''),
                    'record_classification': layout_analysis.get('usage_classification', 'STATIC'),
                    'total_fields': len(enhanced_fields),
                    'component_reference': f"{program_name}_{layout_name}"
                })
                
                program_component['fields'].extend(enhanced_fields)
                
                # Store layout in database with new schema
                try:
                    with self.db_manager.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT OR REPLACE INTO record_layouts 
                            (session_id, layout_name, friendly_name, program_name, level_number, 
                            line_start, line_end, source_code, fields_count, business_purpose,
                            record_classification, record_usage_description, has_whole_record_operations)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            session_id, 
                            layout_name, 
                            layout_analysis.get('friendly_name', layout_name), 
                            program_name,
                            str(layout.level), 
                            layout.line_start, 
                            layout.line_end,
                            layout.source_code, 
                            len(layout.fields), 
                            layout_analysis.get('business_purpose', ''),
                            layout_analysis.get('usage_classification', 'STATIC'),
                            layout_analysis.get('business_purpose', ''),
                            layout_analysis.get('usage_classification', 'STATIC') != 'STATIC'
                        ))
                        
                        layout_id = cursor.lastrowid
                        
                        # Store fields with new schema
                        for field_data in enhanced_fields:
                            field_data['layout_id'] = layout_id
                            self.db_manager.store_field_details(session_id, field_data, program_name, layout_id)
                        
                        logger.info(f"Stored layout {layout_name} with {len(enhanced_fields)} fields")
                        
                except Exception as db_error:
                    logger.error(f"Error storing layout {layout_name}: {str(db_error)}")
                    continue
            
            # Add all layout components to the result
            components.extend(layout_components)
            
            # Update main program totals
            program_component['total_fields'] = len(program_component['fields'])
            program_component['total_layouts'] = len(parsed_data.get('record_layouts', []))
            
            # Extract and store dependencies with enhanced filtering
            self._extract_and_store_dependencies_enhanced(session_id, components, filename)
            
            logger.info(f"Analysis complete: 1 program + {len(layout_components)} layout components, {program_component['total_fields']} total fields")
            
            return components
            
        except Exception as e:
            logger.error(f"Error in COBOL component extraction: {str(e)}")
            return []

    def _complete_field_source_analysis(self, field_name: str, program_content: str, program_name: str) -> Dict:
        """Enhanced field source analysis with proper column handling"""
        analysis = {
            'field_name': field_name,
            'program_name': program_name,
            'all_references': [],
            'definition_line': None,
            'definition_code': '',
            'primary_usage': 'STATIC',
            'primary_source_field': '',
            'receives_data': False,
            'provides_data': False,
            'business_purpose': '',
            'counts': {
                'definition': 0,
                'move_source': 0,
                'move_target': 0,
                'arithmetic': 0,
                'conditional': 0,
                'cics': 0
            }
        }
        
        try:
            lines = program_content.split('\n')
            field_upper = field_name.upper()
            
            for line_idx, line in enumerate(lines, 1):
                # Use proper column extraction
                program_area = self.cobol_parser.extract_program_area_only(line)
                if not program_area:
                    continue
                
                program_upper = program_area.upper()
                
                if field_upper in program_upper:
                    operation_type = 'REFERENCE'
                    source_field = ''
                    target_field = ''
                    
                    # Field definition
                    if ('PIC' in program_upper and 
                        re.match(r'^\s*\d{2}\s+' + re.escape(field_upper), program_upper)):
                        operation_type = 'DEFINITION'
                        analysis['definition_line'] = line_idx
                        analysis['definition_code'] = program_area
                        analysis['counts']['definition'] += 1
                    
                    # Enhanced MOVE detection with proper column handling
                    elif 'MOVE' in program_upper:
                        move_to_pattern = rf'MOVE\s+([A-Z0-9\-\(\)\'\"]+)\s+TO\s+{re.escape(field_upper)}'
                        move_to_match = re.search(move_to_pattern, program_upper)
                        
                        if move_to_match:
                            operation_type = 'MOVE_TARGET'
                            source_field = move_to_match.group(1)
                            analysis['counts']['move_target'] += 1
                            analysis['receives_data'] = True
                            if not analysis['primary_source_field']:
                                analysis['primary_source_field'] = source_field
                        else:
                            move_from_pattern = rf'MOVE\s+{re.escape(field_upper)}\s+TO\s+([A-Z0-9\-\(\)\'\"]+)'
                            move_from_match = re.search(move_from_pattern, program_upper)
                            
                            if move_from_match:
                                operation_type = 'MOVE_SOURCE'
                                target_field = move_from_match.group(1)
                                analysis['counts']['move_source'] += 1
                                analysis['provides_data'] = True
                    
                    # Other operation types
                    elif any(op in program_upper for op in ['COMPUTE', 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE']):
                        operation_type = 'ARITHMETIC'
                        analysis['counts']['arithmetic'] += 1
                    
                    elif any(op in program_upper for op in ['IF', 'WHEN', 'EVALUATE']):
                        operation_type = 'CONDITIONAL'
                        analysis['counts']['conditional'] += 1
                    
                    elif 'EXEC CICS' in program_upper:
                        operation_type = 'CICS'
                        analysis['counts']['cics'] += 1
                    
                    # Create reference entry
                    reference = {
                        'line_number': line_idx,
                        'line_content': program_area,
                        'operation_type': operation_type,
                        'source_field': source_field,
                        'target_field': target_field
                    }
                    
                    analysis['all_references'].append(reference)
            
            # Determine primary usage
            if analysis['receives_data'] and analysis['provides_data']:
                analysis['primary_usage'] = 'INPUT_OUTPUT'
            elif analysis['receives_data']:
                analysis['primary_usage'] = 'INPUT'
            elif analysis['provides_data']:
                analysis['primary_usage'] = 'OUTPUT'
            elif analysis['counts']['arithmetic'] > 0:
                analysis['primary_usage'] = 'DERIVED'
            elif analysis['counts']['conditional'] > 0:
                analysis['primary_usage'] = 'REFERENCE'
            else:
                analysis['primary_usage'] = 'STATIC'
            
            # Generate business purpose
            if analysis['primary_source_field']:
                analysis['business_purpose'] = f"Receives data from {analysis['primary_source_field']}"
            elif analysis['provides_data']:
                analysis['business_purpose'] = f"Provides data to other fields"
            else:
                analysis['business_purpose'] = f"{analysis['primary_usage'].lower().replace('_', ' ')} field"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in field analysis for {field_name}: {str(e)}")
            return analysis

    def _extract_and_store_dependencies_enhanced(self, session_id: str, components: List[Dict], filename: str):
        """Enhanced dependency extraction with proper column handling and junk filtering"""
        try:
            main_program = None
            for component in components:
                if component.get('type') == 'PROGRAM':
                    main_program = component
                    break
            
            if not main_program:
                return
            
            dependencies = []
            program_name = main_program['name']
            
            # Enhanced file dependency extraction
            file_operations = main_program.get('file_operations', [])
            processed_files = set()
            
            for file_op in file_operations:
                file_name = file_op.get('file_name')
                if not file_name or file_name in processed_files:
                    continue
                
                # Additional validation to avoid junk
                if self._is_valid_dependency_target(file_name, program_name):
                    processed_files.add(file_name)
                    
                    io_direction = file_op.get('io_direction', 'UNKNOWN')
                    relationship_type = self._determine_file_relationship_type(io_direction)
                    interface_type = self._determine_interface_type(file_op)
                    
                    dependencies.append({
                        'source_component': program_name,
                        'target_component': file_name,
                        'relationship_type': relationship_type,
                        'interface_type': interface_type,
                        'confidence_score': 0.95,
                        'analysis_details_json': json.dumps({
                            'io_direction': io_direction,
                            'operation': file_op.get('operation'),
                            'line_number': file_op.get('line_number'),
                            'evidence': file_op.get('line_content', '')[:100]
                        }),
                        'source_code_evidence': f"Line {file_op.get('line_number')}: {file_op.get('operation')} {file_name}"
                    })
            
            # Enhanced CICS dependency extraction
            cics_operations = main_program.get('cics_operations', [])
            processed_cics_files = set()
            
            for cics_op in cics_operations:
                file_name = cics_op.get('file_name')
                if not file_name or file_name in processed_cics_files:
                    continue
                
                if self._is_valid_dependency_target(file_name, program_name):
                    processed_cics_files.add(file_name)
                    
                    io_direction = cics_op.get('io_direction', 'UNKNOWN')
                    relationship_type = f"CICS_{self._determine_file_relationship_type(io_direction)}"
                    
                    dependencies.append({
                        'source_component': program_name,
                        'target_component': file_name,
                        'relationship_type': relationship_type,
                        'interface_type': 'CICS',
                        'confidence_score': 0.95,
                        'analysis_details_json': json.dumps({
                            'operation': cics_op.get('operation'),
                            'line_number': cics_op.get('line_number'),
                            'io_direction': io_direction
                        }),
                        'source_code_evidence': f"Line {cics_op.get('line_number')}: {cics_op.get('operation')} {file_name}"
                    })
            
            # Enhanced program call extraction
            program_calls = main_program.get('program_calls', [])
            for call in program_calls:
                target_prog = call.get('program_name')
                if target_prog and self._is_valid_dependency_target(target_prog, program_name):
                    dependencies.append({
                        'source_component': program_name,
                        'target_component': target_prog,
                        'relationship_type': call.get('relationship_type', 'PROGRAM_CALL'),
                        'interface_type': call.get('call_type', 'COBOL'),
                        'confidence_score': 0.98,
                        'analysis_details_json': json.dumps({
                            'line_number': call.get('line_number', 0),
                            'call_type': call.get('call_type', 'STATIC')
                        }),
                        'source_code_evidence': f"Line {call.get('line_number')}: {call.get('operation', 'CALL')} {target_prog}"
                    })
            
            # Enhanced copybook extraction
            copybooks = main_program.get('copybooks', [])
            for copybook in copybooks:
                copybook_name = copybook.get('copybook_name')
                if copybook_name and self._is_valid_dependency_target(copybook_name, program_name):
                    dependencies.append({
                        'source_component': program_name,
                        'target_component': copybook_name,
                        'relationship_type': 'COPYBOOK_INCLUDE',
                        'interface_type': 'COBOL',
                        'confidence_score': 0.99,
                        'analysis_details_json': json.dumps({
                            'line_number': copybook.get('line_number', 0)
                        }),
                        'source_code_evidence': f"Line {copybook.get('line_number')}: COPY {copybook_name}"
                    })
            
            # Store dependencies with enhanced deduplication
            if dependencies:
                self._store_dependencies_with_enhanced_deduplication(session_id, dependencies)
                logger.info(f"Stored {len(dependencies)} unique dependencies for {program_name}")
        
        except Exception as e:
            logger.error(f"Error extracting dependencies: {str(e)}")

    def _is_valid_dependency_target(self, target_name: str, source_program: str) -> bool:
        """Enhanced validation for dependency targets to filter out junk"""
        if not target_name:
            return False
        
        target_upper = target_name.upper().strip()
        source_upper = source_program.upper().strip()
        
        # Don't reference self
        if target_upper == source_upper:
            return False
        
        # Must be reasonable length
        if len(target_upper) < 3 or len(target_upper) > 30:
            return False
        
        # Must start with letter
        if not re.match(r'^[A-Z]', target_upper):
            return False
        
        # Must contain only valid characters
        if not re.match(r'^[A-Z0-9\-_\.]+', target_upper):
            return False
        
        # Exclude COBOL keywords and common junk
        excluded_patterns = {
            'PIC', 'PICTURE', 'VALUE', 'USAGE', 'OCCURS', 'REDEFINES',
            'COMP', 'COMP-3', 'BINARY', 'DISPLAY', 'PACKED-DECIMAL',
            'IF', 'ELSE', 'MOVE', 'COMPUTE', 'PERFORM', 'UNTIL', 'VARYING',
            'THRU', 'THROUGH', 'TIMES', 'GIVING', 'TO', 'FROM', 'INTO', 'BY',
            'WHEN', 'EVALUATE', 'STRING', 'UNSTRING', 'ACCEPT', 'STOP', 'RUN',
            'GOBACK', 'EXIT', 'NEXT', 'SENTENCE', 'PARAGRAPH', 'SECTION',
            'X', 'XX', 'XXX', '9', '99', '999', 'S9', 'V9', 'SPACES', 'ZEROS',
            'HIGH-VALUE', 'LOW-VALUE', 'HIGH-VALUES', 'LOW-VALUES'
        }
        
        if target_upper in excluded_patterns:
            return False
        
        # Exclude obvious PIC patterns
        if re.match(r'^[X9SV\(\),\.]+', target_upper):
            return False
        
        # Exclude obvious working storage prefixes for files (these are usually variables, not files)
        if re.match(r'^(WS|LS|WK|TMP|CTR|IDX|FLG|SW|IND)-', target_upper):
            return False
        
        # Exclude numeric-only patterns
        if re.match(r'^\d+', target_upper):
            return False
        
        return True

    def _determine_file_relationship_type(self, io_direction: str) -> str:
        """Determine relationship type based on I/O direction"""
        if io_direction == 'INPUT':
            return 'INPUT_FILE'
        elif io_direction == 'OUTPUT':
            return 'OUTPUT_FILE'
        elif io_direction == 'INPUT_OUTPUT':
            return 'INPUT_OUTPUT_FILE'
        else:
            return 'FILE_ACCESS'

    def _determine_interface_type(self, file_op: Dict) -> str:
        """Determine interface type based on file operation"""
        file_type = file_op.get('file_type', '')
        if 'CICS' in file_type:
            return 'CICS'
        elif file_op.get('operation') == 'FD':
            return 'FILE_SYSTEM'
        else:
            return 'FILE_SYSTEM'

    def _store_dependencies_with_enhanced_deduplication(self, session_id: str, dependencies: List[Dict]):
        """Store dependencies with enhanced deduplication and validation"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                stored_count = 0
                for dep in dependencies:
                    try:
                        # Check for existing dependency
                        cursor.execute('''
                            SELECT id, confidence_score FROM dependency_relationships
                            WHERE session_id = ? AND UPPER(source_component) = UPPER(?) 
                            AND UPPER(target_component) = UPPER(?) AND relationship_type = ?
                        ''', (
                            session_id, 
                            dep['source_component'], 
                            dep['target_component'], 
                            dep['relationship_type']
                        ))
                        
                        existing = cursor.fetchone()
                        
                        if existing:
                            # Update existing with better confidence score
                            existing_id, existing_conf = existing
                            new_conf = max(existing_conf or 0, dep.get('confidence_score', 0))
                            
                            cursor.execute('''
                                UPDATE dependency_relationships
                                SET confidence_score = ?, analysis_details_json = ?, 
                                    source_code_evidence = ?
                                WHERE id = ?
                            ''', (
                                new_conf, 
                                dep.get('analysis_details_json', '{}'), 
                                dep.get('source_code_evidence', ''), 
                                existing_id
                            ))
                        else:
                            # Insert new dependency
                            cursor.execute('''
                                INSERT INTO dependency_relationships 
                                (session_id, source_component, target_component, relationship_type,
                                interface_type, confidence_score, analysis_details_json, source_code_evidence)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                session_id, 
                                dep['source_component'], 
                                dep['target_component'], 
                                dep['relationship_type'],
                                dep.get('interface_type', ''), 
                                dep.get('confidence_score', 0.0), 
                                dep.get('analysis_details_json', '{}'), 
                                dep.get('source_code_evidence', '')
                            ))
                        
                        stored_count += 1
                        
                    except Exception as dep_error:
                        logger.error(f"Error storing dependency {dep}: {str(dep_error)}")
                        continue
                
                logger.info(f"Successfully stored/updated {stored_count} dependencies")
                        
        except Exception as e:
            logger.error(f"Error in dependency storage: {str(e)}")
            raise

    def _generate_component_summary(self, session_id: str, parsed_data: Dict, component_type: str) -> Dict:
        """Generate LLM summary for component with enhanced context"""
        try:
            context_info = {
                'type': component_type,
                'total_lines': parsed_data.get('total_lines', 0),
                'executable_lines': parsed_data.get('executable_lines', 0),
                'divisions': len(parsed_data.get('divisions', [])),
                'record_layouts': len(parsed_data.get('record_layouts', [])),
                'file_operations': parsed_data.get('file_operations', [])[:5],  # First 5
                'cics_operations': parsed_data.get('cics_operations', [])[:5],
                'program_calls': parsed_data.get('program_calls', [])[:5],
                'copybooks': parsed_data.get('copybooks', [])[:5],
                'business_comments': parsed_data.get('business_comments', [])[:3]
            }
            
            prompt = f"""
Analyze this COBOL {component_type} for a wealth management system and provide business summary.

Component Statistics:
- Total Lines: {context_info['total_lines']}
- Executable Lines: {context_info['executable_lines']}
- Record Layouts: {context_info['record_layouts']}
- File Operations: {len(parsed_data.get('file_operations', []))}
- CICS Operations: {len(parsed_data.get('cics_operations', []))}

Key Business Comments:
{chr(10).join(context_info['business_comments'])}

File Operations:
{chr(10).join([f"- {op.get('operation')}: {op.get('file_name')} ({op.get('io_direction', 'UNKNOWN')})" 
              for op in context_info['file_operations'] if op.get('file_name')])}

Program Calls:
{chr(10).join([f"- {call.get('operation', 'CALL')}: {call.get('program_name')}" 
              for call in context_info['program_calls'] if call.get('program_name')])}

Provide JSON response:
{{
    "business_purpose": "Clear business description of what this component does",
    "primary_function": "BATCH_PROCESSING|ONLINE_TRANSACTION|DATA_CONVERSION|REPORT_GENERATION|VALIDATION|GENERAL",
    "complexity_score": 0.7,
    "key_features": ["feature1", "feature2"],
    "data_sources": ["file1", "file2"],
    "business_domain": "WEALTH_MANAGEMENT|PORTFOLIO|CUSTOMER|TRANSACTION|GENERAL"
}}
"""
            
            response = self.llm_client.call_llm(prompt, max_tokens=600, temperature=0.3)
            
            self.db_manager.log_llm_call(
                session_id, 'component_summary', 1, 1,
                response.prompt_tokens, response.response_tokens, response.processing_time_ms,
                response.success, response.error_message
            )
            
            if response.success:
                summary = self.llm_client.extract_json_from_response(response.content)
                if isinstance(summary, dict):
                    return summary
            
            # Fallback summary
            return {
                'business_purpose': f"{component_type} with {len(parsed_data.get('file_operations', []))} file operations",
                'primary_function': 'GENERAL',
                'complexity_score': min(0.9, (context_info['executable_lines'] / 1000) * 0.5 + 0.3),
                'key_features': [f"{len(parsed_data.get('file_operations', []))} file operations"],
                'data_sources': [op.get('file_name') for op in parsed_data.get('file_operations', [])[:3] if op.get('file_name')],
                'business_domain': 'GENERAL'
            }
            
        except Exception as e:
            logger.error(f"Error generating component summary: {str(e)}")
            return {
                'business_purpose': 'Analysis failed - requires manual review',
                'primary_function': 'UNKNOWN',
                'complexity_score': 0.5,
                'key_features': [],
                'data_sources': [],
                'business_domain': 'GENERAL'
            }