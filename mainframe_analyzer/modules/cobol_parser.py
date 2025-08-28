"""
COBOL Parser Module - FIXED VERSION
Handles COBOL code parsing with proper filtering and deduplication
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DataType(Enum):
    NUMERIC = "NUMERIC"
    ALPHANUMERIC = "ALPHANUMERIC"
    PACKED_DECIMAL = "PACKED_DECIMAL"
    BINARY = "BINARY"
    DISPLAY = "DISPLAY"

@dataclass
class CobolField:
    name: str
    level: int
    picture: str
    usage: str = ""
    occurs: int = 0
    redefines: str = ""
    value: str = ""
    line_number: int = 0
    friendly_name: str = ""
    source_references: List[Dict] = None  # All places this field is referenced

    def __post_init__(self):
        if self.source_references is None:
            self.source_references = []

@dataclass
class RecordLayout:
    name: str
    level: int
    fields: List[CobolField]
    line_start: int
    line_end: int
    source_code: str
    friendly_name: str = ""
    section: str = ""

@dataclass
class CobolComponent:
    name: str
    component_type: str
    line_start: int
    line_end: int
    content: str
    friendly_name: str = ""

class COBOLParser:
    def __init__(self):
        # COBOL patterns
        self.division_pattern = re.compile(r'^\s*(IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION', re.IGNORECASE)
        self.section_pattern = re.compile(r'^\s*(\w+)\s+SECTION\s*\.', re.IGNORECASE)
        self.paragraph_pattern = re.compile(r'^\s*([A-Z0-9][A-Z0-9\-]*)\s*\.', re.IGNORECASE)
        
        # Data definition patterns
        self.data_item_pattern = re.compile(r'^\s*(\d+)\s+([A-Z0-9\-]+)(?:\s+(.*))?$', re.IGNORECASE)
        self.picture_pattern = re.compile(r'PIC(?:TURE)?\s+([X9SVP\(\),\.\+\-\*\$Z]+)', re.IGNORECASE)
        self.usage_pattern = re.compile(r'USAGE\s+(COMP|COMP-3|DISPLAY|BINARY|PACKED-DECIMAL)', re.IGNORECASE)
        self.occurs_pattern = re.compile(r'OCCURS\s+(\d+)', re.IGNORECASE)
        self.redefines_pattern = re.compile(r'REDEFINES\s+([A-Z0-9\-]+)', re.IGNORECASE)
        self.value_pattern = re.compile(r'VALUE\s+([\'"].*?[\'"]|\S+)', re.IGNORECASE)
        
        # Enhanced file operation patterns - more restrictive
        self.file_op_pattern = re.compile(r'\b(READ|WRITE|OPEN|CLOSE|REWRITE|DELETE)\s+([A-Z][A-Z0-9\-]{2,})', re.IGNORECASE)
        self.select_pattern = re.compile(r'SELECT\s+([A-Z][A-Z0-9\-]{2,})\s+ASSIGN\s+TO\s+([A-Z0-9\-\.]+)', re.IGNORECASE)
        self.fd_pattern = re.compile(r'FD\s+([A-Z][A-Z0-9\-]{2,})', re.IGNORECASE)
        
        # CICS patterns - enhanced
        self.cics_file_pattern = re.compile(r'EXEC\s+CICS\s+(READ|WRITE|REWRITE|DELETE)\s+(?:FILE|DATASET)\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', re.IGNORECASE)
        
        # Program calls - more restrictive
        self.call_pattern = re.compile(r'CALL\s+[\'"]([A-Z0-9\-]{3,})[\'"]', re.IGNORECASE)
        
        # Copybook includes - more restrictive  
        self.copy_pattern = re.compile(r'COPY\s+([A-Z0-9\-]{3,})(?:\s+|\.)', re.IGNORECASE)
        
        # Invalid filename patterns to exclude
        self.invalid_file_patterns = {
            'PIC', 'PICTURE', 'VALUE', 'USAGE', 'OCCURS', 'REDEFINES',
            'COMP', 'COMP-3', 'BINARY', 'DISPLAY', 'PACKED-DECIMAL',
            'IF', 'ELSE', 'END-IF', 'MOVE', 'TO', 'FROM', 'GIVING',
            'COMPUTE', 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE',
            'PERFORM', 'UNTIL', 'VARYING', 'TIMES', 'THRU', 'THROUGH',
            'X', 'XX', 'XXX', '9', '99', '999', 'S9', 'V9'
        }
        
        # Field reference patterns for comprehensive source extraction
        self.field_reference_patterns = [
            r'\b({field_name})\b(?!\s*(?:PIC|PICTURE|USAGE|OCCURS|REDEFINES|VALUE))',  # General reference
            r'MOVE\s+([^TO]+)\s+TO\s+({field_name})',  # MOVE TO field
            r'MOVE\s+({field_name})\s+TO\s+([^\s\.]+)',  # MOVE FROM field
            r'COMPUTE\s+({field_name})\s*=\s*(.+)',  # COMPUTE field
            r'ADD\s+([^TO]+)\s+TO\s+({field_name})',  # ADD TO field
            r'IF\s+({field_name})\s*([<>=!]+|\s+(?:EQUAL|GREATER|LESS))',  # IF conditions
            r'({field_name})\s*([<>=!]+|(?:EQUAL|GREATER|LESS))',  # Comparison
        ]

    def generate_friendly_name(self, cobol_name: str, context: str = "") -> str:
        """Generate friendly names for COBOL components"""
        if not cobol_name:
            return ""
        
        # Common COBOL naming conventions to friendly names
        name_mappings = {
            'WS-': 'Working Storage ',
            'LS-': 'Local Storage ',
            'FD-': 'File Description ',
            'WK-': 'Work ',
            'TMP-': 'Temporary ',
            'CTR-': 'Counter ',
            'IDX-': 'Index ',
            'FLG-': 'Flag ',
            'SW-': 'Switch ',
            'IND-': 'Indicator ',
            'DATE-': 'Date ',
            'TIME-': 'Time ',
            'AMT-': 'Amount ',
            'QTY-': 'Quantity ',
            'NBR-': 'Number ',
            'NO-': 'Number ',
            'CD-': 'Code ',
            'DESC-': 'Description ',
            'NAME-': 'Name ',
            'ADDR-': 'Address ',
            'CUST-': 'Customer ',
            'ACCT-': 'Account ',
            'TRAN-': 'Transaction ',
            'REC-': 'Record ',
            'TAB-': 'Table ',
            'FILE-': 'File ',
            'KEY-': 'Key ',
            'EOF-': 'End of File '
        }
        
        friendly = cobol_name.replace('-', ' ')
        
        # Apply common prefix mappings
        for prefix, replacement in name_mappings.items():
            if friendly.upper().startswith(prefix):
                friendly = replacement + friendly[len(prefix):]
                break
        
        # Capitalize words
        friendly = ' '.join(word.capitalize() for word in friendly.split())
        
        # Add context if provided
        if context and context.upper() not in friendly.upper():
            if context.lower() in ['record', 'layout', 'structure']:
                friendly += f" {context.title()}"
            elif context.lower() in ['program', 'procedure', 'module']:
                friendly = f"{context.title()} - {friendly}"
        
        return friendly.strip()

    def _is_valid_filename(self, name: str) -> bool:
        """Check if a name is a valid filename (not a COBOL keyword)"""
        if not name or len(name) < 3:
            return False
            
        name_upper = name.upper()
        
        # Check against invalid patterns
        if name_upper in self.invalid_file_patterns:
            return False
            
        # Check for obvious non-filenames
        if re.match(r'^[X9SV\(\),\.]+$', name_upper):  # PIC clause patterns
            return False
            
        if re.match(r'^\d+$', name):  # Pure numbers
            return False
            
        # Must start with letter
        if not re.match(r'^[A-Z]', name_upper):
            return False
            
        return True

    def extract_file_operations(self, lines: List[str]) -> List[Dict]:
        """Extract file operations with deduplication and proper filtering"""
        operations = []
        seen_operations = set()  # For deduplication
        
        logger.info(f"Extracting file operations from {len(lines)} lines")
        
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            
            # Skip comments and empty lines
            if not line_upper or line_upper.startswith('*'):
                continue
            
            # FD (File Description) entries
            fd_matches = self.fd_pattern.findall(line_upper)
            for file_name in fd_matches:
                if self._is_valid_filename(file_name):
                    op_key = f"FD_{file_name}"
                    if op_key not in seen_operations:
                        seen_operations.add(op_key)
                        operations.append({
                            'operation': 'FD',
                            'file_name': file_name,
                            'friendly_name': self.generate_friendly_name(file_name, 'File'),
                            'line_number': i + 1,
                            'line_content': line.strip(),
                            'file_type': 'FD_FILE'
                        })
            
            # SELECT statements
            select_matches = self.select_pattern.findall(line_upper)
            for logical_name, physical_name in select_matches:
                if self._is_valid_filename(logical_name):
                    op_key = f"SELECT_{logical_name}"
                    if op_key not in seen_operations:
                        seen_operations.add(op_key)
                        operations.append({
                            'operation': 'SELECT',
                            'file_name': logical_name,
                            'physical_name': physical_name,
                            'friendly_name': self.generate_friendly_name(logical_name, 'File'),
                            'line_number': i + 1,
                            'line_content': line.strip(),
                            'file_type': 'SELECT_FILE'
                        })
            
            # READ/WRITE/OPEN/CLOSE operations
            file_op_matches = self.file_op_pattern.findall(line_upper)
            for operation, file_name in file_op_matches:
                if self._is_valid_filename(file_name):
                    op_key = f"{operation}_{file_name}"
                    if op_key not in seen_operations:
                        seen_operations.add(op_key)
                        operations.append({
                            'operation': operation.upper(),
                            'file_name': file_name,
                            'friendly_name': self.generate_friendly_name(file_name, 'File'),
                            'line_number': i + 1,
                            'line_content': line.strip(),
                            'file_type': 'PROCEDURAL_FILE'
                        })
        
        logger.info(f"Found {len(operations)} unique file operations")
        return operations

    def extract_cics_operations(self, lines: List[str]) -> List[Dict]:
        """Extract CICS operations with deduplication and proper filtering"""
        operations = []
        seen_operations = set()
        
        # Enhanced CICS patterns
        cics_patterns = {
            'READ': r'EXEC\s+CICS\s+READ\s+(?:FILE|DATASET)\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)',
            'WRITE': r'EXEC\s+CICS\s+WRITE\s+(?:FILE|DATASET)\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)',
            'REWRITE': r'EXEC\s+CICS\s+REWRITE\s+(?:FILE|DATASET)\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)',
            'DELETE': r'EXEC\s+CICS\s+DELETE\s+(?:FILE|DATASET)\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)',
            'SEND': r'EXEC\s+CICS\s+SEND\s+(?:MAP|TEXT)',
            'RECEIVE': r'EXEC\s+CICS\s+RECEIVE\s+(?:MAP|INTO)',
            'START': r'EXEC\s+CICS\s+START\s+PROGRAM',
            'LINK': r'EXEC\s+CICS\s+LINK\s+PROGRAM\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)',
            'XCTL': r'EXEC\s+CICS\s+XCTL\s+PROGRAM\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)',
            'RETURN': r'EXEC\s+CICS\s+RETURN'
        }
        
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            
            # Skip comments
            if not line_upper or line_upper.startswith('*'):
                continue
                
            for operation, pattern in cics_patterns.items():
                matches = re.findall(pattern, line_upper)
                
                if re.search(pattern, line_upper):
                    operation_info = {
                        'operation': f"CICS {operation}",
                        'line_number': i + 1,
                        'line_content': line.strip(),
                        'friendly_name': f"CICS {operation.title()} Operation"
                    }
                    
                    # Extract file/program names for operations with parameters
                    if matches and operation in ['READ', 'WRITE', 'REWRITE', 'DELETE']:
                        file_name = matches[0]
                        if self._is_valid_filename(file_name):
                            op_key = f"CICS_{operation}_{file_name}"
                            if op_key not in seen_operations:
                                seen_operations.add(op_key)
                                operation_info['file_name'] = file_name
                                operation_info['friendly_name'] = f"CICS {operation.title()} - {self.generate_friendly_name(file_name, 'File')}"
                                operation_info['file_type'] = 'CICS_FILE'
                                
                                if operation == 'READ':
                                    operation_info['operation_type'] = 'INPUT'
                                elif operation in ['WRITE', 'REWRITE']:
                                    operation_info['operation_type'] = 'OUTPUT'
                                else:
                                    operation_info['operation_type'] = 'UPDATE'
                                
                                operations.append(operation_info)
                    
                    elif matches and operation in ['LINK', 'XCTL']:
                        program_name = matches[0]
                        if self._is_valid_filename(program_name):
                            op_key = f"CICS_{operation}_{program_name}"
                            if op_key not in seen_operations:
                                seen_operations.add(op_key)
                                operation_info['program_name'] = program_name
                                operation_info['friendly_name'] = f"CICS {operation} - {self.generate_friendly_name(program_name, 'Program')}"
                                operations.append(operation_info)
                    
                    elif operation in ['SEND', 'RECEIVE', 'START', 'RETURN']:
                        op_key = f"CICS_{operation}_{i}"  # Use line number for uniqueness
                        if op_key not in seen_operations:
                            seen_operations.add(op_key)
                            operations.append(operation_info)
        
        logger.info(f"Found {len(operations)} unique CICS operations")
        return operations

    def extract_program_calls(self, lines: List[str]) -> List[Dict]:
        """Extract program calls with deduplication"""
        calls = []
        seen_calls = set()
        
        for i, line in enumerate(lines):
            # Skip comments
            if line.strip().startswith('*'):
                continue
                
            matches = self.call_pattern.findall(line)
            for program_name in matches:
                if self._is_valid_filename(program_name):
                    if program_name not in seen_calls:
                        seen_calls.add(program_name)
                        calls.append({
                            'program_name': program_name,
                            'friendly_name': self.generate_friendly_name(program_name, 'Program'),
                            'line_number': i + 1,
                            'line_content': line.strip()
                        })
        
        logger.info(f"Found {len(calls)} unique program calls")
        return calls

    def extract_copybooks(self, lines: List[str]) -> List[Dict]:
        """Extract copybook includes with deduplication"""
        copybooks = []
        seen_copybooks = set()
        
        for i, line in enumerate(lines):
            # Skip comments
            if line.strip().startswith('*'):
                continue
                
            matches = self.copy_pattern.findall(line)
            for copybook_name in matches:
                if self._is_valid_filename(copybook_name):
                    if copybook_name not in seen_copybooks:
                        seen_copybooks.add(copybook_name)
                        copybooks.append({
                            'copybook_name': copybook_name,
                            'friendly_name': self.generate_friendly_name(copybook_name, 'Copybook'),
                            'line_number': i + 1,
                            'line_content': line.strip()
                        })
        
        logger.info(f"Found {len(copybooks)} unique copybooks")
        return copybooks

    def extract_comprehensive_field_references(self, lines: List[str], field_name: str) -> List[Dict]:
        """Extract all references to a field with complete source code context"""
        references = []
        field_upper = field_name.upper()
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_upper = line_stripped.upper()
            
            # Skip comments and empty lines
            if not line_stripped or line_stripped.startswith('*'):
                continue
            
            # Check if field is referenced in this line
            if field_upper in line_upper:
                # Determine the type of reference
                reference_type = self._classify_field_reference(line_upper, field_upper)
                
                if reference_type != 'NONE':
                    # Get surrounding context (3 lines before and after)
                    context_start = max(0, i - 3)
                    context_end = min(len(lines), i + 4)
                    context_lines = lines[context_start:context_end]
                    
                    # Extract source and target for MOVE operations
                    source_field, target_field = self._extract_move_fields(line_upper, field_upper)
                    
                    reference = {
                        'line_number': i + 1,
                        'line_content': line_stripped,
                        'reference_type': reference_type,
                        'context_lines': context_lines,
                        'context_start_line': context_start + 1,
                        'source_field': source_field,
                        'target_field': target_field,
                        'business_context': self._infer_business_context(line_upper, reference_type)
                    }
                    
                    references.append(reference)
        
        logger.debug(f"Found {len(references)} references for field {field_name}")
        return references

    def _classify_field_reference(self, line_upper: str, field_upper: str) -> str:
        """Classify the type of field reference"""
        # Field definition
        if re.match(rf'^\s*\d{{2}}\s+{re.escape(field_upper)}\s+PIC', line_upper):
            return 'DEFINITION'
        
        # MOVE operations
        elif f'MOVE' in line_upper:
            if re.search(rf'MOVE\s+.*\s+TO\s+{re.escape(field_upper)}\b', line_upper):
                return 'MOVE_TARGET'
            elif re.search(rf'MOVE\s+{re.escape(field_upper)}\s+TO', line_upper):
                return 'MOVE_SOURCE'
        
        # Arithmetic operations
        elif re.search(rf'COMPUTE\s+{re.escape(field_upper)}\s*=', line_upper):
            return 'COMPUTE_TARGET'
        elif re.search(rf'ADD\s+.*\s+TO\s+{re.escape(field_upper)}\b', line_upper):
            return 'ADD_TARGET'
        
        # Conditional operations
        elif re.search(rf'IF\s+{re.escape(field_upper)}\b', line_upper):
            return 'CONDITION'
        
        # CICS operations
        elif 'EXEC CICS' in line_upper and field_upper in line_upper:
            return 'CICS_REFERENCE'
        
        # General reference
        elif field_upper in line_upper:
            return 'REFERENCE'
        
        return 'NONE'

    def _extract_move_fields(self, line_upper: str, field_upper: str) -> Tuple[str, str]:
        """Extract source and target fields from MOVE operations"""
        source_field = ""
        target_field = ""
        
        # MOVE source TO target
        move_match = re.search(r'MOVE\s+([A-Z0-9\-\(\)\'\"]+)\s+TO\s+([A-Z0-9\-\(\)]+)', line_upper)
        if move_match:
            if move_match.group(2) == field_upper:
                source_field = move_match.group(1)
                target_field = field_upper
            elif move_match.group(1) == field_upper:
                source_field = field_upper
                target_field = move_match.group(2)
        
        return source_field, target_field

    def _infer_business_context(self, line_upper: str, reference_type: str) -> str:
        """Infer business context from the line content and reference type"""
        context_keywords = {
            'CUSTOMER': 'Customer data processing',
            'ACCOUNT': 'Account management',
            'BALANCE': 'Balance calculation',
            'PAYMENT': 'Payment processing',
            'TRANSACTION': 'Transaction handling',
            'DATE': 'Date processing',
            'AMOUNT': 'Amount calculation',
            'TOTAL': 'Total calculation',
            'ERROR': 'Error handling',
            'VALIDATE': 'Data validation'
        }
        
        for keyword, context in context_keywords.items():
            if keyword in line_upper:
                return context
        
        # Default context based on reference type
        default_contexts = {
            'DEFINITION': 'Data structure definition',
            'MOVE_TARGET': 'Data assignment',
            'MOVE_SOURCE': 'Data retrieval',
            'COMPUTE_TARGET': 'Calculation result',
            'CONDITION': 'Business logic condition',
            'CICS_REFERENCE': 'Transaction processing'
        }
        
        return default_contexts.get(reference_type, 'General field usage')

    # Keep all the existing methods from the original parser
    # (parse_cobol_file, process_cobol_lines, extract_business_comments, etc.)
    # but enhance them with the new filtering and deduplication logic
    
    def parse_cobol_file(self, content: str, filename: str) -> Dict:
        """Main entry point to parse COBOL file with enhanced extraction"""
        raw_lines = content.split('\n')
        
        # Process COBOL lines - handle columns 1-8 and comments
        processed_result = self.process_cobol_lines(raw_lines)
        lines = processed_result['executable_lines']
        comments = processed_result['comments']
        
        result = {
            'filename': filename,
            'friendly_name': self.generate_friendly_name(filename, 'Program'),
            'divisions': [],
            'components': [],
            'record_layouts': [],
            'file_operations': [],
            'program_calls': [],
            'copybooks': [],
            'cics_operations': [],
            'mq_operations': [],
            'xml_operations': [],
            'total_lines': len(raw_lines),
            'executable_lines': len(lines),
            'comment_lines': len(comments),
            'comments_summary': comments[:10],
            'business_comments': self.extract_business_comments(comments)
        }
        
        # Parse divisions and components
        result['divisions'] = self.extract_divisions(lines)
        result['components'] = self.extract_components(lines)
        result['record_layouts'] = self.extract_record_layouts_with_field_references(lines, raw_lines)
        
        # Extract operations with new deduplication
        result['file_operations'] = self.extract_file_operations(lines)
        result['program_calls'] = self.extract_program_calls(lines)
        result['copybooks'] = self.extract_copybooks(lines)
        result['cics_operations'] = self.extract_cics_operations(lines)
        result['mq_operations'] = self.extract_mq_operations(lines)
        result['xml_operations'] = self.extract_xml_operations(lines)
        
        return result

    def extract_record_layouts_with_field_references(self, lines: List[str], raw_lines: List[str]) -> List[RecordLayout]:
        """Extract record layouts with comprehensive field reference extraction"""
        layouts = self.extract_record_layouts(lines)  # Use existing method
        
        # Enhance each field with comprehensive references
        for layout in layouts:
            for field in layout.fields:
                field.source_references = self.extract_comprehensive_field_references(raw_lines, field.name)
        
        return layouts
    
    def process_cobol_lines(self, raw_lines: List[str]) -> Dict:
        """Process COBOL lines handling columns 1-8 and comment indicators"""
        executable_lines = []
        comments = []
        
        for i, raw_line in enumerate(raw_lines):
            # Handle empty lines
            if not raw_line.strip():
                continue
                
            # COBOL line format: columns 1-6 (sequence), 7 (indicator), 8+ (content)
            if len(raw_line) < 7:
                # Short line, treat as comment or empty
                if raw_line.strip():
                    comments.append(raw_line.strip())
                continue
            
            # Extract indicator column (column 7)
            indicator = raw_line[6:7] if len(raw_line) > 6 else ' '
            
            # Extract content (columns 8+)
            content = raw_line[7:] if len(raw_line) > 7 else ''
            
            # Check for comment indicators
            if indicator in ['*', '/', 'C', 'c', 'D', 'd']:
                # Comment line - extract for LLM summary
                comment_text = content.strip()
                if comment_text:
                    comments.append(comment_text)
            elif indicator == '-':
                # Continuation line - append to previous executable line
                if executable_lines:
                    executable_lines[-1] += ' ' + content.strip()
            else:
                # Executable line (indicator is space or other)
                if content.strip():
                    executable_lines.append(content.strip())
        
        return {
            'executable_lines': executable_lines,
            'comments': comments
        }
    
    def extract_business_comments(self, comments: List[str]) -> List[str]:
        """Extract business-relevant comments for LLM processing"""
        business_keywords = [
            'FUNCTION', 'PURPOSE', 'DESCRIPTION', 'BUSINESS', 'LOGIC', 'RULE',
            'PROCESS', 'PROCEDURE', 'METHOD', 'ALGORITHM', 'CALCULATION',
            'INPUT', 'OUTPUT', 'FILE', 'TABLE', 'RECORD', 'FIELD',
            'AUTHOR', 'DATE-WRITTEN', 'DATE-COMPILED', 'PROGRAM-ID'
        ]
        
        business_comments = []
        for comment in comments:
            comment_upper = comment.upper()
            if any(keyword in comment_upper for keyword in business_keywords):
                business_comments.append(comment)
            elif len(comment.strip()) > 30:  # Long descriptive comments
                business_comments.append(comment)
        
        return business_comments[:20] 
    

    def extract_data_movements(self, lines: List[str]) -> List[Dict]:
        """Extract data movement operations (MOVE, COMPUTE)"""
        movements = []
        seen_movements = set()
        
        move_pattern = re.compile(r'MOVE\s+([A-Z0-9\-\(\)\'\"]+)\s+TO\s+([A-Z0-9\-\(\)]+)', re.IGNORECASE)
        compute_pattern = re.compile(r'COMPUTE\s+([A-Z0-9\-]+)\s*=\s*(.+)', re.IGNORECASE)
        
        for i, line in enumerate(lines):
            # Skip comments
            if line.strip().startswith('*'):
                continue
                
            # MOVE operations
            move_matches = move_pattern.findall(line)
            for source, target in move_matches:
                movement_key = f"MOVE_{source}_{target}_{i}"
                if movement_key not in seen_movements:
                    seen_movements.add(movement_key)
                    movements.append({
                        'operation': 'MOVE',
                        'source_field': source,
                        'target_field': target,
                        'source_friendly': self.generate_friendly_name(source, 'Field'),
                        'target_friendly': self.generate_friendly_name(target, 'Field'),
                        'line_number': i + 1,
                        'line_content': line.strip()
                    })
            
            # COMPUTE operations
            compute_matches = compute_pattern.findall(line)
            for target, expression in compute_matches:
                movement_key = f"COMPUTE_{target}_{i}"
                if movement_key not in seen_movements:
                    seen_movements.add(movement_key)
                    movements.append({
                        'operation': 'COMPUTE',
                        'target_field': target,
                        'expression': expression,
                        'target_friendly': self.generate_friendly_name(target, 'Field'),
                        'line_number': i + 1,
                        'line_content': line.strip()
                    })
        
        return movements

    def extract_business_logic_patterns(self, lines: List[str]) -> List[Dict]:
        """Extract business logic patterns"""
        patterns = []
        seen_patterns = set()
        
        # Pattern definitions
        business_patterns = [
            {
                'type': 'VALIDATION',
                'pattern': r'IF\s+.*\s+(INVALID|ERROR|FAIL)',
                'description': 'Data validation logic'
            },
            {
                'type': 'CALCULATION',
                'pattern': r'COMPUTE\s+.*\s*=\s*.*[\+\-\*\/]',
                'description': 'Mathematical calculation'
            },
            {
                'type': 'DATE_PROCESSING',
                'pattern': r'(DATE|TIME|TIMESTAMP)',
                'description': 'Date/time processing'
            },
            {
                'type': 'FILE_PROCESSING',
                'pattern': r'(READ|WRITE|OPEN|CLOSE)\s+\w+',
                'description': 'File operation'
            },
            {
                'type': 'DECISION',
                'pattern': r'EVALUATE\s+.*WHEN',
                'description': 'Decision logic'
            }
        ]
        
        for i, line in enumerate(lines):
            # Skip comments
            if line.strip().startswith('*'):
                continue
                
            for pattern_def in business_patterns:
                if re.search(pattern_def['pattern'], line, re.IGNORECASE):
                    pattern_key = f"{pattern_def['type']}_{i}"
                    if pattern_key not in seen_patterns:
                        seen_patterns.add(pattern_key)
                        patterns.append({
                            'type': pattern_def['type'],
                            'description': pattern_def['description'],
                            'line_number': i + 1,
                            'line_content': line.strip(),
                            'pattern_matched': pattern_def['pattern']
                        })
        
        return patterns
    
    def extract_divisions(self, lines: List[str]) -> List[Dict]:
        """Extract COBOL divisions"""
        divisions = []
        current_division = None
        
        for i, line in enumerate(lines):
            match = self.division_pattern.match(line)
            if match:
                if current_division:
                    current_division['line_end'] = i - 1
                    divisions.append(current_division)
                
                division_name = match.group(1).upper()
                current_division = {
                    'name': division_name,
                    'friendly_name': self.generate_friendly_name(division_name, 'Division'),
                    'line_start': i + 1,
                    'line_end': len(lines),
                    'sections': []
                }
        
        # Close last division
        if current_division:
            divisions.append(current_division)
        
        # Extract sections within divisions
        for division in divisions:
            division['sections'] = self.extract_sections(lines, division['line_start'], division['line_end'])
        
        return divisions
    
    def extract_sections(self, lines: List[str], start_line: int, end_line: int) -> List[Dict]:
        """Extract sections within a division"""
        sections = []
        current_section = None
        
        for i in range(start_line, min(end_line, len(lines))):
            line = lines[i]
            match = self.section_pattern.match(line)
            
            if match:
                if current_section:
                    current_section['line_end'] = i
                    sections.append(current_section)
                
                section_name = match.group(1).upper()
                current_section = {
                    'name': section_name,
                    'friendly_name': self.generate_friendly_name(section_name, 'Section'),
                    'line_start': i + 1,
                    'line_end': end_line
                }
        
        # Close last section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def extract_components(self, lines: List[str]) -> List[CobolComponent]:
        """Extract COBOL components (paragraphs, sections, etc.)"""
        components = []
        current_component = None
        
        for i, line in enumerate(lines):
            # Check for paragraph
            paragraph_match = self.paragraph_pattern.match(line)
            if paragraph_match and not self.division_pattern.match(line) and not self.section_pattern.match(line):
                if current_component:
                    current_component.line_end = i - 1
                    components.append(current_component)
                
                name = paragraph_match.group(1)
                current_component = CobolComponent(
                    name=name,
                    component_type='PARAGRAPH',
                    line_start=i + 1,
                    line_end=len(lines),
                    content="",
                    friendly_name=self.generate_friendly_name(name, 'Paragraph')
                )
        
        # Close last component
        if current_component:
            components.append(current_component)
        
        # Extract content for each component
        for component in components:
            content_lines = lines[component.line_start-1:component.line_end]
            component.content = '\n'.join(content_lines)
        
        return components
    
    def extract_record_layouts(self, lines: List[str]) -> List[RecordLayout]:
        """Fixed record layout extraction with proper section handling"""
        layouts = []
        current_layout = None
        current_fields = []
        in_data_division = False
        current_section = None
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_upper = line_stripped.upper()
            
            # Check for division
            if self.division_pattern.match(line_stripped):
                division_name = self.division_pattern.match(line_stripped).group(1).upper()
                in_data_division = (division_name == 'DATA')
                continue
            
            if not in_data_division:
                continue
            
            # Check for section within DATA DIVISION
            if line_upper.endswith(' SECTION.'):
                current_section = line_upper.replace(' SECTION.', '').strip()
                # Finalize any current layout when entering new section
                if current_layout:
                    # Always finalize the current layout (even if it has no fields)
                    current_layout.line_end = i
                    current_layout.fields = current_fields
                    source_lines = lines[current_layout.line_start-1:current_layout.line_end]
                    current_layout.source_code = '\n'.join(source_lines)
                    layouts.append(current_layout)
                    current_layout = None
                    current_fields = []
                continue
            
            # Parse data items
            data_match = self.data_item_pattern.match(line_stripped)
            if data_match:
                level = int(data_match.group(1))
                name = data_match.group(2)
                rest_of_line = data_match.group(3) or ""
                
                if level == 1:
                    # Finalize previous layout before starting new one
                    if current_layout:
                        # Always finalize existing 01-level layout. Previous implementation only
                        # finalized when fields existed which caused earlier 01 groups to be
                        # overwritten and multiple layouts to be consolidated into one.
                        current_layout.line_end = i
                        current_layout.fields = current_fields
                        source_lines = lines[current_layout.line_start-1:current_layout.line_end]
                        current_layout.source_code = '\n'.join(source_lines)
                        layouts.append(current_layout)
                    
                    # Skip FILLER
                    if name.upper() == 'FILLER':
                        current_layout = None
                        current_fields = []
                        continue
                    
                    # Create new layout
                    current_layout = RecordLayout(
                        name=name,
                        level=level,
                        fields=[],
                        line_start=i + 1,
                        line_end=len(lines),
                        source_code="",
                        friendly_name=self.generate_friendly_name(name, 'Record Layout')
                    )
                    # Preserve the section name if one was detected, otherwise keep unknown
                    current_layout.section = current_section or 'WORKING-STORAGE'
                    current_fields = []
                    
                elif current_layout and level > 1:
                    if name.upper() != 'FILLER':
                        field = self.parse_cobol_field(line_stripped, i + 1, level, name, rest_of_line)
                        current_fields.append(field)
        
        # Finalize last layout (if any)
        if current_layout:
            current_layout.line_end = len(lines)
            current_layout.fields = current_fields
            source_lines = lines[current_layout.line_start-1:current_layout.line_end]
            current_layout.source_code = '\n'.join(source_lines)
            layouts.append(current_layout)
        
        return layouts
    
    def parse_cobol_field(self, line: str, line_number: int, level: int, name: str, definition: str) -> CobolField:
        """Parse individual COBOL field definition"""
        field = CobolField(
            name=name,
            level=level,
            picture="",
            line_number=line_number,
            friendly_name=self.generate_friendly_name(name, 'Field')
        )
        
        # Extract PICTURE clause
        pic_match = self.picture_pattern.search(definition)
        if pic_match:
            field.picture = pic_match.group(1)
        
        # Extract USAGE clause
        usage_match = self.usage_pattern.search(definition)
        if usage_match:
            field.usage = usage_match.group(1)
        
        # Extract OCCURS clause
        occurs_match = self.occurs_pattern.search(definition)
        if occurs_match:
            field.occurs = int(occurs_match.group(1))
        
        # Extract REDEFINES clause
        redefines_match = self.redefines_pattern.search(definition)
        if redefines_match:
            field.redefines = redefines_match.group(1)
        
        # Extract VALUE clause
        value_match = self.value_pattern.search(definition)
        if value_match:
            field.value = value_match.group(1).strip('\'"')
        
        return field
    

    def extract_mq_operations(self, lines: List[str]) -> List[Dict]:
        """Extract MQ (Message Queue) operations"""
        operations = []
        
        mq_patterns = {
            'MQOPEN': r'CALL\s+[\'"]MQOPEN[\'"]',
            'MQGET': r'CALL\s+[\'"]MQGET[\'"]',
            'MQPUT': r'CALL\s+[\'"]MQPUT[\'"]',
            'MQPUT1': r'CALL\s+[\'"]MQPUT1[\'"]',
            'MQCLOSE': r'CALL\s+[\'"]MQCLOSE[\'"]',
            'MQCONN': r'CALL\s+[\'"]MQCONN[\'"]',
            'MQDISC': r'CALL\s+[\'"]MQDISC[\'"]',
            'MQINQ': r'CALL\s+[\'"]MQINQ[\'"]',
            'MQSET': r'CALL\s+[\'"]MQSET[\'"]'
        }
        
        # Also check for MQ copybooks
        mq_copybook_pattern = r'COPY\s+(CMQV|CMQP|CMQC|CMQL)'
        
        for i, line in enumerate(lines):
            # Check for MQ API calls
            for operation, pattern in mq_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    operations.append({
                        'operation': f"MQ {operation}",
                        'line_number': i + 1,
                        'line_content': line.strip(),
                        'friendly_name': f"MQ {operation} Operation",
                        'middleware_type': 'MQ'
                    })
            
            # Check for MQ copybooks
            if re.search(mq_copybook_pattern, line, re.IGNORECASE):
                operations.append({
                    'operation': 'MQ COPYBOOK',
                    'line_number': i + 1,
                    'line_content': line.strip(),
                    'friendly_name': 'MQ Copybook Include',
                    'middleware_type': 'MQ'
                })
        
        return operations
    
    def extract_xml_operations(self, lines: List[str]) -> List[Dict]:
        """Extract XML processing operations"""
        operations = []
        
        xml_patterns = {
            'XML_PARSE': r'XML\s+PARSE',
            'XML_GENERATE': r'XML\s+GENERATE',
            'XML_TRANSFORM': r'XML\s+TRANSFORM',
            'JSON_PARSE': r'JSON\s+PARSE',
            'JSON_GENERATE': r'JSON\s+GENERATE'
        }
        
        # XML-related copybooks
        xml_copybook_patterns = [
            r'COPY\s+(DFHXMLX|DFHXML)',
            r'COPY\s+.*XML.*',
            r'COPY\s+.*JSON.*'
        ]
        
        for i, line in enumerate(lines):
            # Check for XML/JSON operations
            for operation, pattern in xml_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    operations.append({
                        'operation': operation.replace('_', ' '),
                        'line_number': i + 1,
                        'line_content': line.strip(),
                        'friendly_name': f"{operation.replace('_', ' ')} Operation",
                        'data_format': 'XML' if 'XML' in operation else 'JSON'
                    })
            
            # Check for XML-related copybooks
            for pattern in xml_copybook_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    operations.append({
                        'operation': 'XML COPYBOOK',
                        'line_number': i + 1,
                        'line_content': line.strip(),
                        'friendly_name': 'XML/JSON Processing Copybook',
                        'data_format': 'XML'
                    })
        
        return operations
    
    def extract_data_movements(self, lines: List[str]) -> List[Dict]:
        """Extract data movement operations (MOVE, COMPUTE)"""
        movements = []
        
        for i, line in enumerate(lines):
            # MOVE operations
            move_matches = self.move_pattern.findall(line)
            for source, target in move_matches:
                movements.append({
                    'operation': 'MOVE',
                    'source_field': source,
                    'target_field': target,
                    'source_friendly': self.generate_friendly_name(source, 'Field'),
                    'target_friendly': self.generate_friendly_name(target, 'Field'),
                    'line_number': i + 1,
                    'line_content': line.strip()
                })
            
            # COMPUTE operations
            compute_matches = self.compute_pattern.findall(line)
            for target, expression in compute_matches:
                movements.append({
                    'operation': 'COMPUTE',
                    'target_field': target,
                    'expression': expression,
                    'target_friendly': self.generate_friendly_name(target, 'Field'),
                    'line_number': i + 1,
                    'line_content': line.strip()
                })
        
        return movements
    
    def convert_pic_to_oracle_type(self, picture: str, usage: str = "") -> Tuple[str, int]:
        """Convert COBOL PIC clause to Oracle data type"""
        if not picture:
            return "VARCHAR2(100)", 100
        
        pic_upper = picture.upper()
        
        # Numeric types
        if re.match(r'9+|\d+', pic_upper):
            # Simple numeric
            length = len(re.sub(r'[^\d9]', '', pic_upper))
            if usage.upper() in ['COMP-3', 'PACKED-DECIMAL']:
                return f"NUMBER({length})", length
            else:
                return f"NUMBER({length})", length
        
        elif 'V' in pic_upper:
            # Decimal numbers
            parts = pic_upper.split('V')
            if len(parts) == 2:
                integer_part = len(re.sub(r'[^\d9]', '', parts[0]))
                decimal_part = len(re.sub(r'[^\d9]', '', parts[1]))
                total_length = integer_part + decimal_part
                return f"NUMBER({total_length},{decimal_part})", total_length
        
        elif re.match(r'S?9+(\(\d+\))?V9+(\(\d+\))?', pic_upper):
            # Pattern like S9(5)V9(2)
            match = re.match(r'S?9+(\((\d+)\))?V9+(\((\d+)\))?', pic_upper)
            if match:
                int_digits = int(match.group(2)) if match.group(2) else 1
                dec_digits = int(match.group(4)) if match.group(4) else 1
                return f"NUMBER({int_digits + dec_digits},{dec_digits})", int_digits + dec_digits
        
        # Alphanumeric types
        elif 'X' in pic_upper:
            # Extract length from X(n) or XXX format
            x_match = re.search(r'X+(\((\d+)\))?', pic_upper)
            if x_match:
                if x_match.group(2):
                    length = int(x_match.group(2))
                else:
                    length = len(re.findall(r'X', pic_upper))
                
                if length <= 4000:
                    return f"VARCHAR2({length})", length
                else:
                    return "CLOB", length
        
        # Default case
        return "VARCHAR2(100)", 100
    
    def analyze_field_usage(self, lines: List[str], field_name: str) -> Dict:
        """Analyze how a field is used in the code"""
        usage_analysis = {
            'field_name': field_name,
            'usage_type': 'UNUSED',
            'operations': [],
            'source_operations': [],
            'target_operations': [],
            'references': []
        }
        
        field_pattern = re.compile(rf'\b{re.escape(field_name)}\b', re.IGNORECASE)
        
        for i, line in enumerate(lines):
            if field_pattern.search(line):
                usage_analysis['references'].append({
                    'line_number': i + 1,
                    'line_content': line.strip()
                })
                
                # Analyze operation type
                if re.search(rf'MOVE\s+.*\bTO\s+{re.escape(field_name)}\b', line, re.IGNORECASE):
                    usage_analysis['target_operations'].append({
                        'operation': 'MOVE_TO',
                        'line_number': i + 1,
                        'line_content': line.strip()
                    })
                    usage_analysis['usage_type'] = 'INPUT'
                
                elif re.search(rf'MOVE\s+{re.escape(field_name)}\s+TO\b', line, re.IGNORECASE):
                    usage_analysis['source_operations'].append({
                        'operation': 'MOVE_FROM',
                        'line_number': i + 1,
                        'line_content': line.strip()
                    })
                    usage_analysis['usage_type'] = 'OUTPUT'
                
                elif re.search(rf'COMPUTE\s+{re.escape(field_name)}\s*=', line, re.IGNORECASE):
                    usage_analysis['target_operations'].append({
                        'operation': 'COMPUTE',
                        'line_number': i + 1,
                        'line_content': line.strip()
                    })
                    usage_analysis['usage_type'] = 'DERIVED'
                
                elif re.search(rf'IF\s+.*{re.escape(field_name)}\b', line, re.IGNORECASE):
                    usage_analysis['operations'].append({
                        'operation': 'CONDITION',
                        'line_number': i + 1,
                        'line_content': line.strip()
                    })
                    if usage_analysis['usage_type'] == 'UNUSED':
                        usage_analysis['usage_type'] = 'REFERENCE'
        
        # Determine final usage type
        if usage_analysis['target_operations'] and usage_analysis['source_operations']:
            usage_analysis['usage_type'] = 'INPUT_OUTPUT'
        elif not usage_analysis['references']:
            usage_analysis['usage_type'] = 'UNUSED'
        elif usage_analysis['references'] and not usage_analysis['target_operations']:
            usage_analysis['usage_type'] = 'STATIC'
        
        return usage_analysis