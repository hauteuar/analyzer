"""
Consolidated COBOL Parser and Component Extractor with LLM-driven naming and proper column handling

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
    annotations: Dict = field(default_factory=dict)


class COBOLParser:
    def __init__(self, llm_client=None):
        # Division/section/paragraph patterns
        self.division_pattern = re.compile(r'^\s*(IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION', re.IGNORECASE)
        self.section_pattern = re.compile(r'^\s*(\w+)\s+SECTION\s*\.', re.IGNORECASE)
        self.paragraph_pattern = re.compile(r'^\s*([A-Z0-9\-]+)\s*\.', re.IGNORECASE)

        # Data item
        #self.data_item_pattern = re.compile(r'^\s*(\d+)\s+([A-Z0-9\-]+)(?:\s+(.*))?$', re.IGNORECASE)
        self.data_item_pattern = re.compile(r'^\s*(\d+)\s+([A-Z0-9\-]+)(?:\s+(.*?))?\s*\.?\s*$', re.IGNORECASE)
        # File / program / copy patterns (operate on program area only)
        self.fd_pattern = re.compile(r'FD\s+([A-Z][A-Z0-9\-]{2,})', re.IGNORECASE)
        self.select_pattern = re.compile(r'SELECT\s+([A-Z][A-Z0-9\-]{2,})\s+ASSIGN\s+TO\s+([A-Z0-9\-\.]+)', re.IGNORECASE)
        self.file_op_pattern = re.compile(r'\b(READ|WRITE|OPEN|CLOSE|REWRITE|DELETE)\s+([A-Z][A-Z0-9\-]{2,})', re.IGNORECASE)
        self.call_pattern = re.compile(r'CALL\s+[\'\"]([A-Z0-9\-]{3,})[\'\"]', re.IGNORECASE)
        self.copy_pattern = re.compile(r'COPY\s+([A-Z0-9\-]{3,})(?:\s+|\.)', re.IGNORECASE)

        # CICS patterns
        self.cics_simple_pattern = re.compile(r'EXEC\s+CICS\s+(READ|WRITE|REWRITE|DELETE)\b', re.IGNORECASE)
        self.cics_file_pattern = re.compile(r'EXEC\s+CICS\s+(READ|WRITE|REWRITE|DELETE)\s+(?:FILE|DATASET)\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', re.IGNORECASE)

        self.llm_client = llm_client
        self.friendly_name_cache = {}

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

    def _is_valid_cics_filename_enhanced(self, name: str) -> bool:
        if not name or len(name) < 3:  # Reduced from 6 to 3
            return False
        
        name_upper = name.upper()
        
        # Must start with letter
        if not re.match(r'^[A-Z]', name_upper):
            return False
        
        # More flexible pattern - allow 3+ character files
        if not re.match(r'^[A-Z][A-Z0-9]{2,29}', name_upper):
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

    def extract_program_calls(self, content: str, filename: str) -> List[Dict]:
        """
        Enhanced program call extraction including CICS LINK/XCTL calls
        FIXED: Now properly calls CICS program parsing methods
        """
        program_calls = []
        lines = content.split('\n')
        
        logger.info(f"Starting program call extraction from {filename}")
        
        try:
            # Extract static calls (existing logic)
            static_calls = self._extract_static_program_calls(content, filename)
            program_calls.extend(static_calls)
            logger.info(f"Found {len(static_calls)} static program calls")
            
            # FIXED: Extract CICS program calls (LINK/XCTL) - this was missing!
            cics_program_calls = self._extract_cics_program_calls(lines)
            program_calls.extend(cics_program_calls)
            logger.info(f"Found {len(cics_program_calls)} CICS program calls")
            
            # Extract dynamic calls (with improved error handling)
            try:
                dynamic_calls = self.extract_dynamic_program_calls(content, filename)
                
                # Convert dynamic calls to program call format
                for dynamic_call in dynamic_calls:
                    for resolved_program in dynamic_call['resolved_programs']:
                        program_calls.append({
                            'operation': dynamic_call['operation'],
                            'program_name': resolved_program['program_name'],
                            'line_number': dynamic_call['line_number'],
                            'call_type': 'dynamic',
                            'variable_name': dynamic_call['variable_name'],
                            'resolution_method': resolved_program['resolution'],
                            'confidence_score': resolved_program['confidence'],
                            'source_info': resolved_program.get('source', ''),
                            'business_context': f"Dynamic call via {dynamic_call['variable_name']} variable"
                        })
                
                logger.info(f"Found {len(dynamic_calls)} dynamic program calls")
                
            except Exception as e:
                logger.error(f"Dynamic program call extraction failed: {str(e)}")
                # Continue without dynamic calls
        
        except Exception as e:
            logger.error(f"Program call extraction failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info(f"Total program calls extracted: {len(program_calls)}")
        return program_calls


    def _extract_cics_program_calls(self, lines: List[str]) -> List[Dict]:
        """
        Extract CICS LINK and XCTL program calls
        FIXED: More comprehensive CICS program call detection
        """
        cics_calls = []
        
        for i, line in enumerate(lines):
            program_area = self.extract_program_area_only(line)
            if not program_area:
                continue
            
            line_upper = program_area.upper()
            
            # Multi-line CICS command handling
            if 'EXEC CICS' in line_upper and ('LINK' in line_upper or 'XCTL' in line_upper):
                logger.debug(f"Found potential CICS program call at line {i+1}: {program_area[:80]}...")
                
                # Extract complete CICS command across multiple lines
                cics_command = self._extract_complete_cics_command(lines, i)
                
                if cics_command:
                    logger.debug(f"Complete CICS command: {cics_command[:150]}...")
                    
                    # Parse the complete command for program calls
                    parsed_calls = self._parse_cics_program_calls(cics_command, i + 1)
                    
                    if parsed_calls:
                        cics_calls.extend(parsed_calls)
                        logger.debug(f"Extracted {len(parsed_calls)} program calls from CICS command")
        
        return cics_calls
    
    def _extract_static_program_calls(self, content: str, filename: str) -> List[Dict]:
        """Extract static/literal program calls (existing functionality)"""
        static_calls = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_upper = line.upper().strip()
            
            if not line_upper or line_upper.startswith('*'):
                continue
            
            # Static CALL statements
            call_match = re.search(r"CALL\s+['\"]([A-Z0-9\-]+)['\"]", line_upper)
            if call_match:
                static_calls.append({
                    'operation': 'CALL',
                    'program_name': call_match.group(1),
                    'line_number': i,
                    'call_type': 'static',
                    'confidence_score': 1.0,
                    'business_context': 'Static program call'
                })
            
            # Static CICS XCTL/LINK with literal program names
            static_cics_match = re.search(r"EXEC\s+CICS\s+(XCTL|LINK)\s+PROGRAM\s*\(\s*['\"]([A-Z0-9\-]+)['\"]\s*\)", line_upper)
            if static_cics_match:
                static_calls.append({
                    'operation': f'CICS_{static_cics_match.group(1)}',
                    'program_name': static_cics_match.group(2),
                    'line_number': i,
                    'call_type': 'static',
                    'confidence_score': 1.0,
                    'business_context': 'Static CICS program call'
                })
        
        return static_calls

    def _parse_cics_program_calls(self, cics_command: str, line_number: int) -> List[Dict]:
        """Parse CICS LINK/XCTL program calls from complete CICS command"""
        operations = []
        cics_upper = cics_command.upper()
        
        logger.debug(f"Parsing CICS command for program calls: {cics_command[:100]}...")
        
        # Enhanced patterns for CICS program calls (LINK/XCTL)
        cics_program_patterns = [
            # LINK patterns with various formats
            (r'EXEC\s+CICS\s+LINK\s+PROGRAM\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', 'CICS_LINK'),
            (r'EXEC\s+CICS\s+LINK\s+.*?PROGRAM\s*\(\s*[\'"]?\s*([A-Z0-9\-]{3,})\s*[\'"]?\s*\)', 'CICS_LINK'),
            
            # XCTL patterns with various formats  
            (r'EXEC\s+CICS\s+XCTL\s+PROGRAM\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', 'CICS_XCTL'),
            (r'EXEC\s+CICS\s+XCTL\s+.*?PROGRAM\s*\(\s*[\'"]?\s*([A-Z0-9\-]{3,})\s*[\'"]?\s*\)', 'CICS_XCTL'),
            
            # START patterns for started transactions
            (r'EXEC\s+CICS\s+START\s+.*?TRANSID\s*\(\s*[\'"]?\s*([A-Z0-9\-]{3,})\s*[\'"]?\s*\)', 'CICS_START'),
            
            # Handle spacing variations - PROGRAM ( 'TMST9P4' )
            (r'EXEC\s+CICS\s+LINK\s+.*?PROGRAM\s*\(\s*[\'"]?\s*([A-Z0-9\-]{3,})\s*[\'"]?\s*\)', 'CICS_LINK'),
            (r'EXEC\s+CICS\s+XCTL\s+.*?PROGRAM\s*\(\s*[\'"]?\s*([A-Z0-9\-]{3,})\s*[\'"]?\s*\)', 'CICS_XCTL'),
            
            # Dynamic program calls with variables
            (r'EXEC\s+CICS\s+LINK\s+.*?PROGRAM\s*\(\s*([A-Z0-9\-]{3,})\s*\)', 'CICS_LINK_DYNAMIC'),
            (r'EXEC\s+CICS\s+XCTL\s+.*?PROGRAM\s*\(\s*([A-Z0-9\-]{3,})\s*\)', 'CICS_XCTL_DYNAMIC'),
        ]
        
        for pattern, operation in cics_program_patterns:
            matches = re.findall(pattern, cics_upper, re.DOTALL)
            for program_name in matches:
                program_name = program_name.strip()
                
                if self._is_valid_cobol_filename(program_name):
                    operations.append({
                        'operation': operation,
                        'program_name': program_name,
                        'call_type': 'CICS',
                        'line_number': line_number,
                        'line_content': cics_command[:100],
                        'relationship_type': operation,
                        'confidence_score': 0.95,
                        'business_context': f"CICS {operation.split('_')[1]} call to {program_name}"
                    })
                    logger.debug(f"Found CICS program call: {operation} -> {program_name}")
        
        return operations
    
    def _extract_cics_record_layout(self, cics_op: Dict) -> Optional[str]:
        """Extract record layout name from CICS INTO/FROM clause"""
        source_line = cics_op.get('source_line', '')
        
        # Extract INTO clause
        into_match = re.search(r'INTO\s*\(\s*([A-Z][A-Z0-9\-]+)\s*\)', source_line.upper())
        if into_match:
            return into_match.group(1)
        
        # Extract FROM clause  
        from_match = re.search(r'FROM\s*\(\s*([A-Z][A-Z0-9\-]+)\s*\)', source_line.upper())
        if from_match:
            return from_match.group(1)
        
        return None
    
    def _extract_cics_programs(self, cics_command: str, line_number: int) -> List[Dict]:
        """Extract program names from CICS LINK/XCTL commands"""
        programs = []
        cics_upper = cics_command.upper()
        
        # Enhanced patterns for CICS program calls
        patterns = [
            # LINK with various formats
            (r'EXEC\s+CICS\s+LINK\s+PROGRAM\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', 'CICS_LINK'),
            (r'EXEC\s+CICS\s+LINK\s+.*?PROGRAM\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', 'CICS_LINK'),
            
            # XCTL with various formats  
            (r'EXEC\s+CICS\s+XCTL\s+PROGRAM\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', 'CICS_XCTL'),
            (r'EXEC\s+CICS\s+XCTL\s+.*?PROGRAM\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)', 'CICS_XCTL'),
            
            # Handle spaces in parentheses: PROGRAM ( 'TMST9P4' )
            (r'EXEC\s+CICS\s+LINK\s+.*?PROGRAM\s*\(\s*[\'"]?\s*([A-Z0-9\-]{3,})\s*[\'"]?\s*\)', 'CICS_LINK'),
            (r'EXEC\s+CICS\s+XCTL\s+.*?PROGRAM\s*\(\s*[\'"]?\s*([A-Z0-9\-]{3,})\s*[\'"]?\s*\)', 'CICS_XCTL'),
        ]
        
        for pattern, operation in patterns:
            matches = re.findall(pattern, cics_upper, re.DOTALL)
            for program_name in matches:
                if self._is_valid_cobol_filename(program_name):
                    programs.append({
                        'operation': operation,
                        'program_name': program_name,
                        'call_type': 'CICS',
                        'line_number': line_number,
                        'line_content': cics_command[:100],
                        'relationship_type': operation
                    })
        
        return programs
    def _extract_complete_cics_command(self, lines: List[str], start: int) -> str:
        """Extract complete CICS command across multiple lines"""
        cics_command = ""
        
        for i in range(start, min(len(lines), start + 15)):  # Increased range
            program_area = self.extract_program_area_only(lines[i])
            if not program_area:
                continue
            
            cics_command += " " + program_area.strip()
            
            # Check for command termination
            if 'END-EXEC' in cics_command.upper():
                break
            
            # Also check for period termination (some CICS commands)
            if cics_command.strip().endswith('.') and len(cics_command) > 20:
                break
        
        return cics_command.strip()

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

    def extract_cics_operations(self, lines: List[str]) -> List[Dict]:
        operations = []
        seen_operations = set()
        
        logger.info(f"Starting CICS extraction from {len(lines)} lines")
        
        for i, line in enumerate(lines):
            program_area = self.extract_program_area_only(line)
            if not program_area:
                continue
            
            if 'EXEC CICS' in program_area.upper():
                logger.debug(f"Found CICS line {i+1}: {program_area[:100]}")
                cics_command = self._extract_complete_cics_command(lines, i)
                if cics_command:
                    logger.debug(f"Complete CICS command: {cics_command[:200]}")
                    cics_ops = self._parse_cics_command_enhanced(cics_command, i + 1)
                    logger.debug(f"Extracted {len(cics_ops)} CICS operations")
                    
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

    def extract_mq_operations(self, lines: List[str]) -> List[Dict]:
        """Extract MQ operations with proper column handling"""
        operations = []
        mq_patterns = [r'CALL\s+[\'\"]MQOPEN[\'\"]', r'CALL\s+[\'\"]MQGET[\'\"]', r'CALL\s+[\'\"]MQPUT[\'\"]']
        for i, raw_line in enumerate(lines):
            program_area = self.extract_program_area_only(raw_line)
            if not program_area:
                continue
            for p in mq_patterns:
                if re.search(p, program_area, re.IGNORECASE):
                    operations.append({
                        'operation': re.findall(r'CALL\s+[\'\"](MQ[A-Z0-9_]*)[\'\"]*', program_area, re.IGNORECASE),
                        'line_number': i+1, 
                        'line_content': program_area[:120]
                    })
        return operations

    def extract_xml_operations(self, lines: List[str]) -> List[Dict]:
        """Extract XML/JSON operations with proper column handling"""
        operations = []
        patterns = [r'XML\s+PARSE', r'XML\s+GENERATE', r'JSON\s+PARSE', r'JSON\s+GENERATE']
        for i, raw_line in enumerate(lines):
            program_area = self.extract_program_area_only(raw_line)
            if not program_area:
                continue
            for p in patterns:
                if re.search(p, program_area, re.IGNORECASE):
                    operations.append({
                        'operation': p.replace('\\s+', ' '), 
                        'line_number': i+1, 
                        'line_content': program_area[:120]
                    })
        return operations

    def extract_db2_operations(self, lines: List[str]) -> List[Dict]:
        """Extract DB2 SQL operations with proper column handling"""
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
        """Extract COBOL divisions"""
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
        """Extract components (paragraphs/sections)"""
        components = []
        para_re = re.compile(r'^\s*([A-Z0-9\-]+)\s*\.', re.IGNORECASE)
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
        """Parse a COBOL field definition"""
        field = CobolField(name=name, level=level, line_number=line_number)
        
        # PIC clause
        pic_m = re.search(r'PIC(?:TURE)?\s+([A-Z0-9SVX\(\)\.\,\-\s]*)', definition, re.IGNORECASE)
        if pic_m:
            field.picture = pic_m.group(1)
            
        # USAGE clause
        u_m = re.search(r'USAGE\s+(COMP|COMP-3|DISPLAY|BINARY|PACKED-DECIMAL)', definition, re.IGNORECASE)
        if u_m:
            field.usage = u_m.group(1)
            
        # OCCURS clause
        o_m = re.search(r'OCCURS\s+(\d+)', definition, re.IGNORECASE)
        if o_m:
            field.occurs = int(o_m.group(1))
            
        # REDEFINES clause
        r_m = re.search(r'REDEFINES\s+([A-Z0-9\-]+)', definition, re.IGNORECASE)
        if r_m:
            field.redefines = r_m.group(1)
            
        # VALUE clause
        v_m = re.search(r'VALUE\s+(["\'].*?["\']|\S+)', definition, re.IGNORECASE)
        if v_m:
            field.value = v_m.group(1).strip('"\'')
            
        return field

    def extract_record_layouts(self, lines: List[str]) -> List[RecordLayout]:
        """Extract record layouts from COBOL source"""
        layouts = []
        current = None
        fields = []
        in_data = False
        current_section = None
        
        for i, raw_line in enumerate(lines):
            program_area = self.extract_program_area_only(raw_line)
            if program_area is None:
                continue
                
            # Division start
            d = self.division_pattern.match(program_area)
            if d:
                in_data = d.group(1).upper() == 'DATA'
                continue
                
            if not in_data:
                continue
                
            # Section
            if program_area.strip().upper().endswith('SECTION.'):
                current_section = program_area.strip().upper().replace('SECTION.', '').strip()
                # finalize previous layout
                if current:
                    current.line_end = i
                    current.fields = fields
                    current.source_code = '\n'.join([
                        self.extract_program_area_only(l) or '' 
                        for l in lines[current.line_start-1:i]
                    ])
                    current.section = current_section
                    layouts.append(current)
                    current = None
                    fields = []
                continue
                
            # data items
            m = self.data_item_pattern.match(program_area.strip())
            if m:
                level = int(m.group(1))
                name = m.group(2)
                rest = m.group(3) or ''
                
                if level == 1:
                    if current:
                        current.line_end = i
                        current.fields = fields
                        current.source_code = '\n'.join([
                            self.extract_program_area_only(l) or '' 
                            for l in lines[current.line_start-1:i]
                        ])
                        layouts.append(current)
                        current = None
                        fields = []
                        
                    if name.upper() != 'FILLER':
                        current = RecordLayout(name=name, level=level, line_start=i+1)
                        current.section = current_section or 'WORKING-STORAGE'
                        fields = []
                else:
                    if current and name.upper() != 'FILLER':
                        f = self.parse_cobol_field(program_area.strip(), i+1, level, name, rest)
                        fields.append(f)
                        
        # Close final layout
        if current:
            current.line_end = len(lines)
            current.fields = fields
            current.source_code = '\n'.join([
                self.extract_program_area_only(l) or '' 
                for l in lines[current.line_start-1:current.line_end]
            ])
            layouts.append(current)
            
        return layouts

    def extract_data_movements(self, lines: List[str]) -> List[Dict]:
        """Extract data movement operations (MOVE, COMPUTE, ADD ... TO) from COBOL program area"""
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

            # ADD ... TO matches
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

    def parse_cobol_file(self, content: str, filename: str) -> Dict:
        """Parse a complete COBOL file and extract all components"""
        raw_lines = content.split('\n')
        
        # Derive program base name
        program_base = filename.split('/')[-1].split('\\')[-1]
        program_base = program_base.rsplit('.', 1)[0] if '.' in program_base else program_base

        parsed = {
            'friendly_name': self.generate_friendly_name(program_base, 'Program'),
            'filename': filename,
            'divisions': self.extract_divisions(raw_lines),
            'components': self.extract_components(raw_lines),
            'record_layouts': self.extract_record_layouts(raw_lines),
            'file_operations': self.extract_file_operations(raw_lines),
            'program_calls': self.extract_program_calls(content, filename),
            'copybooks': self.extract_copybooks(raw_lines),
            'cics_operations': self.extract_cics_operations(raw_lines),
            'mq_operations': self.extract_mq_operations(raw_lines),
            'xml_operations': self.extract_xml_operations(raw_lines),
            'db2_operations': self.extract_db2_operations(raw_lines),
            'data_movements': self.extract_data_movements(raw_lines),
            'db2_operations': self.extract_db2_operations(raw_lines),
            'total_lines': len(raw_lines),
            'executable_lines': sum(1 for l in raw_lines if self.extract_program_area_only(l)),
            'comment_lines': sum(1 for l in raw_lines if len(l)>6 and l[6] in ['*','/','C','c','D','d']),
            'business_comments': [
                l.strip() for l in raw_lines 
                if len(l)>6 and l[6] in ['*','/','C','c','D','d'] and len(l.strip())>30
            ][:20]
        }
        
        return parsed

    def generate_friendly_name(self, technical_name: str, context: str = '', 
                                      business_domain: str = 'WEALTH_MANAGEMENT',
                                      source_code_snippet: str = '', session_id: str = None) -> str:
        """Generate business-friendly names using LLM with context"""
        
        # Check cache first
        cache_key = f"{technical_name}_{context}_{business_domain}"
        if cache_key in self.friendly_name_cache:
            return self.friendly_name_cache[cache_key]
        
        # Fallback to simple generation if no LLM client
        if not self.llm_client:
            return self._generate_simple_friendly_name(technical_name, context)
        
        try:
            # Prepare context for LLM
            prompt = self._build_friendly_name_prompt(
                technical_name, context, business_domain, source_code_snippet
            )
            
            # Call LLM
            response = self.llm_client.call_llm(prompt, max_tokens=200, temperature=0.3)
            
            if response.success:
                friendly_name = self._extract_friendly_name_from_response(response.content)
                if friendly_name and len(friendly_name) > 0:
                    # Cache the result
                    self.friendly_name_cache[cache_key] = friendly_name
                    return friendly_name
            
            # Fallback to simple generation if LLM fails
            return self._generate_simple_friendly_name(technical_name, context)
            
        except Exception as e:
            logger.warning(f"LLM friendly name generation failed for {technical_name}: {str(e)}")
            return self._generate_simple_friendly_name(technical_name, context)
    
    def _build_friendly_name_prompt(self, technical_name: str, context: str, 
                                  business_domain: str, source_code_snippet: str) -> str:
        """Build LLM prompt for friendly name generation"""
        
        context_info = f"Context: {context}" if context else ""
        code_info = f"\nSource Code Context: {source_code_snippet[:200]}" if source_code_snippet else ""
        
        if business_domain == 'WEALTH_MANAGEMENT':
            domain_context = """
This is for a wealth management system. Common patterns:
- ACCT/ACCOUNT = Account Management
- CUST/CUSTOMER = Customer Information  
- PORT/PORTFOLIO = Portfolio Management
- POS/POSITION = Investment Positions
- TXN/TRAN = Transactions
- BAL = Balance Information
- VAL = Valuation/Pricing
- PERF = Performance Analytics
- RISK = Risk Management
- FEE/COMM = Fees and Commissions
- RPT/REPORT = Reporting
- ALLOC = Asset Allocation
- TRADE = Trading Operations
"""
        else:
            domain_context = f"Business Domain: {business_domain}"
        
        prompt = f"""
Generate a clear, business-friendly name for this technical identifier in a {business_domain.lower().replace('_', ' ')} system.

Technical Name: {technical_name}
{context_info}
{code_info}

{domain_context}

Rules:
1. Make it business-meaningful and descriptive
2. Remove technical prefixes (WS-, FD-, etc.)
3. Use proper business terminology
4. Keep it concise but clear (2-6 words)
5. Use title case
6. Focus on business purpose, not technical implementation

Return ONLY the business-friendly name, nothing else.
"""
        return prompt
    
    def _extract_friendly_name_from_response(self, response_content: str) -> str:
        """Extract clean friendly name from LLM response"""
        if not response_content:
            return ""
        
        # Clean up the response
        friendly_name = response_content.strip()
        
        # Remove quotes if present
        friendly_name = friendly_name.strip('"\'')
        
        # Take only the first line if multiple lines
        friendly_name = friendly_name.split('\n')[0].strip()
        
        # Remove any explanatory text after colons or dashes
        if ':' in friendly_name:
            friendly_name = friendly_name.split(':')[-1].strip()
        if ' - ' in friendly_name:
            friendly_name = friendly_name.split(' - ')[0].strip()
        
        # Validate length and content
        if len(friendly_name) > 100 or len(friendly_name) < 3:
            return ""
        
        # Ensure it's title case
        friendly_name = friendly_name.title()
        
        return friendly_name
    
    def _generate_simple_friendly_name(self, technical_name: str, context: str = '') -> str:
        """Fallback simple friendly name generation"""
        if not technical_name:
            return context.title() if context else 'Unknown Component'

        name = str(technical_name).upper().strip()

        # Remove common technical prefixes
        prefixes = ['WS-', 'LS-', 'WK-', 'FD-', 'FD_', 'TB-', 'TB_', 'SRV-', 'PRG-', 'TMS', 'TMST']
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        # Business domain mappings for wealth management
        wm_mappings = {
            'ACCT': 'Account',
            'CUST': 'Customer', 
            'PORT': 'Portfolio',
            'POS': 'Position',
            'TXN': 'Transaction',
            'TRAN': 'Transaction',
            'BAL': 'Balance',
            'VAL': 'Valuation',
            'PERF': 'Performance',
            'RISK': 'Risk',
            'FEE': 'Fee',
            'COMM': 'Commission',
            'RPT': 'Report',
            'ALLOC': 'Allocation',
            'TRADE': 'Trade'
        }
        
        # Apply mappings
        for tech_term, business_term in wm_mappings.items():
            if tech_term in name:
                name = name.replace(tech_term, business_term)
        
        # Clean up formatting
        name = re.sub(r'[_\-\.]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        name = name.title()
        
        if not name:
            return context.title() if context else technical_name
        
        return name

    def generate_batch_friendly_names(self, items: List[Dict], context: str = '', 
                                    business_domain: str = 'WEALTH_MANAGEMENT',
                                    session_id: str = None) -> Dict[str, str]:
        """Generate friendly names for multiple items efficiently"""
        
        if not self.llm_client or len(items) == 0:
            # Fallback to simple generation
            return {
                self._get_item_name(item): self._generate_simple_friendly_name(
                    self._get_item_name(item), context
                )
                for item in items[:20]  # Limit to prevent errors
            }
        
        try:
            # Prepare batch prompt
            item_list = []
            for item in items[:15]:  # Limit for LLM token constraints
                name = self._get_item_name(item)
                item_type = self._get_item_type(item, context)
                source_snippet = self._get_item_source_snippet(item)
                
                item_list.append({
                    'technical_name': name,
                    'type': item_type,
                    'source_snippet': source_snippet[:100] if source_snippet else ''
                })
            
            prompt = self._build_batch_friendly_name_prompt(item_list, business_domain)
            response = self.llm_client.call_llm(prompt, max_tokens=800, temperature=0.3)
            
            if response.success:
                return self._parse_batch_friendly_names(response.content, item_list)
            
        except Exception as e:
            logger.warning(f"Batch friendly name generation failed: {str(e)}")
        
        # Fallback to individual simple generation
        return {
            self._get_item_name(item): self._generate_simple_friendly_name(
                self._get_item_name(item), context
            )
            for item in items[:20]
        }
    
    def _build_batch_friendly_name_prompt(self, item_list: List[Dict], business_domain: str) -> str:
        """Build prompt for batch friendly name generation"""
        
        items_text = ""
        for i, item in enumerate(item_list, 1):
            items_text += f"{i}. {item['technical_name']} ({item['type']})\n"
            if item['source_snippet']:
                items_text += f"   Context: {item['source_snippet']}\n"
        
        domain_context = ""
        if business_domain == 'WEALTH_MANAGEMENT':
            domain_context = """
Wealth Management Domain - Common business terms:
- Account Management, Customer Information, Portfolio Management
- Investment Positions, Transactions, Balance Information
- Valuation/Pricing, Performance Analytics, Risk Management
- Fees/Commissions, Reporting, Asset Allocation, Trading
"""
        
        prompt = f"""
Generate business-friendly names for these technical identifiers in a {business_domain.lower().replace('_', ' ')} system.

{domain_context}

Technical Items:
{items_text}

Return ONLY a JSON object with this exact format:
{{
    "TECHNICAL_NAME_1": "Business Friendly Name",
    "TECHNICAL_NAME_2": "Another Business Name",
    ...
}}

Rules:
- Business-meaningful and descriptive names
- Remove technical prefixes (WS-, FD-, TMS-, etc.)  
- Use proper business terminology
- 2-6 words maximum
- Title case
- Focus on business purpose
"""
        return prompt
    
    def _parse_batch_friendly_names(self, response_content: str, item_list: List[Dict]) -> Dict[str, str]:
        """Parse batch friendly names from LLM response"""
        try:
            # Try to extract JSON
            import json
            
            # Find JSON in response
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_content[start_idx:end_idx+1]
                friendly_names = json.loads(json_str)
                
                if isinstance(friendly_names, dict):
                    # Validate and clean names
                    cleaned_names = {}
                    for tech_name, friendly_name in friendly_names.items():
                        if isinstance(friendly_name, str) and 3 <= len(friendly_name) <= 100:
                            cleaned_names[tech_name.upper()] = friendly_name.strip().title()
                    
                    return cleaned_names
        
        except Exception as e:
            logger.warning(f"Failed to parse batch friendly names: {str(e)}")
        
        # Fallback to simple generation
        return {
            item['technical_name']: self._generate_simple_friendly_name(
                item['technical_name'], item['type']
            )
            for item in item_list
        }
    
    def _get_item_name(self, item) -> str:
        """Extract name from item dict or string"""
        if isinstance(item, dict):
            return item.get('name', item.get('copybook_name', 
                   item.get('file_name', item.get('program_name', 'UNKNOWN'))))
        return str(item)
    
    def _get_item_type(self, item, default_context: str) -> str:
        """Extract type from item dict"""
        if isinstance(item, dict):
            return item.get('type', item.get('operation', default_context))
        return default_context
    
    def _get_item_source_snippet(self, item) -> str:
        """Extract source code snippet from item"""
        if isinstance(item, dict):
            return item.get('line_content', item.get('source_code', ''))
        return ''

    def extract_db2_operations(self, lines: List[str]) -> List[Dict]:
        """Enhanced DB2 SQL operations extraction with table identification"""
        operations = []
        in_exec_sql = False
        buffer = []
        start_line = 0
        
        for i, raw_line in enumerate(lines):
            program_area = self.extract_program_area_only(raw_line)
            if not program_area:
                continue
                
            program_upper = program_area.upper()
            
            if re.search(r'EXEC\s+SQL', program_upper):
                in_exec_sql = True
                buffer = [program_area]
                start_line = i + 1
                
                if re.search(r'END-EXEC', program_upper):
                    in_exec_sql = False
                    complete_sql = ' '.join(buffer)
                    sql_operation = self._analyze_sql_operation(complete_sql, start_line)
                    if sql_operation:
                        operations.append(sql_operation)
                    buffer = []
                    
            elif in_exec_sql:
                buffer.append(program_area)
                if re.search(r'END-EXEC', program_upper):
                    in_exec_sql = False
                    complete_sql = ' '.join(buffer)
                    sql_operation = self._analyze_sql_operation(complete_sql, start_line)
                    if sql_operation:
                        operations.append(sql_operation)
                    buffer = []
            else:
                # Look for inline SQL statements
                if re.search(r'\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b', program_upper):
                    sql_operation = self._analyze_sql_operation(program_area.strip(), i + 1)
                    if sql_operation:
                        operations.append(sql_operation)
        
        return operations

    def _analyze_sql_operation(self, sql_statement: str, line_number: int) -> Dict:
        """Analyze SQL statement to extract tables and operation type"""
        sql_upper = sql_statement.upper()
        
        # Determine operation type
        operation_type = 'UNKNOWN'
        if 'SELECT' in sql_upper:
            operation_type = 'SELECT'
        elif 'INSERT' in sql_upper:
            operation_type = 'INSERT'
        elif 'UPDATE' in sql_upper:
            operation_type = 'UPDATE'
        elif 'DELETE' in sql_upper:
            operation_type = 'DELETE'
        
        # Extract table names
        table_names = self._extract_table_names_from_sql(sql_statement)
        
        return {
            'sql': sql_statement,
            'line_number': line_number,
            'operation_type': operation_type,
            'tables': [table[0] for table in table_names],  # Just table names
            'table_operations': table_names,  # Table name + operation pairs
            'io_direction': 'INPUT' if operation_type == 'SELECT' else 'OUTPUT' if operation_type in ['INSERT', 'UPDATE', 'DELETE'] else 'UNKNOWN'
        }
    
    def _calculate_field_lengths_fixed(self, picture: str, usage: str = "") -> Tuple[int, int, str]:
        """Fixed field length calculation from PIC clause with proper validation"""
        if not picture or picture.strip() == '':
            return 1, 50, "VARCHAR2(50)"  # Default minimum values
        
        pic_upper = picture.upper().strip()
        mainframe_length = 1  # Start with minimum length
        oracle_length = 50
        oracle_type = "VARCHAR2(50)"
        
        try:
            # Numeric fields (9, S9, etc.)
            if re.search(r'[9S]', pic_upper):
                total_digits = 0
                decimal_digits = 0
                
                # Handle parentheses notation like 9(5) or S9(7)V99
                paren_matches = re.findall(r'[9S]\((\d+)\)', pic_upper)
                for match in paren_matches:
                    total_digits += int(match)
                
                # Handle explicit 9s like 99999
                remaining_pic = re.sub(r'[9S]\(\d+\)', '', pic_upper)
                explicit_nines = len(re.findall(r'9', remaining_pic))
                total_digits += explicit_nines
                
                # Handle decimal point (V)
                if 'V' in pic_upper:
                    v_parts = pic_upper.split('V', 1)
                    if len(v_parts) > 1:
                        decimal_part = v_parts[1]
                        # Count decimal digits after V
                        decimal_paren = re.findall(r'9\((\d+)\)', decimal_part)
                        for match in decimal_paren:
                            decimal_digits += int(match)
                        decimal_explicit = len(re.findall(r'9', re.sub(r'9\(\d+\)', '', decimal_part)))
                        decimal_digits += decimal_explicit
                
                # Ensure minimum total digits
                total_digits = max(total_digits, 1)
                
                # Calculate mainframe storage based on usage
                if usage.upper() in ['COMP-3', 'PACKED-DECIMAL']:
                    mainframe_length = max((total_digits + 1) // 2 + 1, 1)
                elif usage.upper() in ['COMP', 'BINARY']:
                    if total_digits <= 4:
                        mainframe_length = 2
                    elif total_digits <= 9:
                        mainframe_length = 4
                    else:
                        mainframe_length = 8
                else:
                    # Display format
                    mainframe_length = total_digits + (1 if 'S' in pic_upper else 0)
                    mainframe_length = max(mainframe_length, 1)
                
                # Oracle type
                if decimal_digits > 0:
                    oracle_type = f"NUMBER({total_digits},{decimal_digits})"
                    oracle_length = total_digits + 1  # +1 for decimal point
                else:
                    oracle_type = f"NUMBER({total_digits})"
                    oracle_length = total_digits
            
            # Alphanumeric fields (X, A)
            elif re.search(r'[XA]', pic_upper):
                # Handle X(n) notation
                paren_matches = re.findall(r'[XA]\((\d+)\)', pic_upper)
                if paren_matches:
                    mainframe_length = sum(int(match) for match in paren_matches)
                else:
                    # Count explicit Xs
                    mainframe_length = max(len(re.findall(r'[XA]', pic_upper)), 1)
                
                oracle_length = mainframe_length
                if oracle_length <= 4000:
                    oracle_type = f"VARCHAR2({oracle_length})"
                else:
                    oracle_type = "CLOB"
            
            # Ensure minimum values
            mainframe_length = max(mainframe_length, 1)
            oracle_length = max(oracle_length, mainframe_length)
            
        except Exception as e:
            logger.warning(f"Error calculating field lengths for PIC {picture}: {str(e)}")
            mainframe_length = 1
            oracle_length = 50
            oracle_type = "VARCHAR2(50)"
        
        return mainframe_length, oracle_length, oracle_type
    
    

    def generate_friendly_name_enhanced(self, technical_name: str, context: str = '', business_domain: str = 'GENERAL') -> str:
        """Enhanced friendly name generation with business domain context"""
        base_friendly = self.generate_friendly_name(technical_name, context)
        
        # Add business domain specific enhancements
        if business_domain == 'WEALTH_MANAGEMENT':
            # Map common wealth management patterns
            wm_patterns = {
                'ACCT': 'Account',
                'CUST': 'Customer',
                'PORT': 'Portfolio',
                'POS': 'Position',
                'TXN': 'Transaction',
                'TRAN': 'Transaction',
                'BAL': 'Balance',
                'VAL': 'Valuation',
                'PERF': 'Performance',
                'RISK': 'Risk',
                'FEE': 'Fee',
                'COMM': 'Commission'
            }
            
            name_upper = technical_name.upper()
            for pattern, replacement in wm_patterns.items():
                if pattern in name_upper:
                    base_friendly = base_friendly.replace(pattern.title(), replacement)
                    break
        
        return base_friendly

    def validate_cobol_identifier(self, identifier: str, identifier_type: str = 'GENERAL') -> bool:
        """Unified validation for COBOL identifiers"""
        if identifier_type == 'FILE':
            return self._is_valid_cobol_filename(identifier)
        elif identifier_type == 'CICS_FILE':
            return self._is_valid_cics_filename_enhanced(identifier)
        else:
            return self._is_valid_cobol_filename(identifier)
        
    def _parse_cics_command(self, cics_command: str, line_number: int) -> List[Dict]:
            """Parse complete CICS command - fallback method for compatibility"""
            # This method is for backward compatibility - delegates to enhanced version
            return self._parse_cics_command_enhanced(cics_command, line_number)
    
    def _calculate_field_lengths(self, picture: str, usage: str = "") -> Tuple[int, int, str]:
        """Calculate field lengths from PIC clause - called by ComponentExtractor"""
        # Use the existing _calculate_field_lengths_fixed method
        return self._calculate_field_lengths_fixed(picture, usage)
    
    def should_include_filler(self, line: str, level: str) -> bool:
        """
        Determine if a FILLER field should be included in analysis
        Include fillers that have VALUE clauses as they represent constants/literals
        """
        line_upper = line.upper().strip()
        
        # Always include fillers with VALUE clauses
        if 'FILLER' in line_upper and 'VALUE' in line_upper:
            return True
        
        # Exclude fillers without VALUE (just spacing/padding)
        if 'FILLER' in line_upper and 'VALUE' not in line_upper:
            return False
        
        return True
    
    def _extract_fields_from_lines(self, lines: List[str], start_idx: int = 0, end_idx: int = None) -> List:
        """Enhanced field extraction with proper filler handling"""
        if end_idx is None:
            end_idx = len(lines)
        
        fields = []
        i = start_idx
        
        while i < end_idx:
            line = lines[i].strip()
            if not line or line.startswith('*'):
                i += 1
                continue
            
            # Check for field definition (level number)
            level_match = re.match(r'^\s*(\d{2})\s+(.+)', line)
            if level_match:
                level = level_match.group(1)
                field_content = level_match.group(2).strip()
                
                # Skip if it's a filler without value (unless it has VALUE)
                if not self.should_include_filler(line, level):
                    i += 1
                    continue
                
                try:
                    field_info = self._parse_field_line(line, i + 1)  # +1 for 1-based line numbers
                    
                    # Enhanced filler processing
                    if 'FILLER' in field_content.upper():
                        # Extract VALUE if present for meaningful filler names
                        value_match = re.search(r"VALUE\s+['\"]([^'\"]*)['\"]", field_content, re.IGNORECASE)
                        if value_match:
                            value_content = value_match.group(1)
                            # Create meaningful name from value
                            field_info.name = f"FILLER-{value_content[:10].replace(' ', '-')}-{level}"
                            field_info.friendly_name = f"Constant: {value_content}"
                            field_info.business_purpose = f"Constant literal value: '{value_content}'"
                            field_info.value = value_content
                            field_info.usage_type = 'CONSTANT'
                        else:
                            field_info.name = f"FILLER-{level}-LINE-{i+1}"
                            field_info.friendly_name = f"Filler Field Level {level}"
                    
                    if field_info:
                        fields.append(field_info)
                        
                except Exception as e:
                    logger.warning(f"Error parsing field at line {i+1}: {e}")
            
            i += 1
        
        return fields

    def _parse_field_line(self, line: str, line_number: int):
        """Enhanced field line parsing with better filler support"""
        try:
            # Extract level number
            level_match = re.match(r'^\s*(\d{2})\s+(.+)', line.strip())
            if not level_match:
                return None
            
            level = int(level_match.group(1))
            field_content = level_match.group(2).strip()
            
            # Initialize field info
            field_info = type('FieldInfo', (), {})()
            field_info.level = level
            field_info.line_number = line_number
            field_info.usage_type = 'STATIC'
            
            # Handle FILLER fields
            if field_content.upper().startswith('FILLER'):
                field_info.name = 'FILLER'
                field_info.friendly_name = 'Filler Field'
                field_info.business_purpose = 'Spacing or constant field'
                remaining_content = field_content[6:].strip()  # Remove 'FILLER'
            else:
                # Extract field name
                name_match = re.match(r'^([A-Za-z][A-Za-z0-9\-_]*)', field_content)
                if name_match:
                    field_info.name = name_match.group(1)
                    field_info.friendly_name = self.generate_friendly_name(field_info.name, 'Field')
                    remaining_content = field_content[len(field_info.name):].strip()
                else:
                    return None
            
            # Extract PIC clause
            pic_match = re.search(r'PIC(?:TURE)?\s+([X9SVP\(\),\.\+\-\*\$Z]+)', remaining_content, re.IGNORECASE)
            if pic_match:
                field_info.picture = pic_match.group(1)
            else:
                field_info.picture = ''
            
            # Extract VALUE clause (especially important for fillers)
            value_match = re.search(r"VALUE\s+(['\"][^'\"]*['\"]|[A-Za-z0-9\-]+)", remaining_content, re.IGNORECASE)
            if value_match:
                value_str = value_match.group(1)
                # Remove quotes if present
                if value_str.startswith('"') or value_str.startswith("'"):
                    field_info.value = value_str[1:-1]
                else:
                    field_info.value = value_str
                
                # For fillers with values, update the field info
                if field_info.name == 'FILLER' and field_info.value:
                    field_info.usage_type = 'CONSTANT'
                    field_info.business_purpose = f'Constant value: {field_info.value}'
            else:
                field_info.value = ''
            
            # Extract USAGE clause
            usage_match = re.search(r'USAGE\s+(COMP|COMP-3|DISPLAY|BINARY|PACKED-DECIMAL)', remaining_content, re.IGNORECASE)
            if usage_match:
                field_info.usage = usage_match.group(1)
            else:
                field_info.usage = ''
            
            # Extract OCCURS clause
            occurs_match = re.search(r'OCCURS\s+(\d+)', remaining_content, re.IGNORECASE)
            if occurs_match:
                field_info.occurs = int(occurs_match.group(1))
            else:
                field_info.occurs = 0
            
            # Extract REDEFINES clause
            redefines_match = re.search(r'REDEFINES\s+([A-Za-z][A-Za-z0-9\-_]*)', remaining_content, re.IGNORECASE)
            if redefines_match:
                field_info.redefines = redefines_match.group(1)
            else:
                field_info.redefines = ''
            
            return field_info
            
        except Exception as e:
            logger.error(f"Error parsing field line '{line}': {e}")
            return None

    def extract_dynamic_program_calls(self, content: str, filename: str) -> List[Dict]:
        """
        Extract dynamic CICS program calls that use variables (XCTL, LINK)
        FIXED: Add error handling for variable map building
        """
        dynamic_calls = []
        lines = content.split('\n')
        
        try:
            # First pass: Build variable value map from working storage
            variable_values = self._build_variable_value_map(lines)
            logger.debug(f"Built variable map with {len(variable_values)} variables")
            
        except Exception as e:
            logger.error(f"Error building variable value map: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue with empty variable map
            variable_values = {}
        
        # Second pass: Find dynamic CICS calls
        # FIXED:
        i = 0
        while i < len(lines):
            line = lines[i]
            line_upper = line.upper().strip()
            
            if 'EXEC CICS' in line_upper and ('XCTL' in line_upper or 'LINK' in line_upper):
                # Extract complete multi-line CICS command
                complete_cics_command = self._extract_complete_cics_command(lines, i)  # Only returns command
                
                if complete_cics_command:
                    logger.debug(f"Complete CICS command: {complete_cics_command}")
                    dynamic_call = self._analyze_dynamic_cics_call(complete_cics_command, i + 1, variable_values)
                    if dynamic_call:
                        dynamic_calls.append(dynamic_call)
                
                # Skip ahead by a reasonable amount since we don't know exact end line
                i += 5  
            else:
                i += 1
        
        logger.info(f"Extracted {len(dynamic_calls)} dynamic program calls")
        return dynamic_calls
    def _build_variable_value_map(self, lines: List[str]) -> Dict[str, Any]:
        """
        Build map of working storage variables and their possible values
        Handles group fields, fillers with values, and computed values
        FIXED: Properly initialize children dictionary
        """
        variable_map = {}
        current_group = None
        
        for i, line in enumerate(lines, 1):
            line_upper = line.upper().strip()
            
            if not line_upper or line_upper.startswith('*'):
                continue
                
            # Check for field definition with level number
            level_match = re.match(r'^\s*(\d{2})\s+(.+)', line_upper)
            if level_match:
                level = int(level_match.group(1))
                field_content = level_match.group(2).strip()
                
                # Handle group fields (01-49 level)
                if level == 1 or (level > 1 and level < 49):
                    # Extract field name
                    name_match = re.match(r'^([A-Z][A-Z0-9\-]*)', field_content)
                    if name_match:
                        field_name = name_match.group(1)
                        
                        if level == 1:
                            current_group = field_name
                            variable_map[field_name] = {
                                'type': 'group',
                                'level': level,
                                'children': {},  # ALWAYS initialize children
                                'line': i,
                                'possible_values': []
                            }
                        else:
                            # Sub-field of group - FIXED: Check if current_group exists and has children
                            if current_group and current_group in variable_map:
                                # Ensure children dict exists
                                if 'children' not in variable_map[current_group]:
                                    variable_map[current_group]['children'] = {}
                                
                                variable_map[current_group]['children'][field_name] = {
                                    'type': 'field',
                                    'level': level,
                                    'line': i,
                                    'parent': current_group
                                }
                            
                            # Also add to main variable map
                            variable_map[field_name] = {
                                'type': 'field',
                                'level': level,
                                'line': i,
                                'parent': current_group,
                                'possible_values': []
                            }
                        
                        # Check for VALUE clause
                        value_match = re.search(r"VALUE\s+['\"]([^'\"]*)['\"]", field_content)
                        if value_match:
                            value = value_match.group(1)
                            variable_map[field_name]['constant_value'] = value
                            variable_map[field_name]['possible_values'].append(value)
                            
                            # For fillers with values in groups
                            if 'FILLER' in field_content and current_group:
                                if current_group in variable_map:
                                    if 'filler_values' not in variable_map[current_group]:
                                        variable_map[current_group]['filler_values'] = []
                                    variable_map[current_group]['filler_values'].append(value)
            
            # Look for MOVE statements that populate variables
            move_match = re.search(r'MOVE\s+([A-Z0-9\-\'\"]+)\s+TO\s+([A-Z0-9\-]+)', line_upper)
            if move_match:
                source_val = move_match.group(1).strip("'\"")
                target_var = move_match.group(2)
                
                if target_var in variable_map:
                    if 'possible_values' not in variable_map[target_var]:
                        variable_map[target_var]['possible_values'] = []
                    if source_val not in variable_map[target_var]['possible_values']:
                        variable_map[target_var]['possible_values'].append(source_val)
        
        return variable_map

    def _resolve_group_field_programs(self, group_name: str, group_info: Dict, variable_map: Dict) -> List[Dict]:
        """
        Resolve program names from group field structure
        FIXED: Add defensive checks for children dictionary
        """
        resolved_programs = []
        filler_values = group_info.get('filler_values', [])
        
        # FIXED: Check if children key exists before accessing
        children = group_info.get('children', {})
        if not isinstance(children, dict):
            logger.warning(f"Group {group_name} has invalid children structure: {type(children)}")
            return resolved_programs
        
        # Look for sub-fields that could contain program suffixes
        for child_name, child_info in children.items():
            if child_name != 'FILLER' and child_name in variable_map:
                child_var_info = variable_map[child_name]
                
                # Combine filler constants with child variable possible values
                for filler_val in filler_values:
                    child_possible_values = child_var_info.get('possible_values', [child_name])
                    for child_val in child_possible_values:
                        if child_val != child_name:  # Skip if it's just the variable name
                            combined_program = f"{filler_val}{child_val}"
                        else:
                            combined_program = filler_val  # Just the filler constant
                        
                        resolved_programs.append({
                            'program_name': combined_program,
                            'resolution': 'group_field_combination',
                            'confidence': 0.8,
                            'source': f"Group {group_name}: '{filler_val}' + {child_name}"
                        })
        
        return resolved_programs

    def _analyze_dynamic_cics_call(self, line: str, line_number: int, variable_map: Dict) -> Optional[Dict]:
        """
        Analyze a line for dynamic CICS XCTL/LINK calls
        FIXED: Better error handling and validation
        """
        if not variable_map:
            logger.debug("No variable map available for dynamic call analysis")
            return None
        
        # Patterns for dynamic CICS calls
        patterns = [
            r'EXEC\s+CICS\s+(XCTL|LINK)\s+PROGRAM\s*\(\s*([A-Z0-9\-]+)\s*\)',
            r'EXEC\s+CICS\s+(XCTL|LINK)\s+.*PROGRAM\s*\(\s*([A-Z0-9\-]+)\s*\)',
            r'(XCTL|LINK)\s+PROGRAM\s*\(\s*([A-Z0-9\-]+)\s*\)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                logger.info(f"Pattern matched: {pattern}")
                logger.info(f"Groups: {match.groups()}")
                operation = match.group(1)
                variable_name = match.group(2)
                logger.info(f"Found: operation={operation} variable={variable_name}")
                # Resolve variable to possible program names
                try:
                    resolved_programs = self._resolve_variable_to_programs(variable_name, variable_map)
                    
                    if resolved_programs:
                        return {
                            'operation': f'CICS_{operation}',
                            'variable_name': variable_name,
                            'resolved_programs': resolved_programs,
                            'line_number': line_number,
                            'line_content': line.strip(),
                            'call_type': 'dynamic',
                            'resolution_confidence': self._calculate_resolution_confidence(
                                variable_name, variable_map, resolved_programs
                            )
                        }
                    else:
                        logger.debug(f"Could not resolve variable {variable_name}")
                        
                except Exception as e:
                    logger.error(f"Error resolving dynamic call variable {variable_name}: {str(e)}")
                    return None
        
        return None


    def _resolve_variable_to_programs(self, variable_name: str, variable_map: Dict) -> List[Dict]:
        """
        Resolve a variable to its possible program name values
        Handles group fields with filler constants, computed values, and HOLD-field patterns
        """
        resolved_programs = []
        
        if variable_name not in variable_map:
            # NEW: Check for HOLD-xxx pattern (for dynamic CICS calls like PROGRAM(TRANX) -> HOLD-TRANX)
            hold_field_name = f"HOLD-{variable_name}"
            if hold_field_name in variable_map:
                logger.debug(f"Found HOLD field for {variable_name}: {hold_field_name}")
                hold_var_info = variable_map[hold_field_name]
                
                # Direct constant value in HOLD field
                if 'constant_value' in hold_var_info:
                    resolved_programs.append({
                        'program_name': hold_var_info['constant_value'],
                        'resolution': 'hold_field_constant',
                        'confidence': 1.0,
                        'source': f"HOLD field {hold_field_name} VALUE '{hold_var_info['constant_value']}'"
                    })
                
                # Possible values from MOVE operations to HOLD field
                for value in hold_var_info.get('possible_values', []):
                    if value not in [p['program_name'] for p in resolved_programs]:
                        resolved_programs.append({
                            'program_name': value,
                            'resolution': 'hold_field_move',
                            'confidence': 0.9,
                            'source': f"MOVE '{value}' TO {hold_field_name}"
                        })
                
                if resolved_programs:
                    logger.debug(f"Resolved {variable_name} via HOLD field to {len(resolved_programs)} programs")
                    return resolved_programs
            
            logger.debug(f"Variable {variable_name} not found in map and no HOLD field available")
            return [{'program_name': variable_name, 'resolution': 'unresolved', 'confidence': 0.1}]
        
        var_info = variable_map[variable_name]
        logger.debug(f"Found variable {variable_name} in map: {var_info.get('type', 'unknown type')}")
        
        # Direct constant value
        if 'constant_value' in var_info:
            resolved_programs.append({
                'program_name': var_info['constant_value'],
                'resolution': 'constant',
                'confidence': 1.0,
                'source': f"VALUE '{var_info['constant_value']}'"
            })
        
        # Possible values from MOVE operations
        for value in var_info.get('possible_values', []):
            if value not in [p['program_name'] for p in resolved_programs]:
                resolved_programs.append({
                    'program_name': value,
                    'resolution': 'move_operation',
                    'confidence': 0.8,
                    'source': f"MOVE '{value}' TO {variable_name}"
                })
        
        # Handle group fields with filler constants
        if var_info.get('type') == 'group' and 'children' in var_info:
            group_programs = self._resolve_group_field_programs(variable_name, var_info, variable_map)
            resolved_programs.extend(group_programs)
            
            # ENHANCED: Also check if this group contains a HOLD-xxx field
            hold_child_name = f"HOLD-{variable_name}"
            if hold_child_name in var_info.get('children', {}):
                logger.debug(f"Group {variable_name} contains HOLD field: {hold_child_name}")
                if hold_child_name in variable_map:
                    hold_child_info = variable_map[hold_child_name]
                    for value in hold_child_info.get('possible_values', []):
                        if value not in [p['program_name'] for p in resolved_programs]:
                            resolved_programs.append({
                                'program_name': value,
                                'resolution': 'group_hold_child',
                                'confidence': 0.9,
                                'source': f"Group child {hold_child_name} = '{value}'"
                            })
        
        # Handle parent group resolution
        if var_info.get('parent') and var_info['parent'] in variable_map:
            parent_info = variable_map[var_info['parent']]
            if 'filler_values' in parent_info:
                for filler_value in parent_info['filler_values']:
                    # Construct potential program name from filler + variable
                    potential_program = f"{filler_value}{variable_name.replace(var_info['parent'] + '-', '')}"
                    resolved_programs.append({
                        'program_name': potential_program,
                        'resolution': 'group_filler_combination',
                        'confidence': 0.7,
                        'source': f"Group {var_info['parent']} filler '{filler_value}' + {variable_name}"
                    })
        
        if resolved_programs:
            logger.debug(f"Successfully resolved {variable_name} to {len(resolved_programs)} programs")
        else:
            logger.debug(f"No resolution found for {variable_name}")
        
        return resolved_programs if resolved_programs else [
            {'program_name': variable_name, 'resolution': 'unresolved', 'confidence': 0.1}
        ]
    

    def _calculate_resolution_confidence(self, variable_name: str, variable_map: Dict, resolved_programs: List[Dict]) -> float:
        """Calculate confidence in variable resolution"""
        if not resolved_programs:
            return 0.0
        
        # Average confidence of all resolved programs
        total_confidence = sum(prog['confidence'] for prog in resolved_programs)
        return min(total_confidence / len(resolved_programs), 1.0)