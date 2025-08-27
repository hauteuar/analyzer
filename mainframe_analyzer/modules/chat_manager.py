"""
Complete Chat Manager Module
Handles intelligent chat with full source code context for mainframe analysis
"""

import re
import json
import logging
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatManager:
    def __init__(self, llm_client, token_manager, db_manager):
        self.llm_client = llm_client
        self.token_manager = token_manager
        self.db_manager = db_manager
        
        # Field name extraction patterns
        self.field_patterns = [
            r'\b([A-Z][A-Z0-9\-]{2,})\b',                    # Standard COBOL fields
            r'field\s+([A-Za-z][A-Za-z0-9\-_]+)',            # "field CUSTOMER-NAME"
            r'about\s+([A-Za-z][A-Za-z0-9\-_]+)',            # "about ACCOUNT-NO"
            r'([A-Za-z][A-Za-z0-9\-_]+)\s+field',            # "CUSTOMER-NAME field"
            r'tell\s+me\s+about\s+([A-Za-z][A-Za-z0-9\-_]+)',# "tell me about FIELD-NAME"
            r'what\s+is\s+([A-Za-z][A-Za-z0-9\-_]+)',        # "what is FIELD-NAME"
            r'how\s+is\s+([A-Za-z][A-Za-z0-9\-_]+)',         # "how is FIELD-NAME"
            r'where\s+is\s+([A-Za-z][A-Za-z0-9\-_]+)',       # "where is FIELD-NAME"
            r'show\s+([A-Za-z][A-Za-z0-9\-_]+)'              # "show FIELD-NAME"
        ]
        
        # COBOL keywords to exclude
        self.cobol_keywords = {
            'MOVE', 'TO', 'FROM', 'PIC', 'PICTURE', 'VALUE', 'OCCURS', 'REDEFINES',
            'USAGE', 'COMP', 'BINARY', 'PACKED', 'DISPLAY', 'COMPUTE', 'ADD',
            'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'IF', 'THEN', 'ELSE', 'END',
            'PERFORM', 'UNTIL', 'VARYING', 'WHEN', 'EVALUATE', 'ACCEPT', 'DISPLAY'
        }
    
    def process_query(self, session_id: str, message: str, conversation_id: str) -> str:
        """Process chat query with better routing logic"""
        logger.info(f"Processing chat query: '{message[:100]}...'")
        
        try:
            # Extract field names more carefully
            field_names = self._extract_field_names(message)
            logger.info(f"Extracted potential field names: {field_names}")
            
            # Check message intent first
            message_lower = message.lower()
            
            # Priority routing based on keywords
            if any(word in message_lower for word in ['help', 'what can you', 'how do i']):
                return self._get_help_response()
            
            elif any(word in message_lower for word in ['summary', 'overview', 'status']):
                return self._handle_summary_query(session_id)
            
            elif any(word in message_lower for word in ['layout', 'record', 'structure']) and not field_names:
                return self._handle_layout_query(session_id, message)
            
            elif any(word in message_lower for word in ['program', 'component', 'module']) and not field_names:
                return self._handle_program_query(session_id, message)
            
            elif field_names:
                # Only if we found actual field names
                return self._handle_field_query(session_id, field_names, message)
            
            else:
                # General query - don't assume field names
                return self._handle_general_query(session_id, message)
                    
        except Exception as e:
            logger.error(f"Error processing chat query: {str(e)}")
            return f"I encountered an error processing your question: {str(e)}"
    
    def _search_for_similar_fields(self, session_id: str, query_term: str) -> List[str]:
        """Search for fields similar to the query term"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Search for fields that contain the query term
                cursor.execute('''
                    SELECT DISTINCT field_name FROM field_analysis_details 
                    WHERE session_id = ? AND (
                        UPPER(field_name) LIKE ? OR
                        UPPER(business_purpose) LIKE ?
                    )
                    ORDER BY field_name 
                    LIMIT 5
                ''', (session_id, f'%{query_term.upper()}%', f'%{query_term.upper()}%'))
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error searching for similar fields: {str(e)}")
            return []

    def _extract_field_names(self, message: str) -> List[str]:
        """Extract COBOL field names more intelligently"""
        field_names = set()
        
        try:
            # More specific field extraction patterns
            specific_patterns = [
                r'\b([A-Z][A-Z0-9\-]{4,})\b',                    # At least 5 chars, COBOL-style
                r'field\s+([A-Za-z][A-Za-z0-9\-_]{3,})',         # "field CUSTOMER-NAME"
                r'about\s+([A-Z][A-Z0-9\-_]{3,})',               # "about ACCOUNT-NO" 
                r'([A-Z][A-Z0-9\-_]{4,})\s+field',               # "CUSTOMER-NAME field"
                r'tell\s+me\s+about\s+([A-Z][A-Z0-9\-_]{3,})',   # "tell me about FIELD-NAME"
                r'what\s+is\s+([A-Z][A-Z0-9\-_]{3,})',           # "what is FIELD-NAME" (only ALL CAPS)
                r'show\s+me\s+([A-Z][A-Z0-9\-_]{3,})',           # "show me FIELD-NAME"
                r'\b([A-Z]{2,}[-_][A-Z0-9\-_]{2,})\b'            # COBOL naming convention
            ]
            
            # English stop words to exclude
            english_words = {
                'WHAT', 'WHERE', 'WHEN', 'HOW', 'WHO', 'WHY', 'WHICH', 'THE', 'THIS', 'THAT',
                'CUSTOMER', 'RECORD', 'STRUCTURE', 'FIELD', 'FIELDS', 'PROGRAM', 'PROGRAMS',
                'LAYOUT', 'LAYOUTS', 'FILE', 'FILES', 'TABLE', 'TABLES', 'SHOW', 'TELL',
                'ABOUT', 'EXPLAIN', 'DESCRIBE', 'LIST', 'FIND', 'SEARCH', 'HELP', 'CAN',
                'WILL', 'WOULD', 'SHOULD', 'COULD', 'AND', 'OR', 'BUT', 'FOR', 'WITH',
                'FROM', 'INTO', 'ONTO', 'OVER', 'UNDER', 'ABOVE', 'BELOW'
            }
            
            for pattern in specific_patterns:
                matches = re.findall(pattern, message, re.IGNORECASE)
                for match in matches:
                    cobol_name = match.upper().replace('_', '-')
                    
                    # More strict filtering
                    if (len(cobol_name) > 3 and 
                        cobol_name not in self.cobol_keywords and
                        cobol_name not in english_words and
                        not cobol_name.isdigit() and
                        '-' in cobol_name or len(cobol_name) > 6):  # Either has dash or is long
                        field_names.add(cobol_name)
            
            result = list(field_names)
            logger.debug(f"Extracted field names from '{message}': {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting field names: {str(e)}")
            return []
    
    def _handle_field_query(self, session_id: str, field_names: List[str], message: str) -> str:
        """Handle queries about specific fields with better error messages"""
        try:
            response_parts = []
            found_fields = []
            not_found_fields = []
            
            for field_name in field_names[:2]:
                field_info = self._get_comprehensive_field_info(session_id, field_name, message)
                if field_info and "was not found" not in field_info:
                    response_parts.append(field_info)
                    found_fields.append(field_name)
                else:
                    not_found_fields.append(field_name)
            
            # If no fields found, search for similar ones
            if not response_parts and not_found_fields:
                similar_suggestions = []
                for field_name in not_found_fields:
                    similar = self._search_for_similar_fields(session_id, field_name)
                    similar_suggestions.extend(similar)
                
                if similar_suggestions:
                    return f"I couldn't find exact matches for {', '.join(not_found_fields)}, but I found these similar fields: {', '.join(similar_suggestions[:5])}. Try asking about one of these specific field names."
                else:
                    available_fields = self._get_available_fields_sample(session_id)
                    return f"I couldn't find fields matching {', '.join(not_found_fields)}.\n\nAvailable fields include: {available_fields}\n\nTry asking about one of these field names, or ask a general question about your programs."
            
            return '\n\n'.join(response_parts)
            
        except Exception as e:
            logger.error(f"Error handling field query: {str(e)}")
            return f"Error analyzing fields: {str(e)}"
    
    def _get_comprehensive_field_info(self, session_id: str, field_name: str, original_message: str) -> str:
        """Get comprehensive field information with source code"""
        try:
            logger.info(f"Getting comprehensive info for field: {field_name}")
            
            # Get field context from database
            context = self.db_manager.get_context_for_field(session_id, field_name)
            
            if context and context.get('field_details'):
                return self._format_database_field_response(field_name, context, original_message)
            else:
                # Perform live analysis
                return self._perform_live_field_analysis(session_id, field_name)
                
        except Exception as e:
            logger.error(f"Error getting field info: {str(e)}")
            return f"Error analyzing {field_name}: {str(e)}"
    
    def _format_database_field_response(self, field_name: str, context: Dict, message: str) -> str:
        """Format comprehensive field response from database context"""
        try:
            field_details = context.get('field_details', [])
            field_mappings = context.get('field_mappings', [])
            
            primary_field = field_details[0]
            response = f"Field Analysis: {field_name}\n"
            response += "=" * (len(field_name) + 16) + "\n"
            
            # Basic information
            response += f"Program: {primary_field.get('program_name', 'Unknown')}\n"
            response += f"Usage Type: {primary_field.get('usage_type', 'Unknown')}\n"
            
            # Business purpose
            business_purpose = primary_field.get('business_purpose', '')
            if business_purpose:
                response += f"Business Purpose: {business_purpose}\n"
            
            # Field definition with source code
            definition_code = primary_field.get('definition_code', '')
            if definition_code:
                response += f"\nField Definition:\n  {definition_code}\n"
            
            # Usage statistics
            total_refs = primary_field.get('total_program_references', 0)
            if total_refs > 0:
                response += f"\nUsage Statistics:\n"
                response += f"  Total References: {total_refs}\n"
                
                # Detailed breakdown
                stats = []
                if primary_field.get('move_target_count', 0) > 0:
                    stats.append(f"Receives data: {primary_field['move_target_count']} operations")
                if primary_field.get('move_source_count', 0) > 0:
                    stats.append(f"Provides data: {primary_field['move_source_count']} operations")
                if primary_field.get('arithmetic_count', 0) > 0:
                    stats.append(f"Calculations: {primary_field['arithmetic_count']} operations")
                if primary_field.get('conditional_count', 0) > 0:
                    stats.append(f"Conditions: {primary_field['conditional_count']} operations")
                if primary_field.get('cics_count', 0) > 0:
                    stats.append(f"CICS operations: {primary_field['cics_count']} operations")
                
                if stats:
                    response += f"  Usage Breakdown: {'; '.join(stats)}\n"
            
            # Source code examples
            field_refs_json = primary_field.get('field_references_json', '[]')
            try:
                references = json.loads(field_refs_json) if field_refs_json else []
                if references:
                    response += f"\nSource Code Examples:\n"
                    
                    # Show definition first
                    def_refs = [ref for ref in references if ref.get('operation_type') == 'DEFINITION']
                    if def_refs:
                        def_ref = def_refs[0]
                        response += f"  Definition (Line {def_ref['line_number']}):\n"
                        response += f"    {def_ref['line_content']}\n"
                    
                    # Show key operations
                    operation_refs = [ref for ref in references if ref.get('operation_type') != 'DEFINITION']
                    operation_refs.sort(key=lambda x: x.get('line_number', 0))
                    
                    for ref in operation_refs[:5]:  # Show up to 5 usage examples
                        response += f"  {ref.get('operation_type', 'Usage')} (Line {ref['line_number']}):\n"
                        response += f"    {ref['line_content']}\n"
                        if ref.get('business_context'):
                            response += f"    -> {ref['business_context']}\n"
                    
                    # Show detailed context for most important operation
                    if operation_refs and any(word in message.lower() for word in ['how', 'where', 'usage', 'context']):
                        important_ref = operation_refs[0]
                        context_block = important_ref.get('context_block', '')
                        if context_block:
                            response += f"\nDetailed Code Context:\n{context_block}\n"
                            
            except Exception as ref_error:
                logger.warning(f"Error parsing field references: {str(ref_error)}")
            
            # Field mappings if available
            if field_mappings:
                mapping = field_mappings[0]
                response += f"\nData Type Mapping:\n"
                response += f"  Mainframe: {mapping.get('mainframe_data_type', 'Unknown')}\n"
                response += f"  Oracle: {mapping.get('oracle_data_type', 'Unknown')}\n"
                if mapping.get('business_logic_description'):
                    response += f"  Logic: {mapping['business_logic_description']}\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting field response: {str(e)}")
            return f"Error formatting response for {field_name}: {str(e)}"
    
    def _perform_live_field_analysis(self, session_id: str, field_name: str) -> str:
        """Perform live field analysis when not in database"""
        try:
            logger.info(f"Performing live analysis for: {field_name}")
            
            components = self.db_manager.get_session_components(session_id)
            
            for component in components:
                if component.get('component_type') != 'PROGRAM':
                    continue
                
                # Get program source code
                analysis_json = component.get('analysis_result_json', '{}')
                analysis_data = json.loads(analysis_json) if analysis_json else {}
                source_content = analysis_data.get('content') or component.get('source_content', '')
                
                if not source_content:
                    continue
                
                # Check if field exists in this program
                if field_name.upper() in source_content.upper():
                    logger.info(f"Found {field_name} in {component['component_name']}")
                    
                    # Analyze field usage
                    field_analysis = self._analyze_field_in_program(
                        field_name, source_content, component['component_name']
                    )
                    
                    return self._format_live_analysis_response(field_name, field_analysis)
            
            return f"Field {field_name} was not found in any analyzed program source code. Please verify the field name and ensure the containing program has been uploaded and analyzed."
            
        except Exception as e:
            logger.error(f"Error in live field analysis: {str(e)}")
            return f"Error performing live analysis for {field_name}: {str(e)}"
    
    def _analyze_field_in_program(self, field_name: str, source_content: str, program_name: str) -> Dict:
        """Complete field analysis in program source code"""
        analysis = {
            'field_name': field_name,
            'program_name': program_name,
            'definition': None,
            'references': [],
            'usage_patterns': {
                'input_operations': [],
                'output_operations': [],
                'arithmetic_operations': [],
                'conditional_operations': [],
                'cics_operations': []
            },
            'business_summary': ''
        }
        
        try:
            lines = source_content.split('\n')
            field_upper = field_name.upper()
            
            for line_idx, line in enumerate(lines, 1):
                line_stripped = line.strip()
                line_upper = line_stripped.upper()
                
                # Skip comments and empty lines
                if not line_stripped or line_stripped.startswith('*'):
                    continue
                
                if field_upper in line_upper:
                    # Determine operation type
                    operation_type, business_context = self._classify_field_operation(line_upper, field_upper)
                    
                    # Get surrounding context
                    context_start = max(0, line_idx - 3)
                    context_end = min(len(lines), line_idx + 2)
                    context_lines = lines[context_start:context_end]
                    
                    reference = {
                        'line_number': line_idx,
                        'line_content': line_stripped,
                        'operation_type': operation_type,
                        'business_context': business_context,
                        'context_lines': context_lines,
                        'context_display': '\n'.join([
                            f"{context_start + i + 1:4d}: {ctx_line}"
                            for i, ctx_line in enumerate(context_lines)
                        ])
                    }
                    
                    # Categorize by operation type
                    if operation_type == 'DEFINITION':
                        analysis['definition'] = reference
                    elif operation_type == 'MOVE_TARGET':
                        analysis['usage_patterns']['input_operations'].append(reference)
                    elif operation_type == 'MOVE_SOURCE':
                        analysis['usage_patterns']['output_operations'].append(reference)
                    elif operation_type == 'ARITHMETIC':
                        analysis['usage_patterns']['arithmetic_operations'].append(reference)
                    elif operation_type == 'CONDITIONAL':
                        analysis['usage_patterns']['conditional_operations'].append(reference)
                    elif operation_type == 'CICS':
                        analysis['usage_patterns']['cics_operations'].append(reference)
                    
                    analysis['references'].append(reference)
            
            # Generate business summary
            analysis['business_summary'] = self._generate_field_business_summary(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing field {field_name}: {str(e)}")
            return analysis
    
    def _classify_field_operation(self, line_upper: str, field_upper: str) -> tuple:
        """Classify the type of operation involving the field"""
        # Field definition
        if ('PIC' in line_upper and 
            re.match(r'^\s*\d{2}\s+' + re.escape(field_upper), line_upper)):
            return 'DEFINITION', 'Data structure definition with type and length specification'
        
        # MOVE operations
        elif 'MOVE' in line_upper:
            # Field receives data (MOVE source TO field)
            if re.search(rf'MOVE\s+.+\s+TO\s+{re.escape(field_upper)}', line_upper):
                source_match = re.search(r'MOVE\s+([A-Z0-9\-\(\)]+)', line_upper)
                source = source_match.group(1) if source_match else 'unknown source'
                return 'MOVE_TARGET', f'Receives data from {source}'
            
            # Field provides data (MOVE field TO target)
            elif re.search(rf'MOVE\s+{re.escape(field_upper)}\s+TO', line_upper):
                target_match = re.search(rf'MOVE\s+{re.escape(field_upper)}\s+TO\s+([A-Z0-9\-\(\)]+)', line_upper)
                target = target_match.group(1) if target_match else 'unknown target'
                return 'MOVE_SOURCE', f'Provides data to {target}'
        
        # Arithmetic operations
        elif any(op in line_upper for op in ['COMPUTE', 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE']):
            return 'ARITHMETIC', 'Used in mathematical calculation or business computation'
        
        # Conditional operations
        elif any(op in line_upper for op in ['IF', 'WHEN', 'EVALUATE']):
            return 'CONDITIONAL', 'Used in business logic decision or program flow control'
        
        # CICS operations
        elif 'CICS' in line_upper:
            return 'CICS', 'Used in CICS transaction processing or screen handling'
        
        # File operations
        elif any(op in line_upper for op in ['READ', 'WRITE', 'REWRITE']):
            return 'FILE_IO', 'Used in file input/output operations'
        
        # General reference
        else:
            return 'REFERENCE', 'Referenced in program logic'
    
    def _generate_field_business_summary(self, analysis: Dict) -> str:
        """Generate business summary from field analysis"""
        patterns = analysis['usage_patterns']
        field_name = analysis['field_name']
        
        summary_parts = []
        
        if patterns['input_operations']:
            summary_parts.append(f"receives data ({len(patterns['input_operations'])} times)")
        
        if patterns['output_operations']:
            summary_parts.append(f"provides data ({len(patterns['output_operations'])} times)")
        
        if patterns['arithmetic_operations']:
            summary_parts.append(f"mathematical calculations ({len(patterns['arithmetic_operations'])} times)")
        
        if patterns['conditional_operations']:
            summary_parts.append(f"business decisions ({len(patterns['conditional_operations'])} times)")
        
        if patterns['cics_operations']:
            summary_parts.append(f"CICS transactions ({len(patterns['cics_operations'])} times)")
        
        if summary_parts:
            return f"{field_name} is actively used for: {', '.join(summary_parts)}"
        elif analysis['definition']:
            return f"{field_name} is defined but not actively used in the main program logic"
        else:
            return f"{field_name} usage pattern could not be determined"
    
    def _format_live_analysis_response(self, field_name: str, analysis: Dict) -> str:
        """Format response from live analysis"""
        response = f"Field Analysis: {field_name} (Live Analysis)\n"
        response += "=" * (len(field_name) + 28) + "\n"
        
        response += f"Program: {analysis['program_name']}\n"
        response += f"Business Summary: {analysis['business_summary']}\n"
        
        # Show definition
        if analysis['definition']:
            def_ref = analysis['definition']
            response += f"\nField Definition:\n"
            response += f"  Line {def_ref['line_number']}: {def_ref['line_content']}\n"
        
        # Show usage patterns
        patterns = analysis['usage_patterns']
        total_operations = sum(len(ops) for ops in patterns.values())
        
        if total_operations > 0:
            response += f"\nUsage Patterns ({total_operations} total operations):\n"
            
            if patterns['input_operations']:
                response += f"  Data Input ({len(patterns['input_operations'])} operations):\n"
                for op in patterns['input_operations'][:2]:
                    response += f"    Line {op['line_number']}: {op['line_content']}\n"
            
            if patterns['output_operations']:
                response += f"  Data Output ({len(patterns['output_operations'])} operations):\n"
                for op in patterns['output_operations'][:2]:
                    response += f"    Line {op['line_number']}: {op['line_content']}\n"
            
            if patterns['arithmetic_operations']:
                response += f"  Calculations ({len(patterns['arithmetic_operations'])} operations):\n"
                for op in patterns['arithmetic_operations'][:2]:
                    response += f"    Line {op['line_number']}: {op['line_content']}\n"
            
            if patterns['conditional_operations']:
                response += f"  Business Logic ({len(patterns['conditional_operations'])} operations):\n"
                for op in patterns['conditional_operations'][:2]:
                    response += f"    Line {op['line_number']}: {op['line_content']}\n"
        
        # Show detailed context for first significant operation
        significant_ops = (patterns['input_operations'] + patterns['output_operations'] + 
                          patterns['arithmetic_operations'])
        if significant_ops:
            important_op = significant_ops[0]
            response += f"\nDetailed Context Example:\n"
            response += important_op['context_display']
        
        return response
    
    def _handle_layout_query(self, session_id: str, message: str) -> str:
        """Handle queries about record layouts"""
        try:
            # Extract layout names from message
            layout_names = re.findall(r'\b([A-Z][A-Z0-9\-]{2,})\b', message.upper())
            
            if layout_names:
                # Get specific layout info
                layouts = self.db_manager.get_record_layouts(session_id)
                matching_layouts = []
                
                for layout_name in layout_names:
                    matches = [l for l in layouts if layout_name in l['layout_name'].upper()]
                    matching_layouts.extend(matches)
                
                if matching_layouts:
                    response = f"Record Layout Analysis:\n\n"
                    for layout in matching_layouts[:2]:
                        response += f"Layout: {layout['layout_name']}\n"
                        response += f"Program: {layout['program_name']}\n"
                        response += f"Level: {layout.get('level_number', '01')}\n"
                        response += f"Fields: {layout.get('fields_count', 0)}\n"
                        if layout.get('business_purpose'):
                            response += f"Purpose: {layout['business_purpose']}\n"
                        response += "\n"
                    
                    return response
            
            # General layout information
            return self._get_general_layout_info(session_id)
            
        except Exception as e:
            logger.error(f"Error handling layout query: {str(e)}")
            return f"Error analyzing layouts: {str(e)}"
    
    def _get_general_layout_info(self, session_id: str) -> str:
        """Get general information about record layouts"""
        try:
            layouts = self.db_manager.get_record_layouts(session_id)
            
            if not layouts:
                return "No record layouts found. Upload COBOL programs with data structures first."
            
            response = f"Record Layouts ({len(layouts)} found):\n\n"
            
            for layout in layouts[:5]:  # Show first 5
                response += f"Layout: {layout['layout_name']}\n"
                response += f"Program: {layout['program_name']}\n"
                response += f"Fields: {layout.get('fields_count', 0)}\n"
                if layout.get('business_purpose'):
                    response += f"Purpose: {layout['business_purpose']}\n"
                response += "\n"
            
            if len(layouts) > 5:
                response += f"... and {len(layouts) - 5} more layouts\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting layout info: {str(e)}")
            return "Error retrieving layout information"
    
    def _handle_program_query(self, session_id: str, message: str) -> str:
        """Handle queries about programs"""
        try:
            components = self.db_manager.get_session_components(session_id)
            programs = [c for c in components if c.get('component_type') == 'PROGRAM']
            
            if not programs:
                return "No programs have been analyzed yet. Please upload COBOL program files first."
            
            response = f"Program Analysis ({len(programs)} programs):\n\n"
            
            for program in programs:
                response += f"Program: {program['component_name']}\n"
                response += f"Lines: {program.get('total_lines', 0)}\n"
                response += f"Fields: {program.get('total_fields', 0)}\n"
                
                if program.get('business_purpose'):
                    response += f"Purpose: {program['business_purpose']}\n"
                
                response += "\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling program query: {str(e)}")
            return f"Error analyzing programs: {str(e)}"
    
    def _handle_summary_query(self, session_id: str) -> str:
        """Handle summary/overview queries"""
        try:
            metrics = self.db_manager.get_session_metrics(session_id)
            components = self.db_manager.get_session_components(session_id)
            
            response = "Project Analysis Summary:\n"
            response += "=" * 25 + "\n"
            
            # Basic metrics
            response += f"Total Components: {metrics.get('total_components', 0)}\n"
            response += f"Total Fields: {metrics.get('total_fields', 0)}\n"
            response += f"Lines of Code: {metrics.get('total_lines', 0)}\n"
            
            # Component breakdown
            if components:
                component_types = {}
                for comp in components:
                    comp_type = comp.get('component_type', 'Unknown')
                    component_types[comp_type] = component_types.get(comp_type, 0) + 1
                
                response += f"\nComponent Breakdown:\n"
                for comp_type, count in component_types.items():
                    response += f"  {comp_type}: {count}\n"
            
            # Token usage
            token_usage = metrics.get('token_usage', {})
            if token_usage:
                total_tokens = token_usage.get('total_prompt_tokens', 0) + token_usage.get('total_response_tokens', 0)
                response += f"\nToken Usage: {total_tokens:,} tokens\n"
                response += f"LLM Calls: {token_usage.get('total_calls', 0)}\n"
            
            response += f"\nAsk me about specific fields, record layouts, or programs!"
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling summary query: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def _handle_general_query(self, session_id: str, message: str) -> str:
        """Handle general queries"""
        try:
            # Check if user is asking for help
            if any(word in message.lower() for word in ['help', 'what can you', 'how do i']):
                return self._get_help_response()
            
            # Provide general guidance
            return ("I can help you analyze your mainframe code! Here's what you can ask:\n\n"
                   "Field Analysis:\n"
                   "  - 'Tell me about CUSTOMER-NAME'\n"
                   "  - 'How is ACCOUNT-NUMBER used?'\n"
                   "  - 'Show me EMPLOYEE-ID field'\n\n"
                   "General Information:\n"
                   "  - 'Show me the project summary'\n"
                   "  - 'What programs have been analyzed?'\n"
                   "  - 'List the record layouts'\n\n"
                   "Ask about any specific field name and I'll show you exactly how it's used in your COBOL programs!")
            
        except Exception as e:
            logger.error(f"Error handling general query: {str(e)}")
            return "I can help analyze your mainframe code. Ask me about specific fields, programs, or record layouts!"
    
    def _get_help_response(self) -> str:
        """Provide help information"""
        return ("Mainframe Code Analyzer Help:\n\n"
               "I can analyze your COBOL programs and provide detailed information about:\n\n"
               "1. Field Analysis:\n"
               "   - Field definitions and data types\n"
               "   - How fields are used (input, output, calculations)\n"
               "   - Source code examples showing field usage\n"
               "   - Business purpose and data flow\n\n"
               "2. Program Structure:\n"
               "   - Record layouts and data structures\n"
               "   - Component relationships and dependencies\n"
               "   - CICS transaction processing\n"
               "   - File operations and data flow\n\n"
               "Example questions:\n"
               "   - 'Tell me about CUSTOMER-NAME field'\n"
               "   - 'How is ACCOUNT-BALANCE calculated?'\n"
               "   - 'Show me the EMPLOYEE-RECORD layout'\n"
               "   - 'What programs use TRANSACTION-CODE?'\n\n"
               "Just ask about any field name and I'll show you the actual COBOL code!")
    
    def _get_available_fields_sample(self, session_id: str) -> str:
        """Get sample of available fields"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT field_name FROM field_analysis_details 
                    WHERE session_id = ? 
                    ORDER BY field_name 
                    LIMIT 10
                ''', (session_id,))
                
                fields = [row[0] for row in cursor.fetchall()]
                return ', '.join(fields) if fields else 'No fields analyzed yet'
                
        except Exception as e:
            logger.error(f"Error getting available fields: {str(e)}")
            return 'Error retrieving field list'
