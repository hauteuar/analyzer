"""
Enhanced Field Analysis Retriever for Complex Field Usage Patterns
Provides comprehensive analysis of field usage, group structures, and business logic flows
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FieldUsageContext:
    """Comprehensive field usage context"""
    field_name: str
    usage_type: str
    line_number: int
    source_line: str
    business_context: str
    condition_context: str
    program_flow_context: str
    related_fields: List[str]
    transaction_identifier: str = ""

@dataclass
class GroupFieldAnalysis:
    """Analysis of group field structures"""
    group_name: str
    level: int
    child_fields: List[Dict]
    parent_field: str
    business_purpose: str
    usage_patterns: List[FieldUsageContext]

class EnhancedFieldAnalyzer:
    """Enhanced field analyzer for complex business logic patterns"""
    
    def __init__(self, db_manager, vector_store):
        self.db_manager = db_manager
        self.vector_store = vector_store
        
    def analyze_field_comprehensive(self, session_id: str, field_name: str, 
                                  query_context: str = "") -> Dict:
        """
        Comprehensive field analysis including:
        - Group structure analysis
        - Business logic flow tracking
        - Conditional value assignments
        - Cross-program references
        - Transaction flow analysis
        """
        try:
            # Get all source code that contains this field
            field_contexts = self._get_field_source_contexts(session_id, field_name)
            
            # Analyze group structure if applicable
            group_analysis = self._analyze_group_structure(field_contexts, field_name)
            
            # Extract business logic patterns
            business_logic = self._extract_business_logic_patterns(field_contexts, field_name)
            
            # Analyze conditional assignments
            conditional_assignments = self._analyze_conditional_assignments(field_contexts, field_name)
            
            # Track program control flow (XCTL, CALL patterns)
            control_flow = self._analyze_control_flow_patterns(field_contexts, field_name)
            
            # Extract transaction identifiers and patterns
            transaction_patterns = self._extract_transaction_patterns(field_contexts, field_name)
            
            return {
                'field_name': field_name,
                'field_type': 'group' if group_analysis else 'elementary',
                'group_analysis': group_analysis,
                'business_logic_summary': self._generate_business_logic_summary(
                    business_logic, conditional_assignments, control_flow, transaction_patterns
                ),
                'conditional_assignments': conditional_assignments,
                'control_flow_patterns': control_flow,
                'transaction_patterns': transaction_patterns,
                'cross_program_usage': self._analyze_cross_program_usage(field_contexts, field_name),
                'semantic_context': self._build_semantic_context(field_contexts, query_context)
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive field analysis: {str(e)}")
            return {'error': str(e)}
    
    def _get_field_source_contexts(self, session_id: str, field_name: str) -> List[Dict]:
        """Enhanced field source context retrieval"""
        try:
            contexts = []
            
            # ENHANCED: Get ALL components that mention this field
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT ca.component_name, ca.analysis_result_json, ca.source_content
                    FROM component_analysis ca
                    WHERE ca.session_id = ? 
                    AND (ca.analysis_result_json LIKE ? OR ca.source_content LIKE ?)
                ''', (session_id, f'%{field_name}%', f'%{field_name}%'))
                
                for row in cursor.fetchall():
                    component_name, analysis_json, source_content = row
                    
                    # Use source_content directly if available
                    if source_content and field_name.upper() in source_content.upper():
                        contexts.append({
                            'component_name': component_name,
                            'source_code': source_content,
                            'similarity_score': 1.0,
                            'source': 'direct_source'
                        })
                    
                    # Also check analysis results
                    if analysis_json:
                        try:
                            analysis = json.loads(analysis_json)
                            if 'content' in analysis and field_name.upper() in analysis['content'].upper():
                                contexts.append({
                                    'component_name': component_name,
                                    'source_code': analysis['content'],
                                    'similarity_score': 0.9,
                                    'source': 'analysis_content'
                                })
                        except:
                            pass
            
            logger.info(f"Found {len(contexts)} contexts for field {field_name}")
            return contexts
            
        except Exception as e:
            logger.error(f"Error getting field contexts: {str(e)}")
            return []
    
    def _extract_business_logic_patterns(self, source_content: str, program_name: str) -> Dict:
        """Extract business logic patterns that drive dynamic behavior"""
        patterns = {
            'conditional_logic': [],
            'decision_trees': [],
            'variable_population_logic': [],
            'dynamic_routing_patterns': []
        }
        
        lines = source_content.split('\n')
        
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            
            # Pattern 1: Conditional variable assignment
            if re.search(r'IF.*MOVE.*TO.*(HOLD-|TRANX)', line_upper):
                patterns['conditional_logic'].append({
                    'line': i+1,
                    'condition': extract_condition(line),
                    'target_variable': extract_target_var(line),
                    'business_rule': infer_business_rule(line)
                })
            
            # Pattern 2: EVALUATE statements for routing
            if 'EVALUATE' in line_upper and any(var in line_upper for var in ['TRANX', 'HOLD-']):
                decision_tree = extract_evaluate_logic(lines, i)
                patterns['decision_trees'].append(decision_tree)
            
            # Pattern 3: Table-driven logic
            if re.search(r'(SEARCH|PERFORM.*VARYING).*TRANX', line_upper):
                patterns['variable_population_logic'].append({
                    'type': 'table_driven',
                    'line': i+1,
                    'logic': extract_table_logic(lines, i)
                })

    def _build_semantic_context(self, contexts: List[Dict], query_context: str) -> str:
        """Build semantic context for better understanding"""
        try:
            if not contexts:
                return "No semantic context available"
            
            context_parts = []
            
            # Summarize contexts
            context_parts.append(f"Found in {len(contexts)} program contexts")
            
            # Extract key programs
            programs = [ctx['component_name'] for ctx in contexts]
            context_parts.append(f"Programs: {', '.join(programs[:3])}")
            
            # Look for business keywords in query
            query_lower = query_context.lower()
            business_keywords = ['business', 'process', 'rule', 'logic', 'calculate', 'validate']
            
            relevant_keywords = [kw for kw in business_keywords if kw in query_lower]
            if relevant_keywords:
                context_parts.append(f"Business focus: {', '.join(relevant_keywords)}")
            
            return '; '.join(context_parts)
            
        except Exception as e:
            logger.error(f"Error building semantic context: {str(e)}")
            return "Semantic context analysis failed"

    def _extract_target_program(self, line_upper: str, keyword: str) -> Optional[str]:
        """Extract target program from control statement"""
        try:
            # Look for PROGRAM clause
            prog_match = re.search(r'PROGRAM\s*\(\s*[\'"]?([A-Z0-9\-]+)[\'"]?\s*\)', line_upper)
            if prog_match:
                return prog_match.group(1)
            
            # Look for direct program name after keyword
            if keyword == 'CALL':
                call_match = re.search(r'CALL\s+[\'"]?([A-Z0-9\-]+)[\'"]?', line_upper)
                if call_match:
                    return call_match.group(1)
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting target program: {str(e)}")
            return None

    def _analyze_field_usage_in_control(self, line: str, field_name: str) -> str:
        """Analyze how field is used in control statement"""
        field_upper = field_name.upper()
        line_upper = line.upper()
        
        if f'PROGRAM({field_upper})' in line_upper or f'PROGRAM ({field_upper})' in line_upper:
            return 'DYNAMIC_PROGRAM_NAME'
        elif field_upper in line_upper:
            return 'PARAMETER_OR_DATA'
        else:
            return 'CONTEXT_REFERENCE'

    def _infer_business_flow(self, control_type: str, target_program: str, context_lines: List[str]) -> str:
        """Infer business flow from control statement"""
        try:
            context_text = ' '.join(context_lines).upper()
            
            # Transaction management patterns
            if any(keyword in target_program.upper() for keyword in ['TMST', 'TMS', 'TRAN']):
                return 'Transaction management system call'
            
            # Error handling patterns
            elif any(keyword in context_text for keyword in ['ERROR', 'FAIL', 'ABEND']):
                return 'Error handling routine'
            
            # Validation patterns
            elif any(keyword in context_text for keyword in ['VALID', 'CHECK', 'VERIFY']):
                return 'Data validation process'
            
            # File processing patterns
            elif any(keyword in context_text for keyword in ['READ', 'WRITE', 'UPDATE']):
                return 'File processing operation'
            
            # Based on control type
            elif control_type == 'XCTL':
                return 'Transfer control to next process'
            elif control_type == 'LINK':
                return 'Call subroutine and return'
            elif control_type == 'CALL':
                return 'Program subroutine call'
            
            return f'{control_type} operation to {target_program}'
            
        except Exception as e:
            logger.error(f"Error inferring business flow: {str(e)}")
            return 'Program control operation'

    def _analyze_group_structure(self, contexts: List[Dict], field_name: str) -> Optional[GroupFieldAnalysis]:
        """Analyze if field is part of a group and extract structure"""
        try:
            for context in contexts:
                source_lines = context['source_code'].split('\n')
                
                # Look for group definition patterns
                for i, line in enumerate(source_lines):
                    line_upper = line.upper().strip()
                    
                    # Check if this is a group definition line
                    group_match = re.match(r'^\s*(\d{2})\s+([A-Z][A-Z0-9\-]+)', line_upper)
                    if group_match and field_name.upper() in line_upper:
                        level = int(group_match.group(1))
                        group_name = group_match.group(2)
                        
                        # Extract child fields
                        child_fields = self._extract_child_fields(source_lines, i, level)
                        
                        # FIXED: Ensure parent_field is a string
                        parent_field = self._find_parent_field(source_lines, i, level)
                        
                        # Safety check - ensure it's a string
                        if isinstance(parent_field, list):
                            parent_field = parent_field[0] if parent_field else ""
                        elif not isinstance(parent_field, str):
                            parent_field = str(parent_field) if parent_field else ""
                        
                        return GroupFieldAnalysis(
                            group_name=group_name,
                            level=level,
                            child_fields=child_fields,
                            parent_field=parent_field,  # Now guaranteed to be string
                            business_purpose=self._infer_group_business_purpose(group_name, child_fields),
                            usage_patterns=[]
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing group structure: {str(e)}")
            return None

    # Add this missing method to the EnhancedFieldAnalyzer class:

    def _find_parent_field(self, source_lines: List[str], current_idx: int, current_level: int) -> str:
        """Find parent field by scanning backwards for lower level number"""
        try:
            # Scan backwards from current position
            for i in range(current_idx - 1, -1, -1):
                line = source_lines[i].strip()
                if not line or line.startswith('*'):
                    continue
                
                # Look for field definition with lower level
                field_match = re.match(r'^\s*(\d{2})\s+([A-Z][A-Z0-9\-]+)', line.upper())
                if field_match:
                    level = int(field_match.group(1))
                    field_name = field_match.group(2)
                    
                    # Found a parent (lower level number)
                    if level < current_level:
                        return field_name
            
            # No parent found
            return ""
            
        except Exception as e:
            logger.error(f"Error finding parent field: {str(e)}")
            return ""

    # Also fix the _analyze_group_structure method to handle the string return:

    
    # Add the missing helper method:
    def _infer_group_business_purpose(self, group_name: str, child_fields: List[Dict]) -> str:
        """Infer business purpose from group name and child fields"""
        try:
            # Analyze group name patterns
            name_upper = group_name.upper()
            
            if 'TRAN' in name_upper or 'TXN' in name_upper:
                return "Transaction processing data structure"
            elif 'CUST' in name_upper:
                return "Customer information record"
            elif 'ACCT' in name_upper:
                return "Account data structure"
            elif any(field.get('has_value') for field in child_fields):
                return "Configuration or constant data group"
            elif len(child_fields) > 5:
                return "Complex data structure with multiple components"
            else:
                return f"Data group containing {len(child_fields)} fields"
                
        except Exception as e:
            logger.error(f"Error inferring group business purpose: {str(e)}")
            return "Data structure group"

    def _extract_child_fields(self, source_lines: List[str], start_idx: int, parent_level: int) -> List[Dict]:
        """Extract child fields from group definition"""
        child_fields = []
        
        try:
            for i in range(start_idx + 1, len(source_lines)):
                line = source_lines[i].strip()
                if not line or line.startswith('*'):
                    continue
                
                # Check for field definition
                field_match = re.match(r'^\s*(\d{2})\s+([A-Z][A-Z0-9\-]+)(?:\s+(.+))?', line.upper())
                if field_match:
                    level = int(field_match.group(1))
                    field_name = field_match.group(2)
                    definition = field_match.group(3) or ""
                    
                    # If level is equal or less than parent, we've reached end of group
                    if level <= parent_level:
                        break
                    
                    # If it's an immediate child
                    if level == parent_level + 5:  # Standard COBOL level increment
                        child_fields.append({
                            'name': field_name,
                            'level': level,
                            'definition': definition,
                            'line_number': i + 1,
                            'is_filler': 'FILLER' in field_name,
                            'has_value': 'VALUE' in definition,
                            'picture': self._extract_picture_clause(definition),
                            'value': self._extract_value_clause(definition)
                        })
        except Exception as e:
            logger.error(f"Error extracting child fields: {str(e)}")
        
        return child_fields
    
    def _analyze_conditional_assignments(self, contexts: List[Dict], field_name: str) -> List[Dict]:
        """Analyze conditional value assignments to field"""
        assignments = []
        
        try:
            for context in contexts:
                source_lines = context['source_code'].split('\n')
                component_name = context['component_name']
                
                # Look for conditional patterns
                in_conditional_block = False
                current_condition = ""
                condition_start_line = 0
                
                for i, line in enumerate(source_lines):
                    line_upper = line.upper().strip()
                    
                    # Detect start of conditional blocks
                    if any(keyword in line_upper for keyword in ['IF ', 'WHEN ', 'EVALUATE']):
                        in_conditional_block = True
                        current_condition = self._extract_condition_logic(line_upper)
                        condition_start_line = i + 1
                    
                    # Detect end of conditional blocks
                    elif any(keyword in line_upper for keyword in ['END-IF', 'END-EVALUATE', 'WHEN OTHER']):
                        in_conditional_block = False
                        current_condition = ""
                    
                    # Look for MOVE statements to our field
                    if field_name.upper() in line_upper and 'MOVE' in line_upper:
                        move_pattern = rf'MOVE\s+([^T]+)\s+TO\s+.*{re.escape(field_name.upper())}'
                        move_match = re.search(move_pattern, line_upper)
                        
                        if move_match:
                            source_value = move_match.group(1).strip()
                            
                            # Get surrounding context
                            context_lines = self._get_context_lines(source_lines, i, 5)
                            
                            assignment = {
                                'program': component_name,
                                'line_number': i + 1,
                                'source_value': source_value,
                                'condition': current_condition if in_conditional_block else "UNCONDITIONAL",
                                'condition_line': condition_start_line if in_conditional_block else 0,
                                'business_context': self._infer_business_context(source_value, current_condition),
                                'context_lines': context_lines,
                                'transaction_identifier': self._extract_transaction_identifier(source_value, context_lines)
                            }
                            
                            assignments.append(assignment)
            
            # Sort by program and line number
            assignments.sort(key=lambda x: (x['program'], x['line_number']))
            
        except Exception as e:
            logger.error(f"Error analyzing conditional assignments: {str(e)}")
        
        return assignments
    
    def _analyze_control_flow_patterns(self, contexts: List[Dict], field_name: str) -> List[Dict]:
        """Analyze how field is used in program control flow (XCTL, CALL, etc.)"""
        control_patterns = []
        
        try:
            for context in contexts:
                source_lines = context['source_code'].split('\n')
                component_name = context['component_name']
                
                for i, line in enumerate(source_lines):
                    line_upper = line.upper().strip()
                    
                    # Look for program control statements that use our field
                    if field_name.upper() in line_upper:
                        control_keywords = ['XCTL', 'LINK', 'CALL', 'EXEC CICS XCTL', 'EXEC CICS LINK']
                        
                        for keyword in control_keywords:
                            if keyword in line_upper:
                                # Extract target program
                                target_program = self._extract_target_program(line_upper, keyword)
                                
                                # Get context around this control statement
                                context_lines = self._get_context_lines(source_lines, i, 3)
                                
                                control_pattern = {
                                    'source_program': component_name,
                                    'control_type': keyword,
                                    'target_program': target_program,
                                    'line_number': i + 1,
                                    'usage_pattern': self._analyze_field_usage_in_control(line_upper, field_name),
                                    'context_lines': context_lines,
                                    'business_flow': self._infer_business_flow(keyword, target_program, context_lines)
                                }
                                
                                control_patterns.append(control_pattern)
        
        except Exception as e:
            logger.error(f"Error analyzing control flow patterns: {str(e)}")
        
        return control_patterns
    
    def _extract_transaction_patterns(self, contexts: List[Dict], field_name: str) -> Dict:
        """Extract transaction-specific patterns and business logic"""
        patterns = {
            'transaction_codes': [],
            'business_flows': [],
            'routing_logic': []
        }
        
        try:
            for context in contexts:
                source_lines = context['source_code'].split('\n')
                
                # Look for transaction code patterns
                for i, line in enumerate(source_lines):
                    line_upper = line.upper().strip()
                    
                    if field_name.upper() in line_upper:
                        # Extract transaction codes from VALUE clauses or MOVE statements
                        if 'VALUE' in line_upper:
                            value_match = re.search(r"VALUE\s+['\"]([^'\"]+)['\"]", line_upper)
                            if value_match:
                                tx_code = value_match.group(1)
                                patterns['transaction_codes'].append({
                                    'code': tx_code,
                                    'context': 'CONSTANT_DEFINITION',
                                    'line': i + 1,
                                    'program': context['component_name']
                                })
                        
                        elif 'MOVE' in line_upper:
                            move_match = re.search(rf"MOVE\s+['\"]([^'\"]+)['\"].*{re.escape(field_name.upper())}", line_upper)
                            if move_match:
                                tx_code = move_match.group(1)
                                business_context = self._get_context_lines(source_lines, i, 3)
                                
                                patterns['transaction_codes'].append({
                                    'code': tx_code,
                                    'context': 'RUNTIME_ASSIGNMENT',
                                    'line': i + 1,
                                    'program': context['component_name'],
                                    'business_context': business_context
                                })
        
        except Exception as e:
            logger.error(f"Error extracting transaction patterns: {str(e)}")
        
        return patterns
    
    def _generate_business_logic_summary(self, business_logic: List, conditional_assignments: List, 
                                       control_flow: List, transaction_patterns: Dict) -> str:
        """Generate comprehensive business logic summary"""
        try:
            summary_parts = []
            
            # Field usage overview
            if conditional_assignments:
                summary_parts.append(f"Field receives {len(conditional_assignments)} different value assignments:")
                
                for assignment in conditional_assignments:
                    condition_text = assignment['condition']
                    value = assignment['source_value']
                    program = assignment['program']
                    
                    if condition_text == "UNCONDITIONAL":
                        summary_parts.append(f"  • In {program}: Always set to '{value}'")
                    else:
                        summary_parts.append(f"  • In {program}: Set to '{value}' when {condition_text}")
            
            # Transaction code analysis
            if transaction_patterns.get('transaction_codes'):
                summary_parts.append("\nTransaction Code Analysis:")
                for tx in transaction_patterns['transaction_codes']:
                    summary_parts.append(f"  • Code '{tx['code']}' ({tx['context']}) in {tx['program']}")
            
            # Control flow analysis
            if control_flow:
                summary_parts.append("\nProgram Control Flow:")
                for flow in control_flow:
                    summary_parts.append(f"  • {flow['source_program']} uses {flow['control_type']} to {flow['target_program']}")
                    if flow.get('business_flow'):
                        summary_parts.append(f"    Business purpose: {flow['business_flow']}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating business logic summary: {str(e)}")
            return "Error generating summary"
    
    # Helper methods
    def _extract_condition_logic(self, line: str) -> str:
        """Extract readable condition logic"""
        # Simplify complex conditions for readability
        condition = line.replace('IF ', '').replace('WHEN ', '').replace('EVALUATE ', '')
        return condition.strip()[:100]  # Limit length
    
    def _get_context_lines(self, source_lines: List[str], center_line: int, radius: int) -> List[str]:
        """Get context lines around a specific line"""
        start = max(0, center_line - radius)
        end = min(len(source_lines), center_line + radius + 1)
        return [f"{i+1:4d}: {source_lines[i]}" for i in range(start, end)]
    
    def _extract_picture_clause(self, definition: str) -> str:
        """Extract PIC clause from field definition"""
        pic_match = re.search(r'PIC(?:TURE)?\s+([^\s]+)', definition.upper())
        return pic_match.group(1) if pic_match else ""
    
    def _extract_value_clause(self, definition: str) -> str:
        """Extract VALUE clause from field definition"""
        value_match = re.search(r"VALUE\s+['\"]([^'\"]+)['\"]", definition.upper())
        return value_match.group(1) if value_match else ""
    
    def _infer_business_context(self, source_value: str, condition: str) -> str:
        """Infer business context from value and condition"""
        if 'TMS' in source_value.upper():
            return "Transaction Management System identifier"
        elif len(source_value) == 4 and source_value.isalnum():
            return "Transaction code assignment"
        else:
            return f"Conditional assignment based on {condition[:50]}"
    
    def _extract_transaction_identifier(self, source_value: str, context_lines: List[str]) -> str:
        """Extract transaction identifier from context"""
        if 'TMS' in source_value.upper():
            return source_value
        
        # Look in context for transaction patterns
        for line in context_lines:
            if 'TRAN' in line.upper() or 'TXN' in line.upper():
                return "Transaction-related"
        
        return ""
    
    def _get_database_field_contexts(self, session_id: str, field_name: str) -> List[Dict]:
        """Get field contexts from database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT ca.component_name, ca.source_content
                    FROM component_analysis ca
                    WHERE ca.session_id = ? AND ca.source_content LIKE ?
                ''', (session_id, f'%{field_name}%'))
                
                contexts = []
                for row in cursor.fetchall():
                    contexts.append({
                        'component_name': row[0],
                        'source_code': row[1],
                        'similarity_score': 1.0
                    })
                
                return contexts
        except Exception as e:
            logger.error(f"Error getting database field contexts: {str(e)}")
            return []