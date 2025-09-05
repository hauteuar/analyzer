"""
Enhanced Program Flow and Data Passing Analyzer
Analyzes complex program interactions, DFHCOMMAREA usage, file operations, and business flows
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProgramCallContext:
    """Context for program calls including data passing"""
    source_program: str
    target_program: str
    call_type: str  # XCTL, LINK, CALL
    data_area: str  # DFHCOMMAREA, LINKAGE SECTION data
    data_structure: Dict
    business_context: str
    call_conditions: List[str]
    line_number: int

@dataclass
class FileOperationContext:
    """Context for file operations"""
    program: str
    file_name: str
    operation_type: str  # READ, WRITE, UPDATE, DELETE
    record_layout: str
    business_purpose: str
    data_flow_direction: str
    related_program_calls: List[str]

@dataclass
class BusinessFlowAnalysis:
    """Complete business flow analysis"""
    flow_name: str
    entry_point: str
    program_chain: List[ProgramCallContext]
    file_operations: List[FileOperationContext]
    data_transformations: List[Dict]
    business_purpose: str
    end_result: str

class EnhancedProgramFlowAnalyzer:
    """Analyzer for complex program flows and data passing patterns"""
    
    def __init__(self, db_manager, vector_store):
        self.db_manager = db_manager
        self.vector_store = vector_store
    
    def analyze_program_flow_comprehensive(self, session_id: str, program_name: str, 
                                         query_context: str = "") -> Dict:
        """
        Comprehensive program flow analysis including:
        - Program call chains and data passing
        - DFHCOMMAREA usage and structure
        - File operations and data flow
        - Business process flow
        - Cross-program data transformations
        """
        try:
            # Get program source and related contexts
            program_contexts = self._get_program_source_contexts(session_id, program_name)
            
            # Analyze program call patterns
            program_calls = self._analyze_program_call_patterns(program_contexts, program_name)
            
            # Analyze DFHCOMMAREA and data passing
            data_passing = self._analyze_data_passing_patterns(program_contexts, program_name)
            
            # Analyze file operations
            file_operations = self._analyze_file_operation_patterns(program_contexts, program_name)
            
            # Build complete business flow
            business_flow = self._build_business_flow_analysis(
                program_calls, data_passing, file_operations, program_name
            )
            
            # Analyze cross-program dependencies
            dependencies = self._analyze_cross_program_dependencies(session_id, program_name)
            
            return {
                'program_name': program_name,
                'program_calls': program_calls,
                'data_passing_analysis': data_passing,
                'file_operations': file_operations,
                'business_flow': business_flow,
                'cross_program_dependencies': dependencies,
                'flow_summary': self._generate_flow_summary(business_flow, query_context)
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive program flow analysis: {str(e)}")
            return {'error': str(e)}
    
    def _get_program_source_contexts(self, session_id: str, program_name: str) -> List[Dict]:
        """Get source code contexts for program and related programs"""
        try:
            contexts = []
            
            # Get main program source
            main_source = self.db_manager.get_component_source_code(
                session_id, program_name, max_size=200000
            )
            
            if main_source.get('success') and main_source.get('components'):
                for comp in main_source['components']:
                    contexts.append({
                        'component_name': comp['component_name'],
                        'source_code': comp.get('source_for_chat', ''),
                        'component_type': 'main_program',
                        'analysis_result': comp.get('analysis_result_json', '{}')
                    })
            
            # Get related programs through dependencies
            dependencies = self.db_manager.get_enhanced_dependencies(session_id)
            related_programs = set()
            
            for dep in dependencies:
                if (dep.get('source_component') == program_name and 
                    dep.get('relationship_type') == 'PROGRAM_CALL'):
                    related_programs.add(dep.get('target_component'))
                elif (dep.get('target_component') == program_name and 
                      dep.get('relationship_type') == 'PROGRAM_CALL'):
                    related_programs.add(dep.get('source_component'))
            
            # Get source for related programs
            for related_prog in list(related_programs)[:3]:  # Limit to 3 related programs
                try:
                    related_source = self.db_manager.get_component_source_code(
                        session_id, related_prog, max_size=100000
                    )
                    
                    if related_source.get('success') and related_source.get('components'):
                        for comp in related_source['components']:
                            contexts.append({
                                'component_name': comp['component_name'],
                                'source_code': comp.get('source_for_chat', ''),
                                'component_type': 'related_program',
                                'analysis_result': comp.get('analysis_result_json', '{}')
                            })
                except Exception as e:
                    logger.warning(f"Error getting related program {related_prog}: {e}")
                    continue
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error getting program source contexts: {str(e)}")
            return []
    
    def _analyze_program_call_patterns(self, contexts: List[Dict], program_name: str) -> List[ProgramCallContext]:
        """Analyze program call patterns with data passing context"""
        call_patterns = []
        
        try:
            for context in contexts:
                source_lines = context['source_code'].split('\n')
                component_name = context['component_name']
                
                # Look for program calls with context
                for i, line in enumerate(source_lines):
                    line_upper = line.upper().strip()
                    
                    # Detect XCTL, LINK, CALL patterns
                    call_keywords = ['EXEC CICS XCTL', 'EXEC CICS LINK', 'CALL']
                    
                    for keyword in call_keywords:
                        if keyword in line_upper:
                            # Extract target program
                            target_program = self._extract_target_program_enhanced(
                                line_upper, keyword, source_lines, i
                            )
                            
                            if target_program:
                                # Analyze data passing context
                                data_context = self._analyze_call_data_context(
                                    source_lines, i, keyword
                                )
                                
                                # Get business context
                                business_context = self._extract_call_business_context(
                                    source_lines, i, 10
                                )
                                
                                # Get call conditions
                                call_conditions = self._extract_call_conditions(
                                    source_lines, i
                                )
                                
                                call_pattern = ProgramCallContext(
                                    source_program=component_name,
                                    target_program=target_program,
                                    call_type=keyword.replace('EXEC CICS ', ''),
                                    data_area=data_context.get('data_area', ''),
                                    data_structure=data_context.get('structure', {}),
                                    business_context=business_context,
                                    call_conditions=call_conditions,
                                    line_number=i + 1
                                )
                                
                                call_patterns.append(call_pattern)
        
        except Exception as e:
            logger.error(f"Error analyzing program call patterns: {str(e)}")
        
        return call_patterns
    
    def _analyze_data_passing_patterns(self, contexts: List[Dict], program_name: str) -> Dict:
        """Analyze DFHCOMMAREA and other data passing mechanisms"""
        data_analysis = {
            'dfhcommarea_usage': [],
            'linkage_section_data': [],
            'working_storage_sharing': [],
            'parameter_passing': []
        }
        
        try:
            for context in contexts:
                source_lines = context['source_code'].split('\n')
                component_name = context['component_name']
                
                # Analyze DFHCOMMAREA usage
                dfhcommarea_analysis = self._analyze_dfhcommarea_usage(
                    source_lines, component_name
                )
                if dfhcommarea_analysis:
                    data_analysis['dfhcommarea_usage'].extend(dfhcommarea_analysis)
                
                # Analyze LINKAGE SECTION
                linkage_analysis = self._analyze_linkage_section(
                    source_lines, component_name
                )
                if linkage_analysis:
                    data_analysis['linkage_section_data'].extend(linkage_analysis)
                
                # Analyze parameter passing in calls
                param_analysis = self._analyze_parameter_passing(
                    source_lines, component_name
                )
                if param_analysis:
                    data_analysis['parameter_passing'].extend(param_analysis)
        
        except Exception as e:
            logger.error(f"Error analyzing data passing patterns: {str(e)}")
        
        return data_analysis
    
    def _analyze_file_operation_patterns(self, contexts: List[Dict], program_name: str) -> List[FileOperationContext]:
        """Analyze file operations with business context"""
        file_operations = []
        
        try:
            for context in contexts:
                source_lines = context['source_code'].split('\n')
                component_name = context['component_name']
                
                # Find file operations
                for i, line in enumerate(source_lines):
                    line_upper = line.upper().strip()
                    
                    # CICS file operations
                    cics_file_ops = ['EXEC CICS READ', 'EXEC CICS WRITE', 'EXEC CICS REWRITE', 'EXEC CICS DELETE']
                    
                    for op in cics_file_ops:
                        if op in line_upper:
                            file_context = self._extract_file_operation_context(
                                source_lines, i, op, component_name
                            )
                            if file_context:
                                file_operations.append(file_context)
                    
                    # Regular file operations
                    regular_file_ops = ['READ ', 'WRITE ', 'REWRITE ']
                    for op in regular_file_ops:
                        if line_upper.startswith(op) or f' {op}' in line_upper:
                            file_context = self._extract_regular_file_context(
                                source_lines, i, op.strip(), component_name
                            )
                            if file_context:
                                file_operations.append(file_context)
        
        except Exception as e:
            logger.error(f"Error analyzing file operation patterns: {str(e)}")
        
        return file_operations
    
    def _build_business_flow_analysis(self, program_calls: List[ProgramCallContext], 
                                    data_passing: Dict, file_operations: List[FileOperationContext],
                                    entry_program: str) -> BusinessFlowAnalysis:
        """Build comprehensive business flow analysis"""
        try:
            # Determine flow name and purpose
            flow_name = f"{entry_program}_BusinessFlow"
            
            # Analyze business purpose from program calls and file operations
            business_purpose = self._infer_business_flow_purpose(
                program_calls, file_operations, entry_program
            )
            
            # Build program chain
            program_chain = sorted(program_calls, key=lambda x: x.line_number)
            
            # Analyze data transformations
            data_transformations = self._analyze_data_transformations(
                data_passing, program_calls, file_operations
            )
            
            # Determine end result
            end_result = self._determine_flow_end_result(
                program_calls, file_operations, data_transformations
            )
            
            return BusinessFlowAnalysis(
                flow_name=flow_name,
                entry_point=entry_program,
                program_chain=program_chain,
                file_operations=file_operations,
                data_transformations=data_transformations,
                business_purpose=business_purpose,
                end_result=end_result
            )
            
        except Exception as e:
            logger.error(f"Error building business flow analysis: {str(e)}")
            return BusinessFlowAnalysis(
                flow_name=f"{entry_program}_Flow",
                entry_point=entry_program,
                program_chain=[],
                file_operations=[],
                data_transformations=[],
                business_purpose="Analysis failed",
                end_result="Unknown"
            )
    
    def _analyze_dfhcommarea_usage(self, source_lines: List[str], program_name: str) -> List[Dict]:
        """Analyze DFHCOMMAREA usage patterns"""
        dfhcommarea_usage = []
        
        try:
            for i, line in enumerate(source_lines):
                line_upper = line.upper().strip()
                
                if 'DFHCOMMAREA' in line_upper:
                    usage_context = {
                        'program': program_name,
                        'line_number': i + 1,
                        'usage_type': self._classify_dfhcommarea_usage(line_upper),
                        'context_lines': self._get_context_lines(source_lines, i, 5),
                        'data_structure': self._extract_commarea_structure(source_lines, i),
                        'business_context': self._infer_commarea_business_context(source_lines, i)
                    }
                    dfhcommarea_usage.append(usage_context)
        
        except Exception as e:
            logger.error(f"Error analyzing DFHCOMMAREA usage: {str(e)}")
        
        return dfhcommarea_usage
    
    def _extract_file_operation_context(self, source_lines: List[str], line_idx: int, 
                                      operation: str, program_name: str) -> Optional[FileOperationContext]:
        """Extract detailed file operation context"""
        try:
            line = source_lines[line_idx].upper()
            
            # Extract file name
            file_match = re.search(r'(?:FILE|DATASET)\s*\(\s*[\'"]?([A-Z0-9\-]+)[\'"]?\s*\)', line)
            file_name = file_match.group(1) if file_match else 'UNKNOWN'
            
            # Extract record layout from INTO/FROM clause
            record_layout = ''
            into_match = re.search(r'INTO\s*\(\s*([A-Z0-9\-]+)\s*\)', line)
            from_match = re.search(r'FROM\s*\(\s*([A-Z0-9\-]+)\s*\)', line)
            
            if into_match:
                record_layout = into_match.group(1)
            elif from_match:
                record_layout = from_match.group(1)
            
            # Determine operation type and data flow
            op_type = operation.replace('EXEC CICS ', '')
            data_flow = 'INPUT' if 'READ' in op_type.lower() else 'OUTPUT'
            
            # Get business context
            business_context = self._get_context_lines(source_lines, line_idx, 3)
            business_purpose = self._infer_file_business_purpose(
                file_name, op_type, business_context
            )
            
            # Find related program calls
            related_calls = self._find_related_program_calls(source_lines, line_idx, 20)
            
            return FileOperationContext(
                program=program_name,
                file_name=file_name,
                operation_type=op_type,
                record_layout=record_layout,
                business_purpose=business_purpose,
                data_flow_direction=data_flow,
                related_program_calls=related_calls
            )
            
        except Exception as e:
            logger.error(f"Error extracting file operation context: {str(e)}")
            return None
    
    def _generate_flow_summary(self, business_flow: BusinessFlowAnalysis, query_context: str) -> str:
        """Generate comprehensive flow summary"""
        try:
            summary_parts = []
            
            # Flow overview
            summary_parts.extend([
                f"Business Flow: {business_flow.flow_name}",
                f"Entry Point: {business_flow.entry_point}",
                f"Purpose: {business_flow.business_purpose}",
                ""
            ])
            
            # Program call chain
            if business_flow.program_chain:
                summary_parts.append("Program Call Chain:")
                for i, call in enumerate(business_flow.program_chain, 1):
                    summary_parts.append(
                        f"{i}. {call.source_program} → {call.target_program} "
                        f"({call.call_type})"
                    )
                    if call.data_area:
                        summary_parts.append(f"   Data: {call.data_area}")
                    if call.business_context:
                        summary_parts.append(f"   Context: {call.business_context}")
                summary_parts.append("")
            
            # File operations
            if business_flow.file_operations:
                summary_parts.append("File Operations:")
                for file_op in business_flow.file_operations:
                    summary_parts.append(
                        f"• {file_op.program}: {file_op.operation_type} {file_op.file_name}"
                    )
                    if file_op.record_layout:
                        summary_parts.append(f"  Layout: {file_op.record_layout}")
                    summary_parts.append(f"  Purpose: {file_op.business_purpose}")
                summary_parts.append("")
            
            # Data transformations
            if business_flow.data_transformations:
                summary_parts.append("Data Transformations:")
                for transform in business_flow.data_transformations:
                    summary_parts.append(f"• {transform.get('description', 'Data transformation')}")
                summary_parts.append("")
            
            # End result
            summary_parts.append(f"End Result: {business_flow.end_result}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating flow summary: {str(e)}")
            return "Error generating flow summary"
    
    # Helper methods
    def _extract_target_program_enhanced(self, line: str, keyword: str, 
                                       source_lines: List[str], line_idx: int) -> Optional[str]:
        """Enhanced target program extraction with context"""
        try:
            # Look for PROGRAM clause
            prog_match = re.search(r'PROGRAM\s*\(\s*[\'"]?([A-Z0-9\-]+)[\'"]?\s*\)', line)
            if prog_match:
                return prog_match.group(1)
            
            # Look for variable containing program name
            var_match = re.search(r'PROGRAM\s*\(\s*([A-Z0-9\-]+)\s*\)', line)
            if var_match:
                var_name = var_match.group(1)
                # Look for variable assignment in surrounding lines
                for i in range(max(0, line_idx - 10), min(len(source_lines), line_idx + 5)):
                    check_line = source_lines[i].upper()
                    if f'MOVE' in check_line and var_name in check_line:
                        move_match = re.search(rf'MOVE\s+[\'"]?([A-Z0-9\-]+)[\'"]?\s+TO\s+{var_name}', check_line)
                        if move_match:
                            return move_match.group(1)
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting target program: {str(e)}")
            return None
    
    def _get_context_lines(self, source_lines: List[str], center_line: int, radius: int) -> List[str]:
        """Get context lines around a specific line"""
        start = max(0, center_line - radius)
        end = min(len(source_lines), center_line + radius + 1)
        return [f"{i+1:4d}: {source_lines[i]}" for i in range(start, end)]
    
    def _classify_dfhcommarea_usage(self, line: str) -> str:
        """Classify how DFHCOMMAREA is being used"""
        if 'MOVE' in line and 'TO DFHCOMMAREA' in line:
            return 'POPULATE_COMMAREA'
        elif 'MOVE DFHCOMMAREA' in line:
            return 'READ_COMMAREA'
        elif 'LENGTH OF DFHCOMMAREA' in line:
            return 'SIZE_CALCULATION'
        elif any(op in line for op in ['XCTL', 'LINK']):
            return 'PASS_TO_PROGRAM'
        else:
            return 'GENERAL_REFERENCE'
    
    def _infer_business_flow_purpose(self, program_calls: List[ProgramCallContext], 
                                   file_operations: List[FileOperationContext], entry_program: str) -> str:
        """Infer business purpose from flow patterns"""
        purposes = []
        
        # Analyze program names for business clues
        if any('CUST' in call.target_program for call in program_calls):
            purposes.append("Customer processing")
        if any('ACCT' in call.target_program for call in program_calls):
            purposes.append("Account management")
        if any('TXN' in call.target_program or 'TRAN' in call.target_program for call in program_calls):
            purposes.append("Transaction processing")
        
        # Analyze file operations
        if file_operations:
            if any('PRO' in op.file_name for op in file_operations):
                purposes.append("Profile/Product file processing")
            if any(op.operation_type in ['WRITE', 'REWRITE'] for op in file_operations):
                purposes.append("Data updates")
            if any(op.operation_type == 'READ' for op in file_operations):
                purposes.append("Data retrieval")
        
        if purposes:
            return f"{entry_program} handles: " + ", ".join(purposes)
        else:
            return f"{entry_program} business processing workflow"