"""
Program Flow Analyzer - Traces complete data flows through program chains
Handles dynamic calls, field passing, and file operations
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger(__name__)

class ProgramFlowAnalyzer:
    def __init__(self, db_manager, component_extractor, llm_client=None):
        self.db_manager = db_manager
        self.component_extractor = component_extractor
        self.llm_client = llm_client
    
    def analyze_complete_program_flow(self, session_id: str, starting_program: str) -> Dict:
        """Analyze complete program flow starting from a given program"""
        logger.info(f"Starting complete flow analysis from {starting_program}")
        
        flow_analysis = {
            'flow_id': str(uuid.uuid4()),
            'starting_program': starting_program,
            'program_chain': [],
            'field_flows': [],
            'file_operations': [],
            'missing_programs': [],
            'data_transformations': [],
            'business_flow_summary': ''
        }
        
        try:
            # Step 1: Build program call chain
            program_chain = self._build_program_call_chain(session_id, starting_program)
            flow_analysis['program_chain'] = program_chain
            
            # Step 2: Trace field data flow through chain
            field_flows = self._trace_field_data_flow(session_id, program_chain)
            flow_analysis['field_flows'] = field_flows
            
            # Step 3: Identify file operations in chain
            file_operations = self._identify_file_operations_in_chain(session_id, program_chain)
            flow_analysis['file_operations'] = file_operations
            
            # Step 4: Identify missing programs and their impact
            missing_programs = self._identify_missing_programs_in_flow(session_id, program_chain)
            flow_analysis['missing_programs'] = missing_programs
            
            # Step 5: Generate business flow summary
            business_summary = self._generate_business_flow_summary(session_id, flow_analysis)
            flow_analysis['business_flow_summary'] = business_summary
            
            # Store in database
            self._store_program_flow_analysis(session_id, flow_analysis)
            
            return flow_analysis
            
        except Exception as e:
            logger.error(f"Error in complete program flow analysis: {str(e)}")
            return flow_analysis
    
    def _build_program_call_chain(self, session_id: str, starting_program: str) -> List[Dict]:
        """Build the complete program call chain"""
        chain = []
        processed_programs = set()
        
        def build_chain_recursive(current_program: str, level: int = 0):
            if current_program in processed_programs or level > 10:  # Prevent infinite loops
                return
            
            processed_programs.add(current_program)
            
            # Get dependencies for current program
            dependencies = self.db_manager.get_enhanced_dependencies(session_id)
            program_calls = [d for d in dependencies 
                           if d['source_component'] == current_program 
                           and d['relationship_type'] in ['PROGRAM_CALL', 'DYNAMIC_PROGRAM_CALL']]
            
            for call in program_calls:
                target_program = call['target_component']
                analysis_details = call.get('analysis_details', {})
                
                chain_step = {
                    'sequence': level + 1,
                    'source_program': current_program,
                    'target_program': target_program,
                    'call_type': analysis_details.get('call_type', 'unknown'),
                    'call_mechanism': call['relationship_type'],
                    'variable_name': analysis_details.get('variable_name', ''),
                    'resolution_method': analysis_details.get('resolution_method', ''),
                    'confidence': call.get('confidence_score', 0.8),
                    'line_number': analysis_details.get('line_number', 0),
                    'is_missing': call.get('display_status') == 'missing'
                }
                
                # Add data flow context
                if not chain_step['is_missing']:
                    data_context = self._analyze_call_data_context(session_id, current_program, target_program)
                    chain_step['data_passed'] = data_context.get('passed_fields', [])
                    chain_step['data_received'] = data_context.get('received_fields', [])
                
                chain.append(chain_step)
                
                # Continue chain if target program is available
                if not chain_step['is_missing']:
                    build_chain_recursive(target_program, level + 1)
        
        build_chain_recursive(starting_program)
        return chain
    
    def _trace_field_data_flow(self, session_id: str, program_chain: List[Dict]) -> List[Dict]:
        """Trace field data flow through the program chain"""
        field_flows = []
        
        for chain_step in program_chain:
            if chain_step.get('is_missing'):
                continue
                
            source_prog = chain_step['source_program']
            target_prog = chain_step['target_program']
            
            # Get source program content
            source_data = self.db_manager.get_component_source_code(session_id, source_prog)
            if not source_data.get('success'):
                continue
            
            source_content = source_data['components'][0].get('source_for_chat', '')
            
            # Analyze field passing in this call
            field_analysis = self._analyze_field_passing_in_call(
                source_content, chain_step, session_id
            )
            
            if field_analysis:
                field_flows.extend(field_analysis)
        
        return field_flows
    
    def _analyze_field_passing_in_call(self, source_content: str, chain_step: Dict, session_id: str) -> List[Dict]:
        """Analyze which fields are passed between programs in a call"""
        field_flows = []
        
        try:
            lines = source_content.split('\n')
            call_line = chain_step.get('line_number', 0)
            
            # Look for USING clause around the call
            for i, line in enumerate(lines):
                if abs(i + 1 - call_line) <= 3:  # Within 3 lines of call
                    line_upper = line.upper().strip()
                    
                    # Check for USING clause
                    if 'USING' in line_upper:
                        using_fields = self._extract_using_fields(line_upper)
                        
                        for field_name in using_fields:
                            # Get field analysis
                            field_info = self._get_field_info_from_db(session_id, field_name, chain_step['source_program'])
                            
                            field_flow = {
                                'field_name': field_name,
                                'source_program': chain_step['source_program'],
                                'target_program': chain_step['target_program'],
                                'flow_type': 'PASSED_TO_PROGRAM',
                                'call_mechanism': chain_step['call_mechanism'],
                                'field_type': field_info.get('usage_type', 'UNKNOWN'),
                                'data_source': field_info.get('source_field', ''),
                                'business_purpose': field_info.get('business_purpose', ''),
                                'transformation_logic': f"Passed via {chain_step['call_type']} call",
                                'sequence': chain_step['sequence'],
                                'confidence': 0.9
                            }
                            field_flows.append(field_flow)
            
            return field_flows
            
        except Exception as e:
            logger.error(f"Error analyzing field passing: {str(e)}")
            return field_flows
    
    def _identify_file_operations_in_chain(self, session_id: str, program_chain: List[Dict]) -> List[Dict]:
        """Identify file operations throughout the program chain"""
        file_operations = []
        
        for chain_step in program_chain:
            if chain_step.get('is_missing'):
                continue
                
            program_name = chain_step['source_program']
            
            # Get program's file operations
            dependencies = self.db_manager.get_enhanced_dependencies(session_id)
            file_deps = [d for d in dependencies 
                        if d['source_component'] == program_name 
                        and ('FILE' in d['relationship_type'] or d['interface_type'] in ['FILE_SYSTEM', 'CICS'])]
            
            for file_dep in file_deps:
                analysis_details = file_dep.get('analysis_details', {})
                
                file_op = {
                    'program_name': program_name,
                    'file_name': file_dep['target_component'],
                    'operations': analysis_details.get('operations', []),
                    'io_direction': analysis_details.get('io_direction', 'UNKNOWN'),
                    'interface_type': file_dep['interface_type'],
                    'sequence_in_flow': chain_step['sequence'],
                    'associated_layouts': analysis_details.get('associated_layouts', []),
                    'has_layout_resolution': analysis_details.get('layout_resolved', False)
                }
                
                # Get fields involved in file operations
                if file_op['has_layout_resolution']:
                    layout_fields = self._get_layout_fields(session_id, file_op['associated_layouts'])
                    file_op['fields_involved'] = layout_fields
                
                file_operations.append(file_op)
        
        return file_operations
    
    def _generate_business_flow_summary(self, session_id: str, flow_analysis: Dict) -> str:
        """Generate business summary of the program flow"""
        try:
            summary_parts = []
            
            starting_prog = flow_analysis['starting_program']
            summary_parts.append(f"Program Flow Analysis starting from {starting_prog}:")
            summary_parts.append("")
            
            # Program chain summary
            chain = flow_analysis.get('program_chain', [])
            if chain:
                summary_parts.append("Program Call Chain:")
                for step in chain:
                    if step.get('is_missing'):
                        summary_parts.append(f"  {step['source_program']} → {step['target_program']} [MISSING]")
                    else:
                        call_type = step.get('call_type', 'CALL')
                        summary_parts.append(f"  {step['source_program']} → {step['target_program']} ({call_type})")
                summary_parts.append("")
            
            # Field flow summary
            field_flows = flow_analysis.get('field_flows', [])
            if field_flows:
                summary_parts.append("Key Data Fields in Flow:")
                field_summary = {}
                for field_flow in field_flows:
                    field_name = field_flow['field_name']
                    if field_name not in field_summary:
                        field_summary[field_name] = []
                    field_summary[field_name].append(f"{field_flow['source_program']} → {field_flow['target_program']}")
                
                for field, flows in list(field_summary.items())[:5]:  # Top 5 fields
                    summary_parts.append(f"  {field}: {' → '.join(flows)}")
                summary_parts.append("")
            
            # File operations summary
            file_ops = flow_analysis.get('file_operations', [])
            if file_ops:
                summary_parts.append("File Operations in Flow:")
                for file_op in file_ops:
                    io_dir = file_op.get('io_direction', 'UNKNOWN')
                    summary_parts.append(f"  {file_op['program_name']} {io_dir} {file_op['file_name']}")
                summary_parts.append("")
            
            # Missing programs impact
            missing = flow_analysis.get('missing_programs', [])
            if missing:
                summary_parts.append("Missing Programs Blocking Flow:")
                for missing_prog in missing:
                    summary_parts.append(f"  {missing_prog['program_name']} - {missing_prog['impact_description']}")
            
            return '\n'.join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating business flow summary: {str(e)}")
            return "Business flow summary generation failed"
    
    def _store_program_flow_analysis(self, session_id: str, flow_analysis: Dict):
        """Store program flow analysis in database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                flow_id = flow_analysis['flow_id']
                
                # Store program flow traces
                for step in flow_analysis.get('program_chain', []):
                    cursor.execute('''
                        INSERT INTO program_flow_traces 
                        (session_id, flow_id, source_program, target_program, call_sequence,
                         call_mechanism, variable_name, resolved_via, data_flow_json,
                         business_context, line_number, confidence_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id, flow_id, step['source_program'], step['target_program'],
                        step['sequence'], step['call_mechanism'], step.get('variable_name', ''),
                        step.get('resolution_method', ''), 
                        json.dumps({'passed_fields': step.get('data_passed', []), 'received_fields': step.get('data_received', [])}),
                        f"Call from {step['source_program']} to {step['target_program']}", 
                        step.get('line_number', 0), step.get('confidence', 0.8)
                    ))
                
                # Store field data flows
                for field_flow in flow_analysis.get('field_flows', []):
                    cursor.execute('''
                        INSERT INTO field_data_flow
                        (session_id, flow_id, field_name, source_program, target_program,
                         flow_type, data_source, data_target, transformation_logic,
                         sequence_in_flow, field_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id, flow_id, field_flow['field_name'],
                        field_flow['source_program'], field_flow['target_program'],
                        field_flow['flow_type'], field_flow.get('data_source', ''),
                        field_flow.get('data_target', ''), field_flow.get('transformation_logic', ''),
                        field_flow.get('sequence', 1), field_flow.get('field_type', 'UNKNOWN')
                    ))
                
                logger.info(f"Stored program flow analysis: {flow_id}")
                
        except Exception as e:
            logger.error(f"Error storing program flow analysis: {str(e)}")
    
    # Helper methods
    def _extract_using_fields(self, line: str) -> List[str]:
        """Extract field names from USING clause"""
        fields = []
        using_match = re.search(r'USING\s+(.*)', line)
        if using_match:
            using_clause = using_match.group(1)
            # Split by comma and clean up
            field_parts = using_clause.split()
            for part in field_parts:
                clean_field = part.strip(',()').strip()
                if clean_field and len(clean_field) > 2 and clean_field.isalpha():
                    fields.append(clean_field)
        return fields
    
    def _get_field_info_from_db(self, session_id: str, field_name: str, program_name: str) -> Dict:
        """Get field information from database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT usage_type, source_field, business_purpose
                    FROM field_analysis_details
                    WHERE session_id = ? AND field_name = ? AND program_name = ?
                    LIMIT 1
                ''', (session_id, field_name, program_name))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'usage_type': result[0],
                        'source_field': result[1],
                        'business_purpose': result[2]
                    }
                return {}
        except Exception as e:
            logger.error(f"Error getting field info: {str(e)}")
            return {}