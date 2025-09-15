"""
Program Flow Analyzer - FIXED VERSION
Traces complete data flows through program chains with proper dependency updates
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
        """Analyze complete program flow starting from a given program - FIXED VERSION"""
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
            # FIXED: Refresh dependencies before analysis to get latest status
            self._refresh_dependency_status(session_id)
            
            # Step 1: Build program call chain with updated dependency status
            program_chain = self._build_program_call_chain_fixed(session_id, starting_program)
            flow_analysis['program_chain'] = program_chain
            
            # Step 2: Trace field data flow through chain
            field_flows = self._trace_field_data_flow_fixed(session_id, program_chain)
            flow_analysis['field_flows'] = field_flows
            
            # Step 3: Identify file operations in chain with layout associations
            file_operations = self._identify_file_operations_with_layouts(session_id, program_chain)
            flow_analysis['file_operations'] = file_operations
            
            # Step 4: Identify missing programs and their impact
            missing_programs = self._identify_missing_programs_in_flow(session_id, program_chain)
            flow_analysis['missing_programs'] = missing_programs
            
            # Step 5: Generate business flow summary
            business_summary = self._generate_business_flow_summary(session_id, flow_analysis)
            flow_analysis['business_flow_summary'] = business_summary
            
            # Store in database
            self._store_program_flow_analysis(session_id, flow_analysis)
            
            
            logger.info(f"Program flow analysis completed: {len(program_chain)} programs, {len(missing_programs)} missing")
            return flow_analysis
            
        except Exception as e:
            logger.error(f"Error in complete program flow analysis: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return flow_analysis
    
    def _refresh_dependency_status(self, session_id: str):
        """Refresh dependency status to reflect newly uploaded programs"""
        try:
            # Get all uploaded programs
            components = self.db_manager.get_session_components(session_id)
            uploaded_programs = set(comp['component_name'].upper() for comp in components)
            
            logger.info(f"Refreshing dependency status. Uploaded programs: {len(uploaded_programs)}")
            
            # Update dependency status in database
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get all dependencies that might need status updates
                cursor.execute('''
                    SELECT id, target_component, dependency_status, interface_type, relationship_type
                    FROM dependency_relationships 
                    WHERE session_id = ? AND relationship_type IN ('PROGRAM_CALL', 'DYNAMIC_PROGRAM_CALL')
                ''', (session_id,))
                
                dependencies = cursor.fetchall()
                updated_count = 0
                
                for dep_id, target_component, current_status, interface_type, rel_type in dependencies:
                    target_upper = target_component.upper()
                    
                    # Determine new status
                    if interface_type == 'COBOL' and rel_type in ['PROGRAM_CALL', 'DYNAMIC_PROGRAM_CALL']:
                        new_status = 'present' if target_upper in uploaded_programs else 'missing'
                    else:
                        # Files, CICS, etc. remain as they are
                        continue
                    
                    # Update if status changed
                    if new_status != current_status:
                        cursor.execute('''
                            UPDATE dependency_relationships 
                            SET dependency_status = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        ''', (new_status, dep_id))
                        updated_count += 1
                        logger.info(f"Updated {target_component}: {current_status} -> {new_status}")
                
                logger.info(f"Refreshed {updated_count} dependency statuses")
                
        except Exception as e:
            logger.error(f"Error refreshing dependency status: {str(e)}")
    
    def _build_program_call_chain_cached(self, starting_program: str, session_data: Dict) -> List[Dict]:
        """Build program chain using cached data"""
        program_chain = []
        programs_to_process = [starting_program]
        processed_programs = set()
        sequence_counter = 1
        
        while programs_to_process and len(processed_programs) < 20:
            current_program = programs_to_process.pop(0)
            if current_program in processed_programs:
                continue
                
            processed_programs.add(current_program)
            
            # Get program calls from cached data
            program_calls = session_data['program_analyses'].get(current_program, {}).get('program_calls', [])
            
            for call in program_calls:
                target_program = call.get('program_name', '')
                if not target_program:
                    continue
                    
                is_missing = target_program.upper() not in session_data['uploaded_programs']
                
                flow_step = {
                    'sequence': sequence_counter,
                    'source_program': current_program,
                    'target_program': target_program,
                    'call_type': call.get('call_type', 'CALL'),
                    'variable_name': call.get('variable_name', ''),
                    'is_missing': is_missing,
                    'confidence': call.get('confidence_score', 0.8),
                    'line_number': call.get('line_number', 0),
                    'business_context': call.get('business_context', f"{call.get('call_type', 'CALL')} from {current_program} to {target_program}")
                }
                
                program_chain.append(flow_step)
                sequence_counter += 1
                
                # Continue processing if program is available
                if not is_missing and target_program not in processed_programs:
                    programs_to_process.append(target_program)
        
        return program_chain

    def _get_layout_associations_cached(self, session_data: Dict, program_chain: List[Dict]) -> List[Dict]:
        """Get detailed layout associations with usage patterns"""
        layout_associations = []
        
        for step in program_chain:
            if step.get('is_missing'):
                continue
                
            program_name = step['source_program']
            program_layouts = [layout for layout in session_data['layouts_data'].values() 
                            if layout['program_name'] == program_name]
            
            for layout in program_layouts:
                # Analyze field usage patterns
                input_fields = [f for f in layout['fields'] if f['usage_type'] in ['INPUT', 'MOVE_TARGET']]
                output_fields = [f for f in layout['fields'] if f['usage_type'] in ['OUTPUT', 'MOVE_SOURCE']]
                derived_fields = [f for f in layout['fields'] if f['usage_type'] == 'DERIVED']
                
                # Determine layout purpose in the call
                layout_purpose = self._determine_layout_call_purpose(layout, step, session_data)
                
                # Find connected files/DB2 tables
                connected_files = self._find_connected_files_for_layout(program_name, layout, session_data)
                
                layout_info = {
                    'layout_name': layout['layout_name'],
                    'friendly_name': layout['friendly_name'],
                    'business_purpose': layout['business_purpose'],
                    'program_name': program_name,
                    'sequence_in_flow': step['sequence'],
                    'fields_count': len(layout['fields']),
                    'key_fields': layout['fields'][:8],
                    'has_fields': len(layout['fields']) > 0,
                    
                    # Enhanced details
                    'usage_in_call': layout_purpose,
                    'field_distribution': {
                        'input_fields': len(input_fields),
                        'output_fields': len(output_fields), 
                        'derived_fields': len(derived_fields),
                        'static_fields': len(layout['fields']) - len(input_fields) - len(output_fields) - len(derived_fields)
                    },
                    'data_flow_role': self._analyze_layout_data_flow_role(input_fields, output_fields, derived_fields),
                    'connected_files': connected_files,
                    'sample_input_fields': [f['field_name'] for f in input_fields[:5]],
                    'sample_output_fields': [f['field_name'] for f in output_fields[:5]],
                    'business_narrative': self._create_layout_business_narrative(layout, input_fields, output_fields, connected_files, step)
                }
                layout_associations.append(layout_info)
        
        return layout_associations

    def _determine_layout_call_purpose(self, layout: Dict, step: Dict, session_data: Dict) -> str:
        """Determine how this layout is used in the program call"""
        fields = layout['fields']
        input_count = len([f for f in fields if f['usage_type'] in ['INPUT', 'MOVE_TARGET']])
        output_count = len([f for f in fields if f['usage_type'] in ['OUTPUT', 'MOVE_SOURCE']])
        
        # Check if this layout might be passed to the called program
        target_program = step['target_program']
        target_layouts = [l for l in session_data['layouts_data'].values() 
                        if l['program_name'] == target_program]
        
        # Look for similar layout in target program
        similar_layout = None
        for target_layout in target_layouts:
            if (layout['layout_name'].replace('-', '').replace('WS', '') in 
                target_layout['layout_name'].replace('-', '').replace('WS', '')):
                similar_layout = target_layout
                break
        
        if similar_layout:
            return f"Passed as parameter to {target_program} (maps to {similar_layout['layout_name']})"
        elif input_count > output_count:
            return f"Receives data before calling {target_program}"
        elif output_count > input_count:
            return f"Processes data after calling {target_program}"
        else:
            return f"Working storage for {target_program} call"

    def _analyze_layout_data_flow_role(self, input_fields: List, output_fields: List, derived_fields: List) -> str:
        """Analyze the layout's role in data flow"""
        if len(input_fields) > 10 and len(output_fields) < 3:
            return "Data Collection Hub - Gathers input from multiple sources"
        elif len(output_fields) > 10 and len(input_fields) < 3:
            return "Data Distribution Hub - Formats output for multiple destinations"
        elif len(derived_fields) > 5:
            return "Calculation Engine - Transforms and computes data"
        elif len(input_fields) > 0 and len(output_fields) > 0:
            return "Data Processing Pipeline - Input-Process-Output pattern"
        else:
            return "Data Structure Definition - Static data container"

    def _find_connected_files_for_layout(self, program_name: str, layout: Dict, session_data: Dict) -> List[Dict]:
        """Find files that might use this layout"""
        connected_files = []
        program_data = session_data['program_analyses'].get(program_name, {})
        
        # Check file operations
        file_ops = program_data.get('file_operations', [])
        for file_op in file_ops:
            file_name = file_op.get('file_name', '')
            operations = file_op.get('operations', [])
            
            # Heuristic: if layout name contains file name or vice versa
            layout_base = layout['layout_name'].replace('-REC', '').replace('-RECORD', '')
            if (layout_base.upper() in file_name.upper() or 
                file_name.replace('-FILE', '').upper() in layout_base.upper()):
                
                connected_files.append({
                    'file_name': file_name,
                    'operations': operations,
                    'io_direction': file_op.get('io_direction', 'UNKNOWN'),
                    'connection_confidence': 'HIGH' if layout_base.upper() in file_name.upper() else 'MEDIUM'
                })
        
        return connected_files

    def _create_layout_business_narrative(self, layout: Dict, input_fields: List, output_fields: List, 
                                        connected_files: List, step: Dict) -> str:
        """Create business narrative for the layout"""
        narrative_parts = []
        
        layout_name = layout['layout_name']
        program_name = step['source_program']
        target_program = step['target_program']
        
        # Start with layout purpose
        if len(input_fields) > len(output_fields):
            narrative_parts.append(f"{layout_name} collects input data in {program_name}")
        elif len(output_fields) > len(input_fields):
            narrative_parts.append(f"{layout_name} formats output data in {program_name}")
        else:
            narrative_parts.append(f"{layout_name} structures working data in {program_name}")
        
        # Add file connections
        if connected_files:
            file_ops = []
            for cf in connected_files:
                ops_str = ', '.join(cf['operations'])
                file_ops.append(f"{ops_str} {cf['file_name']}")
            
            if file_ops:
                narrative_parts.append(f"Used to {' and '.join(file_ops)}")
        
        # Add call context
        if len(input_fields) > 5:
            narrative_parts.append(f"Data prepared before calling {target_program}")
        elif len(output_fields) > 5:
            narrative_parts.append(f"Results processed after {target_program} returns")
        
        return '. '.join(narrative_parts) + '.'
    
    def _trace_field_data_flow_cached(self, session_data: Dict, program_chain: List[Dict]) -> List[Dict]:
        """Enhanced field flow tracing with detailed context"""
        field_flows = []
        
        for step in program_chain:
            if step.get('is_missing'):
                continue
                
            source_prog = step['source_program']
            target_prog = step['target_program']
            
            # Get layouts for both programs
            source_layouts = [l for l in session_data['layouts_data'].values() if l['program_name'] == source_prog]
            target_layouts = [l for l in session_data['layouts_data'].values() if l['program_name'] == target_prog]
            
            # Analyze field flows between programs
            for source_layout in source_layouts:
                for target_layout in target_layouts:
                    field_mappings = self._analyze_detailed_field_mapping(source_layout, target_layout, step, session_data)
                    field_flows.extend(field_mappings)
        
        return field_flows

    def _analyze_detailed_field_mapping(self, source_layout: Dict, target_layout: Dict, 
                                    step: Dict, session_data: Dict) -> List[Dict]:
        """Analyze detailed field mappings between layouts"""
        field_mappings = []
        
        source_fields = {f['field_name']: f for f in source_layout['fields']}
        target_fields = {f['field_name']: f for f in target_layout['fields']}
        
        # Find exact matches and similar fields
        for field_name, source_field in source_fields.items():
            target_field = None
            mapping_type = None
            
            if field_name in target_fields:
                target_field = target_fields[field_name]
                mapping_type = 'EXACT_MATCH'
            else:
                # Look for similar field names
                similar_field = self._find_similar_field(field_name, target_fields)
                if similar_field:
                    target_field = similar_field
                    mapping_type = 'SIMILAR_MATCH'
            
            if target_field:
                # Analyze the data transformation
                transformation_analysis = self._analyze_field_transformation_detailed(
                    source_field, target_field, source_layout, target_layout, step
                )
                
                field_mapping = {
                    'field_name': field_name,
                    'source_program': step['source_program'],
                    'target_program': step['target_program'],
                    'source_layout': source_layout['layout_name'],
                    'target_layout': target_layout['layout_name'],
                    'mapping_type': mapping_type,
                    'target_field_name': target_field['field_name'],
                    'sequence': step['sequence'],
                    
                    # Enhanced details
                    'source_usage': source_field['usage_type'],
                    'target_usage': target_field['usage_type'],
                    'data_transformation': transformation_analysis,
                    'business_impact': self._analyze_field_business_impact(source_field, target_field, transformation_analysis),
                    'flow_narrative': self._create_field_flow_narrative(source_field, target_field, source_layout, target_layout, step)
                }
                field_mappings.append(field_mapping)
        
        return field_mappings

    def _find_similar_field(self, field_name: str, target_fields: Dict) -> Dict:
        """Find similar field names using fuzzy matching"""
        clean_name = field_name.replace('-', '').replace('_', '').upper()
        
        for target_name, target_field in target_fields.items():
            clean_target = target_name.replace('-', '').replace('_', '').upper()
            
            # Check if one contains the other or they share significant portion
            if (clean_name in clean_target or clean_target in clean_name or 
                len(set(clean_name) & set(clean_target)) / max(len(clean_name), len(clean_target)) > 0.6):
                return target_field
        
        return None

    def _analyze_field_transformation_detailed(self, source_field: Dict, target_field: Dict, 
                                            source_layout: Dict, target_layout: Dict, step: Dict) -> Dict:
        """Detailed analysis of field transformation"""
        transformation = {
            'type': 'DIRECT_COPY',
            'changes': [],
            'business_rules': [],
            'data_quality_impact': 'NONE'
        }
        
        # Analyze usage type changes
        source_usage = source_field['usage_type']
        target_usage = target_field['usage_type']
        
        if source_usage != target_usage:
            transformation['type'] = 'USAGE_TRANSFORMATION'
            transformation['changes'].append(f"Usage changed from {source_usage} to {target_usage}")
            
            # Interpret business meaning
            if source_usage == 'INPUT' and target_usage == 'DERIVED':
                transformation['business_rules'].append("Field becomes calculated/derived in target program")
            elif source_usage == 'OUTPUT' and target_usage == 'INPUT':
                transformation['business_rules'].append("Output from source becomes input to target")
            elif target_usage == 'OUTPUT':
                transformation['business_rules'].append("Field prepared for output/reporting")
        
        # Analyze data type changes
        source_type = source_field.get('data_type', 'UNKNOWN')
        target_type = target_field.get('data_type', 'UNKNOWN')
        
        if source_type != target_type and source_type != 'UNKNOWN' and target_type != 'UNKNOWN':
            transformation['changes'].append(f"Data type changed from {source_type} to {target_type}")
            transformation['data_quality_impact'] = 'POTENTIAL_CONVERSION_ISSUES'
        
        # Analyze business purpose changes
        source_purpose = source_field.get('business_purpose', '')
        target_purpose = target_field.get('business_purpose', '')
        
        if source_purpose and target_purpose and source_purpose != target_purpose:
            transformation['business_rules'].append(f"Business purpose evolved: {source_purpose} → {target_purpose}")
        
        return transformation

    def _analyze_field_business_impact(self, source_field: Dict, target_field: Dict, 
                                    transformation: Dict) -> str:
        """Analyze business impact of field transformation"""
        if transformation['type'] == 'DIRECT_COPY':
            return "Data passed unchanged - maintains data integrity"
        elif transformation['type'] == 'USAGE_TRANSFORMATION':
            if target_field['usage_type'] == 'DERIVED':
                return "Field becomes calculated - adds business value through computation"
            elif target_field['usage_type'] == 'OUTPUT':
                return "Field prepared for reporting/external use - supports business reporting"
            else:
                return "Field usage pattern changes - may indicate business logic transformation"
        else:
            return "Field transformation applied - review for business rule compliance"

    def _create_field_flow_narrative(self, source_field: Dict, target_field: Dict, 
                                source_layout: Dict, target_layout: Dict, step: Dict) -> str:
        """Create business narrative for field flow"""
        field_name = source_field['field_name']
        source_prog = step['source_program']
        target_prog = step['target_program']
        
        narrative_parts = []
        
        # Field origin and processing
        if source_field['usage_type'] == 'INPUT':
            narrative_parts.append(f"{field_name} is collected as input in {source_prog}")
        elif source_field['usage_type'] == 'DERIVED':
            narrative_parts.append(f"{field_name} is calculated in {source_prog}")
        else:
            narrative_parts.append(f"{field_name} is processed in {source_prog}")
        
        # Transfer mechanism
        narrative_parts.append(f"passed to {target_prog} via {step['call_type']}")
        
        # Target usage
        if target_field['usage_type'] == 'OUTPUT':
            narrative_parts.append(f"then formatted for output in {target_layout['layout_name']}")
        elif target_field['usage_type'] == 'DERIVED':
            narrative_parts.append(f"then used in calculations in {target_layout['layout_name']}")
        else:
            narrative_parts.append(f"then further processed in {target_layout['layout_name']}")
        
        return ' '.join(narrative_parts) + '.'
    
    def _identify_file_operations_cached(self, session_data: Dict, program_chain: List[Dict]) -> List[Dict]:
        """Identify file operations using cached data"""
        file_operations = []
        
        for step in program_chain:
            if step.get('is_missing'):
                continue
                
            program_name = step['source_program']
            program_data = session_data['program_analyses'].get(program_name, {})
            
            # Get file operations from cached analysis
            file_ops = program_data.get('file_operations', [])
            for file_op in file_ops:
                file_operations.append({
                    'program_name': program_name,
                    'file_name': file_op.get('file_name', ''),
                    'operations': file_op.get('operations', []),
                    'io_direction': file_op.get('io_direction', 'UNKNOWN'),
                    'sequence_in_flow': step['sequence']
                })
        
        return file_operations
    
    def _generate_simple_summary(self, flow_analysis: Dict) -> str:
        """Generate detailed business summary with data flow narrative"""
        chain = flow_analysis.get('program_chain', [])
        layouts = flow_analysis.get('layout_associations', [])
        fields = flow_analysis.get('field_flows', [])
        files = flow_analysis.get('file_operations', [])
        missing = [s for s in chain if s.get('is_missing')]
        
        summary_parts = []
        
        # Executive summary
        summary_parts.append(f"PROGRAM FLOW ANALYSIS: {flow_analysis['starting_program']}")
        summary_parts.append("=" * 60)
        summary_parts.append("")
        
        # Flow overview
        summary_parts.append("BUSINESS PROCESS FLOW:")
        for i, step in enumerate(chain):
            status = "[MISSING]" if step.get('is_missing') else "[AVAILABLE]"
            call_info = f"({step['call_type']}"
            if step.get('variable_name'):
                call_info += f" via {step['variable_name']}"
            call_info += ")"
            
            summary_parts.append(f"  {i+1}. {step['source_program']} calls {step['target_program']} {call_info} {status}")
        summary_parts.append("")
        
        # Data structure analysis
        if layouts:
            summary_parts.append("DATA STRUCTURES & USAGE:")
            layout_by_program = {}
            for layout in layouts:
                prog = layout['program_name']
                if prog not in layout_by_program:
                    layout_by_program[prog] = []
                layout_by_program[prog].append(layout)
            
            for program, prog_layouts in layout_by_program.items():
                summary_parts.append(f"  {program}:")
                for layout in prog_layouts:
                    purpose = layout.get('data_flow_role', 'Data structure')
                    field_dist = layout.get('field_distribution', {})
                    summary_parts.append(f"    • {layout['layout_name']}: {purpose}")
                    summary_parts.append(f"      Fields: {field_dist.get('input_fields', 0)} input, {field_dist.get('output_fields', 0)} output, {field_dist.get('derived_fields', 0)} calculated")
                    
                    if layout.get('connected_files'):
                        file_info = []
                        for cf in layout['connected_files']:
                            file_info.append(f"{cf['operations'][0] if cf['operations'] else 'access'} {cf['file_name']}")
                        summary_parts.append(f"      Files: {', '.join(file_info)}")
                    
                    if layout.get('business_narrative'):
                        summary_parts.append(f"      Purpose: {layout['business_narrative']}")
                summary_parts.append("")
        
        # Field flow analysis
        if fields:
            summary_parts.append("KEY FIELD TRANSFORMATIONS:")
            # Group by field name
            field_groups = {}
            for field in fields:
                fname = field['field_name']
                if fname not in field_groups:
                    field_groups[fname] = []
                field_groups[fname].append(field)
            
            for field_name, field_list in list(field_groups.items())[:10]:  # Top 10 fields
                summary_parts.append(f"  {field_name}:")
                for field_flow in field_list:
                    transformation = field_flow.get('data_transformation', {})
                    impact = field_flow.get('business_impact', 'Standard processing')
                    summary_parts.append(f"    {field_flow['source_program']} → {field_flow['target_program']}: {impact}")
                    if transformation.get('business_rules'):
                        summary_parts.append(f"      Business Rule: {transformation['business_rules'][0]}")
            summary_parts.append("")
        
        # File operations context
        if files:
            summary_parts.append("FILE OPERATIONS CONTEXT:")
            for file_op in files:
                ops_str = ', '.join(file_op.get('operations', []))
                summary_parts.append(f"  {file_op['program_name']}: {ops_str} {file_op['file_name']} ({file_op.get('io_direction', 'unknown')} operation)")
            summary_parts.append("")
        
        # Missing programs impact
        if missing:
            summary_parts.append("MISSING COMPONENTS IMPACT:")
            for miss in missing:
                summary_parts.append(f"  • {miss['target_program']} (called by {miss['source_program']})")
                summary_parts.append(f"    Impact: Flow analysis incomplete - upload to see full data transformations")
            summary_parts.append("")
        
        # Summary statistics
        summary_parts.append("ANALYSIS METRICS:")
        summary_parts.append(f"  Programs Analyzed: {len(chain) - len(missing)}/{len(chain)}")
        summary_parts.append(f"  Data Structures: {len(layouts)}")
        summary_parts.append(f"  Field Transformations: {len(fields)}")
        summary_parts.append(f"  File Operations: {len(files)}")
        
        if len(chain) > len(missing):
            completeness = ((len(chain) - len(missing)) / len(chain)) * 100
            summary_parts.append(f"  Analysis Completeness: {completeness:.1f}%")
        
        return '\n'.join(summary_parts)
        
    def _group_dynamic_calls_by_variable(self, program_calls: List[Dict]) -> List[Dict]:
        """Group dynamic calls by variable name to handle group variables properly"""
        grouped_calls = []
        dynamic_groups = {}
        regular_calls = []
        
        for call in program_calls:
            if call.get('relationship_type') == 'DYNAMIC_PROGRAM_CALL':
                analysis_details = call.get('analysis_details', {})
                variable_name = analysis_details.get('variable_name', 'UNKNOWN')
                is_group_var = analysis_details.get('is_group_variable', False)
                
                if is_group_var and variable_name != 'UNKNOWN':
                    # This is a group variable call - group by variable name
                    if variable_name not in dynamic_groups:
                        dynamic_groups[variable_name] = {
                            'is_group_variable': True,
                            'variable_name': variable_name,
                            'calls': [],
                            'resolved_programs': [],
                            'all_possible_programs': analysis_details.get('all_resolved_programs', [])
                        }
                    
                    dynamic_groups[variable_name]['calls'].append(call)
                    dynamic_groups[variable_name]['resolved_programs'].append({
                        'program_name': call['target_component'],
                        'is_missing': call.get('dependency_status') == 'missing',
                        'confidence': call.get('confidence_score', 0.5),
                        'resolution_method': analysis_details.get('resolution_method', 'unknown'),
                        'analysis_details': analysis_details
                    })
                else:
                    # Regular dynamic call (not a group variable)
                    regular_calls.append(call)
            else:
                # Static call
                regular_calls.append(call)
        
        # Add group variable entries
        for group_info in dynamic_groups.values():
            grouped_calls.append(group_info)
        
        # Add regular calls as individual groups
        if regular_calls:
            grouped_calls.append({
                'is_group_variable': False,
                'calls': regular_calls
            })
        
        return grouped_calls

    def _process_group_variable_calls(self, call_group: Dict, chain: List[Dict], 
                                    current_program: str, level: int, session_id: str):
        """Process group variable calls - create separate chain steps for each resolved program"""
        variable_name = call_group['variable_name']
        resolved_programs = call_group['resolved_programs']
        all_possible = call_group.get('all_possible_programs', [])
        
        logger.info(f"Processing group variable {variable_name} with {len(resolved_programs)} resolved programs")
        
        # Create a chain step for each resolved program
        for i, resolved_prog in enumerate(resolved_programs):
            target_program = resolved_prog['program_name']
            is_missing = resolved_prog.get('is_missing', False)
            analysis_details = resolved_prog.get('analysis_details', {})
            
            # Create enhanced chain step for group variable resolution
            chain_step = {
                'sequence': level + 1,
                'source_program': current_program,
                'target_program': target_program,
                'call_type': 'dynamic_group',
                'call_mechanism': 'DYNAMIC_PROGRAM_CALL',
                'variable_name': variable_name,
                'resolution_method': resolved_prog.get('resolution_method', 'group_construction'),
                'confidence': resolved_prog.get('confidence', 0.5),
                'line_number': analysis_details.get('line_number', 0),
                'is_missing': is_missing,
                'dependency_status': 'missing' if is_missing else 'present',
                'business_context': f"Dynamic call via group variable {variable_name} -> {target_program}",
                'data_passed': [],
                'data_received': [],
                'layout_associations': [],
                
                # FIXED: Add group variable context for proper visualization
                'group_variable_info': {
                    'variable_name': variable_name,
                    'is_group_resolution': True,
                    'resolution_index': i + 1,
                    'total_resolutions': len(resolved_programs),
                    'all_possible_programs': all_possible,
                    'construction_method': analysis_details.get('construction_details', {}),
                    'group_structure_used': analysis_details.get('group_structure_used', False)
                }
            }
            
            # FIXED: Only analyze data context for non-missing programs
            if not is_missing:
                try:
                    data_context = self._analyze_call_data_context_fixed(session_id, current_program, target_program)
                    chain_step['data_passed'] = data_context.get('passed_fields', [])
                    chain_step['data_received'] = data_context.get('received_fields', [])
                    chain_step['layout_associations'] = data_context.get('layout_associations', [])
                except Exception as data_error:
                    logger.error(f"Error analyzing data context for group variable call {current_program} -> {target_program}: {str(data_error)}")
            
            chain.append(chain_step)
            logger.info(f"Added group variable resolution to chain: {current_program} -> {target_program} via {variable_name} (missing: {is_missing})")

    def _create_chain_step(self, current_program: str, target_program: str, call: Dict, 
                        analysis_details: Dict, sequence: int, is_missing: bool, session_id: str) -> Dict:
        """Create a chain step for regular (non-group variable) calls"""
        chain_step = {
            'sequence': sequence,
            'source_program': current_program,
            'target_program': target_program,
            'call_type': analysis_details.get('call_type', 'CALL'),
            'call_mechanism': call['relationship_type'],
            'variable_name': analysis_details.get('variable_name', ''),
            'resolution_method': analysis_details.get('resolution_method', ''),
            'confidence': call.get('confidence_score', 0.8),
            'line_number': analysis_details.get('line_number', 0),
            'is_missing': is_missing,
            'dependency_status': call.get('dependency_status', 'unknown'),
            'business_context': analysis_details.get('business_context', ''),
            'data_passed': [],
            'data_received': [],
            'layout_associations': []
        }
        
        # Only analyze data context for non-missing programs
        if not is_missing:
            try:
                data_context = self._analyze_call_data_context_fixed(session_id, current_program, target_program)
                chain_step['data_passed'] = data_context.get('passed_fields', [])
                chain_step['data_received'] = data_context.get('received_fields', [])
                chain_step['layout_associations'] = data_context.get('layout_associations', [])
            except Exception as data_error:
                logger.error(f"Error analyzing data context for regular call {current_program} -> {target_program}: {str(data_error)}")
        
        return chain_step
    
    def _analyze_call_data_context_fixed(self, session_id: str, source_program: str, target_program: str) -> Dict:
        """FIXED: Analyze data context with proper error handling"""
        try:
            # Get source program content with error handling
            source_data = self.db_manager.get_component_source_code(session_id, source_program)
            if not source_data.get('success'):
                logger.warning(f"Could not get source code for {source_program}")
                return {'passed_fields': [], 'received_fields': [], 'layout_associations': []}
            
            # FIXED: Proper component access with bounds checking
            components = source_data.get('components', [])
            if not components:
                logger.warning(f"No components found for {source_program}")
                return {'passed_fields': [], 'received_fields': [], 'layout_associations': []}
            
            source_content = components[0].get('source_for_chat', '')
            if not source_content:
                logger.warning(f"No source content found for {source_program}")
                return {'passed_fields': [], 'received_fields': [], 'layout_associations': []}
            
            # Get target program content if available
            target_data = self.db_manager.get_component_source_code(session_id, target_program)
            target_content = ''
            if target_data.get('success') and target_data.get('components'):
                target_content = target_data['components'][0].get('source_for_chat', '')
            
            data_context = {
                'passed_fields': [],
                'received_fields': [],
                'call_context': '',
                'linkage_section_fields': [],
                'layout_associations': []
            }
            
            # Analyze source program for fields passed in the call
            passed_fields = self._extract_call_parameters_safe(source_content, target_program)
            data_context['passed_fields'] = passed_fields
            
            # Analyze target program for received fields (LINKAGE SECTION)
            if target_content:
                linkage_fields = self._extract_linkage_section_fields_safe(target_content)
                data_context['linkage_section_fields'] = linkage_fields
                data_context['received_fields'] = linkage_fields
            
            # Extract call context (surrounding lines)
            call_context = self._extract_call_context_safe(source_content, target_program)
            data_context['call_context'] = call_context
            
            # FIXED: Get layout associations for this call
            layout_associations = self._get_call_layout_associations(session_id, source_program, target_program)
            data_context['layout_associations'] = layout_associations
            
            # Correlate passed and received fields
            if passed_fields and data_context['received_fields']:
                data_context['field_correlations'] = self._correlate_passed_received_fields(
                    passed_fields, data_context['received_fields']
                )
            
            return data_context
            
        except Exception as e:
            logger.error(f"Error analyzing call data context for {source_program} -> {target_program}: {str(e)}")
            return {'passed_fields': [], 'received_fields': [], 'layout_associations': []}
    
    def _extract_call_parameters_safe(self, source_content: str, target_program: str) -> List[Dict]:
        """FIXED: Safe extraction of call parameters"""
        passed_fields = []
        
        try:
            if not source_content:
                return passed_fields
                
            lines = source_content.split('\n')
            
            for i, line in enumerate(lines):
                line_upper = line.upper().strip()
                
                # Look for calls to target program
                if target_program.upper() in line_upper:
                    # Check for CALL, LINK, or XCTL statements
                    if any(keyword in line_upper for keyword in ['CALL', 'LINK', 'XCTL']):
                        
                        # Look for USING clause in this line or next few lines
                        using_lines = []
                        for j in range(i, min(i+3, len(lines))):
                            if j < len(lines):  # Bounds check
                                using_lines.append(lines[j])
                        
                        using_text = ' '.join(using_lines).upper()
                        
                        if 'USING' in using_text:
                            # Extract field names from USING clause
                            using_match = re.search(r'USING\s+(.*?)(?:\.|$)', using_text)
                            if using_match:
                                using_clause = using_match.group(1)
                                
                                # Split by common separators and extract field names
                                field_candidates = re.findall(r'\b[A-Z][A-Z0-9\-]{2,30}\b', using_clause)
                                
                                for field_name in field_candidates:
                                    if field_name not in ['USING', 'BY', 'REFERENCE', 'CONTENT', 'VALUE']:
                                        field_info = {
                                            'field_name': field_name,
                                            'line_number': i + 1,
                                            'context': line.strip(),
                                            'parameter_type': 'BY_REFERENCE'  # Default assumption
                                        }
                                        
                                        # Detect parameter passing method
                                        if 'BY CONTENT' in using_text:
                                            field_info['parameter_type'] = 'BY_CONTENT'
                                        elif 'BY VALUE' in using_text:
                                            field_info['parameter_type'] = 'BY_VALUE'
                                        
                                        passed_fields.append(field_info)
        
        except Exception as e:
            logger.error(f"Error extracting call parameters: {str(e)}")
        
        return passed_fields
    
    def _extract_linkage_section_fields_safe(self, target_content: str) -> List[Dict]:
        """FIXED: Safe extraction of linkage section fields"""
        linkage_fields = []
        
        try:
            if not target_content:
                return linkage_fields
                
            lines = target_content.split('\n')
            in_linkage_section = False
            
            for i, line in enumerate(lines):
                line_upper = line.upper().strip()
                
                # Detect start of LINKAGE SECTION
                if 'LINKAGE SECTION' in line_upper:
                    in_linkage_section = True
                    continue
                
                # Detect end of LINKAGE SECTION
                if in_linkage_section and ('PROCEDURE DIVISION' in line_upper or 
                                        (line_upper.endswith('SECTION') and 'LINKAGE' not in line_upper)):
                    in_linkage_section = False
                    break
                
                # Extract field definitions in LINKAGE SECTION
                if in_linkage_section and line_upper:
                    # Look for level numbers (01, 02, 05, etc.)
                    level_match = re.match(r'^\s*(\d{2})\s+([A-Z][A-Z0-9\-]+)', line_upper)
                    if level_match:
                        level_number = level_match.group(1)
                        field_name = level_match.group(2)
                        
                        # Extract PIC clause if present
                        pic_match = re.search(r'PIC\s+([X9A\(\)V\.S\-\+]+)', line_upper)
                        picture_clause = pic_match.group(1) if pic_match else ''
                        
                        field_info = {
                            'field_name': field_name,
                            'level_number': level_number,
                            'picture_clause': picture_clause,
                            'line_number': i + 1,
                            'definition': line.strip()
                        }
                        
                        linkage_fields.append(field_info)
        
        except Exception as e:
            logger.error(f"Error extracting linkage section fields: {str(e)}")
        
        return linkage_fields
    
    def _extract_call_context_safe(self, source_content: str, target_program: str) -> str:
        """FIXED: Safe extraction of call context"""
        try:
            if not source_content:
                return "No source content available"
                
            lines = source_content.split('\n')
            
            for i, line in enumerate(lines):
                if target_program.upper() in line.upper():
                    # Get context: 2 lines before and after the call
                    start_idx = max(0, i - 2)
                    end_idx = min(len(lines), i + 3)
                    
                    context_lines = []
                    for j in range(start_idx, end_idx):
                        if j < len(lines):  # Bounds check
                            prefix = ">>> " if j == i else "    "
                            context_lines.append(f"{prefix}{j+1:4d}: {lines[j].strip()}")
                    
                    return '\n'.join(context_lines)
            
            return f"Call to {target_program} found but context extraction failed"
            
        except Exception as e:
            logger.error(f"Error extracting call context: {str(e)}")
            return "Context extraction failed"
    
    def _get_call_layout_associations(self, session_id: str, source_program: str, target_program: str) -> List[Dict]:
        """Get layout associations for a program call"""
        layout_associations = []
        
        try:
            # Get record layouts for both programs
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get layouts from source program
                cursor.execute('''
                    SELECT layout_name, friendly_name, business_purpose, fields_count
                    FROM record_layouts 
                    WHERE session_id = ? AND program_name = ?
                ''', (session_id, source_program))
                
                source_layouts = [dict(row) for row in cursor.fetchall()]
                
                # Get layouts from target program  
                cursor.execute('''
                    SELECT layout_name, friendly_name, business_purpose, fields_count
                    FROM record_layouts 
                    WHERE session_id = ? AND program_name = ?
                ''', (session_id, target_program))
                
                target_layouts = [dict(row) for row in cursor.fetchall()]
                
                # Create associations
                for source_layout in source_layouts:
                    for target_layout in target_layouts:
                        # Simple name-based association (can be enhanced)
                        if self._layouts_are_related(source_layout['layout_name'], target_layout['layout_name']):
                            layout_associations.append({
                                'source_layout': source_layout['layout_name'],
                                'target_layout': target_layout['layout_name'],
                                'source_friendly_name': source_layout.get('friendly_name', ''),
                                'target_friendly_name': target_layout.get('friendly_name', ''),
                                'association_type': 'PARAMETER_PASSING',
                                'confidence': 0.7
                            })
        
        except Exception as e:
            logger.error(f"Error getting call layout associations: {str(e)}")
        
        return layout_associations
    
    def _layouts_are_related(self, layout1: str, layout2: str) -> bool:
        """Check if two layouts are related (simple heuristic)"""
        # Remove common suffixes and prefixes for comparison
        clean1 = layout1.replace('-REC', '').replace('-RECORD', '').replace('WS-', '')
        clean2 = layout2.replace('-REC', '').replace('-RECORD', '').replace('WS-', '')
        
        # Check for similar base names
        return clean1[:6] == clean2[:6] if len(clean1) >= 6 and len(clean2) >= 6 else False
    
    def _identify_file_operations_with_layouts(self, session_id: str, program_chain: List[Dict]) -> List[Dict]:
        """FIXED: Identify file operations with proper layout associations"""
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
                    'has_layout_resolution': analysis_details.get('layout_resolved', False),
                    'layout_details': []
                }
                
                # FIXED: Get detailed layout information if available
                if file_op['has_layout_resolution'] and file_op['associated_layouts']:
                    layout_details = self._get_detailed_layout_info(session_id, file_op['associated_layouts'])
                    file_op['layout_details'] = layout_details
                    file_op['fields_involved'] = self._extract_fields_from_layouts(layout_details)
                
                file_operations.append(file_op)
        
        return file_operations
    
    def _get_detailed_layout_info(self, session_id: str, layout_names: List[str]) -> List[Dict]:
        """Get detailed information about layouts"""
        layout_details = []
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                for layout_name in layout_names:
                    cursor.execute('''
                        SELECT layout_name, friendly_name, business_purpose, fields_count,
                               record_classification, record_usage_description
                        FROM record_layouts 
                        WHERE session_id = ? AND layout_name = ?
                    ''', (session_id, layout_name))
                    
                    result = cursor.fetchone()
                    if result:
                        layout_info = dict(result)
                        
                        # Get field details for this layout
                        cursor.execute('''
                            SELECT field_name, friendly_name, usage_type, business_purpose
                            FROM field_analysis_details fad
                            JOIN record_layouts rl ON fad.field_id = rl.id
                            WHERE fad.session_id = ? AND rl.layout_name = ?
                            LIMIT 10  -- Limit for performance
                        ''', (session_id, layout_name))
                        
                        fields = [dict(row) for row in cursor.fetchall()]
                        layout_info['key_fields'] = fields
                        layout_details.append(layout_info)
        
        except Exception as e:
            logger.error(f"Error getting detailed layout info: {str(e)}")
        
        return layout_details
    
    def _extract_fields_from_layouts(self, layout_details: List[Dict]) -> List[str]:
        """Extract field names from layout details"""
        fields = []
        for layout in layout_details:
            key_fields = layout.get('key_fields', [])
            fields.extend([field['field_name'] for field in key_fields])
        return fields
    
    def _trace_field_data_flow_fixed(self, session_id: str, program_chain: List[Dict]) -> List[Dict]:
        """FIXED: Trace field data flow with better error handling"""
        field_flows = []
        
        for chain_step in program_chain:
            if chain_step.get('is_missing'):
                continue
                
            try:
                # Analyze field passing in this call
                field_analysis = self._analyze_field_passing_in_call_safe(chain_step, session_id)
                if field_analysis:
                    field_flows.extend(field_analysis)
            except Exception as e:
                logger.error(f"Error analyzing field passing for {chain_step['source_program']} -> {chain_step['target_program']}: {str(e)}")
                continue
        
        return field_flows
    
    def _analyze_field_passing_in_call_safe(self, chain_step: Dict, session_id: str) -> List[Dict]:
        """FIXED: Safe analysis of field passing in calls"""
        field_flows = []
        
        try:
            # Use the data already analyzed in the chain step
            passed_fields = chain_step.get('data_passed', [])
            received_fields = chain_step.get('data_received', [])
            
            for field_info in passed_fields:
                field_name = field_info.get('field_name', '')
                if not field_name:
                    continue
                
                # Get additional field analysis from database
                field_db_info = self._get_field_info_from_db(session_id, field_name, chain_step['source_program'])
                
                field_flow = {
                    'field_name': field_name,
                    'source_program': chain_step['source_program'],
                    'target_program': chain_step['target_program'],
                    'flow_type': 'PASSED_TO_PROGRAM',
                    'call_mechanism': chain_step['call_mechanism'],
                    'field_type': field_db_info.get('usage_type', 'UNKNOWN'),
                    'data_source': field_db_info.get('source_field', ''),
                    'business_purpose': field_db_info.get('business_purpose', ''),
                    'transformation_logic': f"Passed via {chain_step['call_type']} call",
                    'sequence': chain_step['sequence'],
                    'confidence': 0.9,
                    'parameter_type': field_info.get('parameter_type', 'BY_REFERENCE'),
                    'line_number': field_info.get('line_number', 0)
                }
                field_flows.append(field_flow)
        
        except Exception as e:
            logger.error(f"Error in safe field passing analysis: {str(e)}")
        
        return field_flows
    
    # Keep all the existing helper methods but add error handling
    def _get_field_info_from_db(self, session_id: str, field_name: str, program_name: str) -> Dict:
        """Get field information from database with error handling"""
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
    
    def _correlate_passed_received_fields(self, passed_fields: List[Dict], received_fields: List[Dict]) -> List[Dict]:
        """Correlate passed fields with received fields based on position and naming"""
        correlations = []
        
        try:
            # Simple position-based correlation (most common in COBOL)
            for i, passed_field in enumerate(passed_fields):
                correlation = {
                    'passed_field': passed_field['field_name'],
                    'passed_context': passed_field.get('context', ''),
                    'parameter_position': i + 1,
                    'received_field': None,
                    'correlation_confidence': 0.0
                }
                
                # Try to match by position in linkage section
                if i < len(received_fields):
                    received_field = received_fields[i]
                    correlation['received_field'] = received_field['field_name']
                    correlation['received_definition'] = received_field.get('definition', '')
                    correlation['correlation_confidence'] = 0.9  # High confidence for position-based matching
                
                # Try to match by name similarity if no position match
                elif not correlation['received_field']:
                    passed_name = passed_field['field_name']
                    for received_field in received_fields:
                        received_name = received_field['field_name']
                        
                        # Simple name matching
                        if passed_name == received_name:
                            correlation['received_field'] = received_name
                            correlation['correlation_confidence'] = 1.0
                            break
                        elif passed_name.replace('-', '') == received_name.replace('-', ''):
                            correlation['received_field'] = received_name
                            correlation['correlation_confidence'] = 0.8
                            break
                
                correlations.append(correlation)
        
        except Exception as e:
            logger.error(f"Error correlating fields: {str(e)}")
        
        return correlations
    
    def _resolve_dynamic_call_programs(self, session_id: str, source_program: str, variable_name: str) -> List[Dict]:
        """Resolve dynamic call variable to actual program names from database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get all dynamic call dependencies for this variable
                cursor.execute('''
                    SELECT target_component, analysis_details_json, dependency_status
                    FROM dependency_relationships
                    WHERE session_id = ? AND source_component = ? 
                    AND relationship_type = 'DYNAMIC_PROGRAM_CALL'
                    AND analysis_details_json LIKE ?
                ''', (session_id, source_program, f'%{variable_name}%'))
                
                resolved_programs = []
                for row in cursor.fetchall():
                    target_component = row[0]
                    dependency_status = row[2]
                    
                    try:
                        analysis_details = json.loads(row[1]) if row[1] else {}
                    except:
                        analysis_details = {}
                    
                    # Skip generic variable names, only get actual program names
                    if (target_component and 
                        not target_component.startswith('DYNAMIC_CALL_') and 
                        not target_component.startswith('UNRESOLVED_') and
                        target_component != variable_name):
                        
                        resolved_programs.append({
                            'program_name': target_component,
                            'status': dependency_status or 'unknown',
                            'variable_source': variable_name,
                            'resolution_method': analysis_details.get('resolution_method', 'database_lookup'),
                            'confidence': analysis_details.get('confidence', 0.8)
                        })
                
                return resolved_programs
                
        except Exception as e:
            logger.error(f"Error resolving dynamic call programs for {variable_name}: {str(e)}")
            return []

    def _get_program_calls_from_analysis(self, session_id: str, program_name: str) -> List[Dict]:
        # Add caching
        cache_key = f"{session_id}_{program_name}_calls"
        if hasattr(self, '_calls_cache') and cache_key in self._calls_cache:
            return self._calls_cache[cache_key]
        
        if not hasattr(self, '_calls_cache'):
            self._calls_cache = {}
        
        # Single query instead of multiple
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT analysis_result_json FROM component_analysis 
                WHERE session_id = ? AND component_name = ? AND component_type = 'PROGRAM'
            ''', (session_id, program_name))
            
            result = cursor.fetchone()
            if result and result[0]:
                analysis_data = json.loads(result[0])
                program_calls = analysis_data.get('program_calls', [])
                self._calls_cache[cache_key] = program_calls
                return program_calls
        
        return []
    
    def _is_program_missing(self, session_id: str, program_name: str) -> bool:
        """Check if a program is missing from the session"""
        try:
            components = self.db_manager.get_session_components(session_id)
            uploaded_programs = set(comp['component_name'].upper() for comp in components)
            return program_name.upper() not in uploaded_programs
        except Exception as e:
            logger.error(f"Error checking if program is missing: {str(e)}")
            return True  # Assume missing on error
        
        # Keep existing methods for business summary, storage, etc.
    def _generate_business_flow_summary(self, session_id: str, flow_analysis: Dict) -> str:
        """Generate business summary with proper group variable handling"""
        try:
            summary_parts = []
            
            starting_prog = flow_analysis['starting_program']
            summary_parts.append(f"Program Flow Analysis starting from {starting_prog}:")
            summary_parts.append("")
            
            # Program chain summary with group variable details
            chain = flow_analysis.get('program_chain', [])
            if chain:
                summary_parts.append("Program Call Chain:")
                
                # Group by source program to show group variables clearly
                by_source = {}
                for step in chain:
                    source = step['source_program']
                    if source not in by_source:
                        by_source[source] = []
                    by_source[source].append(step)
                
                for source_prog, steps in by_source.items():
                    # Check if this source has group variable calls
                    group_vars = {}
                    regular_calls = []
                    
                    for step in steps:
                        if step.get('group_variable_info', {}).get('is_group_resolution'):
                            var_name = step['variable_name']
                            if var_name not in group_vars:
                                group_vars[var_name] = []
                            group_vars[var_name].append(step)
                        else:
                            regular_calls.append(step)
                    
                    # Display group variable calls
                    for var_name, var_steps in group_vars.items():
                        summary_parts.append(f"  {source_prog} via group variable {var_name}:")
                        all_possible = var_steps[0].get('group_variable_info', {}).get('all_possible_programs', [])
                        
                        for step in var_steps:
                            status_indicator = "[MISSING]" if step.get('is_missing') else "[AVAILABLE]"
                            summary_parts.append(f"    → {step['target_program']} {status_indicator}")
                        
                        # Show additional possibilities if any
                        resolved_programs = [s['target_program'] for s in var_steps]
                        additional = [p for p in all_possible if p not in resolved_programs]
                        if additional:
                            summary_parts.append(f"    Additional possibilities: {', '.join(additional)}")
                    
                    # Display regular calls
                    for step in regular_calls:
                        status_indicator = "[MISSING]" if step.get('is_missing') else "[AVAILABLE]"
                        call_type = step.get('call_type', 'CALL')
                        summary_parts.append(f"  {step['source_program']} → {step['target_program']} ({call_type}) {status_indicator}")
                
                summary_parts.append("")
            
            # Rest of summary remains the same...
            return '\n'.join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating business flow summary: {str(e)}")
            return "Business flow summary generation failed"
    
    def _map_call_mechanism(self, mechanism):
        """Map internal call mechanism to database values"""
        mapping = {
            'PROGRAM_CALL': 'STATIC_CALL',
            'DYNAMIC_PROGRAM_CALL': 'DYNAMIC_CALL',
            'CICS_LINK': 'CICS_LINK',
            'CICS_XCTL': 'CICS_XCTL'
        }
        return mapping.get(mechanism, 'STATIC_CALL') 


    def _get_layout_associations_for_flow(self, session_id: str, program_chain: List[Dict]) -> List[Dict]:
        """Get detailed layout associations for the program flow"""
        layout_associations = []
        
        try:
            for step in program_chain:
                if step.get('is_missing'):
                    continue
                    
                program_name = step['source_program']
                
                # Get layouts used by this program
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Get record layouts for this program
                    cursor.execute('''
                        SELECT rl.layout_name, rl.friendly_name, rl.business_purpose,
                            rl.fields_count, rl.record_classification, rl.record_usage_description,
                            COUNT(fad.id) as actual_field_count
                        FROM record_layouts rl
                        LEFT JOIN field_analysis_details fad ON rl.id = fad.field_id
                        WHERE rl.session_id = ? AND rl.program_name = ?
                        GROUP BY rl.id, rl.layout_name, rl.friendly_name, rl.business_purpose,
                                rl.fields_count, rl.record_classification, rl.record_usage_description
                    ''', (session_id, program_name))
                    
                    program_layouts = cursor.fetchall()
                    
                    for layout_row in program_layouts:
                        layout_info = {
                            'layout_name': layout_row[0],
                            'friendly_name': layout_row[1] or layout_row[0].replace('-', ' ').title(),
                            'business_purpose': layout_row[2] or f'Data structure for {program_name}',
                            'fields_count': layout_row[6] or layout_row[3] or 0,  # Use actual count if available
                            'record_classification': layout_row[4] or 'RECORD',
                            'usage_description': layout_row[5] or 'Program data structure',
                            'program_name': program_name,
                            'sequence_in_flow': step['sequence'],
                            'layout_type': layout_row[4] or 'RECORD',
                            'has_fields': (layout_row[6] or 0) > 0,
                            'usage_context': self._determine_layout_usage_context(session_id, layout_row[0], program_name),
                            'connected_programs': self._get_programs_using_layout(session_id, layout_row[0])
                        }
                        
                        # Get key fields for this layout
                        layout_info['key_fields'] = self._get_layout_key_fields(session_id, layout_row[0])
                        
                        layout_associations.append(layout_info)
            
            logger.info(f"Found {len(layout_associations)} layout associations in flow")
            return layout_associations
            
        except Exception as e:
            logger.error(f"Error getting layout associations: {str(e)}")
            return []

    def _determine_layout_usage_context(self, session_id: str, layout_name: str, program_name: str) -> str:
        """Determine how a layout is used in the context of the program"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check field usage types for this layout in this program
                cursor.execute('''
                    SELECT fad.usage_type, COUNT(*) as count
                    FROM field_analysis_details fad
                    JOIN record_layouts rl ON fad.field_id = rl.id
                    WHERE fad.session_id = ? AND rl.layout_name = ? AND fad.program_name = ?
                    GROUP BY fad.usage_type
                    ORDER BY count DESC
                ''', (session_id, layout_name, program_name))
                
                usage_types = cursor.fetchall()
                
                if not usage_types:
                    return "Data structure definition"
                
                # Determine primary usage
                primary_usage = usage_types[0][0]
                usage_contexts = {
                    'INPUT': 'Input data processing and validation',
                    'OUTPUT': 'Output data formatting and generation',
                    'INPUT_OUTPUT': 'Bidirectional data processing',
                    'DERIVED': 'Calculated field generation',
                    'STATIC': 'Constant data and configuration',
                    'REFERENCE': 'Data lookup and reference'
                }
                
                context = usage_contexts.get(primary_usage, 'General data processing')
                
                # Add additional context if multiple usage types
                if len(usage_types) > 1:
                    other_types = [ut[0] for ut in usage_types[1:3]]  # Top 2 additional types
                    context += f" (also used for {', '.join(other_types).lower()})"
                
                return context
                
        except Exception as e:
            logger.error(f"Error determining layout usage context: {str(e)}")
            return "Data structure processing"

    def _get_programs_using_layout(self, session_id: str, layout_name: str) -> List[str]:
        """Get list of programs that use a specific layout"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT rl.program_name
                    FROM record_layouts rl
                    WHERE rl.session_id = ? AND rl.layout_name = ?
                    AND rl.program_name IS NOT NULL
                ''', (session_id, layout_name))
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting programs using layout: {str(e)}")
            return []

    def _get_layout_key_fields(self, session_id: str, layout_name: str, limit: int = 5) -> List[Dict]:
        """Get key fields for a layout with their usage information"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT fad.field_name, fad.friendly_name, fad.usage_type, 
                        fad.business_purpose, fad.mainframe_data_type
                    FROM field_analysis_details fad
                    JOIN record_layouts rl ON fad.field_id = rl.id
                    WHERE fad.session_id = ? AND rl.layout_name = ?
                    ORDER BY fad.total_program_references DESC, fad.field_name
                    LIMIT ?
                ''', (session_id, layout_name, limit))
                
                key_fields = []
                for row in cursor.fetchall():
                    field_info = {
                        'field_name': row[0],
                        'friendly_name': row[1] or row[0].replace('-', ' ').title(),
                        'usage_type': row[2] or 'STATIC',
                        'business_purpose': row[3] or f'Field {row[0]} processing',
                        'data_type': row[4] or 'UNKNOWN'
                    }
                    key_fields.append(field_info)
                
                return key_fields
                
        except Exception as e:
            logger.error(f"Error getting layout key fields: {str(e)}")
            return []

    def _enhance_field_flows_with_layout_info(self, session_id: str, field_flows: List[Dict]) -> List[Dict]:
        """Enhance field flows with layout information"""
        enhanced_flows = []
        
        for field_flow in field_flows:
            enhanced_flow = field_flow.copy()
            
            # Get source and target layout information
            field_name = field_flow['field_name']
            source_program = field_flow['source_program']
            target_program = field_flow['target_program']
            
            # Find layouts containing this field in source program
            source_layouts = self._find_layouts_containing_field(session_id, field_name, source_program)
            enhanced_flow['source_layouts'] = source_layouts
            
            # Find layouts containing this field in target program
            target_layouts = self._find_layouts_containing_field(session_id, field_name, target_program)
            enhanced_flow['target_layouts'] = target_layouts
            
            # Determine data transformation details
            transformation_details = self._analyze_field_transformation(
                session_id, field_name, source_program, target_program
            )
            enhanced_flow['transformation_details'] = transformation_details
            
            enhanced_flows.append(enhanced_flow)
        
        return enhanced_flows

    def _find_layouts_containing_field(self, session_id: str, field_name: str, program_name: str) -> List[Dict]:
        """Find layouts that contain a specific field in a program"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT rl.layout_name, rl.friendly_name, rl.business_purpose
                    FROM record_layouts rl
                    JOIN field_analysis_details fad ON rl.id = fad.field_id
                    WHERE fad.session_id = ? AND fad.field_name = ? AND rl.program_name = ?
                ''', (session_id, field_name, program_name))
                
                layouts = []
                for row in cursor.fetchall():
                    layout_info = {
                        'layout_name': row[0],
                        'friendly_name': row[1] or row[0].replace('-', ' ').title(),
                        'business_purpose': row[2] or 'Data structure'
                    }
                    layouts.append(layout_info)
                
                return layouts
                
        except Exception as e:
            logger.error(f"Error finding layouts containing field: {str(e)}")
            return []

    def _analyze_field_transformation(self, session_id: str, field_name: str, 
                                    source_program: str, target_program: str) -> Dict:
        """Analyze how a field is transformed between programs"""
        try:
            transformation_details = {
                'transformation_type': 'DIRECT_PASS',
                'data_type_changes': [],
                'validation_rules': [],
                'business_logic_applied': []
            }
            
            # Get field details from both programs
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Source field details
                cursor.execute('''
                    SELECT usage_type, mainframe_data_type, business_purpose, 
                        field_references_json
                    FROM field_analysis_details
                    WHERE session_id = ? AND field_name = ? AND program_name = ?
                ''', (session_id, field_name, source_program))
                
                source_field = cursor.fetchone()
                
                # Target field details  
                cursor.execute('''
                    SELECT usage_type, mainframe_data_type, business_purpose,
                        field_references_json
                    FROM field_analysis_details
                    WHERE session_id = ? AND field_name = ? AND program_name = ?
                ''', (session_id, field_name, target_program))
                
                target_field = cursor.fetchone()
                
                if source_field and target_field:
                    # Analyze data type changes
                    if source_field[1] != target_field[1]:
                        transformation_details['data_type_changes'].append({
                            'from_type': source_field[1] or 'UNKNOWN',
                            'to_type': target_field[1] or 'UNKNOWN',
                            'change_type': 'DATA_TYPE_CONVERSION'
                        })
                        transformation_details['transformation_type'] = 'DATA_TYPE_CONVERSION'
                    
                    # Analyze usage type changes
                    if source_field[0] != target_field[0]:
                        transformation_details['transformation_type'] = 'USAGE_TYPE_CHANGE'
                        transformation_details['business_logic_applied'].append({
                            'logic_type': 'USAGE_TRANSFORMATION',
                            'description': f"Field usage changed from {source_field[0]} to {target_field[0]}"
                        })
                    
                    # Extract business logic from field references
                    source_refs = self._parse_field_references(source_field[3])
                    target_refs = self._parse_field_references(target_field[3])
                    
                    # Look for validation or calculation patterns
                    for ref in source_refs + target_refs:
                        ref_line = ref.get('line_content', '').upper()
                        if any(keyword in ref_line for keyword in ['IF', 'WHEN', 'EVALUATE']):
                            transformation_details['validation_rules'].append({
                                'rule_type': 'CONDITIONAL_LOGIC',
                                'description': ref.get('line_content', '').strip()[:100]
                            })
                        elif any(keyword in ref_line for keyword in ['COMPUTE', 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE']):
                            transformation_details['business_logic_applied'].append({
                                'logic_type': 'CALCULATION',
                                'description': ref.get('line_content', '').strip()[:100]
                            })
            
            return transformation_details
            
        except Exception as e:
            logger.error(f"Error analyzing field transformation: {str(e)}")
            return {'transformation_type': 'UNKNOWN', 'data_type_changes': [], 'validation_rules': [], 'business_logic_applied': []}

    def _parse_field_references(self, references_json: str) -> List[Dict]:
        """Parse field references JSON safely"""
        try:
            if references_json:
                return json.loads(references_json)
            return []
        except:
            return []

    # Update the main analyze_complete_program_flow method to include layout associations
    def analyze_complete_program_flow(self, session_id: str, starting_program: str) -> Dict:
        """Optimized complete program flow analysis"""
        logger.info(f"Starting optimized flow analysis from {starting_program}")
        
        # Batch load all data once
        session_data = self._batch_load_session_data(session_id)
        
        flow_analysis = {
            'flow_id': str(uuid.uuid4()),
            'starting_program': starting_program,
            'program_chain': [],
            'field_flows': [],
            'file_operations': [],
            'layout_associations': [],
            'missing_programs': [],
            'business_flow_summary': ''
        }
        
        try:
            # Build program chain using cached data
            program_chain = self._build_program_call_chain_cached(starting_program, session_data)
            flow_analysis['program_chain'] = program_chain
            
            # Get layout associations using cached data
            layout_associations = self._get_layout_associations_cached(session_data, program_chain)
            flow_analysis['layout_associations'] = layout_associations
            
            # Trace field flows using cached data
            field_flows = self._trace_field_data_flow_cached(session_data, program_chain)
            flow_analysis['field_flows'] = field_flows
            
            # Get file operations using cached data
            file_operations = self._identify_file_operations_cached(session_data, program_chain)
            flow_analysis['file_operations'] = file_operations
            
            # Identify missing programs
            missing_programs = [step for step in program_chain if step.get('is_missing')]
            flow_analysis['missing_programs'] = missing_programs
            
            # Generate simple business summary
            flow_analysis['business_flow_summary'] = self._generate_simple_summary(flow_analysis)
            
            # Store results
            self._store_program_flow_analysis(session_id, flow_analysis)
            
            logger.info(f"Flow analysis completed: {len(program_chain)} programs, {len(layout_associations)} layouts")
            return flow_analysis
            
        except Exception as e:
            logger.error(f"Error in program flow analysis: {str(e)}")
            return flow_analysis

    def _batch_load_session_data(self, session_id: str) -> Dict:
        """Load all session data in one batch to avoid repeated queries"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Load all program analyses
            cursor.execute('''
                SELECT component_name, analysis_result_json 
                FROM component_analysis 
                WHERE session_id = ? AND component_type = 'PROGRAM'
            ''', (session_id,))
            program_analyses = {}
            for row in cursor.fetchall():
                try:
                    program_analyses[row[0]] = json.loads(row[1]) if row[1] else {}
                except:
                    program_analyses[row[0]] = {}
            
            # Load all layouts with their fields
            cursor.execute('''
                SELECT rl.layout_name, rl.program_name, rl.friendly_name, rl.business_purpose,
                    fad.field_name, fad.friendly_name as field_friendly, fad.usage_type,
                    fad.business_purpose as field_purpose, fad.mainframe_data_type
                FROM record_layouts rl
                LEFT JOIN field_analysis_details fad ON rl.id = fad.field_id
                WHERE rl.session_id = ?
                ORDER BY rl.layout_name, fad.field_name
            ''', (session_id,))
            
            layouts_data = {}
            for row in cursor.fetchall():
                layout_name = row[0]
                if layout_name not in layouts_data:
                    layouts_data[layout_name] = {
                        'layout_name': layout_name,
                        'program_name': row[1],
                        'friendly_name': row[2] or layout_name,
                        'business_purpose': row[3] or 'Data structure',
                        'fields': []
                    }
                
                if row[4]:  # field_name exists
                    layouts_data[layout_name]['fields'].append({
                        'field_name': row[4],
                        'friendly_name': row[5] or row[4],
                        'usage_type': row[6] or 'STATIC',
                        'business_purpose': row[7] or 'Field processing',
                        'data_type': row[8] or 'UNKNOWN'
                    })
            
            # Load uploaded programs
            uploaded_programs = set(program_analyses.keys())
            
            return {
                'program_analyses': program_analyses,
                'layouts_data': layouts_data,
                'uploaded_programs': uploaded_programs
            }
    def _generate_enhanced_business_flow_summary(self, session_id: str, flow_analysis: Dict) -> str:
        """Generate enhanced business summary with layout and data transformation details"""
        try:
            summary_parts = []
            
            starting_prog = flow_analysis['starting_program']
            summary_parts.append(f"Enhanced Program Flow Analysis for {starting_prog}")
            summary_parts.append("=" * 60)
            summary_parts.append("")
            
            # Program chain summary
            chain = flow_analysis.get('program_chain', [])
            if chain:
                summary_parts.append("📞 PROGRAM CALL CHAIN:")
                for step in chain:
                    status = "[MISSING]" if step.get('is_missing') else "[AVAILABLE]"
                    call_type = step.get('call_type', 'CALL')
                    
                    if step.get('variable_name'):
                        summary_parts.append(f"  {step['source_program']} --({call_type} via {step['variable_name']})--> {step['target_program']} {status}")
                    else:
                        summary_parts.append(f"  {step['source_program']} --({call_type})--> {step['target_program']} {status}")
                summary_parts.append("")
            
            # Layout associations summary
            layouts = flow_analysis.get('layout_associations', [])
            if layouts:
                summary_parts.append("📋 RECORD LAYOUTS & DATA STRUCTURES:")
                
                # Group by program
                by_program = {}
                for layout in layouts:
                    prog = layout['program_name']
                    if prog not in by_program:
                        by_program[prog] = []
                    by_program[prog].append(layout)
                
                for program, prog_layouts in by_program.items():
                    summary_parts.append(f"  {program}:")
                    for layout in prog_layouts:
                        field_count = layout.get('fields_count', 0)
                        usage = layout.get('usage_context', 'Data processing')
                        summary_parts.append(f"    - {layout['layout_name']} ({field_count} fields) - {usage}")
                    summary_parts.append("")
            
            # Field flows with transformation details
            field_flows = flow_analysis.get('field_flows', [])
            if field_flows:
                summary_parts.append("🔄 FIELD DATA FLOWS:")
                for field_flow in field_flows[:10]:  # Limit to top 10
                    field_name = field_flow['field_name']
                    source = field_flow['source_program']
                    target = field_flow['target_program']
                    transform_type = field_flow.get('transformation_details', {}).get('transformation_type', 'DIRECT_PASS')
                    
                    summary_parts.append(f"  {field_name}: {source} --> {target} ({transform_type})")
                    
                    # Add transformation details if available
                    transform_details = field_flow.get('transformation_details', {})
                    if transform_details.get('business_logic_applied'):
                        logic = transform_details['business_logic_applied'][0]
                        summary_parts.append(f"    Business Logic: {logic.get('description', 'Applied')[:50]}...")
                
                if len(field_flows) > 10:
                    summary_parts.append(f"  ... and {len(field_flows) - 10} more field flows")
                summary_parts.append("")
            
            # File operations summary
            file_ops = flow_analysis.get('file_operations', [])
            if file_ops:
                summary_parts.append("📁 FILE OPERATIONS:")
                for file_op in file_ops:
                    file_name = file_op['file_name']
                    program = file_op['program_name']
                    operations = ', '.join(file_op.get('operations', []))
                    layout_status = "with layouts" if file_op.get('has_layout_resolution') else "no layouts"
                    
                    summary_parts.append(f"  {program} --> {file_name} ({operations}) [{layout_status}]")
                summary_parts.append("")
            
            # Missing programs impact
            missing = flow_analysis.get('missing_programs', [])
            if missing:
                summary_parts.append("⚠️ MISSING PROGRAMS IMPACT:")
                for missing_prog in missing:
                    prog_name = missing_prog['program_name']
                    called_by = missing_prog['called_by']
                    impact = "blocks flow analysis" if missing_prog.get('blocks_further_analysis') else "limits analysis"
                    summary_parts.append(f"  {prog_name} (called by {called_by}) - {impact}")
                summary_parts.append("")
            
            # Summary statistics
            summary_parts.append("📊 ANALYSIS STATISTICS:")
            summary_parts.append(f"  Programs in chain: {len(chain)}")
            summary_parts.append(f"  Record layouts identified: {len(layouts)}")
            summary_parts.append(f"  Field data flows traced: {len(field_flows)}")
            summary_parts.append(f"  File operations found: {len(file_ops)}")
            summary_parts.append(f"  Missing programs: {len(missing)}")
            
            # Data quality assessment
            complete_programs = len([s for s in chain if not s.get('is_missing')])
            if complete_programs > 0:
                completeness_pct = (complete_programs / len(chain)) * 100
                summary_parts.append(f"  Analysis completeness: {completeness_pct:.1f}%")
            
            if missing:
                summary_parts.append("")
                summary_parts.append("🎯 NEXT STEPS:")
                summary_parts.append("  1. Upload missing programs to complete flow analysis")
                summary_parts.append("  2. Verify layout associations are correct")
                summary_parts.append("  3. Validate field transformation logic")
                summary_parts.append("  4. Document business rules discovered")
            
            return '\n'.join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating enhanced business flow summary: {str(e)}")
            return f"Enhanced business flow summary generation failed: {str(e)}"


    def _store_program_flow_analysis(self, session_id: str, flow_analysis: Dict):
        """Store program flow analysis in database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                flow_id = flow_analysis['flow_id']
                
                # Store program flow traces
                for step in flow_analysis.get('program_chain', []):
                    cursor.execute('''
                        INSERT OR REPLACE INTO program_flow_traces 
                        (session_id, flow_id, source_program, target_program, call_sequence,
                         call_mechanism, variable_name, resolved_via, data_flow_json,
                         business_context, line_number, confidence_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id, flow_id, step['source_program'], step['target_program'],
                        step['sequence'], self._map_call_mechanism(step.get('call_mechanism', 'PROGRAM_CALL')),
                        step.get('variable_name', ''),  ''),
                        step.get('resolution_method', ''), 
                        json.dumps({
                            'passed_fields': step.get('data_passed', []), 
                            'received_fields': step.get('data_received', []),
                            'layout_associations': step.get('layout_associations', []),
                            'is_missing': step.get('is_missing', False)
                        }),
                        step.get('business_context', f"Call from {step['source_program']} to {step['target_program']}"), 
                        step.get('line_number', 0), step.get('confidence', 0.8)
                    )
                
                # Store field data flows
                for field_flow in flow_analysis.get('field_flows', []):
                    cursor.execute('''
                        INSERT OR REPLACE INTO field_data_flow
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
    
    def _identify_missing_programs_in_flow(self, session_id: str, program_chain: List[Dict]) -> List[Dict]:
        """Identify missing programs and their impact on the flow"""
        missing_programs = []
        
        try:
            for step in program_chain:
                if step.get('is_missing'):
                    missing_info = {
                        'program_name': step['target_program'],
                        'called_by': step['source_program'],
                        'call_type': step.get('call_type', 'UNKNOWN'),
                        'sequence_position': step.get('sequence', 0),
                        'impact_description': f"Flow blocked: {step['source_program']} cannot complete call to {step['target_program']}",
                        'recommended_action': f"Upload {step['target_program']} to continue flow analysis",
                        'confidence': step.get('confidence', 0.8),
                        'variable_name': step.get('variable_name', ''),
                        'call_mechanism': step.get('call_mechanism', 'PROGRAM_CALL')
                    }
                    
                    # Check if this blocks further analysis
                    blocked_programs = [s['source_program'] for s in program_chain 
                                      if s['source_program'] == step['target_program']]
                    
                    if blocked_programs:
                        missing_info['blocks_further_analysis'] = True
                        missing_info['impact_description'] += f" and blocks analysis of calls from {step['target_program']}"
                    else:
                        missing_info['blocks_further_analysis'] = False
                    
                    missing_programs.append(missing_info)
        
        except Exception as e:
            logger.error(f"Error identifying missing programs: {str(e)}")
        
        return missing_programs