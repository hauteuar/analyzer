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
    
    def _build_program_call_chain_fixed(self, session_id: str, starting_program: str) -> List[Dict]:
        """Build the complete program call chain with proper status checking - FIXED VERSION"""
        chain = []
        processed_programs = set()
        
        def build_chain_recursive(current_program: str, level: int = 0):
            if current_program in processed_programs or level > 10:
                return
            
            processed_programs.add(current_program)
            
            # Get FRESH dependencies for current program
            dependencies = self.db_manager.get_enhanced_dependencies(session_id)
            
            # Filter for program calls from current program
            program_calls = [d for d in dependencies 
                           if d['source_component'].upper() == current_program.upper() 
                           and d['relationship_type'] in ['PROGRAM_CALL', 'DYNAMIC_PROGRAM_CALL']]
            
            logger.info(f"Found {len(program_calls)} program calls from {current_program}")
            
            for call in program_calls:
                target_program = call['target_component']
                analysis_details = call.get('analysis_details', {})
                
                # FIXED: Properly determine missing status from dependency_status field
                dependency_status = call.get('dependency_status', 'unknown')
                is_missing = dependency_status == 'missing'
                
                chain_step = {
                    'sequence': level + 1,
                    'source_program': current_program,
                    'target_program': target_program,
                    'call_type': analysis_details.get('call_type', 'CALL'),
                    'call_mechanism': call['relationship_type'],
                    'variable_name': analysis_details.get('variable_name', ''),
                    'resolution_method': analysis_details.get('resolution_method', ''),
                    'confidence': call.get('confidence_score', 0.8),
                    'line_number': analysis_details.get('line_number', 0),
                    'is_missing': is_missing,  # FIXED: Use proper status
                    'dependency_status': dependency_status,  # FIXED: Add explicit status
                    'business_context': analysis_details.get('business_context', ''),
                    'data_passed': [],
                    'data_received': [],
                    'layout_associations': []
                }
                
                # FIXED: Only analyze data context for non-missing programs
                if not is_missing:
                    try:
                        data_context = self._analyze_call_data_context_fixed(session_id, current_program, target_program)
                        chain_step['data_passed'] = data_context.get('passed_fields', [])
                        chain_step['data_received'] = data_context.get('received_fields', [])
                        chain_step['layout_associations'] = data_context.get('layout_associations', [])
                    except Exception as data_error:
                        logger.error(f"Error analyzing data context for {current_program} -> {target_program}: {str(data_error)}")
                        # Continue with empty data context
                        pass
                
                chain.append(chain_step)
                logger.info(f"Added to chain: {current_program} -> {target_program} (missing: {is_missing})")
                
                # Continue chain if target program is available
                if not is_missing:
                    build_chain_recursive(target_program, level + 1)
        
        build_chain_recursive(starting_program)
        return chain
    
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
    
    # Keep existing methods for business summary, storage, etc.
    def _generate_business_flow_summary(self, session_id: str, flow_analysis: Dict) -> str:
        """Generate business summary of the program flow"""
        try:
            summary_parts = []
            
            starting_prog = flow_analysis['starting_program']
            summary_parts.append(f"Program Flow Analysis starting from {starting_prog}:")
            summary_parts.append("")
            
            # Program chain summary with status
            chain = flow_analysis.get('program_chain', [])
            if chain:
                summary_parts.append("Program Call Chain:")
                for step in chain:
                    status_indicator = "[MISSING]" if step.get('is_missing') else "[AVAILABLE]"
                    call_type = step.get('call_type', 'CALL')
                    summary_parts.append(f"  {step['source_program']} → {step['target_program']} ({call_type}) {status_indicator}")
                    
                    # Add layout information if available
                    if step.get('layout_associations'):
                        for layout in step['layout_associations']:
                            summary_parts.append(f"    └─ Layout: {layout['source_layout']} → {layout['target_layout']}")
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
            
            # File operations summary with layout details
            file_ops = flow_analysis.get('file_operations', [])
            if file_ops:
                summary_parts.append("File Operations in Flow:")
                for file_op in file_ops:
                    io_dir = file_op.get('io_direction', 'UNKNOWN')
                    interface = file_op.get('interface_type', 'FILE')
                    layout_info = ""
                    if file_op.get('has_layout_resolution'):
                        layouts = file_op.get('associated_layouts', [])
                        layout_info = f" (Layouts: {', '.join(layouts)})"
                    summary_parts.append(f"  {file_op['program_name']} {io_dir} {file_op['file_name']} [{interface}]{layout_info}")
                summary_parts.append("")
            
            # Missing programs impact
            missing = flow_analysis.get('missing_programs', [])
            if missing:
                summary_parts.append("Missing Programs Blocking Flow:")
                for missing_prog in missing:
                    summary_parts.append(f"  {missing_prog['program_name']} - {missing_prog['impact_description']}")
                summary_parts.append("")
            
            # Flow statistics
            available_programs = len([s for s in chain if not s.get('is_missing')])
            missing_programs = len([s for s in chain if s.get('is_missing')])
            total_fields = len(field_flows)
            total_files = len(file_ops)
            
            summary_parts.append("Flow Statistics:")
            summary_parts.append(f"  Available Programs: {available_programs}")
            summary_parts.append(f"  Missing Programs: {missing_programs}")
            summary_parts.append(f"  Data Fields Traced: {total_fields}")
            summary_parts.append(f"  File Operations: {total_files}")
            
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
                        INSERT OR REPLACE INTO program_flow_traces 
                        (session_id, flow_id, source_program, target_program, call_sequence,
                         call_mechanism, variable_name, resolved_via, data_flow_json,
                         business_context, line_number, confidence_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id, flow_id, step['source_program'], step['target_program'],
                        step['sequence'], step['call_mechanism'], step.get('variable_name', ''),
                        step.get('resolution_method', ''), 
                        json.dumps({
                            'passed_fields': step.get('data_passed', []), 
                            'received_fields': step.get('data_received', []),
                            'layout_associations': step.get('layout_associations', []),
                            'is_missing': step.get('is_missing', False)
                        }),
                        step.get('business_context', f"Call from {step['source_program']} to {step['target_program']}"), 
                        step.get('line_number', 0), step.get('confidence', 0.8)
                    ))
                
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