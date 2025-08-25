"""
Field Analyzer Module
Handles field mapping analysis and field matrix operations
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from modules.cobol_parser import COBOLParser

logger = logging.getLogger(__name__)

@dataclass
class FieldMapping:
    field_name: str
    friendly_name: str
    mainframe_data_type: str
    oracle_data_type: str
    mainframe_length: int
    oracle_length: int
    population_source: str
    business_logic_type: str
    business_logic_description: str
    programs_involved: List[str]
    confidence_score: float
    derivation_logic: str = ""
    source_record_layout: str = ""

class FieldAnalyzer:
    def __init__(self, llm_client, token_manager, db_manager):
        self.llm_client = llm_client
        self.token_manager = token_manager
        self.db_manager = db_manager
        self.cobol_parser = COBOLParser()
        
        # Business logic patterns
        self.business_logic_patterns = {
            'MOVE': [
                r'MOVE\s+([A-Z0-9\-\(\)]+)\s+TO\s+([A-Z0-9\-\(\)]+)',
                r'MOVE\s+(SPACES|ZEROS|HIGH-VALUES|LOW-VALUES)\s+TO\s+([A-Z0-9\-\(\)]+)'
            ],
            'DERIVED': [
                r'COMPUTE\s+([A-Z0-9\-]+)\s*=\s*(.+)',
                r'ADD\s+([A-Z0-9\-]+)\s+TO\s+([A-Z0-9\-]+)',
                r'SUBTRACT\s+([A-Z0-9\-]+)\s+FROM\s+([A-Z0-9\-]+)',
                r'MULTIPLY\s+([A-Z0-9\-]+)\s+BY\s+([A-Z0-9\-]+)',
                r'DIVIDE\s+([A-Z0-9\-]+)\s+INTO\s+([A-Z0-9\-]+)'
            ],
            'CONDITIONAL': [
                r'IF\s+.*\s+(MOVE|COMPUTE)\s+.*\s+TO\s+([A-Z0-9\-]+)',
                r'EVALUATE\s+.*\s+WHEN\s+.*\s+(MOVE|COMPUTE)',
                r'IF\s+.*\s+THEN\s+.*\s+(MOVE|ADD|SUBTRACT)\s+.*\s+TO\s+([A-Z0-9\-]+)'
            ],
            'STRING_MANIPULATION': [
                r'STRING\s+.*\s+DELIMITED\s+.*\s+INTO\s+([A-Z0-9\-]+)',
                r'UNSTRING\s+([A-Z0-9\-]+)\s+DELIMITED\s+.*\s+INTO\s+([A-Z0-9\-]+)',
                r'INSPECT\s+([A-Z0-9\-]+)\s+REPLACING'
            ],
            'CALCULATED': [
                r'ADD\s+([A-Z0-9\-]+)\s+([A-Z0-9\-]+)\s+GIVING\s+([A-Z0-9\-]+)',
                r'SUBTRACT\s+([A-Z0-9\-]+)\s+FROM\s+([A-Z0-9\-]+)\s+GIVING\s+([A-Z0-9\-]+)',
                r'MULTIPLY\s+([A-Z0-9\-]+)\s+BY\s+([A-Z0-9\-]+)\s+GIVING\s+([A-Z0-9\-]+)',
                r'DIVIDE\s+([A-Z0-9\-]+)\s+BY\s+([A-Z0-9\-]+)\s+GIVING\s+([A-Z0-9\-]+)'
            ]
        }
    
    def analyze_field_mapping(self, session_id: str, target_file: str) -> List[FieldMapping]:
        """Analyze field mappings for target file"""
        logger.info(f"Starting field mapping analysis for {target_file}")
        
        try:
            # Get all components for this session
            components = self.db_manager.get_session_components(session_id)
            
            # Find programs that interact with the target file
            relevant_programs = self._find_programs_for_file(session_id, target_file, components)
            
            if not relevant_programs:
                logger.warning(f"No programs found that interact with {target_file}")
                return []
            
            # Analyze each relevant program
            field_mappings = {}
            
            for program in relevant_programs:
                program_mappings = self._analyze_program_field_mapping(
                    session_id, program, target_file
                )
                
                # Merge mappings
                for mapping in program_mappings:
                    field_key = mapping.field_name
                    if field_key in field_mappings:
                        # Consolidate multiple sources
                        existing = field_mappings[field_key]
                        existing.programs_involved.extend(mapping.programs_involved)
                        existing.programs_involved = list(set(existing.programs_involved))  # Remove duplicates
                        
                        # Handle conflicting business logic
                        if existing.business_logic_type != mapping.business_logic_type:
                            existing.business_logic_type = 'MULTI_SOURCE'
                            existing.business_logic_description += f" | {mapping.business_logic_description}"
                        
                        # Average confidence scores
                        existing.confidence_score = (existing.confidence_score + mapping.confidence_score) / 2
                    else:
                        field_mappings[field_key] = mapping
            
            # Store results in database
            mappings_list = list(field_mappings.values())
            mappings_dict = [self._mapping_to_dict(m) for m in mappings_list]
            self.db_manager.store_field_mappings(session_id, target_file, mappings_dict)
            
            logger.info(f"Completed field mapping analysis for {target_file}: {len(mappings_list)} fields")
            return mappings_list
            
        except Exception as e:
            logger.error(f"Error in field mapping analysis: {str(e)}")
            return []
    
    def _find_programs_for_file(self, session_id: str, target_file: str, components: List[Dict]) -> List[Dict]:
        """Find programs that interact with the target file"""
        relevant_programs = []
        
        for component in components:
            if component['component_type'] == 'PROGRAM':
                # Check if program interacts with target file
                analysis_result = json.loads(component.get('analysis_result_json', '{}'))
                
                # Check file operations (traditional COBOL FD files)
                file_ops = analysis_result.get('file_operations', [])
                for op in file_ops:
                    if target_file.upper() in op.get('file_name', '').upper():
                        relevant_programs.append(component)
                        break
                
                # Check CICS operations (CICS files without FD)
                if not any(prog['component_name'] == component['component_name'] for prog in relevant_programs):
                    cics_ops = analysis_result.get('cics_operations', [])
                    for op in cics_ops:
                        if ('file_name' in op and target_file.upper() in op.get('file_name', '').upper()):
                            relevant_programs.append(component)
                            break
                
                # Check if file name appears in the program content
                if not any(prog['component_name'] == component['component_name'] for prog in relevant_programs):
                    # Get full component analysis to search content
                    if target_file.upper() in str(analysis_result).upper():
                        relevant_programs.append(component)
        
        return relevant_programs
    
    def _create_field_mapping(self, field_detail: Dict, program: Dict, 
                            target_file: str, layout: Dict) -> Optional[FieldMapping]:
        """Create field mapping from field detail"""
        try:
            field_name = field_detail['field_name']
            
            # Determine business logic type from usage
            business_logic_type = self._determine_business_logic_type(field_detail)
            
            # Convert COBOL type to Oracle - enhanced for CICS scenarios
            cobol_type = self._extract_cobol_type_from_detail(field_detail)
            oracle_type, oracle_length = self.cobol_parser.convert_pic_to_oracle_type(
                cobol_type.get('picture', ''), cobol_type.get('usage', '')
            )
            
            # Determine population source - handle CICS files
            population_source = self._determine_population_source(field_detail, program, target_file)
            
            mapping = FieldMapping(
                field_name=field_name,
                friendly_name=self.cobol_parser.generate_friendly_name(field_name, 'Field'),
                mainframe_data_type=f"PIC {cobol_type.get('picture', 'X(10)')} {cobol_type.get('usage', '')}".strip(),
                oracle_data_type=oracle_type,
                mainframe_length=cobol_type.get('length', 10),
                oracle_length=oracle_length,
                population_source=population_source,
                business_logic_type=business_logic_type,
                business_logic_description=self._generate_business_logic_description(field_detail, business_logic_type),
                programs_involved=[program['component_name']],
                confidence_score=0.85,
                source_record_layout=layout['layout_name']
            )
            
            return mapping
            
        except Exception as e:
            logger.error(f"Error creating field mapping: {str(e)}")
            return None
    
    def _determine_population_source(self, field_detail: Dict, program: Dict, target_file: str) -> str:
        """Determine how field is populated - handle enhanced CICS I/O scenarios"""
        usage_type = field_detail.get('usage_type', 'UNKNOWN')
        operation_type = field_detail.get('operation_type', '')
        source_field = field_detail.get('source_field', '')
        program_name = program['component_name']
        
        # Check if this is a CICS-based program and analyze operations
        analysis_result = json.loads(program.get('analysis_result_json', '{}'))
        cics_ops = analysis_result.get('cics_operations', [])
        
        # Analyze CICS operations for this target file
        file_cics_ops = [op for op in cics_ops if op.get('file_name', '').upper() == target_file.upper()]
        
        # Categorize CICS operations
        has_cics_read = any('READ' in op.get('operation', '').upper() for op in file_cics_ops)
        has_cics_write = any('WRITE' in op.get('operation', '').upper() for op in file_cics_ops)
        has_cics_rewrite = any('REWRITE' in op.get('operation', '').upper() for op in file_cics_ops)
        has_cics_delete = any('DELETE' in op.get('operation', '').upper() for op in file_cics_ops)
        
        if usage_type == 'INPUT':
            if has_cics_read:
                return f"CICS READ operation from {target_file}"
            elif has_cics_rewrite:
                return f"CICS REWRITE operation (read phase) from {target_file}"
            elif source_field:
                return f"Input from {source_field}"
            else:
                return f"External input via {program_name}"
                
        elif usage_type == 'OUTPUT':
            if has_cics_write:
                return f"CICS WRITE operation to {target_file}"
            elif has_cics_rewrite:
                return f"CICS REWRITE operation (write phase) to {target_file}"
            else:
                return f"Output from {program_name} processing"
                
        elif usage_type == 'INPUT_OUTPUT':
            if has_cics_rewrite:
                return f"CICS REWRITE operation (read-modify-write) on {target_file}"
            elif has_cics_delete:
                return f"CICS DELETE operation (read-then-delete) on {target_file}"
            elif has_cics_read and has_cics_write:
                return f"CICS READ and WRITE operations on {target_file}"
            else:
                return f"Bidirectional processing in {program_name}"
                
        elif usage_type == 'DERIVED':
            if source_field:
                return f"Computed from {source_field}"
            else:
                return f"Derived through business logic in {program_name}"
                
        elif usage_type == 'STATIC':
            return f"Static value or constant"
        else:
            # Determine from CICS operations if usage_type is unclear
            if has_cics_rewrite or has_cics_delete:
                return f"CICS {('REWRITE' if has_cics_rewrite else 'DELETE')} operation on {target_file}"
            elif has_cics_read:
                return f"CICS READ operation from {target_file}"
            elif has_cics_write:
                return f"CICS WRITE operation to {target_file}"
            else:
                return f"Processed by {program_name}"
    
    def _determine_business_logic_type(self, field_detail: Dict) -> str:
        """Enhanced business logic type determination with CICS I/O awareness"""
        usage_type = field_detail.get('usage_type', 'UNKNOWN')
        operation_type = field_detail.get('operation_type', '')
        code_snippet = field_detail.get('code_snippet', '')
        
        # Check for CICS operations in code snippet
        cics_rewrite_pattern = r'EXEC\s+CICS\s+REWRITE'
        cics_read_pattern = r'EXEC\s+CICS\s+READ'
        cics_write_pattern = r'EXEC\s+CICS\s+WRITE'
        
        if re.search(cics_rewrite_pattern, code_snippet, re.IGNORECASE):
            return 'CICS_REWRITE'  # New business logic type for REWRITE
        elif re.search(cics_read_pattern, code_snippet, re.IGNORECASE):
            return 'CICS_INPUT'
        elif re.search(cics_write_pattern, code_snippet, re.IGNORECASE):
            return 'CICS_OUTPUT'
        
        # Check traditional code snippet patterns
        for logic_type, patterns in self.business_logic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code_snippet, re.IGNORECASE):
                    return logic_type
        
        # Enhanced fallback based on usage type
        usage_mapping = {
            'INPUT': 'MOVE',
            'OUTPUT': 'MOVE', 
            'INPUT_OUTPUT': 'CONDITIONAL',  # I/O operations often involve conditions
            'DERIVED': 'DERIVED',
            'REFERENCE': 'MOVE',
            'STATIC': 'MOVE',
            'UNUSED': 'UNUSED'
        }
        
        return usage_mapping.get(usage_type, 'MOVE')
    
    def _generate_business_logic_description(self, field_detail: Dict, logic_type: str) -> str:
        """Generate enhanced business logic description including CICS operations"""
        descriptions = {
            'MOVE': f"Field populated by moving data from {field_detail.get('source_field', 'source')}",
            'DERIVED': f"Field derived through computation: {field_detail.get('code_snippet', '')}",
            'CONDITIONAL': f"Field populated conditionally based on business rules",
            'CALCULATED': f"Field calculated from other fields using mathematical operations",
            'STRING_MANIPULATION': f"Field populated through string manipulation operations",
            'UNUSED': f"Field defined but not populated in the analyzed programs",
            'CICS_INPUT': f"Field populated via CICS READ operation",
            'CICS_OUTPUT': f"Field written via CICS write operation", 
            'CICS_REWRITE': f"Field involved in CICS rewrite operation (read-modify-write pattern)"
        }
        
        base_description = descriptions.get(logic_type, f"Field processed using {logic_type} logic")
        
        # Add CICS context if available
        code_snippet = field_detail.get('code_snippet', '')
        if 'EXEC CICS' in code_snippet.upper():
            if 'REWRITE' in code_snippet.upper():
                base_description += " - Uses CICS REWRITE for atomic read-modify-write operation"
            elif 'READ' in code_snippet.upper():
                base_description += " - Uses CICS READ for data retrieval"
            elif 'WRITE' in code_snippet.upper():
                base_description += " - Uses CICS write for data persistence"
        
        return base_description
    
    def _analyze_program_field_mapping(self, session_id: str, program: Dict, target_file: str) -> List[FieldMapping]:
        """Analyze field mappings within a specific program"""
        mappings = []
        
        try:
            # Get program analysis
            analysis_result = json.loads(program.get('analysis_result_json', '{}'))
            
            # Get record layouts for this program
            record_layouts = self.db_manager.get_record_layouts(session_id, program['component_name'])
            
            for layout in record_layouts:
                # Get field details for this layout
                field_details = self.db_manager.get_field_matrix(session_id, layout['layout_name'])
                
                for field_detail in field_details:
                    mapping = self._create_field_mapping(
                        field_detail, program, target_file, layout
                    )
                    if mapping:
                        mappings.append(mapping)
            
            # If no layouts found, analyze using LLM
            if not mappings:
                mappings = self._llm_analyze_field_mapping(session_id, program, target_file)
            
        except Exception as e:
            logger.error(f"Error analyzing program field mapping: {str(e)}")
        
        return mappings
    
    def _create_field_mapping(self, field_detail: Dict, program: Dict, 
                            target_file: str, layout: Dict) -> Optional[FieldMapping]:
        """Create field mapping from field detail"""
        try:
            field_name = field_detail['field_name']
            
            # Determine business logic type from usage
            business_logic_type = self._determine_business_logic_type(field_detail)
            
            # Convert COBOL type to Oracle
            cobol_type = self._extract_cobol_type_from_detail(field_detail)
            oracle_type, oracle_length = self.cobol_parser.convert_pic_to_oracle_type(
                cobol_type.get('picture', ''), cobol_type.get('usage', '')
            )
            
            mapping = FieldMapping(
                field_name=field_name,
                friendly_name=self.cobol_parser.generate_friendly_name(field_name, 'Field'),
                mainframe_data_type=f"PIC {cobol_type.get('picture', 'X(10)')} {cobol_type.get('usage', '')}".strip(),
                oracle_data_type=oracle_type,
                mainframe_length=cobol_type.get('length', 10),
                oracle_length=oracle_length,
                population_source=field_detail.get('source_field', 'Unknown'),
                business_logic_type=business_logic_type,
                business_logic_description=self._generate_business_logic_description(field_detail, business_logic_type),
                programs_involved=[program['component_name']],
                confidence_score=0.85,
                source_record_layout=layout['layout_name']
            )
            
            return mapping
            
        except Exception as e:
            logger.error(f"Error creating field mapping: {str(e)}")
            return None
    
    def _determine_business_logic_type(self, field_detail: Dict) -> str:
        """Determine business logic type from field usage"""
        usage_type = field_detail.get('usage_type', 'UNKNOWN')
        operation_type = field_detail.get('operation_type', '')
        code_snippet = field_detail.get('code_snippet', '')
        
        # Check code snippet for patterns
        for logic_type, patterns in self.business_logic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code_snippet, re.IGNORECASE):
                    return logic_type
        
        # Fallback based on usage type
        usage_mapping = {
            'INPUT': 'MOVE',
            'OUTPUT': 'MOVE',
            'DERIVED': 'DERIVED',
            'REFERENCE': 'MOVE',
            'STATIC': 'MOVE',
            'UNUSED': 'UNUSED'
        }
        
        return usage_mapping.get(usage_type, 'MOVE')
    
    def _extract_cobol_type_from_detail(self, field_detail: Dict) -> Dict:
        """Extract COBOL data type information from field detail"""
        # This would typically come from the original record layout parsing
        # For now, return default values
        return {
            'picture': 'X(10)',
            'usage': '',
            'length': 10
        }
    
    def _generate_business_logic_description(self, field_detail: Dict, logic_type: str) -> str:
        """Generate description of business logic"""
        descriptions = {
            'MOVE': f"Field populated by moving data from {field_detail.get('source_field', 'source')}",
            'DERIVED': f"Field derived through computation: {field_detail.get('code_snippet', '')}",
            'CONDITIONAL': f"Field populated conditionally based on business rules",
            'CALCULATED': f"Field calculated from other fields using mathematical operations",
            'STRING_MANIPULATION': f"Field populated through string manipulation operations",
            'UNUSED': f"Field defined but not populated in the analyzed programs"
        }
        
        return descriptions.get(logic_type, f"Field processed using {logic_type} logic")
    
    def _llm_analyze_field_mapping(self, session_id: str, program: Dict, target_file: str) -> List[FieldMapping]:
        """Use LLM to analyze field mappings when structured data is insufficient"""
        mappings = []
        
        try:
            # Get program content
            analysis_result = json.loads(program.get('analysis_result_json', '{}'))
            program_content = analysis_result.get('content', '')
            
            if not program_content:
                return mappings
            
            # Create prompt for field mapping analysis
            prompt = f"""
Analyze this COBOL program to identify field mappings for the target file: {target_file}

Focus on:
1. Fields that are written to {target_file}
2. Data type conversion (COBOL PIC to Oracle)
3. Business logic classification (MOVE, DERIVED, CONDITIONAL, CALCULATED, STRING_MANIPULATION, UNUSED)
4. Source of field population

Return JSON with field mappings:
{{
    "field_mappings": [
        {{
            "field_name": "friendly_field_name",
            "mainframe_data_type": "PIC X(10)",
            "oracle_data_type": "VARCHAR2(10)",
            "mainframe_length": 10,
            "oracle_length": 10,
            "population_source": "source description",
            "business_logic_type": "MOVE|DERIVED|CONDITIONAL|CALCULATED|STRING_MANIPULATION|UNUSED",
            "business_logic_description": "detailed description",
            "confidence_score": 0.85,
            "derivation_logic": "logic description"
        }}
    ]
}}

COBOL Program Content:
{program_content}
"""
            
            # Check if content needs chunking
            if self.token_manager.needs_chunking(prompt):
                chunks = self.token_manager.chunk_cobol_code(program_content)
                all_results = []
                
                for chunk in chunks:
                    chunk_prompt = self.token_manager.get_chunked_analysis_prompt(chunk, 'field_mapping')
                    chunk_prompt += f"\n\nTarget File: {target_file}"
                    
                    response = self.llm_client.call_llm(chunk_prompt)
                    
                    # Log LLM call
                    self.db_manager.log_llm_call(
                        session_id, 'field_mapping', chunk.chunk_number, chunk.total_chunks,
                        response.prompt_tokens, response.response_tokens, response.processing_time_ms,
                        response.success, response.error_message
                    )
                    
                    if response.success:
                        parsed_result = self.llm_client.extract_json_from_response(response.content)
                        if parsed_result:
                            all_results.append(parsed_result)
                
                # Consolidate results
                if all_results:
                    consolidated = self.token_manager.consolidate_chunk_results(all_results, 'field_mapping')
                    mappings = self._convert_llm_response_to_mappings(consolidated, program['component_name'])
            else:
                # Single call
                response = self.llm_client.call_llm(prompt)
                
                # Log LLM call
                self.db_manager.log_llm_call(
                    session_id, 'field_mapping', 1, 1,
                    response.prompt_tokens, response.response_tokens, response.processing_time_ms,
                    response.success, response.error_message
                )
                
                if response.success:
                    parsed_result = self.llm_client.extract_json_from_response(response.content)
                    if parsed_result:
                        mappings = self._convert_llm_response_to_mappings(parsed_result, program['component_name'])
                        
        except Exception as e:
            logger.error(f"Error in LLM field mapping analysis: {str(e)}")
        
        return mappings
    
    def _convert_llm_response_to_mappings(self, llm_response: Dict, program_name: str) -> List[FieldMapping]:
        """Convert LLM response to FieldMapping objects"""
        mappings = []
        
        try:
            field_mappings_data = llm_response.get('field_mappings', [])
            
            for mapping_data in field_mappings_data:
                mapping = FieldMapping(
                    field_name=mapping_data.get('field_name', ''),
                    friendly_name=self.cobol_parser.generate_friendly_name(
                        mapping_data.get('field_name', ''), 'Field'
                    ),
                    mainframe_data_type=mapping_data.get('mainframe_data_type', 'PIC X(10)'),
                    oracle_data_type=mapping_data.get('oracle_data_type', 'VARCHAR2(10)'),
                    mainframe_length=mapping_data.get('mainframe_length', 10),
                    oracle_length=mapping_data.get('oracle_length', 10),
                    population_source=mapping_data.get('population_source', ''),
                    business_logic_type=mapping_data.get('business_logic_type', 'MOVE'),
                    business_logic_description=mapping_data.get('business_logic_description', ''),
                    programs_involved=[program_name],
                    confidence_score=mapping_data.get('confidence_score', 0.8),
                    derivation_logic=mapping_data.get('derivation_logic', ''),
                    source_record_layout=mapping_data.get('source_record_layout', '')
                )
                mappings.append(mapping)
                
        except Exception as e:
            logger.error(f"Error converting LLM response to mappings: {str(e)}")
        
        return mappings
    
    def _mapping_to_dict(self, mapping: FieldMapping) -> Dict:
        """Convert FieldMapping to dictionary for database storage"""
        return {
            'field_name': mapping.field_name,
            'friendly_name': mapping.friendly_name,
            'mainframe_data_type': mapping.mainframe_data_type,
            'oracle_data_type': mapping.oracle_data_type,
            'mainframe_length': mapping.mainframe_length,
            'oracle_length': mapping.oracle_length,
            'population_source': mapping.population_source,
            'business_logic_type': mapping.business_logic_type,
            'business_logic_description': mapping.business_logic_description,
            'programs_involved': mapping.programs_involved,
            'confidence_score': mapping.confidence_score,
            'derivation_logic': mapping.derivation_logic,
            'source_record_layout': mapping.source_record_layout
        }
    
    def get_field_matrix(self, session_id: str, record_layout: str = None, program_name: str = None) -> Dict:
        """Get field matrix data with usage analysis"""
        try:
            # Get raw field matrix data
            field_data = self.db_manager.get_field_matrix(session_id, record_layout, program_name)
            
            if program_name and not record_layout:
                # Group by record layouts for program view
                matrix_data = self._group_fields_by_layout(field_data)
            elif record_layout:
                # Show fields for specific record layout
                matrix_data = {
                    'record_layout': record_layout,
                    'fields': self._enhance_field_data(field_data)
                }
            else:
                # Show all fields grouped by program and layout
                matrix_data = self._group_fields_by_program_and_layout(field_data)
            
            return matrix_data
            
        except Exception as e:
            logger.error(f"Error getting field matrix: {str(e)}")
            return {}
    
    def _group_fields_by_layout(self, field_data: List[Dict]) -> Dict:
        """Group fields by record layout for program view"""
        grouped = {}
        
        for field in field_data:
            layout_name = field.get('record_layout', 'Unknown Layout')
            if layout_name not in grouped:
                grouped[layout_name] = {
                    'layout_name': layout_name,
                    'friendly_name': self.cobol_parser.generate_friendly_name(layout_name, 'Record Layout'),
                    'level': field.get('level_number', '01'),
                    'fields': [],
                    'collapsed': True  # Start collapsed for UI
                }
            
            grouped[layout_name]['fields'].append(self._enhance_single_field_data(field))
        
        return {
            'program_layouts': list(grouped.values()),
            'total_layouts': len(grouped),
            'total_fields': len(field_data)
        }
    
    def _enhance_field_data(self, field_data: List[Dict]) -> List[Dict]:
        """Enhance field data with additional analysis"""
        enhanced_fields = []
        
        for field in field_data:
            enhanced_field = self._enhance_single_field_data(field)
            enhanced_fields.append(enhanced_field)
        
        return enhanced_fields
    
    def _enhance_single_field_data(self, field: Dict) -> Dict:
        """Enhance single field data"""
        enhanced = field.copy()
        
        # Add friendly name
        enhanced['friendly_name'] = self.cobol_parser.generate_friendly_name(
            field.get('field_name', ''), 'Field'
        )
        
        # Enhance usage type description
        usage_descriptions = {
            'INPUT': 'Populated from external source',
            'OUTPUT': 'Used as output to other fields',
            'DERIVED': 'Calculated from other fields',
            'REFERENCE': 'Used in conditions/comparisons',
            'STATIC': 'Has fixed value',
            'UNUSED': 'Defined but not used'
        }
        
        usage_type = field.get('usage_type', 'UNKNOWN')
        enhanced['usage_description'] = usage_descriptions.get(usage_type, 'Usage pattern not determined')
        
        # Add business purpose if available
        if not enhanced.get('business_purpose'):
            enhanced['business_purpose'] = self._infer_business_purpose(field.get('field_name', ''))
        
        return enhanced
    
    def _infer_business_purpose(self, field_name: str) -> str:
        """Infer business purpose from field name"""
        if not field_name:
            return "Purpose not determined"
        
        name_upper = field_name.upper()
        
        purpose_keywords = {
            'CUST': 'Customer information',
            'ACCT': 'Account data',
            'ADDR': 'Address information',
            'DATE': 'Date field',
            'TIME': 'Time field',
            'AMT': 'Amount/monetary value',
            'QTY': 'Quantity field',
            'NBR': 'Numeric identifier',
            'NO': 'Number/identifier',
            'CD': 'Code field',
            'DESC': 'Description field',
            'NAME': 'Name field',
            'TRAN': 'Transaction data',
            'BAL': 'Balance field',
            'RATE': 'Rate/percentage',
            'PCT': 'Percentage',
            'FLG': 'Flag/indicator',
            'IND': 'Indicator field',
            'SW': 'Switch field',
            'CTR': 'Counter field',
            'IDX': 'Index field'
        }
        
        for keyword, purpose in purpose_keywords.items():
            if keyword in name_upper:
                return purpose
        
        return "General data field"
    
    def _group_fields_by_program_and_layout(self, field_data: List[Dict]) -> Dict:
        """Group fields by program and layout"""
        grouped = {}
        
        for field in field_data:
            program_name = field.get('program_name', 'Unknown Program')
            layout_name = field.get('record_layout', 'Unknown Layout')
            
            if program_name not in grouped:
                grouped[program_name] = {
                    'program_name': program_name,
                    'friendly_name': self.cobol_parser.generate_friendly_name(program_name, 'Program'),
                    'layouts': {}
                }
            
            if layout_name not in grouped[program_name]['layouts']:
                grouped[program_name]['layouts'][layout_name] = {
                    'layout_name': layout_name,
                    'friendly_name': self.cobol_parser.generate_friendly_name(layout_name, 'Record Layout'),
                    'fields': []
                }
            
            grouped[program_name]['layouts'][layout_name]['fields'].append(
                self._enhance_single_field_data(field)
            )
        
        # Convert to list format
        programs_list = []
        for program_name, program_data in grouped.items():
            program_data['layouts'] = list(program_data['layouts'].values())
            programs_list.append(program_data)
        
        return {
            'programs': programs_list,
            'total_programs': len(programs_list),
            'total_fields': len(field_data)
        }