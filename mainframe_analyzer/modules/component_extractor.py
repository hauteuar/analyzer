"""
Component Extractor Module
Handles extraction and analysis of COBOL components including record layouts
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from modules.cobol_parser import COBOLParser

logger = logging.getLogger(__name__)

class ComponentExtractor:
    def __init__(self, llm_client, token_manager, db_manager):
        self.llm_client = llm_client
        self.token_manager = token_manager
        self.db_manager = db_manager
        self.cobol_parser = COBOLParser()
    
    def extract_components(self, session_id: str, file_content: str, file_name: str, file_type: str) -> List[Dict]:
        """Extract all components from uploaded file"""
        logger.info(f"Extracting components from {file_name}")
        
        try:
            components = []
            
            if file_type.upper() == 'COBOL':
                components = self._extract_cobol_components(session_id, file_content, file_name)
            elif file_type.upper() == 'JCL':
                components = self._extract_jcl_components(session_id, file_content, file_name)
            elif file_type.upper() == 'COPYBOOK':
                components = self._extract_copybook_components(session_id, file_content, file_name)
            else:
                # Generic analysis
                components = self._extract_generic_components(session_id, file_content, file_name, file_type)
            
            logger.info(f"Extracted {len(components)} components from {file_name}")
            return components
            
        except Exception as e:
            logger.error(f"Error extracting components: {str(e)}")
            return []
    
    def _extract_cobol_components(self, session_id: str, content: str, filename: str) -> List[Dict]:
        """Extract components from COBOL program"""
        components = []
        
        try:
            # Use COBOL parser for initial parsing
            parsed_data = self.cobol_parser.parse_cobol_file(content, filename)
            
            # Generate LLM summary for the main program
            program_summary = self._generate_component_summary(session_id, parsed_data, 'PROGRAM')
            
            # Create main program component with LLM summary
            program_component = {
                'name': filename.replace('.cob', '').replace('.CBL', '').replace('.cbl', ''),
                'friendly_name': parsed_data['friendly_name'],
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
                'file_operations': parsed_data['file_operations'],
                'program_calls': parsed_data['program_calls'],
                'copybooks': parsed_data['copybooks'],
                'cics_operations': parsed_data['cics_operations'],
                'mq_operations': parsed_data.get('mq_operations', []),
                'xml_operations': parsed_data.get('xml_operations', []),
                'derived_components': []  # Will store derived component names
            }
            components.append(program_component)
            
            # Extract record layouts as separate components with summaries
            for layout in parsed_data['record_layouts']:
                layout_summary = self._generate_layout_summary(session_id, layout, parsed_data)
                
                layout_component = {
                    'name': layout.name,
                    'friendly_name': layout.friendly_name,
                    'type': 'RECORD_LAYOUT',
                    'parent_program': program_component['name'],
                    'level': layout.level,
                    'line_start': layout.line_start,
                    'line_end': layout.line_end,
                    'source_code': layout.source_code,
                    'fields': [self._field_to_dict(field) for field in layout.fields],
                    'llm_summary': layout_summary,
                    'business_purpose': layout_summary.get('business_purpose', ''),
                    'usage_pattern': layout_summary.get('usage_pattern', 'UNKNOWN')
                }
                components.append(layout_component)
                program_component['derived_components'].append(layout.name)
                
                # Store record layout in database
                self.db_manager.store_record_layout(session_id, {
                    'name': layout.name,
                    'friendly_name': layout.friendly_name,
                    'level': str(layout.level),
                    'line_start': layout.line_start,
                    'line_end': layout.line_end,
                    'source_code': layout.source_code,
                    'fields': [self._field_to_dict(field) for field in layout.fields]
                }, program_component['name'])
            
            # Extract copybook references as components
            for copybook in parsed_data['copybooks']:
                copybook_component = {
                    'name': copybook['copybook_name'],
                    'friendly_name': copybook['friendly_name'],
                    'type': 'COPYBOOK_REFERENCE',
                    'parent_program': program_component['name'],
                    'line_number': copybook['line_number'],
                    'line_content': copybook['line_content'],
                    'business_purpose': f"Copybook include providing shared data structures or procedures"
                }
                components.append(copybook_component)
                program_component['derived_components'].append(copybook['copybook_name'])
            
            # Extract CICS file operations as logical file components
            cics_files = {}
            for cics_op in parsed_data['cics_operations']:
                if 'file_name' in cics_op:
                    file_name = cics_op['file_name']
                    if file_name not in cics_files:
                        cics_files[file_name] = {
                            'name': file_name,
                            'friendly_name': self.cobol_parser.generate_friendly_name(file_name, 'CICS File'),
                            'type': 'CICS_FILE',
                            'parent_program': program_component['name'],
                            'operations': [],
                            'access_pattern': 'UNKNOWN',
                            'io_operations': {
                                'read_ops': [],
                                'write_ops': [],
                                'rewrite_ops': [],
                                'delete_ops': []
                            }
                        }
                    cics_files[file_name]['operations'].append(cics_op)
                    
                    # Categorize operations by type
                    op_type = cics_op.get('operation', '').upper()
                    if 'READ' in op_type:
                        cics_files[file_name]['io_operations']['read_ops'].append(cics_op)
                    elif 'WRITE' in op_type:
                        cics_files[file_name]['io_operations']['write_ops'].append(cics_op)
                    elif 'REWRITE' in op_type:
                        cics_files[file_name]['io_operations']['rewrite_ops'].append(cics_op)
                    elif 'DELETE' in op_type:
                        cics_files[file_name]['io_operations']['delete_ops'].append(cics_op)
            
            # Add CICS files as components with proper I/O classification
            for file_name, file_info in cics_files.items():
                # Determine comprehensive access pattern
                io_ops = file_info['io_operations']
                has_read = len(io_ops['read_ops']) > 0
                has_write = len(io_ops['write_ops']) > 0
                has_rewrite = len(io_ops['rewrite_ops']) > 0
                has_delete = len(io_ops['delete_ops']) > 0
                
                # Classify access pattern based on operations
                if has_rewrite or has_delete:
                    # REWRITE and DELETE are inherently I/O operations
                    file_info['access_pattern'] = 'INPUT_OUTPUT'
                    file_info['io_classification'] = 'BIDIRECTIONAL'
                elif has_read and has_write:
                    file_info['access_pattern'] = 'READ_WRITE'
                    file_info['io_classification'] = 'BIDIRECTIONAL'
                elif has_read:
                    file_info['access_pattern'] = 'READ_ONLY'
                    file_info['io_classification'] = 'INPUT_ONLY'
                elif has_write:
                    file_info['access_pattern'] = 'WRITE_ONLY'
                    file_info['io_classification'] = 'OUTPUT_ONLY'
                
                # Generate business purpose based on I/O pattern
                operation_summary = []
                if has_read:
                    operation_summary.append(f"{len(io_ops['read_ops'])} read operation(s)")
                if has_write:
                    operation_summary.append(f"{len(io_ops['write_ops'])} write operation(s)")
                if has_rewrite:
                    operation_summary.append(f"{len(io_ops['rewrite_ops'])} rewrite operation(s)")
                if has_delete:
                    operation_summary.append(f"{len(io_ops['delete_ops'])} delete operation(s)")
                
                file_info['business_purpose'] = f"CICS managed file with {', '.join(operation_summary)} - {file_info['io_classification'].lower().replace('_', ' ')} access pattern"
                file_info['operation_count'] = len(file_info['operations'])
                
                components.append(file_info)
                program_component['derived_components'].append(file_name)
            
            # If content is large, use LLM for enhanced analysis
            if self.token_manager.needs_chunking(content):
                enhanced_components = self._llm_analyze_large_program(session_id, content, filename)
                components.extend(enhanced_components)
            
            # Extract field relationships and store in database
            self._extract_and_store_field_relationships(session_id, components, filename)
            
        except Exception as e:
            logger.error(f"Error extracting COBOL components: {str(e)}")
            # Fallback to basic parsing
            components = [{
                'name': filename,
                'friendly_name': self.cobol_parser.generate_friendly_name(filename, 'Program'),
                'type': 'PROGRAM',
                'content': content,
                'total_lines': len(content.split('\n')),
                'llm_summary': {'business_purpose': 'Analysis failed - manual review required'},
                'error': str(e)
            }]
        
        return components
    
    def _generate_component_summary(self, session_id: str, parsed_data: Dict, component_type: str) -> Dict:
        """Generate LLM summary for component"""
        try:
            # Build context from parsed data and comments
            context_info = {
                'type': component_type,
                'total_lines': parsed_data.get('total_lines', 0),
                'executable_lines': parsed_data.get('executable_lines', 0),
                'divisions': len(parsed_data.get('divisions', [])),
                'record_layouts': len(parsed_data.get('record_layouts', [])),
                'file_operations': parsed_data.get('file_operations', []),
                'cics_operations': parsed_data.get('cics_operations', []),
                'mq_operations': parsed_data.get('mq_operations', []),
                'xml_operations': parsed_data.get('xml_operations', []),
                'program_calls': parsed_data.get('program_calls', []),
                'copybooks': parsed_data.get('copybooks', []),
                'business_comments': parsed_data.get('business_comments', [])
            }
            
            # Create summary prompt
            prompt = f"""
Analyze this COBOL {component_type} and provide a business summary.

Component Analysis:
- Total Lines: {context_info['total_lines']}
- Executable Lines: {context_info['executable_lines']}
- Divisions: {context_info['divisions']}
- Record Layouts: {context_info['record_layouts']}
- File Operations: {len(context_info['file_operations'])}
- CICS Operations: {len(context_info['cics_operations'])}
- MQ Operations: {len(context_info['mq_operations'])}
- XML Operations: {len(context_info['xml_operations'])}

Business Comments:
{chr(10).join(context_info['business_comments'][:5])}

File Operations:
{chr(10).join([f"- {op.get('operation', 'N/A')}: {op.get('file_name', 'N/A')}" for op in context_info['file_operations'][:5]])}

CICS Operations:
{chr(10).join([f"- {op.get('operation', 'N/A')}: {op.get('file_name', op.get('operation', 'N/A'))}" for op in context_info['cics_operations'][:5]])}

Please provide a JSON response with:
{{
    "business_purpose": "What this component does from a business perspective",
    "primary_function": "Main function category (e.g., BATCH_PROCESSING, ONLINE_TRANSACTION, DATA_CONVERSION, REPORT_GENERATION)",
    "complexity_score": 0.7,
    "key_features": ["feature1", "feature2", "feature3"],
    "integration_points": ["system1", "system2"],
    "data_sources": ["source1", "source2"],
    "business_domain": "FINANCIAL|INSURANCE|RETAIL|MANUFACTURING|GENERAL"
}}
"""
            
            response = self.llm_client.call_llm(prompt, max_tokens=800, temperature=0.3)
            
            # Log LLM call
            self.db_manager.log_llm_call(
                session_id, 'component_summary', 1, 1,
                response.prompt_tokens, response.response_tokens, response.processing_time_ms,
                response.success, response.error_message
            )
            
            if response.success:
                summary = self.llm_client.extract_json_from_response(response.content)
                if summary:
                    return summary
            
            # Fallback summary
            return {
                'business_purpose': f"{component_type} component with {len(context_info['file_operations'])} file operations",
                'primary_function': 'GENERAL_PROCESSING',
                'complexity_score': min(0.9, (context_info['executable_lines'] / 1000) * 0.5 + 0.3),
                'key_features': [f"{len(context_info['file_operations'])} file operations", 
                               f"{len(context_info['cics_operations'])} CICS operations"],
                'integration_points': [],
                'data_sources': [op.get('file_name', 'Unknown') for op in context_info['file_operations'][:3]],
                'business_domain': 'GENERAL'
            }
            
        except Exception as e:
            logger.error(f"Error generating component summary: {str(e)}")
            return {
                'business_purpose': 'Summary generation failed - manual review required',
                'primary_function': 'UNKNOWN',
                'complexity_score': 0.5,
                'key_features': [],
                'integration_points': [],
                'data_sources': [],
                'business_domain': 'GENERAL'
            }
    
    def _generate_layout_summary(self, session_id: str, layout, parsed_data: Dict) -> Dict:
        """Generate summary for record layout"""
        try:
            field_count = len(layout.fields)
            field_types = {}
            
            # Analyze field types
            for field in layout.fields:
                if 'X' in field.picture:
                    field_types['alphanumeric'] = field_types.get('alphanumeric', 0) + 1
                elif '9' in field.picture:
                    field_types['numeric'] = field_types.get('numeric', 0) + 1
                else:
                    field_types['other'] = field_types.get('other', 0) + 1
            
            return {
                'business_purpose': f"Data structure with {field_count} fields containing {', '.join([f'{count} {type}' for type, count in field_types.items()])} fields",
                'usage_pattern': 'DATA_STRUCTURE',
                'field_analysis': field_types,
                'complexity_score': min(0.9, field_count / 50)
            }
            
        except Exception as e:
            logger.error(f"Error generating layout summary: {str(e)}")
            return {
                'business_purpose': 'Data structure - analysis failed',
                'usage_pattern': 'UNKNOWN',
                'field_analysis': {},
                'complexity_score': 0.5
            }
            
    
    def _extract_jcl_components(self, session_id: str, content: str, filename: str) -> List[Dict]:
        """Extract components from JCL"""
        components = []
        
        try:
            lines = content.split('\n')
            
            # Main JCL component
            jcl_component = {
                'name': filename.replace('.jcl', '').replace('.JCL', ''),
                'friendly_name': self.cobol_parser.generate_friendly_name(filename, 'JCL Job'),
                'type': 'JCL',
                'file_path': filename,
                'content': content,
                'total_lines': len(lines),
                'job_steps': [],
                'datasets': [],
                'programs_called': []
            }
            
            # Parse JCL structure
            current_step = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('//*'):
                    continue
                
                # Job card
                if line.startswith('//') and ' JOB ' in line:
                    job_name = line.split()[0][2:]  # Remove //
                    jcl_component['job_name'] = job_name
                    jcl_component['friendly_name'] = self.cobol_parser.generate_friendly_name(job_name, 'JCL Job')
                
                # Step card
                elif line.startswith('//') and ' EXEC ' in line:
                    if current_step:
                        jcl_component['job_steps'].append(current_step)
                    
                    step_parts = line.split()
                    step_name = step_parts[0][2:]  # Remove //
                    
                    current_step = {
                        'step_name': step_name,
                        'friendly_name': self.cobol_parser.generate_friendly_name(step_name, 'JCL Step'),
                        'line_number': i + 1,
                        'datasets': [],
                        'program': None
                    }
                    
                    # Extract program name
                    for part in step_parts:
                        if part.startswith('PGM='):
                            current_step['program'] = part.split('=')[1]
                            jcl_component['programs_called'].append({
                                'program_name': current_step['program'],
                                'friendly_name': self.cobol_parser.generate_friendly_name(current_step['program'], 'Program'),
                                'step': step_name,
                                'line_number': i + 1
                            })
                
                # DD card
                elif line.startswith('//') and ' DD ' in line and current_step:
                    dd_parts = line.split()
                    dd_name = dd_parts[0][2:]  # Remove //
                    
                    dataset_info = {
                        'dd_name': dd_name,
                        'friendly_name': self.cobol_parser.generate_friendly_name(dd_name, 'Dataset'),
                        'line_number': i + 1
                    }
                    
                    # Extract DSN if present
                    for part in dd_parts:
                        if part.startswith('DSN='):
                            dataset_info['dsn'] = part.split('=')[1].rstrip(',')
                            break
                    
                    current_step['datasets'].append(dataset_info)
                    jcl_component['datasets'].append(dataset_info)
            
            # Close last step
            if current_step:
                jcl_component['job_steps'].append(current_step)
            
            components.append(jcl_component)
            
        except Exception as e:
            logger.error(f"Error extracting JCL components: {str(e)}")
            components = [{
                'name': filename,
                'friendly_name': self.cobol_parser.generate_friendly_name(filename, 'JCL'),
                'type': 'JCL',
                'content': content,
                'error': str(e)
            }]
        
        return components
    
    def _extract_copybook_components(self, session_id: str, content: str, filename: str) -> List[Dict]:
        """Extract components from copybook"""
        components = []
        
        try:
            # Parse as COBOL structure
            parsed_data = self.cobol_parser.parse_cobol_file(content, filename)
            
            # Main copybook component
            copybook_component = {
                'name': filename.replace('.cpy', '').replace('.CPY', ''),
                'friendly_name': self.cobol_parser.generate_friendly_name(filename, 'Copybook'),
                'type': 'COPYBOOK',
                'file_path': filename,
                'content': content,
                'total_lines': parsed_data['total_lines'],
                'record_layouts': []
            }
            
            # Extract record layouts from copybook
            for layout in parsed_data['record_layouts']:
                layout_component = {
                    'name': layout.name,
                    'friendly_name': layout.friendly_name,
                    'type': 'RECORD_LAYOUT',
                    'parent_copybook': copybook_component['name'],
                    'level': layout.level,
                    'line_start': layout.line_start,
                    'line_end': layout.line_end,
                    'source_code': layout.source_code,
                    'fields': [self._field_to_dict(field) for field in layout.fields]
                }
                components.append(layout_component)
                copybook_component['record_layouts'].append(layout_component['name'])
                
                # Store in database
                self.db_manager.store_record_layout(session_id, {
                    'name': layout.name,
                    'friendly_name': layout.friendly_name,
                    'level': str(layout.level),
                    'line_start': layout.line_start,
                    'line_end': layout.line_end,
                    'source_code': layout.source_code,
                    'fields': [self._field_to_dict(field) for field in layout.fields]
                }, copybook_component['name'])
            
            components.insert(0, copybook_component)  # Main component first
            
        except Exception as e:
            logger.error(f"Error extracting copybook components: {str(e)}")
            components = [{
                'name': filename,
                'friendly_name': self.cobol_parser.generate_friendly_name(filename, 'Copybook'),
                'type': 'COPYBOOK',
                'content': content,
                'error': str(e)
            }]
        
        return components
    
    def _extract_generic_components(self, session_id: str, content: str, filename: str, file_type: str) -> List[Dict]:
        """Extract components from generic file type using LLM"""
        components = []
        
        try:
            # Use LLM to analyze unknown file types
            enhanced_data = self._llm_analyze_generic_file(session_id, content, filename, file_type)
            
            if enhanced_data:
                components = enhanced_data.get('components', [])
            
            # Fallback component
            if not components:
                components = [{
                    'name': filename,
                    'friendly_name': self.cobol_parser.generate_friendly_name(filename, file_type),
                    'type': file_type.upper(),
                    'content': content,
                    'total_lines': len(content.split('\n'))
                }]
                
        except Exception as e:
            logger.error(f"Error extracting generic components: {str(e)}")
            components = [{
                'name': filename,
                'friendly_name': filename,
                'type': file_type.upper(),
                'content': content,
                'error': str(e)
            }]
        
        return components
    
    def _field_to_dict(self, field) -> Dict:
        """Convert CobolField to dictionary"""
        return {
            'name': field.name,
            'friendly_name': field.friendly_name,
            'level': field.level,
            'picture': field.picture,
            'usage': field.usage,
            'occurs': field.occurs,
            'redefines': field.redefines,
            'value': field.value,
            'line_number': field.line_number
        }
    
    def _llm_analyze_large_program(self, session_id: str, content: str, filename: str) -> List[Dict]:
        """Analyze large programs using chunked LLM calls"""
        components = []
        
        try:
            # Chunk the content
            chunks = self.token_manager.chunk_cobol_code(content, preserve_structure=True)
            all_results = []
            
            for chunk in chunks:
                chunk_prompt = self.token_manager.get_chunked_analysis_prompt(chunk, 'component_extraction')
                
                response = self.llm_client.call_llm(chunk_prompt)
                
                # Log LLM call
                self.db_manager.log_llm_call(
                    session_id, 'component_extraction', chunk.chunk_number, chunk.total_chunks,
                    response.prompt_tokens, response.response_tokens, response.processing_time_ms,
                    response.success, response.error_message
                )
                
                if response.success:
                    parsed_result = self.llm_client.extract_json_from_response(response.content)
                    if parsed_result:
                        all_results.append(parsed_result)
            
            # Consolidate results
            if all_results:
                consolidated = self.token_manager.consolidate_chunk_results(all_results, 'component_extraction')
                components = self._convert_llm_components(consolidated, filename)
                
        except Exception as e:
            logger.error(f"Error in LLM large program analysis: {str(e)}")
        
        return components
    
    def _llm_enhance_component_analysis(self, session_id: str, content: str, filename: str) -> Optional[Dict]:
        """Enhance component analysis using LLM for smaller files"""
        try:
            prompt = f"""
Analyze this COBOL program and provide enhanced component analysis.
Focus on:
1. Business logic identification
2. Data flow analysis  
3. Interface points
4. Field relationships
5. Program dependencies

Provide analysis in JSON format:
{{
    "program_analysis": {{
        "business_functions": [],
        "interface_points": [],
        "data_flows": [],
        "complexity_score": 0.8
    }},
    "additional_components": []
}}

COBOL Program ({filename}):
{content}
"""
            
            response = self.llm_client.call_llm(prompt)
            
            # Log LLM call
            self.db_manager.log_llm_call(
                session_id, 'component_enhancement', 1, 1,
                response.prompt_tokens, response.response_tokens, response.processing_time_ms,
                response.success, response.error_message
            )
            
            if response.success:
                return self.llm_client.extract_json_from_response(response.content)
                
        except Exception as e:
            logger.error(f"Error in LLM component enhancement: {str(e)}")
        
        return None
    
    def _llm_analyze_generic_file(self, session_id: str, content: str, filename: str, file_type: str) -> Optional[Dict]:
        """Analyze generic file types using LLM"""
        try:
            prompt = f"""
Analyze this {file_type} file and extract components/structure.
Identify:
1. Main components or sections
2. Data definitions
3. Processing logic
4. Dependencies or relationships

Return JSON format:
{{
    "components": [
        {{
            "name": "component_name",
            "friendly_name": "Friendly Component Name",
            "type": "{file_type.upper()}",
            "description": "what this component does",
            "line_start": 1,
            "line_end": 10
        }}
    ]
}}

File Content ({filename}):
{content}
"""
            
            response = self.llm_client.call_llm(prompt)
            
            # Log LLM call
            self.db_manager.log_llm_call(
                session_id, 'generic_analysis', 1, 1,
                response.prompt_tokens, response.response_tokens, response.processing_time_ms,
                response.success, response.error_message
            )
            
            if response.success:
                return self.llm_client.extract_json_from_response(response.content)
                
        except Exception as e:
            logger.error(f"Error in LLM generic file analysis: {str(e)}")
        
        return None
    
    def _convert_llm_components(self, llm_response: Dict, filename: str) -> List[Dict]:
        """Convert LLM response to component format"""
        components = []
        
        try:
            llm_components = llm_response.get('components', [])
            
            for comp_data in llm_components:
                component = {
                    'name': comp_data.get('name', 'Unknown'),
                    'friendly_name': comp_data.get('friendly_name', 
                        self.cobol_parser.generate_friendly_name(comp_data.get('name', ''), 'Component')),
                    'type': comp_data.get('type', 'COMPONENT'),
                    'file_path': filename,
                    'description': comp_data.get('description', ''),
                    'line_start': comp_data.get('line_start', 1),
                    'line_end': comp_data.get('line_end', 1),
                    'business_functions': comp_data.get('business_functions', []),
                    'data_flows': comp_data.get('data_flows', [])
                }
                components.append(component)
                
        except Exception as e:
            logger.error(f"Error converting LLM components: {str(e)}")
        
        return components
    
    def _extract_and_store_field_relationships(self, session_id: str, components: List[Dict], filename: str):
        """Extract and store field relationships in database"""
        try:
            # Get program component
            program_component = None
            record_layouts = []
            
            for component in components:
                if component['type'] == 'PROGRAM':
                    program_component = component
                elif component['type'] == 'RECORD_LAYOUT':
                    record_layouts.append(component)
            
            if not program_component:
                return
            
            # Extract data movements from program content
            program_content = program_component.get('content', '')
            data_movements = self.cobol_parser.extract_data_movements(program_content.split('\n'))
            
            # Store field relationships
            for movement in data_movements:
                if movement['operation'] == 'MOVE':
                    # Store field details for both source and target
                    self._store_field_operation(
                        session_id, movement['source_field'], program_component['name'],
                        'SOURCE', movement['line_number'], movement['line_content']
                    )
                    self._store_field_operation(
                        session_id, movement['target_field'], program_component['name'],
                        'TARGET', movement['line_number'], movement['line_content']
                    )
                    
                elif movement['operation'] == 'COMPUTE':
                    self._store_field_operation(
                        session_id, movement['target_field'], program_component['name'],
                        'COMPUTED', movement['line_number'], movement['line_content']
                    )
            
            # Analyze field usage for each field in record layouts
            for layout in record_layouts:
                for field_data in layout.get('fields', []):
                    field_usage = self.cobol_parser.analyze_field_usage(
                        program_content.split('\n'), field_data['name']
                    )
                    
                    # Determine usage type
                    usage_type = self._determine_field_usage_type(field_usage)
                    
                    # Store enhanced field details
                    self.db_manager.store_field_details(session_id, {
                        'name': field_data['name'],
                        'operation_type': 'DEFINITION',
                        'line_number': field_data.get('line_number', 0),
                        'code_snippet': f"Level {field_data['level']} {field_data['name']} {field_data.get('picture', '')}",
                        'usage': usage_type,
                        'business_purpose': self._infer_business_purpose_from_usage(field_usage, field_data['name'])
                    }, program_component['name'])
                    
        except Exception as e:
            logger.error(f"Error extracting field relationships: {str(e)}")
    
    def _store_field_operation(self, session_id: str, field_name: str, program_name: str,
                             operation_type: str, line_number: int, code_snippet: str):
        """Store individual field operation"""
        try:
            self.db_manager.store_field_details(session_id, {
                'name': field_name,
                'operation_type': operation_type,
                'line_number': line_number,
                'code_snippet': code_snippet,
                'usage': self._map_operation_to_usage(operation_type),
                'business_purpose': f"Field used in {operation_type.lower()} operation"
            }, program_name)
        except Exception as e:
            logger.error(f"Error storing field operation: {str(e)}")
    
    def _determine_field_usage_type(self, field_usage: Dict) -> str:
        """Determine field usage type from analysis"""
        if field_usage['target_operations']:
            if field_usage['source_operations']:
                return 'INPUT_OUTPUT'
            else:
                return 'INPUT'
        elif field_usage['source_operations']:
            return 'OUTPUT'
        elif field_usage['operations']:
            return 'REFERENCE'
        elif field_usage['references']:
            return 'REFERENCE'
        else:
            return 'UNUSED'
    
    def _map_operation_to_usage(self, operation_type: str) -> str:
        """Map operation type to usage type"""
        mapping = {
            'SOURCE': 'OUTPUT',
            'TARGET': 'INPUT',
            'COMPUTED': 'DERIVED',
            'DEFINITION': 'STATIC'
        }
        return mapping.get(operation_type, 'REFERENCE')
    
    def _infer_business_purpose_from_usage(self, field_usage: Dict, field_name: str) -> str:
        """Infer business purpose from field usage analysis"""
        if field_usage['target_operations']:
            return f"Populated field - receives data from other sources"
        elif field_usage['source_operations']:
            return f"Source field - provides data to other fields"
        elif field_usage['operations']:
            operation_types = [op['operation'] for op in field_usage['operations']]
            if 'CONDITION' in operation_types:
                return f"Control field - used in conditional logic"
            else:
                return f"Processing field - used in {', '.join(set(operation_types)).lower()} operations"
        else:
            return f"Static field - defined but not actively processed"