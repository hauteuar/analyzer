"""
Component Extractor Module
Handles extraction and analysis of COBOL components including record layouts
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from modules.cobol_parser import COBOLParser
import time
import traceback

logger = logging.getLogger(__name__)

class ComponentExtractor:
    def __init__(self, llm_client, token_manager, db_manager):
        self.llm_client = llm_client
        self.token_manager = token_manager
        self.db_manager = db_manager
        self.cobol_parser = COBOLParser()
    
    def extract_components(self, session_id: str, file_content: str, file_name: str, file_type: str) -> List[Dict]:
        """Extract all components from uploaded file"""
        logger.info(f"🔧 Starting component extraction for {file_name} (Type: {file_type})")
        logger.info(f"📝 File statistics: {len(file_content)} chars, {len(file_content.split())} words, {len(file_content.splitlines())} lines")
        
        try:
            components = []
            
            if file_type.upper() == 'COBOL':
                logger.info("🏗️  Processing as COBOL program...")
                components = self._extract_cobol_components(session_id, file_content, file_name)
                
            elif file_type.upper() == 'JCL':
                logger.info("📋 Processing as JCL job...")
                components = self._extract_jcl_components(session_id, file_content, file_name)
                
            elif file_type.upper() == 'COPYBOOK':
                logger.info("📚 Processing as copybook...")
                components = self._extract_copybook_components(session_id, file_content, file_name)
                
            else:
                logger.info(f"🔍 Processing as generic {file_type} file...")
                components = self._extract_generic_components(session_id, file_content, file_name, file_type)
            
            logger.info(f"✅ Component extraction completed: {len(components)} components extracted")
            
            for component in components:
                try:
                    # Ensure component has the proper structure for database storage
                    component_data = {
                        # Core component information
                        'name': component.get('name', 'UNKNOWN'),
                        'friendly_name': component.get('friendly_name', component.get('name', 'UNKNOWN')),
                        'type': component.get('type', 'UNKNOWN'),
                        'file_path': file_name,
                        
                        # Content and metrics
                        'content': component.get('content', ''),
                        'total_lines': component.get('total_lines', 0),
                        'executable_lines': component.get('executable_lines', 0),
                        'comment_lines': component.get('comment_lines', 0),
                        'total_fields': component.get('total_fields', 0),
                        
                        # Business analysis
                        'business_purpose': component.get('business_purpose', ''),
                        'complexity_score': component.get('complexity_score', 0.5),
                        'llm_summary': component.get('llm_summary', {}),
                        
                        # Structural data
                        'divisions': component.get('divisions', []),
                        'file_operations': component.get('file_operations', []),
                        'program_calls': component.get('program_calls', []),
                        'copybooks': component.get('copybooks', []),
                        'cics_operations': component.get('cics_operations', []),
                        'mq_operations': component.get('mq_operations', []),
                        'xml_operations': component.get('xml_operations', []),
                        
                        # Relationships
                        'derived_components': component.get('derived_components', []),
                        'record_layouts': component.get('record_layouts', []),
                        'fields': component.get('fields', []),
                        
                        # Component-specific data
                        'parent_program': component.get('parent_program', ''),
                        'parent_copybook': component.get('parent_copybook', ''),
                        'level': component.get('level', ''),
                        'line_start': component.get('line_start', 0),
                        'line_end': component.get('line_end', 0),
                        'source_code': component.get('source_code', ''),
                        'access_pattern': component.get('access_pattern', ''),
                        'io_classification': component.get('io_classification', ''),
                        'operations': component.get('operations', []),
                        'usage_pattern': component.get('usage_pattern', '')
                    }
                    
                    self.db_manager.store_component_analysis(
                        session_id, 
                        component_data['name'], 
                        component_data['type'], 
                        file_name, 
                        component_data
                    )
                    
                    logger.info(f"✅ Stored component: {component_data['name']} ({component_data['type']})")
                    
                except Exception as e:
                    logger.error(f"❌ Error storing component {component.get('name', 'UNKNOWN')}: {str(e)}")
                    # Continue with other components even if one fails
                    continue
            # Log component summary
            if components:
                main_components = [c for c in components if c.get('type') in ['PROGRAM', 'JCL', 'COPYBOOK']]
                derived_components = [c for c in components if c.get('type') not in ['PROGRAM', 'JCL', 'COPYBOOK']]
                
                logger.info(f"📈 Component summary:")
                logger.info(f"   • Main components: {len(main_components)}")
                logger.info(f"   • Derived components: {len(derived_components)}")
                
                
                for component in main_components:
                    derived_count = len(component.get('derived_components', []))
                    logger.info(f"   📦 {component['name']} → {derived_count} derived components")
            
            return components
            
        except Exception as e:
            logger.error(f"❌ Error extracting components from {file_name}: {str(e)}")
            logger.error(f"📍 Stack trace: {traceback.format_exc()}")
            return []
    
    def _extract_cobol_components(self, session_id: str, content: str, filename: str) -> List[Dict]:
        """Complete COBOL component extraction with full source code analysis"""
        logger.info(f"Starting complete COBOL analysis for {filename}")
        
        try:
            # Parse COBOL structure
            start_time = time.time()
            parsed_data = self.cobol_parser.parse_cobol_file(content, filename)
            logger.info(f"COBOL parsing completed: {len(parsed_data['record_layouts'])} layouts found")
            
            # Generate LLM summary
            program_summary = self._generate_component_summary(session_id, parsed_data, 'PROGRAM')
            
            # Create main program component with complete source
            program_name = filename.replace('.cob', '').replace('.CBL', '').replace('.cbl', '')
            program_component = {
                'name': program_name,
                'friendly_name': parsed_data['friendly_name'],
                'type': 'PROGRAM',
                'file_path': filename,
                'content': content,  # CRITICAL: Store complete source code
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
                'derived_components': [],
                'record_layouts': [],
                'fields': []
            }
            
            components = [program_component]
            
            # Process each record layout with complete field analysis
            for layout_idx, layout in enumerate(parsed_data['record_layouts'], 1):
                layout_name = layout.name
                logger.info(f"Processing layout {layout_idx}/{len(parsed_data['record_layouts'])}: {layout_name}")
                
                # Generate layout summary
                layout_summary = self._generate_layout_summary(session_id, layout, parsed_data)
                
                # Store record layout in database first
                layout_data = {
                    'name': layout_name,
                    'friendly_name': layout.friendly_name,
                    'level': str(layout.level),
                    'line_start': layout.line_start,
                    'line_end': layout.line_end,
                    'source_code': layout.source_code,
                    'fields': []  # Will be populated below
                }
                
                # Store layout in database and get ID
                try:
                    with self.db_manager.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT OR REPLACE INTO record_layouts 
                            (session_id, layout_name, friendly_name, program_name, level_number, 
                            line_start, line_end, source_code, fields_count, business_purpose)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (session_id, layout_name, layout.friendly_name, program_name,
                            str(layout.level), layout.line_start, layout.line_end,
                            layout.source_code, len(layout.fields),
                            layout_summary.get('business_purpose', '')))
                        
                        layout_id = cursor.lastrowid
                        logger.info(f"Stored layout {layout_name} with ID {layout_id}")
                except Exception as db_error:
                    logger.error(f"Error storing layout {layout_name}: {str(db_error)}")
                    layout_id = None
                
                # Process each field with complete source code analysis
                enhanced_fields = []
                for field_idx, field in enumerate(layout.fields, 1):
                    try:
                        logger.debug(f"Analyzing field {field_idx}/{len(layout.fields)}: {field.name}")
                        
                        # Perform complete source code analysis for this field
                        field_source_analysis = self._complete_field_source_analysis(
                            field.name, content, program_name
                        )
                        
                        # Create enhanced field data with complete context
                        enhanced_field = {
                            'name': field.name,
                            'friendly_name': field.friendly_name or field.name.replace('-', ' ').title(),
                            'level': field.level,
                            'picture': field.picture,
                            'usage': field.usage,
                            'occurs': field.occurs,
                            'redefines': field.redefines,
                            'value': field.value,
                            'line_number': field.line_number,
                            'code_snippet': f"{field.level:02d} {field.name}" + (f" PIC {field.picture}" if field.picture else ""),
                            'usage_type': field_source_analysis['primary_usage'],
                            'operation_type': 'COMPREHENSIVE_DEFINITION',
                            'business_purpose': field_source_analysis['business_purpose'],
                            'confidence': 0.95,
                            'source_field': field_source_analysis.get('primary_source_field', ''),
                            'target_field': field.name if field_source_analysis.get('receives_data', False) else '',
                            
                            # Complete source code context for database
                            'definition_line_number': field_source_analysis.get('definition_line', field.line_number),
                            'definition_code': field_source_analysis.get('definition_code', ''),
                            'program_source_content': content,  # Store complete program source
                            'field_references_json': json.dumps(field_source_analysis['all_references']),
                            'usage_summary_json': json.dumps(field_source_analysis['usage_summary']),
                            'total_program_references': len(field_source_analysis['all_references']),
                            'move_source_count': field_source_analysis['counts']['move_source'],
                            'move_target_count': field_source_analysis['counts']['move_target'],
                            'arithmetic_count': field_source_analysis['counts']['arithmetic'],
                            'conditional_count': field_source_analysis['counts']['conditional'],
                            'cics_count': field_source_analysis['counts']['cics']
                        }
                        
                        enhanced_fields.append(enhanced_field)
                        
                        # Store field details in database with complete context
                        if layout_id:
                            try:
                                self.db_manager.store_field_details(session_id, enhanced_field, program_name, layout_id)
                                logger.debug(f"Stored field {field.name} with {len(field_source_analysis['all_references'])} source references")
                            except Exception as field_db_error:
                                logger.error(f"Error storing field {field.name}: {str(field_db_error)}")
                        
                    except Exception as field_error:
                        logger.error(f"Error analyzing field {field.name}: {str(field_error)}")
                        # Create basic field entry as fallback
                        enhanced_field = {
                            'name': field.name,
                            'level': field.level,
                            'picture': field.picture,
                            'line_number': field.line_number,
                            'error': str(field_error)
                        }
                        enhanced_fields.append(enhanced_field)
                
                # Update layout data with enhanced fields
                layout_data['fields'] = enhanced_fields
                
                # Create layout component
                layout_component = {
                    'name': layout_name,
                    'friendly_name': layout.friendly_name,
                    'type': 'RECORD_LAYOUT',
                    'parent_program': program_name,
                    'level': layout.level,
                    'line_start': layout.line_start,
                    'line_end': layout.line_end,
                    'source_code': layout.source_code,
                    'fields': enhanced_fields,
                    'llm_summary': layout_summary,
                    'business_purpose': layout_summary.get('business_purpose', ''),
                    'usage_pattern': layout_summary.get('usage_pattern', 'UNKNOWN')
                }
                
                components.append(layout_component)
                program_component['derived_components'].append(layout_name)
                program_component['record_layouts'].append(layout_data)
                program_component['fields'].extend(enhanced_fields)
                
                logger.info(f"Completed layout {layout_name}: {len(enhanced_fields)} fields analyzed")
            
            # Update final counts
            program_component['total_fields'] = len(program_component['fields'])
            
            total_time = time.time() - start_time
            logger.info(f"COBOL extraction completed in {total_time:.2f}s: {len(components)} components, {program_component['total_fields']} fields")
            
            return components
            
        except Exception as e:
            logger.error(f"Error in COBOL component extraction: {str(e)}")
            return []


    def store_field_details(self, session_id: str, field_data: Dict, program_name: str, layout_id: int = None):
        """Store complete field details with source code context"""
        field_name = field_data.get('name', 'UNNAMED')
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Store complete field analysis
                cursor.execute('''
                    INSERT OR REPLACE INTO field_analysis_details 
                    (session_id, field_id, field_name, friendly_name, program_name, layout_name,
                    operation_type, line_number, code_snippet, usage_type, source_field, target_field,
                    business_purpose, analysis_confidence,
                    definition_line_number, definition_code, program_source_content,
                    field_references_json, usage_summary_json, total_program_references,
                    move_source_count, move_target_count, arithmetic_count, conditional_count, cics_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id, layout_id, field_name,
                    field_data.get('friendly_name', field_name),
                    program_name, None,
                    field_data.get('operation_type', 'DEFINITION'),
                    field_data.get('line_number', 0),
                    field_data.get('code_snippet', ''),
                    field_data.get('usage_type', 'STATIC'),
                    field_data.get('source_field', ''),
                    field_data.get('target_field', ''),
                    field_data.get('business_purpose', ''),
                    field_data.get('confidence', 0.8),
                    field_data.get('definition_line_number', 0),
                    field_data.get('definition_code', ''),
                    field_data.get('program_source_content', ''),
                    field_data.get('field_references_json', '[]'),
                    field_data.get('usage_summary_json', '{}'),
                    field_data.get('total_program_references', 0),
                    field_data.get('move_source_count', 0),
                    field_data.get('move_target_count', 0),
                    field_data.get('arithmetic_count', 0),
                    field_data.get('conditional_count', 0),
                    field_data.get('cics_count', 0)
                ))
                
                logger.debug(f"Stored complete field details: {field_name}")
                
        except Exception as e:
            logger.error(f"Error storing field details for {field_name}: {str(e)}")
            raise

    # Add this method to ComponentExtractor class:

    def _complete_field_source_analysis(self, field_name: str, program_content: str, program_name: str) -> Dict:
        """Perform complete source code analysis for a field"""
        analysis = {
            'field_name': field_name,
            'program_name': program_name,
            'all_references': [],
            'definition_line': None,
            'definition_code': '',
            'primary_usage': 'STATIC',
            'primary_source_field': '',
            'receives_data': False,
            'provides_data': False,
            'business_purpose': '',
            'counts': {
                'definition': 0,
                'move_source': 0,
                'move_target': 0,
                'arithmetic': 0,
                'conditional': 0,
                'cics': 0
            },
            'usage_summary': {}
        }
        
        try:
            lines = program_content.split('\n')
            field_upper = field_name.upper()
            
            # Analyze every line for field usage
            for line_idx, line in enumerate(lines, 1):
                line_stripped = line.strip()
                line_upper = line_stripped.upper()
                
                # Skip empty lines and comments
                if not line_stripped or line_stripped.startswith('*'):
                    continue
                
                # Check if field is referenced in this line
                if field_upper in line_upper:
                    operation_type = 'REFERENCE'
                    business_context = ''
                    source_field = ''
                    target_field = ''
                    
                    # Field definition with PIC clause
                    if ('PIC' in line_upper and 
                        re.match(r'^\s*\d{2}\s+' + re.escape(field_upper), line_upper)):
                        operation_type = 'DEFINITION'
                        analysis['definition_line'] = line_idx
                        analysis['definition_code'] = line_stripped
                        analysis['counts']['definition'] += 1
                        business_context = 'Field data structure definition'
                    
                    # MOVE operations - field receives data
                    elif 'MOVE' in line_upper:
                        # Pattern: MOVE source TO field_name
                        move_to_match = re.search(rf'MOVE\s+([A-Z0-9\-\(\)]+)\s+TO\s+{re.escape(field_upper)}', line_upper)
                        if move_to_match:
                            operation_type = 'MOVE_TARGET'
                            source_field = move_to_match.group(1)
                            analysis['counts']['move_target'] += 1
                            analysis['receives_data'] = True
                            if not analysis['primary_source_field']:
                                analysis['primary_source_field'] = source_field
                            business_context = f'Receives data from {source_field}'
                        
                        # Pattern: MOVE field_name TO target
                        else:
                            move_from_match = re.search(rf'MOVE\s+{re.escape(field_upper)}\s+TO\s+([A-Z0-9\-\(\)]+)', line_upper)
                            if move_from_match:
                                operation_type = 'MOVE_SOURCE'
                                target_field = move_from_match.group(1)
                                analysis['counts']['move_source'] += 1
                                analysis['provides_data'] = True
                                business_context = f'Provides data to {target_field}'
                    
                    # Arithmetic operations
                    elif any(op in line_upper for op in ['COMPUTE', 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE']):
                        operation_type = 'ARITHMETIC'
                        analysis['counts']['arithmetic'] += 1
                        business_context = 'Used in mathematical computation'
                    
                    # Conditional operations
                    elif any(op in line_upper for op in ['IF', 'WHEN', 'EVALUATE']):
                        operation_type = 'CONDITIONAL'
                        analysis['counts']['conditional'] += 1
                        business_context = 'Used in business logic decision'
                    
                    # CICS operations
                    elif 'CICS' in line_upper:
                        operation_type = 'CICS'
                        analysis['counts']['cics'] += 1
                        business_context = 'Used in CICS transaction processing'
                    
                    # Get context lines for complete understanding
                    context_start = max(0, line_idx - 4)
                    context_end = min(len(lines), line_idx + 3)
                    context_lines = lines[context_start:context_end]
                    
                    # Create comprehensive reference entry
                    reference = {
                        'line_number': line_idx,
                        'line_content': line_stripped,
                        'operation_type': operation_type,
                        'business_context': business_context,
                        'source_field': source_field,
                        'target_field': target_field,
                        'context_lines': context_lines,
                        'context_block': '\n'.join([
                            f"{context_start + i + 1:4d}: {ctx_line}"
                            for i, ctx_line in enumerate(context_lines)
                        ])
                    }
                    
                    analysis['all_references'].append(reference)
            
            # Determine primary usage based on actual operations
            counts = analysis['counts']
            total_ops = sum(counts.values()) - counts['definition']  # Exclude definition from usage count
            
            if analysis['receives_data'] and analysis['provides_data']:
                analysis['primary_usage'] = 'INPUT_OUTPUT'
            elif analysis['receives_data']:
                analysis['primary_usage'] = 'INPUT'
            elif analysis['provides_data']:
                analysis['primary_usage'] = 'OUTPUT'
            elif counts['arithmetic'] > 0:
                analysis['primary_usage'] = 'DERIVED'
            elif counts['conditional'] > 0:
                analysis['primary_usage'] = 'REFERENCE'
            elif counts['cics'] > 0:
                analysis['primary_usage'] = 'CICS_FIELD'
            elif total_ops == 0:
                analysis['primary_usage'] = 'STATIC'
            else:
                analysis['primary_usage'] = 'PROCESSED'
            
            # Generate comprehensive business purpose
            purpose_elements = []
            if counts['move_target'] > 0:
                purpose_elements.append(f"receives data ({counts['move_target']} operations)")
            if counts['move_source'] > 0:
                purpose_elements.append(f"provides data ({counts['move_source']} operations)")
            if counts['arithmetic'] > 0:
                purpose_elements.append(f"mathematical calculations ({counts['arithmetic']} operations)")
            if counts['conditional'] > 0:
                purpose_elements.append(f"business decisions ({counts['conditional']} operations)")
            if counts['cics'] > 0:
                purpose_elements.append(f"CICS transactions ({counts['cics']} operations)")
            
            if purpose_elements:
                analysis['business_purpose'] = f"{field_name} - {', '.join(purpose_elements)}"
            elif counts['definition'] > 0:
                analysis['business_purpose'] = f"{field_name} - Static data field (defined but not actively used)"
            else:
                analysis['business_purpose'] = f"{field_name} - Field usage could not be determined"
            
            # Create usage summary
            analysis['usage_summary'] = {
                'total_references': len(analysis['all_references']),
                'definition_found': analysis['definition_line'] is not None,
                'actively_used': total_ops > 0,
                'primary_pattern': analysis['primary_usage'],
                'operation_breakdown': dict(counts)
            }
            
            logger.debug(f"Field {field_name}: {analysis['primary_usage']}, {len(analysis['all_references'])} refs")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in complete field analysis for {field_name}: {str(e)}")
            return {
                'field_name': field_name,
                'all_references': [],
                'primary_usage': 'ERROR',
                'business_purpose': f'Analysis failed: {str(e)}',
                'counts': {},
                'usage_summary': {}
            }

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
            # Fallback to basic parsing
            components = [{
                'name': filename,
                'friendly_name': self.cobol_parser.generate_friendly_name(filename, 'Program'),
                'type': 'PROGRAM',
                'content': content,
                'total_lines': len(content.split('\n')),
                'error': str(e)
            }]
        
        return components
    
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
                    'fields': [self._field_to_dict_enhanced(field, content) for field in layout.fields]  # NEW METHOD
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
                    'fields': [self._field_to_dict_enhanced(field, content) for field in layout.fields]  # NEW METHOD
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
    
    def _field_to_dict_enhanced(self, field, program_content: str, program_name: str = "PROGRAM") -> Dict:
        """Convert field with complete source code analysis"""
        try:
            field_name = field.name
            
            # Perform complete source analysis
            source_analysis = self._complete_field_source_analysis(field_name, program_content, program_name)
            
            return {
                'name': field_name,
                'friendly_name': field.friendly_name or field_name.replace('-', ' ').title(),
                'level': field.level,
                'picture': field.picture,
                'usage': field.usage,
                'occurs': field.occurs,
                'redefines': field.redefines,
                'value': field.value,
                'line_number': field.line_number,
                'code_snippet': f"{field.level:02d} {field_name}" + (f" PIC {field.picture}" if field.picture else ""),
                'usage_type': source_analysis['primary_usage'],
                'operation_type': 'COMPREHENSIVE_DEFINITION',
                'business_purpose': source_analysis['business_purpose'],
                'confidence': 0.95,
                'source_field': source_analysis.get('primary_source_field', ''),
                'target_field': field_name if source_analysis.get('receives_data', False) else '',
                
                # Complete source code storage
                'definition_line_number': source_analysis.get('definition_line', field.line_number),
                'definition_code': source_analysis.get('definition_code', ''),
                'program_source_content': program_content,
                'field_references_json': json.dumps(source_analysis['all_references']),
                'usage_summary_json': json.dumps(source_analysis['usage_summary']),
                'total_program_references': len(source_analysis['all_references']),
                'move_source_count': source_analysis['counts']['move_source'],
                'move_target_count': source_analysis['counts']['move_target'],
                'arithmetic_count': source_analysis['counts']['arithmetic'],
                'conditional_count': source_analysis['counts']['conditional'],
                'cics_count': source_analysis['counts']['cics']
            }
            
        except Exception as e:
            logger.error(f"Error enhancing field {field.name}: {str(e)}")
            return {
                'name': field.name,
                'level': field.level,
                'picture': getattr(field, 'picture', ''),
                'line_number': getattr(field, 'line_number', 0),
                'error': str(e)
            }
    
    def _infer_field_business_purpose(self, field_name: str, usage_analysis: Dict) -> str:
        """Infer business purpose from field name and usage analysis"""
        name_upper = field_name.upper()
        
        # Check usage patterns first
        if usage_analysis.get('target_operations'):
            return f"Input field - receives data from external sources"
        elif usage_analysis.get('source_operations'):
            return f"Output field - provides data to other processes"
        elif usage_analysis.get('operations'):
            op_types = [op.get('operation', '') for op in usage_analysis['operations']]
            if 'CONDITION' in op_types:
                return f"Control field - used in business logic decisions"
            return f"Processing field - used in {', '.join(set(op_types)).lower()} operations"
        
        # Fallback to name-based inference
        purpose_keywords = {
            'CUST': 'Customer information field',
            'ACCT': 'Account data field',
            'ADDR': 'Address information field',
            'DATE': 'Date field for temporal data',
            'TIME': 'Time field for temporal data',
            'AMT': 'Monetary amount field',
            'QTY': 'Quantity field',
            'NBR': 'Numeric identifier field',
            'NO': 'Number/identifier field',
            'CD': 'Code field for classification',
            'DESC': 'Description field',
            'NAME': 'Name field',
            'TRAN': 'Transaction data field',
            'BAL': 'Balance field',
            'RATE': 'Rate/percentage field',
            'FLG': 'Flag/indicator field',
            'IND': 'Indicator field',
            'SW': 'Switch field for control logic'
        }
        
        for keyword, purpose in purpose_keywords.items():
            if keyword in name_upper:
                return purpose
        
        return f"Data field - {field_name} used in program processing"

    def _extract_source_field(self, usage_analysis: Dict) -> str:
        """Extract source field from usage analysis"""
        source_ops = usage_analysis.get('source_operations', [])
        if source_ops:
            # Look for MOVE operations
            for op in source_ops:
                if 'MOVE' in op.get('line_content', ''):
                    # Extract source from MOVE statement
                    import re
                    move_match = re.search(r'MOVE\s+([A-Z0-9\-\(\)]+)\s+TO', op['line_content'], re.IGNORECASE)
                    if move_match:
                        return move_match.group(1)
        return ''

    # Also add this method to store components with enhanced data
    
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
    
    def _determine_enhanced_usage_type(self, usage_analysis: Dict) -> str:
        """Determine enhanced usage type from comprehensive analysis"""
        if usage_analysis.get('target_operations') and usage_analysis.get('source_operations'):
            return 'INPUT_OUTPUT'
        elif usage_analysis.get('target_operations'):
            return 'INPUT'
        elif usage_analysis.get('source_operations'):
            return 'OUTPUT'
        elif usage_analysis.get('operations'):
            # Check if used in conditions
            conditions = [op for op in usage_analysis['operations'] if op.get('operation') == 'CONDITION']
            if conditions:
                return 'REFERENCE'
            return 'DERIVED'
        elif usage_analysis.get('references'):
            return 'REFERENCE'
        else:
            return 'STATIC'

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
        
    def _extract_and_store_dependencies(self, session_id: str, components: List[Dict], filename: str):
        """Extract and store component dependencies"""
        try:
            logger.info(f"Extracting dependencies for {len(components)} components")
            
            dependencies = []
            
            # Get main program component
            main_program = None
            for component in components:
                if component.get('type') == 'PROGRAM':
                    main_program = component
                    break
            
            if not main_program:
                logger.warning("No main program found for dependency extraction")
                return
            
            # Extract program calls as dependencies
            for call in main_program.get('program_calls', []):
                dependency = {
                    'source_component': main_program['name'],
                    'target_component': call.get('program_name', call.get('name', 'Unknown')),
                    'relationship_type': 'PROGRAM_CALL',
                    'interface_type': 'COBOL_CALL',
                    'confidence_score': 0.9,
                    'analysis_details_json': json.dumps({
                        'line_number': call.get('line_number', 0),
                        'call_type': call.get('call_type', 'STATIC'),
                        'parameters': call.get('parameters', [])
                    })
                }
                dependencies.append(dependency)
            
            # Extract copybook dependencies
            for copybook in main_program.get('copybooks', []):
                dependency = {
                    'source_component': main_program['name'],
                    'target_component': copybook.get('copybook_name', copybook.get('name', 'Unknown')),
                    'relationship_type': 'COPYBOOK_INCLUDE',
                    'interface_type': 'COBOL_COPY',
                    'confidence_score': 0.95,
                    'analysis_details_json': json.dumps({
                        'line_number': copybook.get('line_number', 0),
                        'include_type': 'COPY',
                        'library': copybook.get('library', '')
                    })
                }
                dependencies.append(dependency)
            
            # Extract CICS file dependencies
            for cics_op in main_program.get('cics_operations', []):
                if 'file_name' in cics_op:
                    dependency = {
                        'source_component': main_program['name'],
                        'target_component': cics_op['file_name'],
                        'relationship_type': 'CICS_FILE_ACCESS',
                        'interface_type': 'CICS',
                        'confidence_score': 0.9,
                        'analysis_details_json': json.dumps({
                            'operation': cics_op.get('operation', 'UNKNOWN'),
                            'access_type': cics_op.get('access_type', 'READ_WRITE'),
                            'line_number': cics_op.get('line_number', 0)
                        })
                    }
                    dependencies.append(dependency)
            
            # Extract record layout dependencies
            for layout in main_program.get('record_layouts', []):
                dependency = {
                    'source_component': main_program['name'],
                    'target_component': layout.get('name', 'Unknown'),
                    'relationship_type': 'USES_RECORD_LAYOUT',
                    'interface_type': 'DATA_STRUCTURE',
                    'confidence_score': 0.95,
                    'analysis_details_json': json.dumps({
                        'layout_level': layout.get('level', '01'),
                        'field_count': len(layout.get('fields', [])),
                        'line_start': layout.get('line_start', 0)
                    })
                }
                dependencies.append(dependency)
            
            # Store all dependencies
            if dependencies:
                logger.info(f"Storing {len(dependencies)} dependencies")
                
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    for dep in dependencies:
                        cursor.execute('''
                            INSERT OR REPLACE INTO dependency_relationships 
                            (session_id, source_component, target_component, relationship_type,
                            interface_type, confidence_score, analysis_details_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (session_id, dep['source_component'], dep['target_component'],
                            dep['relationship_type'], dep['interface_type'],
                            dep['confidence_score'], dep['analysis_details_json']))
                
                logger.info(f"Successfully stored {len(dependencies)} dependencies")
            
        except Exception as e:
            logger.error(f"Error extracting dependencies: {str(e)}")