"""
Component Extractor Module
Handles extraction and analysis of COBOL components including record layouts
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from modules.cobol_parser import COBOLParser
import time
import traceback

logger = logging.getLogger(__name__)

class ComponentExtractor:
    def __init__(self, llm_client, token_manager, db_manager):
        self.llm_client = llm_client
        self.token_manager = token_manager
        self.db_manager = db_manager
        self.cobol_parser = COBOLParser(llm_client) 
    
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
            
            # In extract_components method, wrap the storage in better error handling:

            # In component_extractor.py, in the extract_components method:

            for component in components:
                try:
                    # CHANGE 1: Ensure source content is always stored
                    main_component_data = {
                        'name': component.get('name', 'UNKNOWN'),
                        'friendly_name': component.get('friendly_name', component.get('name', 'UNKNOWN')),
                        'type': component.get('type', 'UNKNOWN'),
                        'file_path': file_name,
                        'content': file_content,  # ALWAYS store full source
                        'total_lines': component.get('total_lines', 0),
                        'business_purpose': component.get('business_purpose', ''),
                        #'total_lines': component.get('total_lines', 0),
                        'executable_lines': component.get('executable_lines', 0),
                        'comment_lines': component.get('comment_lines', 0),
                        'total_fields': len(component.get('fields', [])),
                        #'business_purpose': component.get('business_purpose', ''),
                        'complexity_score': component.get('complexity_score', 0.5),
                        'llm_summary': component.get('llm_summary', {}),
                        # Don't include derived_components in main JSON
                        'divisions': component.get('divisions', []),
                        'file_operations': component.get('file_operations', []),
                        'program_calls': component.get('program_calls', []),
                        'copybooks': component.get('copybooks', []),
                        'cics_operations': component.get('cics_operations', []),
                        'fields': component.get('fields', []),
                    }
                    
                    # Store main component
                    self.db_manager.store_component_analysis_with_full_source(
                        session_id, 
                        main_component_data['name'], 
                        main_component_data['type'], 
                        file_name, 
                        main_component_data,
                        full_source_content=file_content  # Explicit parameter
                    )
                    
                    llm_summary = component.get('llm_summary', {})
                    if llm_summary:
                        self.db_manager.store_llm_summary(
                            session_id, 
                            component['name'], 
                            component['type'], 
                            llm_summary
                        )
                    # Store derived components separately
                    derived_components = component.get('derived_components', [])
                    if derived_components:
                        derived_components = [
                            dc if isinstance(dc, dict) else {'name': dc}
                            for dc in derived_components
                        ]
                        logger.info(f"Storing {len(derived_components)} derived components for {main_component_data['name']}")
                        self.db_manager.store_derived_components(
                            session_id, 
                            main_component_data['name'], 
                            derived_components
                        )
                    
                    # Also store record layouts as derived components
                    record_layouts = component.get('record_layouts', [])
                    if record_layouts:
                        layout_components = []
                        for layout in record_layouts:
                            layout_component = {
                                'name': layout.get('name', 'UNKNOWN_LAYOUT'),
                                'type': 'RECORD_LAYOUT',
                                'friendly_name': layout.get('friendly_name', ''),
                                'business_purpose': layout.get('business_purpose', ''),
                                'line_start': layout.get('line_start', 0),
                                'line_end': layout.get('line_end', 0),
                                'source_code': layout.get('source_code', ''),
                                'fields': layout.get('fields', [])
                            }
                            layout_components.append(layout_component)
                        
                        logger.info(f"Storing {len(layout_components)} record layouts for {main_component_data['name']}")
                        self.db_manager.store_derived_components(
                            session_id, 
                            main_component_data['name'], 
                            layout_components
                        )
                    
                    logger.info(f"Successfully stored component: {main_component_data['name']}")
                    
                except Exception as e:
                    logger.error(f"Error storing component {component.get('name', 'UNKNOWN')}: {str(e)}")
                    continue
            
            return components
            
        except Exception as e:
            logger.error(f"❌ Error extracting components from {file_name}: {str(e)}")
            logger.error(f"📍 Stack trace: {traceback.format_exc()}")
            return []
    
    def _extract_cobol_components(self, session_id: str, content: str, filename: str) -> List[Dict]:
        """Extract COBOL components with proper record layout separation"""
        logger.info(f"Starting COBOL analysis for {filename}")
        
        try:
            parsed_data = self.cobol_parser.parse_cobol_file(content, filename)
            
            # Get record-level classifications
            record_classifications = self._analyze_record_level_operations(content, parsed_data['record_layouts'])
            
            # Generate program summary
            program_summary = self._generate_component_summary(session_id, parsed_data, 'PROGRAM')
            
            program_name = filename.replace('.cbl', '').replace('.CBL', '').replace('.cob', '').replace('.COB', '')
            
            # Create main program component
            program_component = {
                'name': program_name,
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
                'derived_components': [],  # Will contain layout names
                'record_layouts': [],      # Will contain layout data
                'fields': []               # All fields from all layouts
            }
            
            components = [program_component]  # Start with main program
            
            # Process each record layout as a separate derived component
            layout_components = []
            
            for layout_idx, layout in enumerate(parsed_data['record_layouts'], 1):
                layout_name = layout.name
                record_classification = record_classifications.get(layout_name, 'STATIC')
                
                logger.info(f"Processing layout {layout_idx}: {layout_name} - Classification: {record_classification}")
                
                # Generate layout-specific summary
                layout_summary = self._generate_layout_summary(session_id, layout, parsed_data)
                
                # Create separate component for this record layout
                layout_component = {
                    'name': f"{program_name}_{layout_name}",
                    'friendly_name': layout.friendly_name or layout_summary.get('friendly_name', layout_name),
                    'type': 'RECORD_LAYOUT',
                    'parent_component': program_name,
                    'file_path': filename,
                    'content': layout.source_code,  # Only the layout's source code
                    'total_lines': layout.line_end - layout.line_start + 1,
                    'line_start': layout.line_start,
                    'line_end': layout.line_end,
                    'level': layout.level,
                    'section': layout.section,
                    'business_purpose': layout_summary.get('business_purpose', ''),
                    'record_classification': record_classification,
                    'record_usage_description': self._get_record_usage_description(record_classification),
                    'fields': [],  # Will be populated below
                    'total_fields': len(layout.fields)
                }
                
                # Process fields for this specific layout
                enhanced_fields = []
                for field in layout.fields:
                    try:
                        field_analysis = self._complete_field_source_analysis(field.name, content, program_name)
                        
                        enhanced_field = {
                            'name': field.name,
                            'friendly_name': field.friendly_name or field.name.replace('-', ' ').title(),
                            'level': field.level,
                            'picture': field.picture,
                            'usage': field.usage,
                            'line_number': field.line_number,
                            'usage_type': field_analysis['primary_usage'],
                            'business_purpose': field_analysis['business_purpose'],
                            'total_program_references': len(field_analysis['all_references']),
                            'source_field': field_analysis.get('primary_source_field', ''),
                            'source_references': field.source_references if hasattr(field, 'source_references') else [],
                            'total_references': len(field.source_references) if hasattr(field, 'source_references') else 0,
                            'confidence_score': 0.9,
                            # Record-level context
                            'record_classification': record_classification,
                            'parent_layout': layout_name,
                            'inherited_from_record': self._should_inherit_record_classification(
                                field_analysis, record_classification
                            ),
                            'effective_classification': self._get_effective_field_classification(
                                field_analysis['primary_usage'], record_classification
                            )
                        }
                        
                        enhanced_fields.append(enhanced_field)
                        
                    except Exception as field_error:
                        logger.error(f"Error analyzing field {field.name}: {str(field_error)}")
                        continue
                
                # Add fields to the layout component
                layout_component['fields'] = enhanced_fields
                
                # Add this layout as a separate component
                layout_components.append(layout_component)
                
                # Update main program component with references
                program_component['derived_components'].append(layout_name)
                program_component['record_layouts'].append({
                    'name': layout_name,
                    'friendly_name': layout.friendly_name,
                    'business_purpose': layout_summary.get('business_purpose', ''),
                    'record_classification': record_classification,
                    'record_usage_description': self._get_record_usage_description(record_classification),
                    'total_fields': len(enhanced_fields),
                    'component_reference': f"{program_name}_{layout_name}"  # Reference to separate component
                })
                
                # Add fields to main program's field list (for aggregate counts)
                program_component['fields'].extend(enhanced_fields)
                
                # Store layout in database
                try:
                    with self.db_manager.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT OR REPLACE INTO record_layouts 
                            (session_id, layout_name, friendly_name, program_name, level_number, 
                            line_start, line_end, source_code, fields_count, business_purpose,
                            record_classification, record_usage_description, has_whole_record_operations)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            session_id, 
                            layout_name, 
                            layout.friendly_name, 
                            program_name,
                            str(layout.level), 
                            layout.line_start, 
                            layout.line_end,
                            layout.source_code, 
                            len(layout.fields), 
                            layout_summary.get('business_purpose', ''),
                            record_classification,
                            self._get_record_usage_description(record_classification),
                            record_classification != 'STATIC'
                        ))
                        
                        layout_id = cursor.lastrowid
                        
                        # Store fields for this layout
                        for field_data in enhanced_fields:
                            field_data['layout_id'] = layout_id
                            self.db_manager.store_field_details_with_lengths(session_id, field_data, program_name, layout_id)
                        
                        logger.info(f"Stored layout {layout_name} with {len(enhanced_fields)} fields")
                        
                except Exception as db_error:
                    logger.error(f"Error storing layout {layout_name}: {str(db_error)}")
                    continue
            
            # Add all layout components to the result
            components.extend(layout_components)
            
            # Update main program totals
            program_component['total_fields'] = len(program_component['fields'])
            program_component['total_layouts'] = len(parsed_data['record_layouts'])
            
            # Extract and store dependencies
            self._extract_and_store_enhanced_dependencies(session_id, components, filename)
            
            logger.info(f"Analysis complete: 1 program + {len(layout_components)} layout components, {program_component['total_fields']} total fields")
            
            return components
            
        except Exception as e:
            logger.error(f"Error in COBOL component extraction: {str(e)}")
            traceback.print_exc()
            return []
        
    def _determine_field_usage_from_references(self, field) -> str:
        """Determine field usage type from comprehensive references"""
        if not hasattr(field, 'source_references') or not field.source_references:
            return 'STATIC'
        
        reference_types = [ref.get('reference_type', '') for ref in field.source_references]
        
        has_move_target = any('MOVE_TARGET' in rt for rt in reference_types)
        has_move_source = any('MOVE_SOURCE' in rt for rt in reference_types)
        has_compute = any('COMPUTE' in rt for rt in reference_types)
        has_condition = any('CONDITION' in rt for rt in reference_types)
        
        if has_move_target and has_move_source:
            return 'INPUT_OUTPUT'
        elif has_move_target or has_compute:
            return 'INPUT'
        elif has_move_source:
            return 'OUTPUT'
        elif has_compute:
            return 'DERIVED'
        elif has_condition:
            return 'REFERENCE'
        else:
            return 'STATIC'

    def _generate_field_business_purpose(self, field) -> str:
        """Generate business purpose from field name and references"""
        if hasattr(field, 'source_references') and field.source_references:
            contexts = [ref.get('business_context', '') for ref in field.source_references]
            if contexts and contexts[0]:
                return contexts[0]
        
        # Fallback to name-based inference
        return self._infer_field_business_purpose(field.name, {})

    def _extract_primary_source_field(self, field) -> str:
        """Extract primary source field from references"""
        if hasattr(field, 'source_references') and field.source_references:
            for ref in field.source_references:
                if ref.get('reference_type') == 'MOVE_TARGET' and ref.get('source_field'):
                    return ref['source_field']
        return ''

    def _field_to_dict_with_lengths(self, field, program_content: str, program_name: str = "PROGRAM") -> Dict:
        """
        MAIN field conversion method - handles all field types including valuable fillers
        Converts a field object to a detailed dictionary with proper length calculations
        """
        try:
            field_name = field.name
            
            # Special handling for valuable fillers (fillers with VALUE clause)
            if field_name.startswith('FILLER') and hasattr(field, 'value') and field.value:
                return self._process_valuable_filler(field, program_name)
            
            # Regular field processing
            return self._process_regular_field(field, program_content, program_name)
            
        except Exception as e:
            logger.error(f"Error processing field {getattr(field, 'name', 'UNKNOWN')}: {str(e)}")
            return self._create_fallback_field_dict(field, str(e))

    def _process_valuable_filler(self, field, program_name: str) -> Dict:
        """Process fillers that have VALUE clauses (constants)"""
        try:
            # Create meaningful name from value content
            value_content = field.value[:10].replace(' ', '-').replace("'", "").replace('"', '')
            meaningful_name = f"FILLER-{value_content}-L{field.level}"
            
            # Calculate length from value
            value_length = len(field.value) if field.value else 1
            
            return {
                'name': meaningful_name,
                'friendly_name': f"Constant: {field.value}",
                'level': field.level,
                'picture': field.picture or f"X({value_length})",
                'usage': field.usage or '',
                'value': field.value,
                'line_number': field.line_number,
                'code_snippet': f"{field.level:02d} FILLER PIC {field.picture or f'X({value_length})'} VALUE '{field.value}'",
                'usage_type': 'CONSTANT',
                'operation_type': 'CONSTANT_DEFINITION',
                'business_purpose': f"Constant literal value: '{field.value}'",
                'confidence': 1.0,
                'source_field': '',
                'target_field': '',
                
                # Proper field lengths for constants
                'mainframe_length': value_length,
                'oracle_length': value_length,
                'oracle_data_type': f"VARCHAR2({value_length})",
                'mainframe_data_type': f"PIC X({value_length}) VALUE '{field.value}'",
                
                # Minimal source analysis for constants
                'definition_line_number': field.line_number,
                'definition_code': f"{field.level:02d} FILLER VALUE '{field.value}'",
                'program_source_content': '',  # Not needed for constants
                'field_references_json': '[]',
                'usage_summary_json': '{"type": "constant", "value": "' + field.value + '"}',
                'total_program_references': 0,  # Constants aren't referenced
                'move_source_count': 0,
                'move_target_count': 0,
                'arithmetic_count': 0,
                'conditional_count': 0,
                'cics_count': 0
            }
        except Exception as e:
            logger.error(f"Error processing valuable filler: {str(e)}")
            return self._create_fallback_field_dict(field, str(e))

    def _process_regular_field(self, field, program_content: str, program_name: str) -> Dict:
        """Process regular fields with full source analysis"""
        try:
            field_name = field.name
            
            # Perform comprehensive source analysis
            source_analysis = self._complete_field_source_analysis(field_name, program_content, program_name)
            
            # Calculate proper field lengths
            mainframe_length, oracle_length, oracle_type = self.cobol_parser._calculate_field_lengths_fixed(
                field.picture, field.usage
            )
            
            return {
                'name': field_name,
                'friendly_name': field.friendly_name or self.cobol_parser.generate_friendly_name(field_name, 'Field'),
                'level': field.level,
                'picture': field.picture,
                'usage': field.usage,
                'occurs': field.occurs,
                'redefines': field.redefines,
                'value': getattr(field, 'value', ''),
                'line_number': field.line_number,
                'code_snippet': f"{field.level:02d} {field_name}" + (f" PIC {field.picture}" if field.picture else ""),
                'usage_type': source_analysis['primary_usage'],
                'operation_type': 'COMPREHENSIVE_DEFINITION',
                'business_purpose': source_analysis['business_purpose'],
                'confidence': 0.95,
                'source_field': source_analysis.get('primary_source_field', ''),
                'target_field': field_name if source_analysis.get('receives_data', False) else '',
                
                # Proper field length calculation
                'mainframe_length': mainframe_length,
                'oracle_length': oracle_length,
                'oracle_data_type': oracle_type,
                'mainframe_data_type': f"PIC {field.picture}" if field.picture else "UNKNOWN",
                
                # Complete source code analysis
                'definition_line_number': source_analysis.get('definition_line', field.line_number),
                'definition_code': source_analysis.get('definition_code', ''),
                'program_source_content': program_content,
                'field_references_json': json.dumps(source_analysis['all_references']),
                'usage_summary_json': json.dumps(source_analysis.get('usage_summary', {})),
                'total_program_references': len(source_analysis['all_references']),
                'move_source_count': source_analysis['counts']['move_source'],
                'move_target_count': source_analysis['counts']['move_target'],
                'arithmetic_count': source_analysis['counts']['arithmetic'],
                'conditional_count': source_analysis['counts']['conditional'],
                'cics_count': source_analysis['counts']['cics']
            }
        except Exception as e:
            logger.error(f"Error processing regular field {field.name}: {str(e)}")
            return self._create_fallback_field_dict(field, str(e))

    def _create_fallback_field_dict(self, field, error_msg: str) -> Dict:
        """Create fallback field dictionary when processing fails"""
        return {
            'name': getattr(field, 'name', 'UNKNOWN_FIELD'),
            'friendly_name': getattr(field, 'friendly_name', 'Unknown Field'),
            'level': getattr(field, 'level', 5),
            'picture': getattr(field, 'picture', ''),
            'usage': getattr(field, 'usage', ''),
            'line_number': getattr(field, 'line_number', 0),
            'usage_type': 'UNKNOWN',
            'business_purpose': 'Field processing failed',
            'mainframe_length': 1,  # Minimum valid length
            'oracle_length': 50,    # Safe default
            'oracle_data_type': 'VARCHAR2(50)',
            'mainframe_data_type': 'UNKNOWN',
            'confidence': 0.1,
            'error': error_msg,
            'field_references_json': '[]',
            'total_program_references': 0,
            'move_source_count': 0,
            'move_target_count': 0,
            'arithmetic_count': 0,
            'conditional_count': 0,
            'cics_count': 0
        }

    def generate_friendly_names_batch(self, session_id: str, items: List[Dict], context: str) -> Dict[str, str]:
        """Generate friendly names for multiple items efficiently"""
        try:
            # Use parser's method for consistent naming
            friendly_names = {}
            for item in items:
                if isinstance(item, dict):
                    name = item.get('name', item.get('copybook_name', item.get('file_name', item.get('program_name', 'UNKNOWN'))))
                else:
                    name = str(item)
                
                if name and name != 'UNKNOWN':
                    friendly_names[name] = self.cobol_parser.generate_friendly_name_enhanced(name, context, 'WEALTH_MANAGEMENT')
            
            return friendly_names
        except Exception as e:
            logger.error(f"Error generating batch friendly names: {str(e)}")
            return {}

    def validate_component_data(self, component: Dict) -> bool:
        """Validate component data before storage"""
        required_fields = ['name', 'type', 'file_path']
        
        for field in required_fields:
            if not component.get(field):
                logger.warning(f"Component missing required field: {field}")
                return False
        
        # Validate component name using parser
        if not self.cobol_parser.validate_cobol_identifier(component['name'], 'GENERAL'):
            logger.warning(f"Invalid component name: {component['name']}")
            return False
        
        return True

    def enhanced_dependency_extraction(self, session_id: str, components: List[Dict], filename: str):
        """Enhanced dependency extraction with better validation"""
        try:
            main_program = None
            for component in components:
                if component.get('type') == 'PROGRAM':
                    main_program = component
                    break
            
            if not main_program:
                return
            
            dependencies = []
            program_name = main_program['name']
            
            # Extract file dependencies with enhanced validation
            file_operations = main_program.get('file_operations', [])
            for file_op in file_operations:
                file_name = file_op.get('file_name')
                if file_name and self.cobol_parser.validate_cobol_identifier(file_name, 'FILE'):
                    # Determine I/O direction and relationship type
                    io_direction = file_op.get('io_direction', 'UNKNOWN')
                    relationship_type = self._map_io_to_relationship(io_direction)
                    
                    dependencies.append({
                        'source_component': program_name,
                        'target_component': file_name,
                        'relationship_type': relationship_type,
                        'interface_type': self._determine_interface_type(file_op),
                        'confidence_score': 0.95,
                        'analysis_details_json': json.dumps({
                            'io_direction': io_direction,
                            'operation': file_op.get('operation'),
                            'line_number': file_op.get('line_number')
                        })
                    })
            
            # Extract CICS dependencies with enhanced validation
            cics_operations = main_program.get('cics_operations', [])
            for cics_op in cics_operations:
                file_name = cics_op.get('file_name')
                if file_name and self.cobol_parser.validate_cobol_identifier(file_name, 'CICS_FILE'):
                    dependencies.append({
                        'source_component': program_name,
                        'target_component': file_name,
                        'relationship_type': 'CICS_FILE',
                        'interface_type': 'CICS',
                        'confidence_score': 0.95,
                        'analysis_details_json': json.dumps({
                            'operation': cics_op.get('operation'),
                            'line_number': cics_op.get('line_number')
                        })
                    })
            
            # Store dependencies
            if dependencies:
                self._store_validated_dependencies(session_id, dependencies)
            
        except Exception as e:
            logger.error(f"Error in enhanced dependency extraction: {str(e)}")

    def _map_io_to_relationship(self, io_direction: str) -> str:
        """Map I/O direction to relationship type"""
        mapping = {
            'INPUT': 'INPUT_FILE',
            'OUTPUT': 'OUTPUT_FILE',
            'INPUT_OUTPUT': 'INPUT_OUTPUT_FILE',
            'DECLARATION': 'FILE_DECLARATION'
        }
        return mapping.get(io_direction, 'FILE_ACCESS')

    def _determine_interface_type(self, file_op: Dict) -> str:
        """Determine interface type from file operation"""
        file_type = file_op.get('file_type', '')
        if 'CICS' in file_type:
            return 'CICS'
        elif file_op.get('operation') in ['FD', 'SELECT']:
            return 'FILE_SYSTEM'
        else:
            return 'COBOL'

    def _store_validated_dependencies(self, session_id: str, dependencies: List[Dict]):
        """Store dependencies with validation and deduplication"""
        try:
            unique_dependencies = {}
            
            for dep in dependencies:
                # Create unique key
                key = (
                    dep['source_component'],
                    dep['target_component'],
                    dep['relationship_type'],
                    dep['interface_type']
                )
                
                # Keep highest confidence entry
                if key not in unique_dependencies or dep['confidence_score'] > unique_dependencies[key]['confidence_score']:
                    unique_dependencies[key] = dep
            
            # Store in database
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                stored_count = 0
                
                for dep in unique_dependencies.values():
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO dependency_relationships 
                            (session_id, source_component, target_component, relationship_type,
                            interface_type, confidence_score, analysis_details_json, source_code_evidence)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            session_id,
                            dep['source_component'],
                            dep['target_component'],
                            dep['relationship_type'],
                            dep['interface_type'],
                            dep['confidence_score'],
                            dep['analysis_details_json'],
                            dep.get('source_code_evidence', '')
                        ))
                        stored_count += 1
                    except Exception as store_error:
                        logger.error(f"Error storing dependency: {store_error}")
                        continue
                
                logger.info(f"Stored {stored_count} validated dependencies")
                
        except Exception as e:
            logger.error(f"Error storing validated dependencies: {str(e)}")

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
        """Enhanced field source analysis with better MOVE detection"""
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
            
            for line_idx, line in enumerate(lines, 1):
                line_stripped = line.strip()
                line_upper = line_stripped.upper()
                
                if not line_stripped or line_stripped.startswith('*'):
                    continue
                
                if field_upper in line_upper:
                    operation_type = 'REFERENCE'
                    business_context = ''
                    source_field = ''
                    target_field = ''
                    
                    # Field definition
                    if ('PIC' in line_upper and 
                        re.match(r'^\s*\d{2}\s+' + re.escape(field_upper), line_upper)):
                        operation_type = 'DEFINITION'
                        analysis['definition_line'] = line_idx
                        analysis['definition_code'] = line_stripped
                        analysis['counts']['definition'] += 1
                        business_context = 'Field data structure definition'
                    
                    # Enhanced MOVE detection
                    elif 'MOVE' in line_upper:
                        # Pattern: MOVE source TO field_name
                        move_to_pattern = rf'MOVE\s+([A-Z0-9\-\(\)\'\"]+)\s+TO\s+{re.escape(field_upper)}'
                        move_to_match = re.search(move_to_pattern, line_upper)
                        
                        if move_to_match:
                            operation_type = 'MOVE_TARGET'
                            source_field = move_to_match.group(1)
                            analysis['counts']['move_target'] += 1
                            analysis['receives_data'] = True
                            if not analysis['primary_source_field']:
                                analysis['primary_source_field'] = source_field
                            business_context = f'Receives data from {source_field}'
                        else:
                            # Pattern: MOVE field_name TO target
                            move_from_pattern = rf'MOVE\s+{re.escape(field_upper)}\s+TO\s+([A-Z0-9\-\(\)\'\"]+)'
                            move_from_match = re.search(move_from_pattern, line_upper)
                            
                            if move_from_match:
                                operation_type = 'MOVE_SOURCE'
                                target_field = move_from_match.group(1)
                                analysis['counts']['move_source'] += 1
                                analysis['provides_data'] = True
                                business_context = f'Provides data to {target_field}'
                    
                    # Add other operation types...
                    elif any(op in line_upper for op in ['COMPUTE', 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE']):
                        operation_type = 'ARITHMETIC'
                        analysis['counts']['arithmetic'] += 1
                        business_context = 'Used in mathematical computation'
                    
                    elif any(op in line_upper for op in ['IF', 'WHEN', 'EVALUATE']):
                        operation_type = 'CONDITIONAL'
                        analysis['counts']['conditional'] += 1
                        business_context = 'Used in business logic decision'
                    
                    # Create reference entry
                    reference = {
                        'line_number': line_idx,
                        'line_content': line_stripped,
                        'operation_type': operation_type,
                        'business_context': business_context,
                        'source_field': source_field,
                        'target_field': target_field
                    }
                    
                    analysis['all_references'].append(reference)
            
            # Determine primary usage and business purpose
                    if analysis['receives_data'] and analysis['provides_data']:
                            analysis['primary_usage'] = 'INPUT_OUTPUT'
                    elif analysis['receives_data']:
                        analysis['primary_usage'] = 'INPUT'
                    elif analysis['provides_data']:
                        analysis['primary_usage'] = 'OUTPUT'
                    elif analysis['counts']['arithmetic'] > 0:
                        analysis['primary_usage'] = 'DERIVED'
                    elif analysis['counts']['conditional'] > 0:
                        analysis['primary_usage'] = 'REFERENCE'
                    elif len(analysis['all_references']) == 0:
                        analysis['primary_usage'] = 'STATIC'
                    else:
                        analysis['primary_usage'] = 'UNUSED'
                        
                
            # Generate business purpose
            if analysis['primary_source_field']:
                analysis['business_purpose'] = f"{field_name} - receives data from {analysis['primary_source_field']}"
            elif analysis['provides_data']:
                analysis['business_purpose'] = f"{field_name} - provides data to other fields"
            else:
                analysis['business_purpose'] = f"{field_name} - {analysis['primary_usage'].lower()} field"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in field analysis for {field_name}: {str(e)}")
            return analysis

    def _generate_component_summary(self, session_id: str, parsed_data: Dict, component_type: str) -> Dict:
        """Generate LLM summary with enhanced COBOL context"""
        try:
            # Get the actual source code for analysis
            source_content = parsed_data.get('content', '')
            
            # Extract key COBOL sections
            key_sections = self._extract_key_cobol_sections(source_content)
            
            # Build enhanced context
            context_info = {
                'type': component_type,
                'total_lines': parsed_data.get('total_lines', 0),
                'executable_lines': parsed_data.get('executable_lines', 0),
                'divisions': len(parsed_data.get('divisions', [])),
                'record_layouts': len(parsed_data.get('record_layouts', [])),
                'file_operations': parsed_data.get('file_operations', []),
                'cics_operations': parsed_data.get('cics_operations', []),
                'program_calls': parsed_data.get('program_calls', []),
                'business_comments': parsed_data.get('business_comments', [])[:10],
                'key_sections': key_sections
            }
            
            # Create enhanced prompt with actual COBOL content
            prompt = f"""
    You are analyzing a COBOL program in a wealth management/financial services system.

    PROGRAM ANALYSIS:
    - Type: {component_type}
    - Total Lines: {context_info['total_lines']} ({context_info['executable_lines']} executable)
    - Data Structures: {context_info['record_layouts']} record layouts
    - File Operations: {len(context_info['file_operations'])} operations
    - CICS Operations: {len(context_info['cics_operations'])} operations  
    - Program Calls: {len(context_info['program_calls'])} calls

    KEY PROGRAM SECTIONS:
    {self._format_key_sections(key_sections)}

    BUSINESS COMMENTS FROM SOURCE:
    {chr(10).join(context_info['business_comments'])}

    FILE OPERATIONS:
    {self._format_operations(context_info['file_operations'], 'File')}

    CICS OPERATIONS:
    {self._format_operations(context_info['cics_operations'], 'CICS')}

    PROGRAM CALLS:
    {self._format_program_calls(context_info['program_calls'])}

    Based on this COBOL program analysis, provide a JSON response describing what this program actually does in business terms:

    {{
        "business_purpose": "Specific description of what this program does in the business process (be specific about financial/wealth management functions)",
        "primary_function": "CUSTOMER_PROCESSING|ACCOUNT_MANAGEMENT|TRANSACTION_PROCESSING|PORTFOLIO_MANAGEMENT|REPORTING|BATCH_PROCESSING|ONLINE_TRANSACTION|DATA_CONVERSION",
        "complexity_score": 0.7,
        "key_features": ["specific feature 1", "specific feature 2", "specific feature 3"],
        "integration_points": ["system1", "system2"],
        "data_sources": ["file1", "file2"],
        "business_domain": "WEALTH_MANAGEMENT"
    }}

    Focus on SPECIFIC business functionality, not generic descriptions.
    """
            
            response = self.llm_client.call_llm(prompt, max_tokens=1200, temperature=0.2)  # Increased tokens, lower temperature
            
            # Log the LLM call
            self.db_manager.log_llm_call(
                session_id, 'component_summary', 1, 1,
                response.prompt_tokens, response.response_tokens, response.processing_time_ms,
                response.success, response.error_message
            )
            
            if response.success:
                summary = self.llm_client.extract_json_from_response(response.content)
                
                if isinstance(summary, dict):
                    logger.info(f"Generated specific summary: {summary.get('business_purpose', 'No purpose')[:100]}")
                    return summary
                else:
                    logger.warning(f"LLM returned non-JSON: {response.content[:200]}")
                    # Better fallback with actual analysis
                    return self._create_fallback_summary_with_analysis(context_info)
            else:
                logger.error(f"LLM call failed: {response.error_message}")
                return self._create_fallback_summary_with_analysis(context_info)
                
        except Exception as e:
            logger.error(f"Error generating component summary: {str(e)}")
            return self._create_fallback_summary_with_analysis(context_info)

    def _extract_key_cobol_sections(self, source_content: str) -> Dict:
        """Extract key sections from COBOL source for LLM analysis"""
        lines = source_content.split('\n')
        key_sections = {
            'identification': [],
            'procedure_division_start': [],
            'main_logic': [],
            'file_definitions': []
        }
        
        try:
            current_section = None
            procedure_started = False
            
            for i, line in enumerate(lines):
                line_upper = line.upper().strip()
                
                # Identification Division
                if 'PROGRAM-ID' in line_upper:
                    key_sections['identification'].append(line.strip())
                
                # File definitions
                if line_upper.startswith('FD ') or line_upper.startswith('SELECT '):
                    key_sections['file_definitions'].append(line.strip())
                
                # Procedure Division start
                if 'PROCEDURE DIVISION' in line_upper:
                    procedure_started = True
                    # Get next 10 lines of procedure division
                    for j in range(i, min(i + 10, len(lines))):
                        if lines[j].strip():
                            key_sections['procedure_division_start'].append(lines[j].strip())
                    break
            
            # Get main processing logic (look for PERFORM, IF, CALL statements)
            if procedure_started:
                for line in lines[i:i+50]:  # Look in first 50 lines of procedure division
                    line_stripped = line.strip()
                    if any(keyword in line.upper() for keyword in ['PERFORM', 'IF ', 'CALL ', 'EXEC CICS']):
                        key_sections['main_logic'].append(line_stripped)
                        if len(key_sections['main_logic']) >= 10:
                            break
            
        except Exception as e:
            logger.warning(f"Error extracting key sections: {e}")
        
        return key_sections

    def _format_key_sections(self, key_sections: Dict) -> str:
        """Format key sections for LLM prompt"""
        formatted = []
        
        if key_sections.get('identification'):
            formatted.append("IDENTIFICATION:")
            formatted.extend([f"  {line}" for line in key_sections['identification']])
        
        if key_sections.get('file_definitions'):
            formatted.append("FILE DEFINITIONS:")
            formatted.extend([f"  {line}" for line in key_sections['file_definitions'][:5]])
        
        if key_sections.get('procedure_division_start'):
            formatted.append("PROCEDURE DIVISION START:")
            formatted.extend([f"  {line}" for line in key_sections['procedure_division_start'][:8]])
        
        if key_sections.get('main_logic'):
            formatted.append("MAIN PROCESSING LOGIC:")
            formatted.extend([f"  {line}" for line in key_sections['main_logic'][:8]])
        
        return '\n'.join(formatted) if formatted else "No key sections extracted"

    def _format_operations(self, operations: List[Dict], op_type: str) -> str:
        """Format operations for LLM prompt"""
        if not operations:
            return f"No {op_type} operations found"
        
        formatted = []
        for op in operations[:5]:  # Limit to first 5
            if op_type == 'File':
                formatted.append(f"  {op.get('operation', 'UNKNOWN')}: {op.get('file_name', 'UNKNOWN')} (Line {op.get('line_number', 0)})")
            else:  # CICS
                formatted.append(f"  {op.get('operation', 'UNKNOWN')}: {op.get('file_name', 'UNKNOWN')} (Line {op.get('line_number', 0)})")
        
        return '\n'.join(formatted)

    def _format_program_calls(self, program_calls: List[Dict]) -> str:
        """Format program calls for LLM prompt"""
        if not program_calls:
            return "No program calls found"
        
        formatted = []
        for call in program_calls[:5]:
            formatted.append(f"  {call.get('operation', 'CALL')}: {call.get('program_name', 'UNKNOWN')} (Line {call.get('line_number', 0)})")
        
        return '\n'.join(formatted)

    def _create_fallback_summary_with_analysis(self, context_info: Dict) -> Dict:
        """Create intelligent fallback summary based on analysis"""
        
        # Analyze operations to determine function
        file_ops = context_info.get('file_operations', [])
        cics_ops = context_info.get('cics_operations', [])
        program_calls = context_info.get('program_calls', [])
        
        primary_function = 'GENERAL_PROCESSING'
        key_features = []
        
        if len(cics_ops) > len(file_ops):
            primary_function = 'ONLINE_TRANSACTION'
            key_features.append(f"{len(cics_ops)} CICS operations")
        elif len(file_ops) > 3:
            primary_function = 'BATCH_PROCESSING'  
            key_features.append(f"{len(file_ops)} file operations")
        
        if program_calls:
            key_features.append(f"Calls {len(program_calls)} other programs")
        
        # Generate business purpose based on operations
        business_purpose = f"COBOL program with {context_info.get('executable_lines', 0)} executable lines"
        if file_ops:
            business_purpose += f", processes {len(file_ops)} files"
        if cics_ops:
            business_purpose += f", {len(cics_ops)} CICS transactions"
        
        return {
            'business_purpose': business_purpose,
            'primary_function': primary_function,
            'complexity_score': min(0.9, context_info.get('executable_lines', 0) / 1000),
            'key_features': key_features,
            'integration_points': [op.get('file_name', '') for op in file_ops[:3]],
            'data_sources': [op.get('file_name', '') for op in (file_ops + cics_ops)[:3]],
            'business_domain': 'WEALTH_MANAGEMENT',
            'analysis_method': 'fallback_with_analysis'
        }
    
    def _analyze_record_level_operations(self, program_content: str, record_layouts: List) -> Dict[str, str]:
        """
        Analyze record-level operations and classify 01-level layouts as INPUT, OUTPUT, or INPUT_OUTPUT
        when whole record operations are detected.
        """
        record_classifications = {}
        
        lines = program_content.split('\n')
        
        for layout in record_layouts:
            layout_name = layout.name
            classification = self._classify_record_layout(layout_name, lines)
            record_classifications[layout_name] = classification
        
        return record_classifications

    def _classify_record_layout(self, layout_name: str, program_lines: List[str]) -> str:
        """
        Classify a record layout based on whole-record operations
        """
        import re
        
        receives_data = False
        provides_data = False
        
        # Patterns for whole record operations
        record_patterns = {
            'input': [
                rf'MOVE\s+([A-Z0-9\-]+)\s+TO\s+{re.escape(layout_name)}',
                rf'READ\s+[A-Z0-9\-]+\s+INTO\s+{re.escape(layout_name)}',
                rf'EXEC\s+CICS\s+READ[^E]*INTO\s*\(\s*{re.escape(layout_name)}\s*\)',
                rf'ACCEPT\s+{re.escape(layout_name)}',
                rf'UNSTRING\s+[^I]*INTO\s+{re.escape(layout_name)}',
            ],
            'output': [
                rf'MOVE\s+{re.escape(layout_name)}\s+TO\s+([A-Z0-9\-]+)',
                rf'WRITE\s+[A-Z0-9\-]+\s+FROM\s+{re.escape(layout_name)}',
                rf'EXEC\s+CICS\s+WRITE[^E]*FROM\s*\(\s*{re.escape(layout_name)}\s*\)',
                rf'DISPLAY\s+{re.escape(layout_name)}',
                rf'STRING\s+{re.escape(layout_name)}[^I]*INTO',
            ],
            'bidirectional': [
                rf'EXEC\s+CICS\s+REWRITE[^E]*FROM\s*\(\s*{re.escape(layout_name)}\s*\)',
                rf'REWRITE\s+[A-Z0-9\-]+\s+FROM\s+{re.escape(layout_name)}',
            ]
        }
        
        for line in program_lines:
            line_upper = line.strip().upper()
            
            # Check for bidirectional operations first
            for pattern in record_patterns['bidirectional']:
                if re.search(pattern, line_upper, re.IGNORECASE):
                    return 'INPUT_OUTPUT'
            
            # Check for input operations
            for pattern in record_patterns['input']:
                if re.search(pattern, line_upper, re.IGNORECASE):
                    receives_data = True
                    break
                    
            # Check for output operations  
            for pattern in record_patterns['output']:
                if re.search(pattern, line_upper, re.IGNORECASE):
                    provides_data = True
                    break
        
        # Determine final classification
        if receives_data and provides_data:
            return 'INPUT_OUTPUT'
        elif receives_data:
            return 'INPUT'
        elif provides_data:
            return 'OUTPUT'
        else:
            return 'STATIC'

    def _get_record_usage_description(self, classification: str) -> str:
        """Get description for record-level usage"""
        descriptions = {
            'INPUT': 'Record receives data as a complete unit (READ INTO, MOVE TO record)',
            'OUTPUT': 'Record provides data as a complete unit (WRITE FROM, MOVE record TO)',
            'INPUT_OUTPUT': 'Record used in bidirectional operations (REWRITE, read-modify-write patterns)',
            'STATIC': 'Record defined but no whole-record operations detected'
        }
        return descriptions.get(classification, 'Unknown record usage pattern')

    def _should_inherit_record_classification(self, field_analysis: Dict, record_classification: str) -> bool:
        """
        Determine if a field should inherit the record-level classification
        """
        total_field_operations = (
            field_analysis['counts']['move_source'] +
            field_analysis['counts']['move_target'] +
            field_analysis['counts']['arithmetic'] +
            field_analysis['counts']['conditional']
        )
        
        return (total_field_operations <= 1 and record_classification != 'STATIC')

    def _get_effective_field_classification(self, field_classification: str, record_classification: str) -> str:
        """
        Get the effective classification considering both field-level and record-level analysis
        """
        if field_classification in ['INPUT_OUTPUT', 'DERIVED']:
            return field_classification
        
        if record_classification != 'STATIC' and field_classification in ['STATIC', 'REFERENCE']:
            return f"{record_classification}_INHERITED"
        
        return field_classification
    
    def generate_component_friendly_names(self, session_id: str, components: List[Dict]) -> List[Dict]:
        """Generate friendly names for components using LLM"""
        
        try:
            # Separate different component types
            programs = [c for c in components if c.get('type') == 'PROGRAM']
            record_layouts = [c for c in components if c.get('type') == 'RECORD_LAYOUT']
            files = [c for c in components if c.get('type') in ['FILE', 'CICS_FILE']]
            
            # Generate friendly names by type
            enhanced_components = []
            
            # Programs
            if programs:
                program_names = self.cobol_parser.generate_batch_friendly_names(
                    programs, 'PROGRAM', 'WEALTH_MANAGEMENT', session_id
                )
                for component in programs:
                    name = self._get_component_name(component)
                    component['friendly_name'] = program_names.get(name, 
                        self.cobol_parser._generate_simple_friendly_name(name, 'Program'))
                    enhanced_components.append(component)
            
            # Record Layouts - with additional context
            if record_layouts:
                layout_names = {}
                for layout in record_layouts:
                    name = self._get_component_name(layout)
                    # Add field information as context for better naming
                    field_info = f"Fields: {len(layout.get('fields', []))}"
                    source_snippet = layout.get('source_code', '')[:200]
                    
                    friendly_name = self.cobol_parser.generate_business_friendly_name(
                        name, 'RECORD_LAYOUT', 'WEALTH_MANAGEMENT', 
                        f"{field_info}\n{source_snippet}", session_id
                    )
                    layout['friendly_name'] = friendly_name
                    enhanced_components.append(layout)
            
            # Files
            if files:
                file_names = self.cobol_parser.generate_batch_friendly_names(
                    files, 'FILE', 'WEALTH_MANAGEMENT', session_id
                )
                for component in files:
                    name = self._get_component_name(component)
                    component['friendly_name'] = file_names.get(name,
                        self.cobol_parser._generate_simple_friendly_name(name, 'File'))
                    enhanced_components.append(component)
            
            # Add any remaining components
            for component in components:
                if component not in enhanced_components:
                    name = self._get_component_name(component)
                    component['friendly_name'] = self.cobol_parser.generate_business_friendly_name(
                        name, component.get('type', 'Component'), 'WEALTH_MANAGEMENT', '', session_id
                    )
                    enhanced_components.append(component)
            
            return enhanced_components
            
        except Exception as e:
            logger.error(f"Error generating component friendly names: {str(e)}")
            
            # Fallback: use simple friendly names
            for component in components:
                name = self._get_component_name(component)
                component['friendly_name'] = self.cobol_parser._generate_simple_friendly_name(
                    name, component.get('type', 'Component')
                )
            
            return components
    
    def _get_component_name(self, component) -> str:
        """Extract component name consistently"""
        if isinstance(component, dict):
            return component.get('name', component.get('layout_name', 
                   component.get('file_name', 'UNKNOWN')))
        return str(component)

    def enhance_record_layout_with_llm_naming(self, session_id: str, layout, program_context: str) -> Dict:
        """Enhanced record layout analysis with LLM-generated friendly names"""
        try:
            field_count = len(layout.fields) if hasattr(layout, 'fields') else 0
            layout_name = layout.name if hasattr(layout, 'name') else str(layout)
            
            # Prepare context for LLM
            field_names = []
            field_types = {}
            
            if hasattr(layout, 'fields'):
                field_names = [f.name for f in layout.fields[:10]]  # First 10 fields
                for field in layout.fields:
                    if hasattr(field, 'picture') and field.picture:
                        if 'X' in field.picture:
                            field_types['alphanumeric'] = field_types.get('alphanumeric', 0) + 1
                        elif '9' in field.picture:
                            field_types['numeric'] = field_types.get('numeric', 0) + 1
            
            # Generate business-friendly name using LLM
            context_info = f"""
Layout: {layout_name}
Fields ({field_count}): {', '.join(field_names)}
Field Types: {', '.join([f'{count} {ftype}' for ftype, count in field_types.items()])}
Program Context: {program_context}
"""
            
            friendly_name = self.cobol_parser.generate_business_friendly_name(
                layout_name, 'RECORD_LAYOUT', 'WEALTH_MANAGEMENT', context_info, session_id
            )
            
            # Generate business purpose using LLM
            business_purpose = self._generate_layout_business_purpose(
                session_id, layout_name, friendly_name, field_names, field_types, program_context
            )
            
            return {
                'friendly_name': friendly_name,
                'business_purpose': business_purpose,
                'usage_pattern': self._infer_usage_pattern(field_names, field_types),
                'business_domain': 'WEALTH_MANAGEMENT',
                'field_analysis': field_types,
                'complexity_score': min(0.9, field_count / 50)
            }
            
        except Exception as e:
            logger.error(f"Error in LLM record layout enhancement: {str(e)}")
            return {
                'friendly_name': self.cobol_parser._generate_simple_friendly_name(layout_name, 'Record Layout'),
                'business_purpose': f"Data structure with {field_count} fields",
                'usage_pattern': 'WORKING_STORAGE',
                'business_domain': 'WEALTH_MANAGEMENT',
                'field_analysis': field_types,
                'complexity_score': 0.5
            }

    def _generate_layout_business_purpose(self, session_id: str, layout_name: str, 
                                        friendly_name: str, field_names: List[str], 
                                        field_types: Dict, program_context: str) -> str:
        """Generate business purpose for record layout using LLM"""
        try:
            prompt = f"""
Analyze this COBOL record layout and describe its business purpose in 1-2 concise sentences.

Record Layout: {layout_name}
Business Name: {friendly_name}
Program: {program_context}
Field Count: {len(field_names)}
Field Types: {field_types}
Sample Fields: {', '.join(field_names[:8])}

This is for a wealth management system. Focus on:
- What business data this record represents
- How it's used in wealth management operations
- Keep it business-focused, not technical

Return only the business purpose description (1-2 sentences).
"""
            
            response = self.llm_client.call_llm(prompt, max_tokens=150, temperature=0.3)
            
            if response.success and response.content:
                purpose = response.content.strip()
                # Clean up the response
                purpose = purpose.replace('\n', ' ').strip()
                if len(purpose) > 200:
                    purpose = purpose[:197] + "..."
                return purpose
            
        except Exception as e:
            logger.warning(f"LLM business purpose generation failed: {str(e)}")
        
        # Fallback
        return f"{friendly_name} - Data structure containing {len(field_names)} fields for {program_context.lower()}"
    
    def _infer_usage_pattern(self, field_names: List[str], field_types: Dict) -> str:
        """Infer usage pattern from field characteristics"""
        name_patterns = ' '.join(field_names).upper()
        
        if any(pattern in name_patterns for pattern in ['INPUT', 'IN-', 'FROM']):
            return 'INPUT'
        elif any(pattern in name_patterns for pattern in ['OUTPUT', 'OUT-', 'TO']):
            return 'OUTPUT'
        elif any(pattern in name_patterns for pattern in ['PARM', 'PARAMETER']):
            return 'PARAMETER'
        else:
            return 'WORKING_STORAGE'

    def _generate_layout_summary(self, session_id: str, layout, parsed_data: Dict) -> Dict:
        """Generate LLM summary for record layout with friendly name"""
        try:
            field_count = len(layout.fields)
            field_types = {}
            
            # Analyze field types
            for field in layout.fields:
                if 'X' in field.picture:
                    field_types['alphanumeric'] = field_types.get('alphanumeric', 0) + 1
                elif '9' in field.picture:
                    field_types['numeric'] = field_types.get('numeric', 0) + 1
            
            # Create LLM prompt for layout analysis
            field_names = [f.name for f in layout.fields[:10]]  # First 10 fields
            
            prompt = f"""
    Analyze this COBOL record layout and provide business context.

    Layout: {layout.name}
    Fields ({field_count}): {', '.join(field_names)}
    Field Types: {', '.join([f'{count} {type}' for type, count in field_types.items()])}

    Provide JSON response:
    {{
        "friendly_name": "Business-friendly name for this record layout with respect to wealth management",
        "business_purpose": "What this record structure represents in business terms",
        "usage_pattern": "How this record is typically used (INPUT|OUTPUT|WORKING_STORAGE|PARAMETER)",
        "business_domain": "Business area (CUSTOMER|ACCOUNT|TRANSACTION|EMPLOYEE|GENERAL)"
    }}
    """
            
            response = self.llm_client.call_llm(prompt, max_tokens=400, temperature=0.3)
            
            # Log LLM call
            self.db_manager.log_llm_call(
                session_id, 'layout_summary', 1, 1,
                response.prompt_tokens, response.response_tokens, response.processing_time_ms,
                response.success, response.error_message
            )
            
            if response.success:
                llm_summary = self.llm_client.extract_json_from_response(response.content)
                
                # Defensive handling: normalize string outputs to dict where possible
                if isinstance(llm_summary, str):
                    try:
                        parsed = json.loads(llm_summary)
                        if isinstance(parsed, dict):
                            llm_summary = parsed
                        else:
                            logger.warning('Layout LLM returned a string that JSON-parsed to non-dict; ignoring parsed value')
                            llm_summary = None
                    except Exception:
                        logger.warning('Layout LLM returned a non-JSON string; treating as no structured result')
                        llm_summary = None

                if isinstance(llm_summary, dict):
                    return {
                        'friendly_name': llm_summary.get('friendly_name', layout.friendly_name),
                        'business_purpose': llm_summary.get('business_purpose', f"Data structure with {field_count} fields"),
                        'usage_pattern': llm_summary.get('usage_pattern', 'WORKING_STORAGE'),
                        'business_domain': llm_summary.get('business_domain', 'GENERAL'),
                        'field_analysis': field_types,
                        'complexity_score': min(0.9, field_count / 50)
                    }
                else:
                    # Preserve raw text when structured JSON is not available
                    if response.content and isinstance(response.content, str) and len(response.content.strip()) > 0:
                        raw_text = response.content.strip()
                        logger.info('Using raw LLM text as fallback layout summary (stored under business_purpose)')
                        return {
                            'friendly_name': layout.friendly_name,
                            'business_purpose': raw_text.split('\n')[0][:500],
                            'usage_pattern': 'WORKING_STORAGE',
                            'business_domain': 'GENERAL',
                            'field_analysis': field_types,
                            'complexity_score': min(0.9, field_count / 50),
                            'raw': raw_text
                        }
            
            # Fallback
            return {
                'friendly_name': layout.friendly_name,
                'business_purpose': f"Data structure with {field_count} fields",
                'usage_pattern': 'WORKING_STORAGE',
                'business_domain': 'GENERAL',
                'field_analysis': field_types,
                'complexity_score': min(0.9, field_count / 50)
            }
            
        except Exception as e:
            logger.error(f"Error generating layout summary: {str(e)}")
            return {
                'friendly_name': layout.friendly_name,
                'business_purpose': 'Analysis failed',
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
                    'fields': [self._field_to_dict_with_lengths(field, content) for field in layout.fields]  # NEW METHOD
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
                    'fields': [self._field_to_dict_with_lengths(field, content) for field in layout.fields]  # NEW METHOD
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
    
    
    def _calculate_field_lengths(self, picture: str, usage: str = "") -> Tuple[int, int, str]:
        """Fixed field length calculation from PIC clause"""
        if not picture:
            return 0, 50, "VARCHAR2(50)"
        
        pic_upper = picture.upper().strip()
        mainframe_length = 0
        oracle_length = 50
        oracle_type = "VARCHAR2(50)"
        
        try:
            # Numeric fields
            if re.search(r'[9S]', pic_upper):
                total_digits = 0
                decimal_digits = 0
                
                # Handle parentheses notation like 9(5) or S9(7)
                paren_matches = re.findall(r'[9S]\((\d+)\)', pic_upper)
                for match in paren_matches:
                    total_digits += int(match)
                
                # Handle explicit 9s like 99999
                explicit_nines = len(re.findall(r'9', re.sub(r'9\(\d+\)', '', pic_upper)))
                total_digits += explicit_nines
                
                # Handle decimal point (V)
                if 'V' in pic_upper:
                    parts = pic_upper.split('V')
                    if len(parts) > 1:
                        decimal_part = parts[1]
                        # Count decimal digits
                        decimal_paren = re.findall(r'9\((\d+)\)', decimal_part)
                        for match in decimal_paren:
                            decimal_digits += int(match)
                        decimal_explicit = len(re.findall(r'9', re.sub(r'9\(\d+\)', '', decimal_part)))
                        decimal_digits += decimal_explicit
                
                # Calculate mainframe storage
                if usage.upper() in ['COMP-3', 'PACKED-DECIMAL']:
                    mainframe_length = (total_digits + 1) // 2 + 1
                elif usage.upper() in ['COMP', 'BINARY']:
                    if total_digits <= 4:
                        mainframe_length = 2
                    elif total_digits <= 9:
                        mainframe_length = 4
                    else:
                        mainframe_length = 8
                else:
                    mainframe_length = total_digits + (1 if 'S' in pic_upper else 0)
                
                # Oracle type
                if decimal_digits > 0:
                    oracle_type = f"NUMBER({total_digits},{decimal_digits})"
                    oracle_length = total_digits
                else:
                    oracle_type = f"NUMBER({total_digits})"
                    oracle_length = total_digits
            
            # Alphanumeric fields
            elif re.search(r'[XA]', pic_upper):
                # Handle X(n) notation
                paren_matches = re.findall(r'X\((\d+)\)', pic_upper)
                if paren_matches:
                    mainframe_length = sum(int(match) for match in paren_matches)
                else:
                    # Count explicit Xs
                    mainframe_length = len(re.findall(r'X', pic_upper))
                
                oracle_length = mainframe_length
                if oracle_length <= 4000:
                    oracle_type = f"VARCHAR2({oracle_length})"
                else:
                    oracle_type = "CLOB"
            
            # Default for unknown patterns
            else:
                mainframe_length = 10
                oracle_length = 50
                oracle_type = "VARCHAR2(50)"
                
        except Exception as e:
            logger.warning(f"Error calculating field lengths for PIC {picture}: {str(e)}")
            mainframe_length = 10
            oracle_length = 50
            oracle_type = "VARCHAR2(50)"
        
        return mainframe_length, oracle_length, oracle_type


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
Analyze this COBOL program and provide enhanced component analysis with respect to wealth management.
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
    
    def _extract_and_store_enhanced_dependencies(self, session_id: str, components: List[Dict], filename: str):
        """Enhanced dependency extraction with dynamic call resolution and missing detection"""
        try:
            main_program = None
            for component in components:
                if component.get('type') == 'PROGRAM':
                    main_program = component
                    break
            
            logger.info(f"DEPENDENCY DEBUG: Processing program: {main_program['name']}")
        
            dependencies = []
            program_name = main_program['name']
            
            # Get all uploaded components for missing dependency detection
            all_session_components = self.db_manager.get_session_components(session_id)
            uploaded_programs = set(comp['component_name'].upper() for comp in all_session_components)
            
            # Enhanced program calls with dynamic resolution
            program_calls = main_program.get('program_calls', [])
            
            for call in program_calls:
                call_type = call.get('call_type', 'static')
                
                if call_type == 'dynamic':
                    # Handle dynamic calls with multiple possible targets
                    dependencies.extend(self._create_dynamic_call_dependencies(
                        session_id, program_name, call, uploaded_programs
                    ))
                else:
                    # Handle static calls
                    target_prog = call.get('program_name')
                    if target_prog and self._is_valid_dependency_target(target_prog, program_name):
                        dependency = self._create_program_call_dependency(
                            program_name, call, uploaded_programs
                        )
                        if dependency:
                            dependencies.append(dependency)
            
            # File dependencies (unchanged logic)
            file_operations = main_program.get('file_operations', [])
            file_classifications = self._classify_file_io_direction_enhanced(file_operations)
            processed_files = set()
            
            for file_op in file_operations:
                file_name = file_op.get('file_name')
                if file_name and file_name not in processed_files and self._is_valid_dependency_target(file_name, program_name):
                    processed_files.add(file_name)
                    io_direction = file_classifications.get(file_name, file_op.get('io_direction', 'UNKNOWN'))
                    
                    dependencies.append({
                        'source_component': program_name,
                        'target_component': file_name,
                        'relationship_type': self._map_io_to_relationship_type(io_direction),
                        'interface_type': 'FILE_SYSTEM',
                        'confidence_score': 0.95,
                        'dependency_status': 'file',  # Files don't have upload status
                        'analysis_details_json': json.dumps({
                            'io_direction': io_direction,
                            'operations': [op.get('operation') for op in file_operations if op.get('file_name') == file_name]
                        })
                    })
            
            # CICS dependencies with I/O classification
            cics_operations = main_program.get('cics_operations', [])
            processed_cics_files = set()
            
            for cics_op in cics_operations:
                file_name = cics_op.get('file_name')
                if file_name and file_name not in processed_cics_files and self._is_valid_dependency_target(file_name, program_name):
                    processed_cics_files.add(file_name)
                    
                    cics_file_ops = [op for op in cics_operations if op.get('file_name') == file_name]
                    io_classification = self._classify_cics_io_operations(cics_file_ops)
                    
                    dependencies.append({
                        'source_component': program_name,
                        'target_component': file_name,
                        'relationship_type': io_classification['relationship_type'],
                        'interface_type': 'CICS',
                        'confidence_score': 0.95,
                        'dependency_status': 'cics_file',  # CICS files don't have upload status
                        'analysis_details_json': json.dumps({
                            'io_direction': io_classification['io_direction'],
                            'operations': [op.get('operation') for op in cics_file_ops]
                        })
                    })
            
             # After storing file dependencies, store file-layout associations
            file_layout_associations = self.db_manager.get_file_record_layout_associations(session_id)

            for file_name, layout_info in file_layout_associations.items():
                # Store as a special dependency type
                dependencies.append({
                    'source_component': layout_info['program_name'],
                    'target_component': f"{file_name}::{layout_info['layout_name']}",
                    'relationship_type': f"FILE_LAYOUT_{layout_info['io_type']}",
                    'interface_type': 'RECORD_LAYOUT',
                    'confidence_score': 0.95,
                    'dependency_status': 'layout_association',
                    'analysis_details_json': json.dumps({
                        'file_name': file_name,
                        'layout_name': layout_info['layout_name'],
                        'io_type': layout_info['io_type'],
                        'association_method': layout_info['method']
                    })
                })

            logger.info(f"DEPENDENCY DEBUG: Found {len(dependencies)} dependencies to store")
            for dep in dependencies[:3]:  # Log first 3 for debugging
                logger.info(f"DEPENDENCY DEBUG: Sample dependency: {dep['source_component']} -> {dep['target_component']} ({dep['relationship_type']})")

            # Store enhanced dependencies
            if dependencies:
                self.db_manager._store_enhanced_dependencies_with_status(session_id, dependencies)
                logger.info(f"Stored {len(dependencies)} enhanced dependencies for {program_name}")

           

        except Exception as e:
            logger.error(f"Error in enhanced dependency extraction: {str(e)}")


    def _get_file_record_layout_association(self, session_id: str, file_name: str, program_name: str) -> Optional[Dict]:
        """Get record layout associated with a specific file"""
        try:
            associations = self.db_manager.get_file_record_layout_associations(session_id)
            
            if file_name in associations:
                layout_info = associations[file_name]
                
                # Get the actual fields from the layout
                layout_fields = self.db_manager.get_field_matrix(
                    session_id, record_layout=layout_info['layout_name']
                )
                
                return {
                    'layout_name': layout_info['layout_name'],
                    'io_type': layout_info['io_type'],
                    'fields': layout_fields,
                    'program_name': layout_info['program_name']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting file-layout association for {file_name}: {str(e)}")
            return None


    def _create_dynamic_call_dependencies(self, session_id: str, program_name: str, 
                                   dynamic_call: Dict, uploaded_programs: set) -> List[Dict]:
        """Create dependencies for dynamic calls with multiple possible targets"""
        dependencies = []
        variable_name = dynamic_call.get('variable_name', 'UNKNOWN')
        
        # Get resolved programs from the dynamic call analysis
        resolved_programs = dynamic_call.get('resolved_programs', [])
        if not resolved_programs:
            resolved_programs = [{'program_name': variable_name, 'resolution': 'unresolved', 'confidence': 0.1}]
        
        for resolved_program in resolved_programs:
            target_prog = resolved_program.get('program_name')
            if target_prog and self._is_valid_dependency_target(target_prog, program_name):
                
                # Check if target program is uploaded
                dependency_status = 'present' if target_prog.upper() in uploaded_programs else 'missing'
                
                dependency = {
                    'source_component': program_name,
                    'target_component': target_prog,
                    'relationship_type': 'DYNAMIC_PROGRAM_CALL',
                    'interface_type': 'CICS',
                    'confidence_score': resolved_program.get('confidence', 0.5),
                    'dependency_status': dependency_status,
                    'analysis_details_json': json.dumps({
                        'call_type': 'dynamic',
                        'variable_name': variable_name,
                        'resolution_method': resolved_program.get('resolution', 'unknown'),
                        'source_info': resolved_program.get('source', ''),
                        'line_number': dynamic_call.get('line_number', 0),
                        'business_context': f"Dynamic call via {variable_name} variable"
                    }),
                    'source_code_evidence': f"Line {dynamic_call.get('line_number', 0)}: {dynamic_call.get('operation')} PROGRAM({variable_name}) -> {target_prog}"
                }
                dependencies.append(dependency)
        
        return dependencies

    def _create_program_call_dependency(self, program_name: str, call: Dict, uploaded_programs: set) -> Optional[Dict]:
        """Create dependency for static program call with missing detection"""
        target_prog = call.get('program_name')
        if not target_prog:
            return None
        
        # Check if target program is uploaded
        dependency_status = 'present' if target_prog.upper() in uploaded_programs else 'missing'
        
        return {
            'source_component': program_name,
            'target_component': target_prog,
            'relationship_type': 'PROGRAM_CALL',
            'interface_type': 'COBOL',
            'confidence_score': call.get('confidence_score', 0.98),
            'dependency_status': dependency_status,
            'analysis_details_json': json.dumps({
                'call_type': call.get('call_type', 'static'),
                'line_number': call.get('line_number', 0),
                'business_context': call.get('business_context', 'Program call')
            }),
            'source_code_evidence': f"Line {call.get('line_number', 0)}: {call.get('operation', 'CALL')} {target_prog}"
        }

    def _store_enhanced_dependencies_with_status(self, session_id: str, dependencies: List[Dict]):
        """Store dependencies with enhanced status tracking"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Add dependency_status column if it doesn't exist
                cursor.execute('''
                    ALTER TABLE dependency_relationships 
                    ADD COLUMN dependency_status TEXT DEFAULT 'unknown'
                ''')
                
                for dep in dependencies:
                    try:
                        # Check for existing dependency
                        cursor.execute('''
                            SELECT id, confidence_score FROM dependency_relationships
                            WHERE session_id = ? AND source_component = ? AND target_component = ? 
                                AND relationship_type = ?
                        ''', (session_id, dep['source_component'], dep['target_component'], dep['relationship_type']))
                        
                        existing = cursor.fetchone()
                        
                        if existing:
                            # Update existing with enhanced information
                            cursor.execute('''
                                UPDATE dependency_relationships
                                SET confidence_score = ?, analysis_details_json = ?, 
                                    source_code_evidence = ?, dependency_status = ?
                                WHERE id = ?
                            ''', (
                                max(existing[1] or 0, dep.get('confidence_score', 0)),
                                dep.get('analysis_details_json', '{}'),
                                dep.get('source_code_evidence', ''),
                                dep.get('dependency_status', 'unknown'),
                                existing[0]
                            ))
                        else:
                            # Insert new dependency
                            cursor.execute('''
                                INSERT INTO dependency_relationships 
                                (session_id, source_component, target_component, relationship_type,
                                interface_type, confidence_score, analysis_details_json, 
                                source_code_evidence, dependency_status)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                session_id, dep['source_component'], dep['target_component'], 
                                dep['relationship_type'], dep.get('interface_type', ''),
                                dep.get('confidence_score', 0.0), dep.get('analysis_details_json', '{}'),
                                dep.get('source_code_evidence', ''), dep.get('dependency_status', 'unknown')
                            ))
                            
                    except sqlite3.OperationalError as e:
                        if "duplicate column" in str(e).lower():
                            # Column already exists, continue with insert/update
                            pass
                        else:
                            raise
                            
                    except Exception as store_error:
                        logger.error(f"Error storing enhanced dependency: {store_error}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error in enhanced dependency storage: {str(e)}")

    def _map_io_to_relationship_type(self, io_direction: str) -> str:
        """Map I/O direction to relationship type"""
        mapping = {
            'INPUT': 'INPUT_FILE',
            'OUTPUT': 'OUTPUT_FILE', 
            'INPUT_OUTPUT': 'INPUT_OUTPUT_FILE',
            'UNKNOWN': 'FILE_ACCESS'
        }
        return mapping.get(io_direction, 'FILE_ACCESS')

    
    
    def _classify_cics_io_operations(self, cics_ops: List[Dict]) -> Dict:
        """Classify CICS operations to determine I/O direction"""
        operations = [op.get('operation', '').upper() for op in cics_ops]
        
        has_read = any('READ' in op for op in operations)
        has_write = any('WRITE' in op for op in operations)
        has_rewrite = any('REWRITE' in op for op in operations)
        has_delete = any('DELETE' in op for op in operations)
        
        if has_rewrite or (has_read and has_write):
            return {
                'relationship_type': 'CICS_INPUT_OUTPUT_FILE',
                'io_direction': 'INPUT_OUTPUT'
            }
        elif has_read or has_delete:  # DELETE typically reads first
            return {
                'relationship_type': 'CICS_INPUT_FILE', 
                'io_direction': 'INPUT'
            }
        elif has_write:
            return {
                'relationship_type': 'CICS_OUTPUT_FILE',
                'io_direction': 'OUTPUT'
            }
        else:
            return {
                'relationship_type': 'CICS_FILE',
                'io_direction': 'UNKNOWN'
            }
        
    def _classify_file_io_direction_enhanced(self, file_operations: List[Dict]) -> Dict[str, str]:
        """Enhanced file I/O direction classification"""
        file_directions = {}
        
        for op in file_operations:
            file_name = op.get('file_name')
            operation = op.get('operation', '').upper()
            
            if not file_name:
                continue
                
            if file_name not in file_directions:
                file_directions[file_name] = set()
            
            # More precise operation mapping
            if operation in ['READ', 'OPEN INPUT', 'SELECT INPUT']:
                file_directions[file_name].add('INPUT')
            elif operation in ['WRITE', 'REWRITE', 'OPEN OUTPUT', 'OPEN EXTEND']:
                file_directions[file_name].add('OUTPUT')
            elif operation in ['OPEN I-O', 'OPEN IO']:
                file_directions[file_name].add('INPUT_OUTPUT')
        
        # Determine final classification
        classifications = {}
        for file_name, directions in file_directions.items():
            if 'INPUT_OUTPUT' in directions or ('INPUT' in directions and 'OUTPUT' in directions):
                classifications[file_name] = 'INPUT_OUTPUT'
            elif 'INPUT' in directions:
                classifications[file_name] = 'INPUT'
            elif 'OUTPUT' in directions:
                classifications[file_name] = 'OUTPUT'
            else:
                classifications[file_name] = 'UNKNOWN'
        
        return classifications

    def _classify_file_io_direction(self, file_operations: List[Dict]) -> Dict[str, str]:
        """Classify files as INPUT, OUTPUT, or INPUT_OUTPUT based on all operations"""
        file_directions = {}
        
        for op in file_operations:
            file_name = op.get('file_name')
            io_direction = op.get('io_direction')
            
            if not file_name or not io_direction or io_direction in ['DECLARATION', 'NEUTRAL', 'UNKNOWN']:
                continue
            
            if file_name not in file_directions:
                file_directions[file_name] = set()
            
            file_directions[file_name].add(io_direction)
        
        # Determine final classification
        classifications = {}
        for file_name, directions in file_directions.items():
            if 'INPUT' in directions and 'OUTPUT' in directions:
                classifications[file_name] = 'INPUT_OUTPUT'
            elif 'INPUT_OUTPUT' in directions:
                classifications[file_name] = 'INPUT_OUTPUT'
            elif 'INPUT' in directions:
                classifications[file_name] = 'INPUT'
            elif 'OUTPUT' in directions:
                classifications[file_name] = 'OUTPUT'
            else:
                classifications[file_name] = 'UNKNOWN'
        
        return classifications

    def _store_dependencies_with_deduplication(self, session_id: str, dependencies: List[Dict]):
        """Store dependencies with enhanced deduplication logic"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                for dep in dependencies:
                    # Check for existing dependency
                    cursor.execute('''
                        SELECT id, confidence_score, analysis_details_json 
                        FROM dependency_relationships
                        WHERE session_id = ? AND source_component = ? AND target_component = ? AND relationship_type = ?
                    ''', (
                        session_id, 
                        dep['source_component'], 
                        dep['target_component'], 
                        dep['relationship_type']
                    ))
                    
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing with better information
                        existing_id, existing_conf, existing_details = existing
                        new_conf = max(existing_conf or 0, dep.get('confidence_score', 0))
                        
                        # Merge analysis details
                        merged_details = self._merge_analysis_details(
                            existing_details, 
                            dep.get('analysis_details_json', '{}')
                        )
                        
                        cursor.execute('''
                            UPDATE dependency_relationships
                            SET confidence_score = ?, analysis_details_json = ?, 
                                source_code_evidence = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        ''', (
                            new_conf, 
                            merged_details, 
                            dep.get('source_code_evidence', ''), 
                            existing_id
                        ))
                    else:
                        # Insert new dependency
                        cursor.execute('''
                            INSERT INTO dependency_relationships 
                            (session_id, source_component, target_component, relationship_type,
                            interface_type, confidence_score, analysis_details_json, source_code_evidence)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            session_id, 
                            dep['source_component'], 
                            dep['target_component'], 
                            dep['relationship_type'],
                            dep.get('interface_type', ''), 
                            dep.get('confidence_score', 0.0), 
                            dep.get('analysis_details_json', '{}'), 
                            dep.get('source_code_evidence', '')
                        ))
                        
        except Exception as e:
            logger.error(f"Error storing dependencies: {str(e)}")
            raise

    def _merge_analysis_details(self, existing_json: str, new_json: str) -> str:
        """Merge analysis details JSON objects"""
        try:
            existing = json.loads(existing_json) if existing_json else {}
            new = json.loads(new_json) if new_json else {}
            
            # Merge file operations or CICS operations lists
            for key in ['file_operations', 'cics_operations']:
                if key in new:
                    if key in existing:
                        # Combine and deduplicate operations
                        combined = existing[key] + new[key]
                        # Deduplicate by line_number
                        seen_lines = set()
                        deduplicated = []
                        for op in combined:
                            line_num = op.get('line_number', 0)
                            if line_num not in seen_lines:
                                seen_lines.add(line_num)
                                deduplicated.append(op)
                        existing[key] = deduplicated
                    else:
                        existing[key] = new[key]
            
            # Merge other fields
            for key, value in new.items():
                if key not in ['file_operations', 'cics_operations']:
                    existing[key] = value
            
            return json.dumps(existing)
            
        except Exception as e:
            logger.error(f"Error merging analysis details: {str(e)}")
            return new_json if new_json else existing_json
    
    def _store_field_operation(self, session_id: str, field_name: str, program_name: str,
                             operation_type: str, line_number: int, code_snippet: str):
        """Store individual field operation"""
        try:
            self.db_manager.store_field_details_with_lengths(session_id, {
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
        
    def _extract_and_store_dependencies_upd(self, session_id: str, components: List[Dict], filename: str):
        """Enhanced dependency extraction including file operations"""
        try:
            main_program = None
            for component in components:
                if component.get('type') == 'PROGRAM':
                    main_program = component
                    break
            
            if not main_program:
                return
            
            dependencies = []
            program_name = main_program['name']

            # Helper: decide if a discovered target is valid as a dependency
            def _is_valid_target(target_name: Optional[str], source_program: str) -> bool:
                if not target_name:
                    return False
                t = str(target_name).strip()
                if not t:
                    return False

                # Normalize for checks
                t_up = t.upper()

                # Ignore exact same-as-source targets
                if t_up == str(source_program).upper():
                    return False

                # Ignore short or noisy tokens
                if len(t_up) < 3:
                    return False

                # Ignore tokens made of punctuation/numbers only
                if re.match(r'^[0-9\W_]+$', t_up):
                    return False

                # Common COBOL keywords and tokens that should not be treated as files/programs
                IGNORE_KEYWORDS = {
                    'VALUE', 'PIC', 'FUNCTION', 'SECTION', 'PARAGRAPH', 'MODE', 'USAGE', 'REDEFINES',
                    'COPY', 'END', 'IF', 'ELSE', 'THEN', 'MOVE', 'READ', 'WRITE', 'OPEN', 'CLOSE'
                }

                # If the token is exactly a keyword or contains a keyword-only suffix like ' VALUE', ignore
                tokens = re.split(r'\s+|[,;:\(\)]+', t_up)
                # If any token is a sole-keyword that indicates a COBOL construct, ignore
                if any(tok in IGNORE_KEYWORDS for tok in tokens if tok):
                    return False

                return True
            
            # Extract FILE dependencies (FD files)
            file_operations = main_program.get('file_operations', [])
            for file_op in file_operations:
                file_name = file_op.get('file_name')
                operation = file_op.get('operation', 'UNKNOWN')
                if file_name and _is_valid_target(file_name, program_name):
                    # Determine relationship type based on operation
                    fn_up = str(file_name).upper()
                    if operation in ['READ', 'OPEN'] or 'INPUT' in fn_up:
                        relationship_type = 'INPUT_FILE'
                    elif operation in ['WRITE', 'REWRITE'] or 'OUTPUT' in fn_up:
                        relationship_type = 'OUTPUT_FILE'
                    else:
                        relationship_type = 'FILE_ACCESS'

                    dependencies.append({
                        'source_component': program_name,
                        'target_component': file_name,
                        'relationship_type': relationship_type,
                        'interface_type': 'FILE_SYSTEM',
                        'confidence_score': 0.9,
                        'analysis_details_json': json.dumps({
                            'operation': operation,
                            'line_number': file_op.get('line_number', 0),
                            'file_type': file_op.get('file_type', 'UNKNOWN')
                        })
                    })
            
            # Extract CICS dependencies
            cics_operations = main_program.get('cics_operations', [])
            for cics_op in cics_operations:
                if 'file_name' in cics_op and _is_valid_target(cics_op['file_name'], program_name):
                    dependencies.append({
                        'source_component': program_name,
                        'target_component': cics_op['file_name'],
                        'relationship_type': 'CICS_FILE',
                        'interface_type': 'CICS',
                        'confidence_score': 0.9,
                        'analysis_details_json': json.dumps({
                            'operation': cics_op.get('operation', 'READ'),
                            'line_number': cics_op.get('line_number', 0)
                        })
                    })
            
            # Extract PROGRAM CALL dependencies
            program_calls = main_program.get('program_calls', [])
            for call in program_calls:
                target_prog = call.get('program_name', 'Unknown')
                if _is_valid_target(target_prog, program_name):
                    dependencies.append({
                        'source_component': program_name,
                        'target_component': target_prog,
                        'relationship_type': 'PROGRAM_CALL',
                        'interface_type': 'COBOL',
                        'confidence_score': 0.95,
                        'analysis_details_json': json.dumps({
                            'line_number': call.get('line_number', 0)
                        })
                    })
            
            # Deduplicate dependencies by (source, target, relationship_type, interface_type)
            unique_deps = {}
            for dep in dependencies:
                # Use normalized (case-insensitive) keys to prevent duplicates that differ only by case
                key = (
                    str(dep.get('source_component')).upper(),
                    str(dep.get('target_component')).upper(),
                    str(dep.get('relationship_type')).upper(),
                    str(dep.get('interface_type', '')).upper()
                )
                # Keep the entry with highest confidence_score or merge analysis details
                if key in unique_deps:
                    existing = unique_deps[key]
                    # Prefer higher confidence and merge analysis JSONs conservatively
                    if dep.get('confidence_score', 0) > existing.get('confidence_score', 0):
                        existing['confidence_score'] = dep.get('confidence_score')
                    # Merge analysis details by appending if different
                    if dep.get('analysis_details_json') and dep.get('analysis_details_json') != existing.get('analysis_details_json'):
                        merged = existing.get('analysis_details_json', '') + '\n' + dep.get('analysis_details_json')
                        existing['analysis_details_json'] = merged
                else:
                    unique_deps[key] = dict(dep)

            # Store unique dependencies, updating existing DB rows if present to avoid duplicates
            if unique_deps:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    stored_count = 0
                    for dep in unique_deps.values():
                        try:
                            cursor.execute('''
                                SELECT id, confidence_score FROM dependency_relationships
                                WHERE session_id = ? AND source_component = ? AND target_component = ? AND relationship_type = ?
                            ''', (session_id, dep['source_component'], dep['target_component'], dep['relationship_type']))
                            row = cursor.fetchone()
                            if row:
                                # Update existing row: keep the higher confidence and update analysis details
                                existing_id = row[0]
                                existing_conf = row[1] or 0
                                new_conf = max(existing_conf, dep.get('confidence_score', 0))
                                cursor.execute('''
                                    UPDATE dependency_relationships
                                    SET confidence_score = ?, analysis_details_json = ?, source_code_evidence = ?, created_at = CURRENT_TIMESTAMP
                                    WHERE id = ?
                                ''', (new_conf, dep.get('analysis_details_json'), dep.get('source_code_evidence', ''), existing_id))
                            else:
                                cursor.execute('''
                                    INSERT INTO dependency_relationships 
                                    (session_id, source_component, target_component, relationship_type,
                                    interface_type, confidence_score, analysis_details_json, source_code_evidence)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    session_id, dep['source_component'], dep['target_component'], dep['relationship_type'],
                                    dep.get('interface_type'), dep.get('confidence_score', 0.0), dep.get('analysis_details_json'), dep.get('source_code_evidence', '')
                                ))
                            stored_count += 1
                        except Exception as ins_e:
                            logger.error(f"Error storing dependency {dep}: {ins_e}")
                            continue

                logger.info(f"Stored/updated {stored_count} unique dependencies for {program_name}")
            
        except Exception as e:
            logger.error(f"Error extracting dependencies: {str(e)}")