from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import json
import uuid
import datetime
import time
import requests
import os
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import re
import traceback

# Import our modules
from modules.token_manager import TokenManager
from modules.llm_client import LLMClient
from modules.database_manager import DatabaseManager
from modules.cobol_parser import COBOLParser
from modules.field_analyzer import FieldAnalyzer
from modules.component_extractor import ComponentExtractor
from modules.chat_manager import ChatManager

app = Flask(__name__)
CORS(app)

# Configure logging with more detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('mainframe_analyzer.log', mode='a')  # File output
    ]
)
logger = logging.getLogger(__name__)

# Set specific loggers to INFO level
logging.getLogger('modules.component_extractor').setLevel(logging.INFO)
logging.getLogger('modules.field_analyzer').setLevel(logging.INFO) 
logging.getLogger('modules.cobol_parser').setLevel(logging.INFO)
logging.getLogger('modules.llm_client').setLevel(logging.INFO)
logging.getLogger('modules.token_manager').setLevel(logging.INFO)
logging.getLogger('modules.database_manager').setLevel(logging.INFO)

class MainframeAnalyzer:
    def __init__(self, llm_endpoint: str = "http://localhost:8100/generate"):
        self.token_manager = TokenManager()
        self.llm_client = LLMClient(llm_endpoint)
        self.db_manager = DatabaseManager()
        self.cobol_parser = COBOLParser()
        self.field_analyzer = FieldAnalyzer(self.llm_client, self.token_manager, self.db_manager)
        self.component_extractor = ComponentExtractor(self.llm_client, self.token_manager, self.db_manager)
        self.chat_manager = ChatManager(self.llm_client, self.token_manager, self.db_manager)
        
        # Initialize database
        self.db_manager.initialize_database()
    
    def update_llm_config(self, config: Dict) -> bool:
        """Update LLM client configuration"""
        try:
            # Update LLM client endpoint
            self.llm_client.endpoint = config.get('endpoint', self.llm_client.endpoint)
            
            # Update token manager limits
            max_tokens = config.get('maxTokens', 6000)
            self.token_manager.MAX_TOKENS_PER_CALL = max_tokens
            self.token_manager.EFFECTIVE_CONTENT_LIMIT = max_tokens - 500  # Reserve for system prompts
            
            # Update LLM client settings
            if hasattr(self.llm_client, 'default_max_tokens'):
                self.llm_client.default_max_tokens = max_tokens
            if hasattr(self.llm_client, 'default_temperature'):
                self.llm_client.default_temperature = config.get('temperature', 0.1)
            if hasattr(self.llm_client, 'timeout'):
                self.llm_client.timeout = config.get('timeout', 60)
            if hasattr(self.llm_client, 'max_retries'):
                self.llm_client.max_retries = config.get('retries', 3)
                
            logger.info(f"Updated LLM configuration: endpoint={self.llm_client.endpoint}, max_tokens={max_tokens}")
            return True
        except Exception as e:
            logger.error(f"Error updating LLM config: {str(e)}")
            return False
    
    def test_llm_connection(self, config: Dict) -> Dict:
        """Test connection to LLM server with provided configuration"""
        try:
            # Create temporary LLM client with new config
            from modules.llm_client import LLMClient
            
            test_client = LLMClient(config.get('endpoint', 'http://localhost:8100/generate'))
            test_client.timeout = config.get('timeout', 60)
            test_client.default_max_tokens = config.get('maxTokens', 100)
            test_client.default_temperature = config.get('temperature', 0.1)
            
            # Test with simple prompt
            test_prompt = "Hello! Please respond with 'VLLM_TEST_OK' to confirm you are working properly."
            
            start_time = time.time()
            response = test_client.call_llm(
                test_prompt,
                max_tokens=config.get('maxTokens', 100),
                temperature=config.get('temperature', 0.1)
            )
            response_time = int((time.time() - start_time) * 1000)
            
            if response.success:
                # Try to get model information
                model_info = self._get_model_info(test_client)
                
                return {
                    'success': True,
                    'response_time_ms': response_time,
                    'model_name': model_info.get('model_name', 'Unknown'),
                    'max_context_length': model_info.get('max_context_length', 'Unknown'),
                    'response_preview': response.content[:100] + ('...' if len(response.content) > 100 else ''),
                    'prompt_tokens': response.prompt_tokens,
                    'response_tokens': response.response_tokens
                }
            else:
                return {
                    'success': False,
                    'error': response.error_message,
                    'response_time_ms': response_time
                }
                
        except Exception as e:
            logger.error(f"Error testing LLM connection: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'response_time_ms': 0
            }
    
    def _get_model_info(self, llm_client: LLMClient) -> Dict:
        """Try to get model information from LLM server"""
        try:
            # Try different endpoints to get model info
            info_endpoints = ['/v1/models', '/models', '/info']
            
            for endpoint in info_endpoints:
                try:
                    base_url = llm_client.endpoint.replace('/generate', '')
                    info_url = f"{base_url}{endpoint}"
                    
                    response = requests.get(info_url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Extract model information
                        if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                            model = data['data'][0]
                            return {
                                'model_name': model.get('id', 'Unknown'),
                                'max_context_length': model.get('max_model_len', 'Unknown')
                            }
                        elif 'model' in data:
                            return {
                                'model_name': data.get('model', 'Unknown'),
                                'max_context_length': data.get('max_context_length', 'Unknown')
                            }
                except:
                    continue
            
            # Fallback - return basic info
            return {
                'model_name': 'VLLM Server',
                'max_context_length': 'Unknown'
            }
            
        except Exception as e:
            logger.warning(f"Could not retrieve model info: {str(e)}")
            return {
                'model_name': 'Unknown',
                'max_context_length': 'Unknown'
            }
    
    def create_session(self, project_name: str) -> str:
        """Create new analysis session"""
        session_id = str(uuid.uuid4())
        self.db_manager.create_session(session_id, project_name)
        return session_id
    
    def upload_file(self, session_id: str, file_content: str, file_name: str, file_type: str) -> Dict:
        """Upload and analyze mainframe file"""
        logger.info(f"🚀 Starting analysis of {file_name} ({file_type}) - Session: {session_id[:8]}...")
        logger.info(f"📏 File size: {len(file_content)} characters, {len(file_content.split())} words")
        
        try:
            # Parse file to extract components
            logger.info(f"🔍 Extracting components from {file_name}...")
            start_time = time.time()
            
            components = self.component_extractor.extract_components(session_id, file_content, file_name, file_type)
            
            extraction_time = time.time() - start_time
            logger.info(f"✅ Component extraction completed in {extraction_time:.2f}s - Found {len(components)} components")
            
            # Log component breakdown
            component_types = {}
            for component in components:
                comp_type = component.get('type', 'UNKNOWN')
                component_types[comp_type] = component_types.get(comp_type, 0) + 1
            
            logger.info(f"📊 Component breakdown: {dict(component_types)}")
            
            # Store analysis results
            logger.info(f"💾 Storing analysis results in database...")
            storage_start = time.time()
            
            for i, component in enumerate(components, 1):
                logger.debug(f"Storing component {i}/{len(components)}: {component['name']} ({component['type']})")
                self.db_manager.store_component_analysis(
                    session_id, component['name'], component['type'], 
                    file_name, component
                )
            
            storage_time = time.time() - storage_start
            logger.info(f"💾 Database storage completed in {storage_time:.2f}s")
            
            total_time = time.time() - start_time
            logger.info(f"🎉 Analysis of {file_name} completed successfully in {total_time:.2f}s")
            
            return {
                'success': True,
                'components': components,
                'message': f'Successfully analyzed {len(components)} components from {file_name}',
                'processing_time': total_time,
                'component_breakdown': component_types
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {file_name}: {str(e)}")
            logger.error(f"📍 Error details: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'file_name': file_name
            }
    
    def analyze_field_mapping(self, session_id: str, target_file: str) -> Dict:
        """Analyze field mapping for target file"""
        logger.info(f"🎯 Starting field mapping analysis for target file: {target_file}")
        logger.info(f"📂 Session: {session_id[:8]}...")
        
        try:
            start_time = time.time()
            
            # Find relevant programs
            logger.info(f"🔍 Finding programs that interact with {target_file}...")
            components = self.db_manager.get_session_components(session_id)
            relevant_programs = self.field_analyzer._find_programs_for_file(session_id, target_file, components)
            
            logger.info(f"📋 Found {len(relevant_programs)} relevant programs: {[p['component_name'] for p in relevant_programs]}")
            
            if not relevant_programs:
                logger.warning(f"⚠️  No programs found that interact with {target_file}")
                return {
                    'success': True,
                    'field_mappings': [],
                    'message': f'No programs found that interact with {target_file}'
                }
            
            # Analyze field mappings
            logger.info(f"🔬 Analyzing field mappings across {len(relevant_programs)} programs...")
            mapping_results = self.field_analyzer.analyze_field_mapping(session_id, target_file)
            
            analysis_time = time.time() - start_time
            logger.info(f"✅ Field mapping analysis completed in {analysis_time:.2f}s")
            logger.info(f"📊 Generated {len(mapping_results)} field mappings")
            
            # Log field mapping summary
            if mapping_results:
                logic_types = {}
                for mapping in mapping_results:
                    logic_type = mapping.business_logic_type
                    logic_types[logic_type] = logic_types.get(logic_type, 0) + 1
                logger.info(f"🧠 Business logic breakdown: {dict(logic_types)}")
            
            return {
                'success': True,
                'field_mappings': mapping_results,
                'message': f'Successfully analyzed field mappings for {target_file}',
                'processing_time': analysis_time,
                'programs_analyzed': len(relevant_programs)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing field mapping for {target_file}: {str(e)}")
            logger.error(f"📍 Error details: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'target_file': target_file
            }
    
    def get_field_matrix(self, session_id: str, record_layout: str = None, program_name: str = None) -> Dict:
        """Get field matrix for selected record layout or program"""
        try:
            matrix_data = self.field_analyzer.get_field_matrix(session_id, record_layout, program_name)
            return {
                'success': True,
                'matrix_data': matrix_data
            }
        except Exception as e:
            logger.error(f"Error getting field matrix: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def chat_query(self, session_id: str, message: str, conversation_id: str) -> Dict:
        """Process chat query with context"""
        try:
            response = self.chat_manager.process_query(session_id, message, conversation_id)
            return {
                'success': True,
                'response': response
            }
        except Exception as e:
            logger.error(f"Error processing chat query: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# Initialize analyzer
analyzer = MainframeAnalyzer()

# API Routes
@app.route('/')
def index():
    """Serve main application page"""
    return render_template('index.html')

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create new analysis session"""
    data = request.json
    project_name = data.get('project_name', f'Project_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    session_id = analyzer.create_session(project_name)
    return jsonify({'session_id': session_id, 'project_name': project_name})

# Add this to your main.py Flask routes:
# Add this route to reset database if needed
@app.route('/api/reset-database', methods=['POST'])
def reset_database():
    """Reset database - USE WITH CAUTION"""
    try:
        import os
        if os.path.exists(analyzer.db_manager.db_path):
            os.remove(analyzer.db_manager.db_path)
        
        # Reinitialize
        analyzer.db_manager.init_executed = False
        analyzer.db_manager.initialize_database()
        
        return jsonify({
            'success': True,
            'message': 'Database reset successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Add to main.py
@app.route('/api/debug-storage/<session_id>')
def debug_storage(session_id):
    """Debug what's actually stored in the database"""
    try:
        with analyzer.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT component_name, business_purpose, complexity_score, 
                       LENGTH(analysis_result_json) as json_length,
                       SUBSTR(analysis_result_json, 1, 200) as json_preview
                FROM component_analysis 
                WHERE session_id = ?
            ''', (session_id,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'name': row[0],
                    'business_purpose': row[1],
                    'complexity_score': row[2],
                    'json_length': row[3],
                    'json_preview': row[4]
                })
            
            return jsonify({
                'success': True,
                'stored_components': results
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/derived-components/<session_id>')
def get_derived_components_api(session_id):
    """Get derived components for session"""
    parent_component = request.args.get('parent_component')
    
    try:
        derived_components = analyzer.db_manager.get_derived_components(session_id, parent_component)
        return jsonify({
            'success': True,
            'derived_components': derived_components,
            'count': len(derived_components)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/components-with-derived/<session_id>')
def get_components_with_derived_counts(session_id):
    """Get components with derived component counts"""
    try:
        # Get main components
        components = analyzer.db_manager.get_session_components(session_id)
        
        # Add derived counts to each component
        for component in components:
            derived_count = analyzer.db_manager.get_derived_components_count(
                session_id, component['component_name']
            )
            component['derived_count'] = derived_count
        
        return jsonify({
            'success': True,
            'components': components
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/debug-components/<session_id>')
def debug_components(session_id):
    """Debug endpoint to see raw component data"""
    try:
        analyzer.db_manager.debug_database_schema()
        
        components = analyzer.db_manager.get_session_components(session_id)
        
        return jsonify({
            'success': True,
            'components_count': len(components),
            'components': components,
            'session_id': session_id
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/component-source/<session_id>/<component_name>')
def get_component_source_api(session_id, component_name):
    """Get component source code for chat/display"""
    try:
        source_data = analyzer.db_manager.get_component_source_code(
            session_id, component_name, max_size=100000
        )
        return jsonify(source_data)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/test-llm', methods=['POST'])
def test_llm_connection():
    """Test LLM connection with provided configuration"""
    try:
        data = request.json
        config = {
            'endpoint': data.get('endpoint', 'http://localhost:8100/generate'),
            'maxTokens': data.get('maxTokens', 6000),
            'temperature': data.get('temperature', 0.1),
            'timeout': data.get('timeout', 60)
        }
        
        result = analyzer.test_llm_connection(config)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in test LLM endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and analyze mainframe file"""
    data = request.json
    session_id = data.get('session_id')
    file_content = data.get('content')
    file_name = data.get('filename')
    file_type = data.get('file_type', 'COBOL')
    
    result = analyzer.upload_file(session_id, file_content, file_name, file_type)
    return jsonify(result)

@app.route('/api/field-mapping', methods=['POST'])
def analyze_field_mapping():
    """Fixed field mapping analysis with proper field lengths, types, and validation"""
    data = request.json
    session_id = data.get('session_id')
    target_file = data.get('target_file')
    
    logger.info(f"Field mapping request - Session: {session_id[:8]}, Target: {target_file}")
    
    try:
        # Enhanced field query with proper length calculation and filtering
        with analyzer.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT fad.field_name, fad.friendly_name, fad.mainframe_data_type, 
                       fad.oracle_data_type, fad.mainframe_length, fad.oracle_length,
                       fad.source_field, fad.business_purpose, fad.usage_type,
                       fad.program_name, rl.layout_name,
                       fad.field_references_json, fad.definition_code,
                       fad.total_program_references, fad.move_source_count, fad.move_target_count
                FROM field_analysis_details fad
                LEFT JOIN record_layouts rl ON fad.field_id = rl.id
                WHERE fad.session_id = ?
                AND fad.mainframe_length > 0  -- Filter out invalid length calculations
                AND fad.field_name NOT IN ('VALUE', 'VALUES', 'PIC', 'PICTURE', 'USAGE', 'COMP')
                AND (UPPER(rl.layout_name) LIKE UPPER(?) 
                     OR UPPER(fad.program_name) LIKE UPPER(?)
                     OR UPPER(fad.field_name) LIKE UPPER(?))
                ORDER BY rl.layout_name, fad.field_name
            ''', (session_id, f'%{target_file}%', f'%{target_file}%', f'%{target_file}%'))
            
            field_results = cursor.fetchall()
        
        if not field_results:
            return jsonify({
                'success': True,
                'field_mappings': [],
                'message': f'No valid fields found matching "{target_file}". Please check the target file name.',
                'summary': {'total_fields': 0, 'input_fields': 0, 'output_fields': 0, 'static_fields': 0, 'calculated_fields': 0}
            })
        
        # Transform to field mappings with proper business logic classification
        field_mappings = []
        for row in field_results:
            field_references = []
            try:
                if row[11]:  # field_references_json
                    field_references = json.loads(row[11])
            except:
                pass
            logger.info(f"Processing field: {row[8]}, references: {field_references}")
            # Proper business logic type determination
            usage_type = row[8] or 'STATIC'  # usage_type
            move_source_count = row[14] or 0
            move_target_count = row[15] or 0
            
            # Enhanced business logic classification
            if move_target_count > 0 and move_source_count > 0:
                business_logic_type = 'INPUT_OUTPUT'
            elif move_target_count > 0:
                business_logic_type = 'DIRECT_MOVE'
            elif move_source_count > 0:
                business_logic_type = 'OUTPUT_SOURCE'
            elif usage_type == 'DERIVED':
                business_logic_type = 'CALCULATED'
            elif usage_type == 'STATIC':
                business_logic_type = 'CONSTANT'
            else:
                business_logic_type = 'REFERENCE'
            
            # Fixed mainframe length - ensure it's properly calculated
            mainframe_length = max(row[4] or 0, 1)  # Ensure minimum length of 1
            oracle_length = max(row[5] or mainframe_length, mainframe_length)  # Oracle length should be at least mainframe length
            
            field_mapping = {
                'field_name': row[0],
                'friendly_name': row[1] or row[0].replace('-', ' ').title(),
                'mainframe_data_type': row[2] or 'UNKNOWN',
                'oracle_data_type': row[3] or f'VARCHAR2({oracle_length})',
                'mainframe_length': mainframe_length,
                'oracle_length': oracle_length,
                'population_source': row[6] or 'Program logic',
                'source_record_layout': row[10] or 'Unknown',
                'business_logic_type': business_logic_type,
                'business_logic_description': row[7] or f'Field {row[0]} processing',
                'derivation_logic': f"Populated from {row[6] or 'program logic'}",
                'programs_involved': [row[9]] if row[9] else [],
                'confidence_score': 0.9,
                'field_references': field_references,
                'definition_code': row[12] or '',
                'total_references': row[13] or 0,
                'move_operations': {
                    'source_count': move_source_count,
                    'target_count': move_target_count
                }
            }
            
            field_mappings.append(field_mapping)
        
        logger.info(f"Generated {len(field_mappings)} field mappings for {target_file}")
        
        # Corrected summary calculations
        summary = {
            'total_fields': len(field_mappings),
            'input_fields': len([f for f in field_mappings if 'INPUT' in f['business_logic_type'] or f['business_logic_type'] == 'DIRECT_MOVE']),
            'output_fields': len([f for f in field_mappings if 'OUTPUT' in f['business_logic_type']]),
            'static_fields': len([f for f in field_mappings if f['business_logic_type'] == 'CONSTANT']),
            'calculated_fields': len([f for f in field_mappings if f['business_logic_type'] == 'CALCULATED'])
        }
        
        return jsonify({
            'success': True,
            'field_mappings': field_mappings,
            'message': f'Successfully analyzed {len(field_mappings)} field mappings for "{target_file}".',
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error in field mapping analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

    
@app.route('/api/field-matrix', methods=['GET'])
def get_field_matrix():
    """Get field matrix data"""
    session_id = request.args.get('session_id')
    record_layout = request.args.get('record_layout')
    program_name = request.args.get('program_name')
    
    result = analyzer.get_field_matrix(session_id, record_layout, program_name)
    return jsonify(result)

@app.route('/api/chat', methods=['POST'])
def chat_query():
    try:
        data = request.json
        session_id = data.get('session_id')
        message = data.get('message')
        conversation_id = data.get('conversation_id', str(uuid.uuid4()))
        
        # Process query - returns string now
        chat_response = analyzer.chat_manager.process_query(session_id, message, conversation_id)
        
        return jsonify({
            'success': True,
            'response': chat_response,
            'content': chat_response
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'response': 'Chat service error'
        })
    
@app.route('/api/record-layouts/<session_id>')
def get_record_layouts_api(session_id):
    """Get record layouts with friendly names - FIXED VERSION"""
    try:
        layouts = analyzer.db_manager.get_record_layouts(session_id)
        
        # Debug logging
        logger.info(f"Retrieved {len(layouts)} record layouts for session {session_id}")
        for layout in layouts[:3]:  # Log first 3 for debugging
            logger.info(f"Layout: {layout.get('layout_name')} from {layout.get('program_name')}")
        
        return jsonify({'success': True, 'layouts': layouts})
    except Exception as e:
        logger.error(f"Error getting record layouts: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

    
@app.route('/api/field-source-code/<session_id>/<field_name>')
def get_field_source_code(session_id, field_name):
    """Get source code context for a field"""
    try:
        context = analyzer.db_manager.get_context_for_field(session_id, field_name)
        
        if context and context.get('field_details'):
            field_detail = context['field_details'][0]
            source_code = field_detail.get('program_source_content', '')
            
            if not source_code:
                # Get from component analysis
                components = analyzer.db_manager.get_session_components(session_id)
                program_name = field_detail.get('program_name', '')
                
                for comp in components:
                    if comp['component_name'] == program_name:
                        if comp.get('analysis_result_json'):
                            analysis = json.loads(comp['analysis_result_json'])
                            source_code = analysis.get('content', 'Source not available')
                        break
            
            return jsonify({
                'success': True,
                'source_code': source_code or 'Source code not available',
                'field_name': field_name
            })
        else:
            return jsonify({
                'success': False,
                'source_code': 'Field not found in analysis',
                'field_name': field_name
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'source_code': 'Error loading source code'
        })
@app.route('/api/components/<session_id>')
def get_components(session_id):
    """Fixed component retrieval with proper LLM summary extraction and friendly names"""
    try:
        with analyzer.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT component_name, component_type, total_lines, total_fields, 
                       business_purpose, complexity_score, analysis_result_json, 
                       file_path, created_at
                FROM component_analysis 
                WHERE session_id = ?
                ORDER BY component_name
            ''', (session_id,))
            
            raw_components = cursor.fetchall()
        
        transformed_components = []
        for row in raw_components:
            component = dict(row)
            
            # Enhanced LLM summary extraction with proper fallbacks
            business_purpose = 'Analysis in progress...'
            llm_summary = {}
            complexity_score = 0.5
            friendly_name = component['component_name']
            
            # Parse analysis_result_json thoroughly
            if component.get('analysis_result_json'):
                try:
                    parsed_analysis = json.loads(component['analysis_result_json'])
                    
                    # Extract friendly name first
                    if parsed_analysis.get('friendly_name'):
                        friendly_name = parsed_analysis['friendly_name']
                    else:
                        # Generate friendly name using parser
                        friendly_name = analyzer.cobol_parser.generate_friendly_name(
                            component['component_name'], 'Program'
                        )
                    
                    # Extract LLM summary with multiple fallback paths
                    if parsed_analysis.get('llm_summary'):
                        llm_summary = parsed_analysis['llm_summary']
                        
                        # Primary: business_purpose from LLM summary
                        if llm_summary.get('business_purpose') and llm_summary['business_purpose'] not in ['undefined', 'null', '']:
                            business_purpose = llm_summary['business_purpose']
                        # Secondary: raw response from LLM
                        elif llm_summary.get('raw'):
                            raw_response = llm_summary['raw']
                            business_purpose = (raw_response[:200] + '...' if len(raw_response) > 200 else raw_response)
                            llm_summary['is_raw'] = True
                        # Tertiary: primary_function as description
                        elif llm_summary.get('primary_function'):
                            business_purpose = f"{llm_summary['primary_function']} component"
                        
                        # Extract complexity score
                        if isinstance(llm_summary.get('complexity_score'), (int, float)):
                            complexity_score = float(llm_summary['complexity_score'])
                    
                    # Fallback to direct business_purpose in parsed data
                    elif parsed_analysis.get('business_purpose') and parsed_analysis['business_purpose'] not in ['undefined', 'null', '']:
                        business_purpose = parsed_analysis['business_purpose']
                        
                except Exception as e:
                    logger.warning(f"Error parsing analysis JSON for {component['component_name']}: {e}")
                    # Generate friendly name even if JSON parsing fails
                    friendly_name = analyzer.cobol_parser.generate_friendly_name(
                        component['component_name'], 'Program'
                    )
            else:
                # No analysis JSON - generate friendly name
                friendly_name = analyzer.cobol_parser.generate_friendly_name(
                    component['component_name'], 'Program'
                )
            
            # Use database business_purpose if available and valid
            if (component.get('business_purpose') and 
                component['business_purpose'] not in [None, '', 'undefined', 'null', 'Analysis pending...', 'Analysis in progress...']):
                business_purpose = component['business_purpose']
            
            # Use database complexity_score if available
            if component.get('complexity_score') and component['complexity_score'] > 0:
                complexity_score = float(component['complexity_score'])
            
            # Get derived components count
            derived_count = analyzer.db_manager.get_derived_components_count(session_id, component['component_name'])
            
            transformed = {
                'component_name': component['component_name'],
                'component_type': component['component_type'],
                'friendly_name': friendly_name,
                'total_lines': component.get('total_lines', 0),
                'total_fields': component.get('total_fields', 0),
                'business_purpose': business_purpose,
                'llm_summary': llm_summary,
                'complexity_score': complexity_score,
                'derived_count': derived_count,
                'file_path': component.get('file_path', ''),
                'created_at': component.get('created_at', ''),
                'analysis_status': 'completed' if business_purpose != 'Analysis in progress...' else 'pending'
            }
            transformed_components.append(transformed)

        logger.info(f"Transformed {len(transformed_components)} components with proper LLM summaries")

        return jsonify({
            'success': True,
            'components': transformed_components
        })
        
    except Exception as e:
        logger.error(f"Error retrieving components: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dependencies/<session_id>')
def get_dependencies(session_id):
    """Get dependency relationships with enhanced I/O categorization and friendly names"""
    try:
        logger.info(f"Getting dependencies for session: {session_id}")
        
        with analyzer.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Enhanced query with proper I/O direction analysis
            cursor.execute('''
                SELECT dr.source_component, dr.target_component, dr.relationship_type,
                       dr.interface_type, dr.confidence_score, dr.analysis_details_json,
                       dr.source_code_evidence, dr.created_at,
                       ca.business_purpose as source_purpose,
                       ca2.business_purpose as target_purpose
                FROM dependency_relationships dr
                LEFT JOIN component_analysis ca ON (dr.source_component = ca.component_name AND ca.session_id = ?)
                LEFT JOIN component_analysis ca2 ON (dr.target_component = ca2.component_name AND ca2.session_id = ?)
                WHERE dr.session_id = ?
                ORDER BY dr.source_component, dr.relationship_type, dr.target_component
            ''', (session_id, session_id, session_id))
            
            raw_dependencies = cursor.fetchall()
        
        if not raw_dependencies:
            components = analyzer.db_manager.get_session_components(session_id)
            return jsonify({
                'success': True, 
                'dependencies': [],
                'dependency_groups': {},
                'categorized_dependencies': {},
                'total_count': 0,
                'debug_info': f"No dependencies found. Session has {len(components)} components."
            })
        
        # Process and categorize dependencies
        transformed_deps = []
        cics_files = {}  # Track CICS files for I/O categorization
        
        for dep in raw_dependencies:
            details = {}
            if dep[5]:  # analysis_details_json
                try:
                    details = json.loads(dep[5])
                except:
                    pass
            
            # Generate friendly names using the parser's method
            target_friendly = analyzer.cobol_parser.generate_friendly_name(dep[1], 'Component')
            source_friendly = analyzer.cobol_parser.generate_friendly_name(dep[0], 'Program')
            
            # Determine I/O direction for CICS files
            io_direction = 'UNKNOWN'
            if dep[2].startswith('CICS') and 'FILE' in dep[2]:
                if 'INPUT' in dep[2]:
                    io_direction = 'INPUT'
                elif 'OUTPUT' in dep[2]:
                    io_direction = 'OUTPUT'
                else:
                    # Analyze from details
                    io_direction = details.get('io_direction', 'UNKNOWN')
            elif dep[2] in ['INPUT_FILE', 'OUTPUT_FILE', 'INPUT_OUTPUT_FILE']:
                io_direction = dep[2].replace('_FILE', '')
            
            # Track CICS files for deduplication
            if dep[3] == 'CICS':
                file_key = dep[1]  # target_component (file name)
                if file_key not in cics_files:
                    cics_files[file_key] = {
                        'file_name': dep[1],
                        'friendly_name': target_friendly,
                        'operations': [],
                        'io_directions': set(),
                        'source_programs': set()
                    }
                
                cics_files[file_key]['operations'].append(dep[2])
                cics_files[file_key]['io_directions'].add(io_direction)
                cics_files[file_key]['source_programs'].add(dep[0])
            
            transformed_dep = {
                'source_component': dep[0],
                'source_friendly': source_friendly,
                'target_component': dep[1],
                'target_friendly': target_friendly,
                'relationship_type': dep[2],
                'interface_type': dep[3],
                'io_direction': io_direction,
                'confidence_score': dep[4] or 0.0,
                'details': details,
                'source_code_evidence': dep[6],
                'created_at': dep[7]
            }
            transformed_deps.append(transformed_dep)
        
        # Create categorized view of CICS files
        cics_categorized = {}
        for file_info in cics_files.values():
            directions = file_info['io_directions']
            if 'INPUT' in directions and 'OUTPUT' in directions:
                category = 'INPUT_OUTPUT'
            elif 'INPUT' in directions:
                category = 'INPUT'
            elif 'OUTPUT' in directions:
                category = 'OUTPUT'
            else:
                category = 'UNKNOWN'
            
            if category not in cics_categorized:
                cics_categorized[category] = []
            
            cics_categorized[category].append({
                'file_name': file_info['file_name'],
                'friendly_name': file_info['friendly_name'],
                'operations': list(file_info['operations']),
                'source_programs': list(file_info['source_programs'])
            })
        
        # Group dependencies by type
        dependency_groups = {}
        for dep in transformed_deps:
            rel_type = dep['relationship_type']
            if rel_type not in dependency_groups:
                dependency_groups[rel_type] = []
            dependency_groups[rel_type].append(dep)
        
        logger.info(f"Returning {len(transformed_deps)} dependencies with {len(cics_categorized)} CICS file categories")
        
        return jsonify({
            'success': True, 
            'dependencies': transformed_deps,
            'dependency_groups': dependency_groups,
            'cics_categorized': cics_categorized,
            'total_count': len(transformed_deps),
            'cics_file_count': len(cics_files)
        })
        
    except Exception as e:
        logger.error(f"Error getting dependencies: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
        
@app.route('/api/session-metrics/<session_id>')
def get_session_metrics(session_id):
    """Get session metrics"""
    try:
        metrics = analyzer.db_manager.get_session_metrics(session_id)
        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/layout-summary/<session_id>/<layout_name>')
def get_layout_summary(session_id, layout_name):
    """Get summary statistics for a record layout"""
    try:
        with analyzer.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get layout info and field counts by usage type
            cursor.execute('''
                SELECT rl.layout_name, rl.friendly_name, rl.business_purpose,
                       COUNT(fad.id) as total_fields,
                       SUM(CASE WHEN fad.usage_type = 'INPUT' THEN 1 ELSE 0 END) as input_fields,
                       SUM(CASE WHEN fad.usage_type = 'OUTPUT' THEN 1 ELSE 0 END) as output_fields,
                       SUM(CASE WHEN fad.usage_type = 'STATIC' THEN 1 ELSE 0 END) as static_fields,
                       SUM(CASE WHEN fad.usage_type = 'UNUSED' OR fad.total_program_references = 0 THEN 1 ELSE 0 END) as unused_fields,
                       SUM(CASE WHEN fad.usage_type = 'DERIVED' THEN 1 ELSE 0 END) as derived_fields
                FROM record_layouts rl
                LEFT JOIN field_analysis_details fad ON rl.id = fad.field_id
                WHERE rl.session_id = ? AND rl.layout_name = ?
                GROUP BY rl.id, rl.layout_name, rl.friendly_name, rl.business_purpose
            ''', (session_id, layout_name))
            
            result = cursor.fetchone()
            
            if not result:
                return jsonify({
                    'success': False,
                    'error': f'Layout {layout_name} not found'
                })
            
            summary = {
                'layout_name': result[0],
                'friendly_name': result[1] or result[0],
                'business_purpose': result[2] or 'No description available',
                'total_fields': result[3],
                'input_fields': result[4],
                'output_fields': result[5], 
                'static_fields': result[6],
                'unused_fields': result[7],
                'derived_fields': result[8]
            }
            
            return jsonify({
                'success': True,
                'summary': summary
            })
            
    except Exception as e:
        logger.error(f"Error getting layout summary: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

# ====================
# 6. ADD FIELD DETAILS API FOR MODAL
# ====================

@app.route('/api/field-details/<session_id>/<field_name>')
def get_field_details(session_id, field_name):
    """Get complete field details including operations and source code - FIXED VERSION"""
    try:
        with analyzer.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Enhanced query with proper joins and validation
            cursor.execute('''
                SELECT fad.field_name, fad.friendly_name, fad.business_purpose,
                       fad.usage_type, fad.source_field, fad.target_field,
                       fad.mainframe_data_type, fad.oracle_data_type,
                       fad.mainframe_length, fad.oracle_length,
                       fad.definition_code, fad.program_source_content,
                       fad.field_references_json, fad.total_program_references,
                       fad.move_source_count, fad.move_target_count,
                       fad.arithmetic_count, fad.conditional_count,
                       rl.layout_name, fad.program_name, fad.line_number
                FROM field_analysis_details fad
                LEFT JOIN record_layouts rl ON fad.field_id = rl.id
                WHERE fad.session_id = ? AND UPPER(fad.field_name) = UPPER(?)
                AND fad.field_name NOT IN ('VALUE', 'VALUES', 'PIC', 'PICTURE')
                ORDER BY fad.program_name, fad.line_number
            ''', (session_id, field_name))
            
            results = cursor.fetchall()
            
            if not results:
                return jsonify({
                    'success': False,
                    'error': f'Field "{field_name}" not found in analysis results'
                })
            
            # Process field references and operations
            all_references = []
            field_operations = []
            
            for row in results:
                # Parse field references safely
                try:
                    if row[12]:  # field_references_json
                        references = json.loads(row[12])
                        if isinstance(references, list):
                            all_references.extend(references)
                except Exception as ref_error:
                    logger.warning(f"Error parsing references for {field_name}: {ref_error}")
                
                # Create operation summary
                field_operations.append({
                    'program_name': row[19],
                    'layout_name': row[18],
                    'usage_type': row[3],
                    'move_source_count': row[15] or 0,
                    'move_target_count': row[16] or 0,
                    'arithmetic_count': row[17] or 0,
                    'conditional_count': row[18] or 0,
                    'line_number': row[20] or 0
                })
            
            # Get the primary field info (first result)
            primary = results[0]
            
            # Ensure mainframe length is properly calculated
            mainframe_length = max(primary[8] or 0, 1)  # Minimum length 1
            oracle_length = max(primary[9] or mainframe_length, mainframe_length)
            
            # Get source code snippet with error handling
            source_snippet = 'Source not available'
            if primary[11]:  # program_source_content
                source_snippet = primary[11][:2000] if len(primary[11]) > 2000 else primary[11]
            
            field_details = {
                'field_name': primary[0],
                'friendly_name': primary[1] or primary[0].replace('-', ' ').title(),
                'business_purpose': primary[2] or f'Field {primary[0]} - analysis pending',
                'usage_type': primary[3] or 'STATIC',
                'source_field': primary[4] or '',
                'target_field': primary[5] or '',
                'mainframe_data_type': primary[6] or 'UNKNOWN',
                'oracle_data_type': primary[7] or f'VARCHAR2({oracle_length})',
                'mainframe_length': mainframe_length,
                'oracle_length': oracle_length,
                'definition_code': primary[10] or f'Field definition for {primary[0]}',
                'source_code_snippet': source_snippet,
                'total_references': primary[13] or len(all_references),
                'operations': field_operations,
                'references': all_references[:15],  # Limit to first 15 references
                'layout_name': primary[18] or 'Unknown',
                'program_name': primary[19] or 'Unknown'
            }
            
            return jsonify({
                'success': True,
                'field_details': field_details
            })
            
    except Exception as e:
        logger.error(f"Error getting field details for {field_name}: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error retrieving field details: {str(e)}'
        })

@app.route('/api/db-health', methods=['GET'])
def check_database_health():
    """Check database health and fix common issues"""
    try:
        health_info = {
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Check database file permissions
        db_path = analyzer.db_manager.db_path
        
        if not os.path.exists(db_path):
            health_info['issues'].append('Database file does not exist')
            health_info['recommendations'].append('Database will be created automatically on first use')
        else:
            # Check file permissions
            if not os.access(db_path, os.R_OK):
                health_info['issues'].append('Database file not readable')
                health_info['status'] = 'error'
            
            if not os.access(db_path, os.W_OK):
                health_info['issues'].append('Database file not writable')
                health_info['status'] = 'error'
                health_info['recommendations'].append('Check file permissions: chmod 664 mainframe_analyzer.db')
        
        # Check directory permissions
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if not os.access(db_dir, os.W_OK):
            health_info['issues'].append('Database directory not writable')
            health_info['status'] = 'error'
            health_info['recommendations'].append(f'Check directory permissions: chmod 755 {db_dir}')
        
        # Check for lock files
        lock_files = [
            db_path + "-wal",
            db_path + "-shm", 
            db_path + "-journal"
        ]
        
        active_locks = []
        for lock_file in lock_files:
            if os.path.exists(lock_file):
                try:
                    size = os.path.getsize(lock_file)
                    active_locks.append(f"{os.path.basename(lock_file)} ({size} bytes)")
                except OSError:
                    active_locks.append(os.path.basename(lock_file))
        
        if active_locks:
            health_info['issues'].append(f"Active lock files: {', '.join(active_locks)}")
            health_info['recommendations'].append('Lock files are normal during operation, but persistent locks may indicate issues')
        
        # Test database connection
        try:
            with analyzer.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
            health_info['connection_test'] = 'success'
        except Exception as e:
            health_info['issues'].append(f"Database connection failed: {str(e)}")
            health_info['status'] = 'error'
            health_info['connection_test'] = 'failed'
            
            if "database is locked" in str(e).lower():
                health_info['recommendations'].append('Try restarting the application')
                health_info['recommendations'].append('Check if another process is using the database')
        
        return jsonify({
            'success': True,
            'health': health_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/fix-database-locks', methods=['POST'])
def fix_database_locks():
    """Attempt to fix database lock issues (use with caution)"""
    try:
        logger.warning("Database lock fix requested - this should only be used when the application is idle")
        
        # Force unlock database
        analyzer.db_manager.force_unlock_database()
        
        # Test connection after unlock
        try:
            with analyzer.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
            
            return jsonify({
                'success': True,
                'message': 'Database locks cleared successfully'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Database still locked after cleanup: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/llm-config', methods=['POST'])
def update_llm_config():
    """Update LLM configuration"""
    try:
        data = request.json
        success = analyzer.update_llm_config(data)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'LLM configuration updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to update LLM configuration'
            }), 500
            
    except Exception as e:
        logger.error(f"Error updating LLM config: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e) 
        }), 500

# Add to your Flask app
@app.route('/api/llm-summaries/<session_id>', methods=['GET'])
def get_llm_summaries(session_id):
    try:
        summaries = db_manager.get_all_llm_summaries(session_id)
        return jsonify({
            'success': True,
            'summaries': summaries,
            'count': len(summaries)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/llm-summary/<session_id>/<component_name>', methods=['GET'])
def get_component_llm_summary(session_id, component_name):
    try:
        summary = db_manager.get_llm_summary(session_id, component_name)
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)