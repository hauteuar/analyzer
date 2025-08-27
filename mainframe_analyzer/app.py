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
            test_client = LLMClient(config.get('endpoint', 'http://localhost:8100/generate'))
            
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
    """Fixed field mapping analysis"""
    data = request.json
    session_id = data.get('session_id')
    target_file = data.get('target_file')
    
    logger.info(f"Field mapping request - Session: {session_id[:8]}, Target: {target_file}")
    
    try:
        # First check if we have any components
        components = analyzer.db_manager.get_session_components(session_id)
        if not components:
            return jsonify({
                'success': True,
                'field_mappings': [],
                'message': 'No components found in session. Please upload and analyze files first.'
            })
        
        # Get record layouts for the session
        record_layouts = analyzer.db_manager.get_record_layouts(session_id)
        if not record_layouts:
            return jsonify({
                'success': True,
                'field_mappings': [],
                'message': f'No record layouts found in session. The target file "{target_file}" was not found in any analyzed programs.'
            })
        
        logger.info(f"Found {len(record_layouts)} record layouts to analyze")
        
        # Look for layouts that might match the target file
        matching_layouts = []
        target_upper = target_file.upper().replace('-', '').replace('_', '')
        
        for layout in record_layouts:
            layout_name = layout['layout_name'].upper().replace('-', '').replace('_', '')
            if target_upper in layout_name or layout_name in target_upper:
                matching_layouts.append(layout)
        
        if not matching_layouts:
            # Try fuzzy matching
            for layout in record_layouts:
                layout_words = layout['layout_name'].upper().split('-')
                target_words = target_file.upper().split('-')
                
                # Check for word overlap
                overlap = set(layout_words) & set(target_words)
                if overlap:
                    matching_layouts.append(layout)
        
        if not matching_layouts:
            available_layouts = [layout['layout_name'] for layout in record_layouts[:10]]  # Show first 10
            return jsonify({
                'success': True,
                'field_mappings': [],
                'message': f'No record layouts found matching "{target_file}". Available layouts include: {", ".join(available_layouts)}{"..." if len(record_layouts) > 10 else ""}'
            })
        
        logger.info(f"Found {len(matching_layouts)} matching layouts: {[l['layout_name'] for l in matching_layouts]}")
        
        # Analyze each matching layout
        all_field_mappings = []
        
        for layout in matching_layouts:
            try:
                # Get field matrix data for this layout
                field_data = analyzer.db_manager.get_field_matrix(session_id, layout['layout_name'])
                
                logger.info(f"Processing layout {layout['layout_name']} with {len(field_data)} field entries")
                
                for field_entry in field_data:
                    # Create field mapping entry
                    field_mapping = {
                        'field_name': field_entry['field_name'],
                        'friendly_name': f"Field from {layout['layout_name']}",
                        'mainframe_data_type': 'COBOL-PIC',  # Default, could be enhanced
                        'oracle_data_type': 'VARCHAR2',       # Default mapping
                        'mainframe_length': 0,                # Could be extracted from picture clause
                        'oracle_length': 255,                 # Default
                        'population_source': field_entry.get('source_field', 'Program logic'),
                        'source_record_layout': layout['layout_name'],
                        'business_logic_type': field_entry.get('usage_type', 'UNKNOWN'),
                        'business_logic_description': field_entry.get('business_purpose', ''),
                        'derivation_logic': f"Derived from {field_entry.get('source_field', 'program logic')}",
                        'programs_involved': [field_entry.get('program_name', 'Unknown')],
                        'confidence_score': 0.8
                    }
                    
                    all_field_mappings.append(field_mapping)
                
            except Exception as layout_error:
                logger.error(f"Error processing layout {layout['layout_name']}: {str(layout_error)}")
                continue
        
        # Store field mappings in database
        if all_field_mappings:
            analyzer.db_manager.store_field_mappings(session_id, target_file, all_field_mappings)
        
        logger.info(f"Generated {len(all_field_mappings)} field mappings for {target_file}")
        
        return jsonify({
            'success': True,
            'field_mappings': all_field_mappings,
            'message': f'Successfully analyzed {len(all_field_mappings)} field mappings for "{target_file}" from {len(matching_layouts)} matching record layouts.',
            'layouts_analyzed': [layout['layout_name'] for layout in matching_layouts]
        })
        
    except Exception as e:
        logger.error(f"Error in field mapping analysis: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'target_file': target_file
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
    """Get record layouts with friendly names"""
    try:
        layouts = analyzer.db_manager.get_record_layouts(session_id)
        return jsonify({'success': True, 'layouts': layouts})
    except Exception as e:
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
    """Debug version to see what's happening"""
    try:
        # Get raw components
        raw_components = analyzer.db_manager.get_session_components(session_id)
        logger.info(f"Retrieved {len(raw_components)} raw components for session {session_id}")
        
        # Log each component
        for comp in raw_components:
            logger.info(f"Component: {comp['component_name']} ({comp['component_type']})")
        
        # Transform for UI
        transformed_components = []
        for component in raw_components:
            transformed = {
                'component_name': component['component_name'],
                'component_type': component['component_type'],
                'total_lines': component.get('total_lines', 0),
                'total_fields': component.get('total_fields', 0),
                'business_purpose': component.get('business_purpose', 'Analysis pending'),
                'analysis_result_json': component.get('analysis_result_json', '{}')
            }
            transformed_components.append(transformed)
        
        logger.info(f"Returning {len(transformed_components)} transformed components")
        
        return jsonify({
            'success': True, 
            'components': transformed_components,
            'debug_info': f"Found {len(raw_components)} raw components"
        })
        
    except Exception as e:
        logger.error(f"Error retrieving components: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})    
    
@app.route('/api/dependencies/<session_id>')
def get_dependencies(session_id):
    """Get dependency relationships with enhanced data and debugging"""
    try:
        logger.info(f"Getting dependencies for session: {session_id}")
        
        dependencies = analyzer.db_manager.get_dependencies(session_id)
        logger.info(f"Found {len(dependencies)} dependencies")
        
        if not dependencies:
            # Debug: Check if we have any components at all
            components = analyzer.db_manager.get_session_components(session_id)
            logger.info(f"Session has {len(components)} components")
            
            return jsonify({
                'success': True, 
                'dependencies': [],
                'dependency_groups': {},
                'total_count': 0,
                'debug_info': f"No dependencies found. Session has {len(components)} components."
            })
        
        # Transform dependencies for UI
        transformed_deps = []
        for dep in dependencies:
            details = {}
            if dep.get('analysis_details_json'):
                try:
                    details = json.loads(dep['analysis_details_json'])
                except:
                    pass
            
            transformed_dep = {
                'source_component': dep['source_component'],
                'target_component': dep['target_component'],
                'relationship_type': dep['relationship_type'],
                'interface_type': dep.get('interface_type', 'UNKNOWN'),
                'confidence_score': dep.get('confidence_score', 0.0),
                'details': details,
                'created_at': dep.get('created_at', '')
            }
            transformed_deps.append(transformed_dep)
        
        # Group dependencies by type
        dependency_groups = {}
        for dep in transformed_deps:
            rel_type = dep['relationship_type']
            if rel_type not in dependency_groups:
                dependency_groups[rel_type] = []
            dependency_groups[rel_type].append(dep)
        
        logger.info(f"Returning {len(transformed_deps)} dependencies in {len(dependency_groups)} groups")
        
        return jsonify({
            'success': True, 
            'dependencies': transformed_deps,
            'dependency_groups': dependency_groups,
            'total_count': len(transformed_deps)
        })
        
    except Exception as e:
        logger.error(f"Error getting dependencies: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)})
        
@app.route('/api/session-metrics/<session_id>')
def get_session_metrics(session_id):
    """Get session metrics"""
    try:
        metrics = analyzer.db_manager.get_session_metrics(session_id)
        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)