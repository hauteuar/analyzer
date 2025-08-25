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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        try:
            # Parse file to extract components
            components = self.component_extractor.extract_components(session_id, file_content, file_name, file_type)
            
            # Store analysis results
            for component in components:
                self.db_manager.store_component_analysis(
                    session_id, component['name'], component['type'], 
                    file_name, component
                )
            
            return {
                'success': True,
                'components': components,
                'message': f'Successfully analyzed {len(components)} components from {file_name}'
            }
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_field_mapping(self, session_id: str, target_file: str) -> Dict:
        """Analyze field mapping for target file"""
        try:
            mapping_results = self.field_analyzer.analyze_field_mapping(session_id, target_file)
            return {
                'success': True,
                'field_mappings': mapping_results,
                'message': f'Successfully analyzed field mappings for {target_file}'
            }
        except Exception as e:
            logger.error(f"Error analyzing field mapping: {str(e)}")
            return {
                'success': False,
                'error': str(e)
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
    """Analyze field mapping for target file"""
    data = request.json
    session_id = data.get('session_id')
    target_file = data.get('target_file')
    
    result = analyzer.analyze_field_mapping(session_id, target_file)
    return jsonify(result)

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
    """Process chat query"""
    data = request.json
    session_id = data.get('session_id')
    message = data.get('message')
    conversation_id = data.get('conversation_id', str(uuid.uuid4()))
    
    result = analyzer.chat_query(session_id, message, conversation_id)
    return jsonify(result)

@app.route('/api/components/<session_id>')
def get_components(session_id):
    """Get all components for session"""
    try:
        components = analyzer.db_manager.get_session_components(session_id)
        return jsonify({'success': True, 'components': components})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dependencies/<session_id>')
def get_dependencies(session_id):
    """Get dependency relationships"""
    try:
        dependencies = analyzer.db_manager.get_dependencies(session_id)
        return jsonify({'success': True, 'dependencies': dependencies})
    except Exception as e:
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)