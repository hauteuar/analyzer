from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import json
import uuid
import datetime
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

@app.route('/api/export/<session_id>/<export_type>')
def export_data(session_id, export_type):
    """Export analysis data"""
    try:
        if export_type == 'field_mapping':
            data = analyzer.db_manager.export_field_mappings(session_id)
        elif export_type == 'components':
            data = analyzer.db_manager.export_components(session_id)
        elif export_type == 'dependencies':
            data = analyzer.db_manager.export_dependencies(session_id)
        else:
            return jsonify({'success': False, 'error': 'Invalid export type'})
        
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4500, debug=True)