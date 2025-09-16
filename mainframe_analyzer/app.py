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
from modules.program_flow_analyzer import ProgramFlowAnalyzer
logger = logging.getLogger(__name__)
# CHANGE 1: Import the Agentic RAG system instead of regular chat
AGENTIC_RAG_AVAILABLE = False
try:
    # Try to import the rewritten Agentic RAG system
    from modules.agentic_rag_chat import AgenticRAGChatManager, VectorStore
    AGENTIC_RAG_AVAILABLE = True
    logger.info("✅ Agentic RAG system available")
    
    # Try to import enhanced analyzers
    try:
        from modules.agentic_rag_chat import EnhancedFieldAnalyzer, EnhancedProgramFlowAnalyzer
        from modules.enhanced_field_query_analyzer import EnhancedFieldQueryAnalyzer
        from modules.program_flow_query_analyzer import ProgramFlowQueryAnalyzer
        ENHANCED_ANALYZERS_AVAILABLE = True
        logger.info("✅ Enhanced analyzers available")
    except ImportError as enhanced_error:
        ENHANCED_ANALYZERS_AVAILABLE = False
        logger.warning(f"⚠️  Enhanced analyzers not available: {enhanced_error}")
        
except ImportError as e:
    logger.warning(f"⚠️  Agentic RAG not available, falling back to basic chat: {e}")
    AGENTIC_RAG_AVAILABLE = False
    ENHANCED_ANALYZERS_AVAILABLE = False
    
    # Import fallback chat manager
    try:
        from modules.chat_manager import ChatManager
        logger.info("✅ Basic chat manager available")
    except ImportError as fallback_error:
        logger.error(f"❌ Even basic chat manager failed: {fallback_error}")
        raise Exception("No chat system available")


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
        self.db_manager.migrate_database_for_program_flow()
        # Add after line ~40 where other analyzers are initialized
        self.program_flow_analyzer = ProgramFlowAnalyzer(
            self.db_manager, 
            self.component_extractor, 
            self.llm_client
        )
                # CHANGE 2: Initialize RAG or fallback chat system
        self.chat_system_type = "unknown"
        try:
            if AGENTIC_RAG_AVAILABLE:
                logger.info("🤖 Initializing Enhanced RAG System...")
                
                # Create vector store for RAG
                from modules.agentic_rag_chat import VectorStore
                vector_store = VectorStore(self.db_manager)
                
                # Create fallback chat manager first
                from modules.chat_manager import ChatManager
                fallback_chat = ChatManager(self.llm_client, self.token_manager, self.db_manager)
                
                # Initialize the rewritten RAG system
                self.chat_manager = AgenticRAGChatManager(
                    self.llm_client, 
                    self.db_manager,
                    vector_store=vector_store
                )
                
                # Try to initialize enhanced components
                enhanced_field_ready = self._initialize_enhanced_field_analysis()
                enhanced_flow_ready = self._initialize_enhanced_program_flow()
                
                # Set system type based on what's available
                if enhanced_field_ready and enhanced_flow_ready:
                    self.chat_system_type = "agentic_rag_super_enhanced"
                    logger.info("✅ Super Enhanced RAG: Field + Program Flow Analysis")
                elif enhanced_field_ready:
                    self.chat_system_type = "agentic_rag_field_enhanced" 
                    logger.info("✅ Field Enhanced RAG")
                elif enhanced_flow_ready:
                    self.chat_system_type = "agentic_rag_flow_enhanced"
                    logger.info("✅ Program Flow Enhanced RAG")
                else:
                    self.chat_system_type = "agentic_rag_basic"
                    logger.info("✅ Basic RAG System")
                
            else:
                logger.info("📝 RAG not available, using basic chat")
                from modules.chat_manager import ChatManager
                self.chat_manager = ChatManager(self.llm_client, self.token_manager, self.db_manager)
                self.chat_system_type = "basic_chat"
                
        except Exception as e:
            logger.error(f"❌ Chat system initialization failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Ultimate fallback
            try:
                from modules.chat_manager import ChatManager
                self.chat_manager = ChatManager(self.llm_client, self.token_manager, self.db_manager)
                self.chat_system_type = "fallback_chat"
                logger.info("🔄 Using fallback chat system")
            except Exception as fallback_error:
                logger.error(f"❌ Even fallback failed: {str(fallback_error)}")
                raise Exception("Failed to initialize any chat system")
        
        # Initialize database
        self.db_manager.initialize_database()
        
        # Update database for RAG if using enhanced system
        if "agentic_rag" in self.chat_system_type:
            try:
                self._update_database_for_rag()
                logger.info("📊 Database updated for RAG system")
            except Exception as e:
                logger.warning(f"⚠️  Could not update database for RAG: {e}")
    
    def _initialize_enhanced_field_analysis(self) -> bool:
        """Initialize enhanced field analysis components"""
        try:
            # Import enhanced field analyzer from the rewritten RAG system
            from modules.agentic_rag_chat import EnhancedFieldAnalyzer
            
            # Create enhanced field analyzer
            enhanced_field_analyzer = EnhancedFieldAnalyzer(
                self.db_manager, 
                getattr(self.chat_manager, 'vector_store', None)
            )
            
            # Attach to chat manager
            self.chat_manager.enhanced_field_analyzer = enhanced_field_analyzer
            
            logger.info("🔬 Enhanced field analysis initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced field analysis failed: {str(e)}")
            return False
    
    def _initialize_enhanced_program_flow(self) -> bool:
        """Initialize enhanced program flow analysis components"""
        try:
            # Import enhanced program flow analyzer
            from modules.agentic_rag_chat import EnhancedProgramFlowAnalyzer
            
            # Create enhanced program flow analyzer
            enhanced_program_flow = EnhancedProgramFlowAnalyzer(
                self.db_manager,
                getattr(self.chat_manager, 'vector_store', None)
            )
            
            # Attach to chat manager
            self.chat_manager.enhanced_program_flow_analyzer = enhanced_program_flow
            
            logger.info("🔄 Enhanced program flow analysis initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced program flow analysis failed: {str(e)}")
            return False
        
    def _update_database_for_rag(self):
        """Update database schema for RAG system"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Add RAG-specific tables
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS rag_query_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        query TEXT NOT NULL,
                        query_type TEXT,
                        entities_found INTEGER DEFAULT 0,
                        contexts_retrieved INTEGER DEFAULT 0,
                        confidence_score REAL DEFAULT 0.0,
                        processing_time_ms INTEGER DEFAULT 0,
                        response_length INTEGER DEFAULT 0,
                        retrieval_methods TEXT,
                        user_satisfaction INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS query_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        pattern_type TEXT,
                        pattern_data TEXT,
                        frequency INTEGER DEFAULT 1,
                        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                logger.info("RAG database schema updated successfully")
                
        except Exception as e:
            logger.error(f"Error updating database for RAG: {str(e)}")
    
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
        """Upload and analyze mainframe file with enhanced RAG initialization"""
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
            
            # ENHANCED: Initialize RAG system for this session after upload
            rag_features_status = self._initialize_session_rag_features(session_id)
            
            total_time = time.time() - start_time
            logger.info(f"🎉 Analysis of {file_name} completed successfully in {total_time:.2f}s")
            
            return {
                'success': True,
                'components': components,
                'message': f'Successfully analyzed {len(components)} components from {file_name}',
                'processing_time': total_time,
                'component_breakdown': component_types,
                'chat_system_ready': True,
                'field_analysis_ready': rag_features_status['field_analysis_ready'],
                'program_flow_ready': rag_features_status['program_flow_ready'], 
                'comprehensive_analysis': rag_features_status['comprehensive_analysis'],
                'rag_system_type': self.chat_system_type,
                'rag_features_initialized': rag_features_status['features_initialized']
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {file_name}: {str(e)}")
            logger.error(f"📍 Error details: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'file_name': file_name
            }

    def _initialize_session_rag_features(self, session_id: str) -> Dict:
        """Initialize RAG features after file upload"""
        
        features_status = {
            'features_initialized': False,
            'field_analysis_ready': False,
            'program_flow_ready': False,
            'comprehensive_analysis': False,
            'initialization_time': 0
        }
        
        try:
            init_start = time.time()
            
            if "agentic_rag" in self.chat_system_type:
                logger.info("🤖 Initializing enhanced RAG features for session...")
                
                # Initialize session in chat manager
                if hasattr(self.chat_manager, 'initialize_session'):
                    self.chat_manager.initialize_session(session_id)
                    features_status['features_initialized'] = True
                    logger.info("✅ RAG session initialized")
                
                # Check enhanced field analysis
                if hasattr(self.chat_manager, 'enhanced_field_analyzer'):
                    features_status['field_analysis_ready'] = True
                    logger.info("✅ Enhanced field analysis ready")
                
                # Check enhanced program flow analysis  
                if hasattr(self.chat_manager, 'enhanced_program_flow_analyzer'):
                    features_status['program_flow_ready'] = True
                    logger.info("✅ Enhanced program flow analysis ready")
                
                # Set comprehensive status
                if features_status['field_analysis_ready'] and features_status['program_flow_ready']:
                    features_status['comprehensive_analysis'] = True
                    logger.info("✅ Comprehensive analysis capabilities active")
                
            else:
                logger.info("📝 Basic chat system - no enhanced features to initialize")
                features_status['features_initialized'] = True  # Basic system is ready
            
            features_status['initialization_time'] = time.time() - init_start
            logger.info(f"🎯 RAG feature initialization completed in {features_status['initialization_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error initializing RAG features: {str(e)}")
            # Don't fail the whole upload for RAG issues
            features_status['features_initialized'] = False
        
        return features_status
    def initialize_session(self, session_id: str):
        """Initialize RAG session - called after file upload"""
        try:
            if hasattr(self.chat_manager, '_initialize_session'):
                self.chat_manager._initialize_session(session_id)
                logger.info(f"🔄 RAG session {session_id[:8]} initialized")
            else:
                logger.info(f"📝 Basic chat - no session initialization needed")
        except Exception as e:
            logger.error(f"❌ Session initialization failed: {str(e)}")
    
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
    
    # CHANGE 5: Enhanced chat query processing for RAG system
    def chat_query(self, session_id: str, message: str, conversation_id: str) -> Dict:
        """Process chat query with enhanced RAG support"""
        try:
            if self.chat_system_type == "agentic_rag":
                # Use enhanced RAG processing
                result = self.chat_manager.process_query_with_full_features(
                    session_id, message, conversation_id
                )
                
                # Handle RAG result format
                if isinstance(result, dict):
                    return {
                        'success': True,
                        'response': result.get('response', 'No response generated'),
                        'query_plan': result.get('query_plan', {}),
                        'contexts_used': result.get('contexts_used', 0),
                        'processing_time': result.get('processing_time', 0),
                        'cached': result.get('cached', False),
                        'routed': result.get('routed', False),
                        'system_type': 'agentic_rag'
                    }
                else:
                    # Fallback for string response
                    return {
                        'success': True,
                        'response': str(result),
                        'system_type': 'agentic_rag'
                    }
            else:
                # Use basic chat processing
                response = self.chat_manager.process_query(session_id, message, conversation_id)
                return {
                    'success': True,
                    'response': response,
                    'system_type': self.chat_system_type
                }
                
        except Exception as e:
            logger.error(f"Error processing chat query: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'system_type': self.chat_system_type
            }
    
    # CHANGE 6: Add method to get chat system health
    def get_chat_system_health(self) -> Dict:
        """Enhanced chat system health with proper status"""
        try:
            health = {
                'system_type': self.chat_system_type,
                'status': 'healthy',
                'features': []
            }
            
            if "agentic_rag" in self.chat_system_type:
                # Get RAG system health
                if hasattr(self.chat_manager, 'get_system_health'):
                    rag_health = self.chat_manager.get_system_health()
                    health.update(rag_health)
                
                # Enhanced features based on system type
                if self.chat_system_type == "agentic_rag_super_enhanced":
                    health['features'] = [
                        'Vector Search',
                        'Query Classification', 
                        'Context Retrieval',
                        'Enhanced Field Analysis',
                        'Program Flow Analysis', 
                        'Dynamic Call Resolution',
                        'Business Logic Extraction',
                        'Cross-Program Analysis',
                        'Specialized Handlers',
                        'Performance Monitoring'
                    ]
                elif self.chat_system_type == "agentic_rag_field_enhanced":
                    health['features'] = [
                        'Vector Search',
                        'Query Classification',
                        'Enhanced Field Analysis',
                        'Group Structure Analysis',
                        'Conditional Assignment Tracking',
                        'Business Logic Extraction'
                    ]
                elif self.chat_system_type == "agentic_rag_flow_enhanced":
                    health['features'] = [
                        'Vector Search',
                        'Query Classification', 
                        'Program Flow Analysis',
                        'Dynamic Call Resolution',
                        'Cross-Program Analysis'
                    ]
                else:  # basic RAG
                    health['features'] = [
                        'Vector Search',
                        'Query Classification',
                        'Context Retrieval',
                        'Basic Analysis'
                    ]
                
                # Component status
                health['enhanced_components'] = {
                    'field_analyzer': hasattr(self.chat_manager, 'enhanced_field_analyzer'),
                    'program_flow_analyzer': hasattr(self.chat_manager, 'enhanced_program_flow_analyzer'),
                    'vector_store': hasattr(self.chat_manager, 'vector_store') and self.chat_manager.vector_store is not None
                }
                
            else:
                health['features'] = ['Basic Chat', 'Field Analysis', 'Source Code Context']
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting chat system health: {e}")
            return {
                'system_type': self.chat_system_type,
                'status': 'error',
                'error': str(e)
            }

    def _generate_dynamic_call_summary(self, source_code: str, dynamic_calls: List[Dict]) -> str:
        """Generate business logic summary for dynamic calls"""
        try:
            # Extract key business logic points
            summary_points = []
            
            for call in dynamic_calls:
                var_name = call.get('variable_name', '')
                resolved_programs = call.get('resolved_programs', [])
                
                summary_points.append(f"Variable {var_name} can resolve to: {[p.get('program_name') for p in resolved_programs]}")
                
                # Look for business logic that determines the variable value
                lines = source_code.split('\n')
                for i, line in enumerate(lines):
                    if var_name in line.upper() and any(logic in line.upper() for logic in ['IF', 'WHEN', 'EVALUATE']):
                        context = ' '.join(lines[max(0, i-1):i+2]).strip()
                        summary_points.append(f"Business logic: {context}")
                        break
            
            return '; '.join(summary_points)
            
        except Exception as e:
            logger.error(f"Error generating dynamic call summary: {str(e)}")
            return "Dynamic call analysis available but summary generation failed"
        
    def _get_db2_table_mapping(self, session_id: str, field_name: str) -> Dict:
        """Get DB2 table mapping for a field"""
        # Check if field appears in SQL operations
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ca.component_name, ca.analysis_result_json
                FROM component_analysis ca
                WHERE ca.session_id = ? AND ca.analysis_result_json LIKE ?
            ''', (session_id, f'%{field_name}%'))
            
            results = cursor.fetchall()
            
            for row in results:
                analysis = json.loads(row[1])
                db2_ops = analysis.get('db2_operations', [])
                
                for op in db2_ops:
                    if field_name.upper() in op.get('sql', '').upper():
                        return {
                            'mapped_to_db2': True,
                            'table_operations': self.component_extractor._extract_table_names_from_sql(op['sql']),
                            'sql_context': op['sql'][:200]
                        }
        
        return {'mapped_to_db2': False}
    
    def _get_field_sql_operations(self, session_id: str, field_name: str) -> List[Dict]:
        """Get SQL operations involving a specific field"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT ca.component_name, ca.analysis_result_json
                    FROM component_analysis ca
                    WHERE ca.session_id = ? AND ca.analysis_result_json LIKE ?
                ''', (session_id, f'%{field_name}%'))
                
                sql_operations = []
                for row in cursor.fetchall():
                    try:
                        analysis = json.loads(row[1])
                        db2_ops = analysis.get('db2_operations', [])
                        
                        for op in db2_ops:
                            if field_name.upper() in op.get('sql', '').upper():
                                sql_operations.append({
                                    'program': row[0],
                                    'sql_operation': op.get('operation_type', 'UNKNOWN'),
                                    'sql_statement': op.get('sql', ''),
                                    'table_name': self._extract_table_from_sql(op.get('sql', '')),
                                    'field_context': self._extract_field_context(op.get('sql', ''), field_name)
                                })
                    except Exception as e:
                        logger.warning(f"Error parsing SQL operations for {row[0]}: {e}")
                        continue
                
                return sql_operations
        except Exception as e:
            logger.error(f"Error getting SQL operations for field {field_name}: {e}")
            return []

    def _extract_table_from_sql(self, sql: str) -> str:
        """Extract table name from SQL statement"""
        import re
        # Simple regex to extract table names from FROM, UPDATE, INSERT INTO clauses
        patterns = [
            r'FROM\s+([A-Z0-9_]+)',
            r'UPDATE\s+([A-Z0-9_]+)',
            r'INSERT\s+INTO\s+([A-Z0-9_]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sql.upper())
            if match:
                return match.group(1)
        
        return 'UNKNOWN_TABLE'

    def _extract_field_context(self, sql: str, field_name: str) -> str:
        """Extract context around field usage in SQL"""
        lines = sql.split('\n')
        for line in lines:
            if field_name.upper() in line.upper():
                return line.strip()
        return ''
    
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

# CHANGE 7: Add chat system status endpoint
@app.route('/api/chat/status', methods=['GET'])
def get_chat_status():
    """Get chat system status and capabilities"""
    try:
        health = analyzer.get_chat_system_health()
        return jsonify({
            'success': True,
            'chat_system': health
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/test-field-analysis/<session_id>/<field_name>', methods=['GET'])
def test_field_analysis(session_id, field_name):
    """Test enhanced field analysis capabilities"""
    try:
        if (analyzer.chat_system_type == "agentic_rag_enhanced" and 
            hasattr(analyzer.chat_manager, 'enhanced_field_analyzer')):
            
            # Test comprehensive field analysis
            field_analysis = analyzer.chat_manager.enhanced_field_analyzer.analyze_field_comprehensive(
                session_id, field_name, f"Analyze the field {field_name}"
            )
            
            return jsonify({
                'success': True,
                'field_analysis': field_analysis,
                'analysis_type': 'enhanced',
                'features_detected': {
                    'group_structure': bool(field_analysis.get('group_analysis')),
                    'conditional_assignments': len(field_analysis.get('conditional_assignments', [])),
                    'control_flow_patterns': len(field_analysis.get('control_flow_patterns', [])),
                    'transaction_patterns': len(field_analysis.get('transaction_patterns', {}).get('transaction_codes', []))
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Enhanced field analysis not available',
                'system_type': analyzer.chat_system_type
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# CHANGE 8: Enhanced chat endpoint with better error handling
@app.route('/api/chat', methods=['POST'])
def enhanced_chat_query():
    try:
        data = request.json
        session_id = data.get('session_id')
        message = data.get('message')
        conversation_id = data.get('conversation_id', str(uuid.uuid4()))
        
        # ENHANCED: Better query type detection
        message_lower = message.lower()
        
        # Log the query for debugging
        logger.info(f"Chat query: {message[:100]}...")
        logger.info(f"System type: {analyzer.chat_system_type}")
        
        # Force enhanced processing for program/field queries
        if analyzer.chat_system_type in ["agentic_rag_super_enhanced", "agentic_rag_field_enhanced", "agentic_rag_flow_enhanced"]:
            
            # Check if this should be enhanced
            is_program_query = any(word in message_lower for word in [
                'program', 'call', 'xctl', 'link', 'flow', 'calls'
            ])
            
            is_field_query = any(word in message_lower for word in [
                'field', 'variable', 'move', 'value', 'assign'
            ])
            
            if is_program_query or is_field_query:
                logger.info(f"🔍 Enhanced processing triggered: program={is_program_query}, field={is_field_query}")
                
                # Force session initialization if not done
                try:
                    analyzer.chat_manager.initialize_session(session_id)
                except:
                    pass  # May already be initialized
        
        # Process with full features
        result = analyzer.chat_query(session_id, message, conversation_id)
        
        # Enhanced response logging
        if result.get('success'):
            query_plan = result.get('query_plan', {})
            logger.info(f"Query processed: type={query_plan.get('query_type')}, contexts={result.get('contexts_used', 0)}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'response': 'Chat service encountered an error'
        })    
# CHANGE 9: Add RAG analytics endpoint
@app.route('/api/chat/analytics/<session_id>', methods=['GET'])
def get_chat_analytics(session_id):
    """Get chat analytics for RAG system"""
    try:
        if analyzer.chat_system_type == "agentic_rag" and hasattr(analyzer.chat_manager, 'analytics'):
            analytics = analyzer.chat_manager.analytics.generate_session_analytics(session_id)
            return jsonify({
                'success': True,
                'analytics': analytics
            })
        else:
            return jsonify({
                'success': True,
                'analytics': {'message': 'Analytics not available for current chat system'},
                'system_type': analyzer.chat_system_type
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Add this to your Flask routes:
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
        
        # CHANGE 10: Re-update database for RAG after reset
        if analyzer.chat_system_type == "agentic_rag":
            analyzer._update_database_for_rag()
        
        return jsonify({
            'success': True,
            'message': 'Database reset successfully',
            'chat_system': analyzer.chat_system_type
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Keep all your existing routes unchanged - they will work with both systems

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
                'stored_components': results,
                'chat_system': analyzer.chat_system_type
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# [Keep all your existing routes - they remain unchanged]
# ... (all your existing routes from @app.route('/api/derived-components/<session_id>') onwards)

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
                'usage_type': row[8] or 'STATIC',
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
                },
                'db2_table_mapping': analyzer._get_db2_table_mapping(session_id, row[0]),
                'sql_operations': analyzer._get_field_sql_operations(session_id, row[0])
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
def get_dependencies_enhanced_complete(session_id):
    """Complete enhanced dependency endpoint with CICS layout resolution"""
    try:
        logger.info(f"Getting complete enhanced dependencies for session: {session_id}")
        
        # Use the complete enhanced method
        dependencies = analyzer.db_manager.get_dependencies_with_layout_resolution_complete(session_id)
        
        if not dependencies:
            components = analyzer.db_manager.get_session_components(session_id)
            return jsonify({
                'success': True, 
                'dependencies': [],
                'enhanced_features': True,
                'cics_layout_resolution': True,
                'total_count': 0,
                'debug_info': f"No dependencies found. Session has {len(components)} components."
            })
        
        # Enhanced categorization with complete layout resolution
        categorized_dependencies = {
            'input_files': [],
            'output_files': [],
            'input_output_files': [],
            'physical_file_mappings': [], 
            'cics_files_with_layouts': [],
            'cics_files_only': [],
            'db2_input_tables': [],    # NEW
            'db2_output_tables': [],   # NEW
            'jcl_confirmed_calls': [], 
            'program_calls': [],
            'missing_programs': []
        }
        
        status_counts = {
            'present': 0,
            'missing': 0,
            'file': 0,
            'file_with_layout': 0,
            'unknown': 0
        }
        
        # Process each dependency with enhanced categorization
        for dep in dependencies:
            status = dep.get('dependency_status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            rel_type = dep['relationship_type']
            interface_type = dep['interface_type']
            analysis = dep.get('analysis_details', {})
            
            # Enhanced CICS file processing
            if interface_type == 'CICS' and 'FILE' in rel_type:
                file_info = {
                    'file_name': dep['target_component'],
                    'source_program': dep['source_component'],
                    'operations': analysis.get('operations', []),
                    'io_direction': analysis.get('io_direction', 'UNKNOWN'),
                    'associated_layouts': analysis.get('associated_layouts', []),
                    'has_layout_resolution': analysis.get('layout_resolved', False),
                    'interface': 'CICS',
                    'relationship_type': rel_type,
                    'confidence': dep.get('confidence_score', 0.9)
                }
                
                if file_info['has_layout_resolution']:
                    categorized_dependencies['cics_files_with_layouts'].append(file_info)
                else:
                    categorized_dependencies['cics_files_only'].append(file_info)
                    
                # Also categorize by I/O direction
                io_dir = file_info['io_direction']
                if io_dir == 'INPUT':
                    categorized_dependencies['input_files'].append(file_info)
                elif io_dir == 'OUTPUT':
                    categorized_dependencies['output_files'].append(file_info)
                elif io_dir == 'INPUT_OUTPUT':
                    categorized_dependencies['input_output_files'].append(file_info)
            
            # Program call processing with direction arrows
            elif rel_type == 'PROGRAM_CALL':
                call_info = {
                    'source': dep['source_component'],
                    'target': dep['target_component'],
                    'call_type': analysis.get('call_type', 'CALL'),
                    'call_direction': analysis.get('call_direction', 'bidirectional'),
                    'arrow_direction': analysis.get('arrow_direction', 'bidirectional'),
                    'status': status,
                    'confidence': dep.get('confidence_score', 0.9),
                    'line_number': analysis.get('line_number', 0)
                }
                
                if status == 'missing':
                    categorized_dependencies['missing_programs'].append(call_info)
                else:
                    categorized_dependencies['program_calls'].append(call_info)
            
            if interface_type == 'DB2' and 'TABLE' in rel_type:
                table_info = {
                'table_name': dep['target_component'],
                'source_program': dep['source_component'],
                'sql_operation': analysis.get('sql_operation', 'UNKNOWN'),
                'io_direction': analysis.get('io_direction', 'UNKNOWN'),
                'interface': 'DB2',
                'relationship_type': rel_type,
                'confidence': dep.get('confidence_score', 0.95)
            }
            
            if rel_type == 'DB2_INPUT_TABLE':
                categorized_dependencies['db2_input_tables'].append(table_info)
            elif rel_type == 'DB2_OUTPUT_TABLE':
                categorized_dependencies['db2_output_tables'].append(table_info)

            if dep.get('relationship_type') == 'PHYSICAL_FILE_MAPPING':
                    categorized_dependencies['physical_file_mappings'].append({
                    'logical_file': dep.get('logical_name'),
                    'physical_dataset': dep['target_component'],
                    'dd_name': analysis.get('jcl_dd_name'),
                    'io_direction': analysis.get('io_direction')
                })
            elif dep.get('relationship_type') == 'PROGRAM_CALL':
                analysis = dep.get('analysis_details', {})
                if analysis.get('jcl_confirmation'):
                    categorized_dependencies['jcl_confirmed_calls'].append(dep)
                else:
                    categorized_dependencies['program_calls'].append(dep)
        
        logger.info(f"Enhanced categorization complete: "
                   f"{len(categorized_dependencies['cics_files_with_layouts'])} CICS files with layouts, "
                   f"{len(categorized_dependencies['missing_programs'])} missing programs")
        
        return jsonify({
            'success': True, 
            'dependencies': dependencies,
            'categorized_dependencies': categorized_dependencies,
            'status_counts': status_counts,
            'enhanced_features': True,
            'cics_layout_resolution': True,
            'total_count': len(dependencies),
            'layout_resolution_count': len(categorized_dependencies['cics_files_with_layouts'])
        })
        
    except Exception as e:
        logger.error(f"Error in complete enhanced dependencies: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'enhanced_features': True
        })

@app.route('/api/dependency-details/<session_id>/<source>/<target>')
def get_dependency_details(session_id, source, target):
    """Get detailed information about a specific dependency"""
    try:
        with analyzer.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT dr.*, ca1.source_content as source_program_content
                FROM dependency_relationships dr
                LEFT JOIN component_analysis ca1 ON (dr.source_component = ca1.component_name AND ca1.session_id = ?)
                WHERE dr.session_id = ? AND dr.source_component = ? AND dr.target_component = ?
                ORDER BY dr.created_at DESC LIMIT 1
            ''', (session_id, session_id, source, target))
            
            result = cursor.fetchone()
            
            if result:
                dependency_detail = dict(result)
                
                # Parse analysis details
                if dependency_detail['analysis_details_json']:
                    dependency_detail['analysis_details'] = json.loads(dependency_detail['analysis_details_json'])
                
                # Extract relevant source code context
                source_content = dependency_detail.get('source_program_content', '')
                if source_content and target.upper() in source_content.upper():
                    context_lines = []
                    for i, line in enumerate(source_content.split('\n')):
                        if target.upper() in line.upper():
                            start_idx = max(0, i-2)
                            end_idx = min(len(source_content.split('\n')), i+3)
                            s_cont = source_content.split('\n')[j]
                            context_lines.extend([
                                f"{j+1}: {s_cont}" 
                                for j in range(start_idx, end_idx)
                            ])
                            break
                    
                    dependency_detail['source_context'] = '\n'.join(context_lines[:10])  # Limit context
                
                return jsonify({
                    'success': True,
                    'dependency_detail': dependency_detail
                })
            
            return jsonify({
                'success': False,
                'error': f'Dependency not found: {source} -> {target}'
            })
            
    except Exception as e:
        logger.error(f"Error getting dependency details: {str(e)}")
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
                       SUM(CASE WHEN fad.usage_type = 'INPUT_OUTPUT' THEN 1 ELSE 0 END) as io_fields
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
                'io_fields': result[8]
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

# Keep all remaining routes...

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

@app.route('/api/llm-summaries/<session_id>', methods=['GET'])
def get_llm_summaries(session_id):
    """Get all LLM summaries for a session"""
    try:
        summaries = analyzer.db_manager.get_all_llm_summaries(session_id)
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
    """Get LLM summary for specific component"""
    try:
        summary = analyzer.db_manager.get_llm_summary(session_id, component_name)
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    

@app.route('/api/validate-dependencies/<session_id>')
def validate_dependencies_api(session_id):
    """Validate all dependencies and identify missing ones"""
    try:
        # Get all components in session
        components = analyzer.db_manager.get_session_components(session_id)
        uploaded_program_names = set(comp['component_name'].upper() for comp in components)
        
        # Get all dependencies
        dependencies = analyzer.db_manager.get_enhanced_dependencies(session_id)
        
        validation_results = {
            'total_dependencies': len(dependencies),
            'validated_dependencies': [],
            'missing_programs': [],
            'validation_summary': {
                'present': 0,
                'missing': 0,
                'files': 0,
                'unknown': 0
            }
        }
        
        for dep in dependencies:
            target = dep.get('target_component', '')
            interface_type = dep.get('interface_type', '')
            
            # Files and CICS files are always considered available
            if interface_type in ['FILE_SYSTEM', 'CICS'] or 'FILE' in dep.get('relationship_type', ''):
                status = 'file'
                validation_results['validation_summary']['files'] += 1
            # Programs need to be checked
            elif interface_type == 'COBOL':
                if target.upper() in uploaded_program_names:
                    status = 'present'
                    validation_results['validation_summary']['present'] += 1
                else:
                    status = 'missing'
                    validation_results['validation_summary']['missing'] += 1
                    if target not in validation_results['missing_programs']:
                        validation_results['missing_programs'].append(target)
            else:
                status = 'unknown'
                validation_results['validation_summary']['unknown'] += 1
            
            validation_results['validated_dependencies'].append({
                'source': dep.get('source_component'),
                'target': target,
                'relationship': dep.get('relationship_type'),
                'interface': interface_type,
                'status': status,
                'confidence': dep.get('confidence_score', 0)
            })
        
        return jsonify({
            'success': True,
            'validation_results': validation_results,
            'recommendations': [
                f"Upload {prog} to complete the analysis" 
                for prog in validation_results['missing_programs']
            ]
        })
        
    except Exception as e:
        logger.error(f"Error validating dependencies: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/dependency-impact/<session_id>/<program_name>')
def get_dependency_impact_api(session_id, program_name):
    """Get impact analysis of uploading a missing program"""
    try:
        # Get current missing dependencies
        missing_summary = analyzer.db_manager.get_missing_dependencies_summary(session_id)
        
        if program_name.upper() not in [p.upper() for p in missing_summary['missing_programs']]:
            return jsonify({
                'success': False,
                'error': f'Program {program_name} is not in the missing dependencies list'
            })
        
        # Find what would be resolved by uploading this program
        impact_analysis = {
            'program_name': program_name,
            'would_resolve': [],
            'calling_programs': [],
            'estimated_new_dependencies': 0
        }
        
        # Find which programs call this missing program
        for source_prog, missing_calls in missing_summary['missing_by_source'].items():
            for call in missing_calls:
                if call['program'].upper() == program_name.upper():
                    impact_analysis['calling_programs'].append({
                        'caller': source_prog,
                        'relationship': call['relationship'],
                        'call_type': call['call_type']
                    })
                    impact_analysis['would_resolve'].append(f"{source_prog} -> {program_name}")
        
        # Estimate potential new dependencies (typical COBOL program)
        impact_analysis['estimated_new_dependencies'] = len(impact_analysis['calling_programs']) * 3  # Rough estimate
        
        return jsonify({
            'success': True,
            'impact_analysis': impact_analysis,
            'recommendation': f"Uploading {program_name} would resolve {len(impact_analysis['would_resolve'])} missing dependencies and potentially discover {impact_analysis['estimated_new_dependencies']} new dependencies."
        })
        
    except Exception as e:
        logger.error(f"Error analyzing dependency impact: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/dependencies-enhanced-with-dynamic/<session_id>')
def get_enhanced_dependencies_with_dynamic_api(session_id):
    """FIXED: Get enhanced dependencies with proper dynamic call resolution to show actual program names"""
    try:
        dependencies = analyzer.db_manager.get_enhanced_dependencies(session_id)
        
        # Debug logging
        dynamic_deps = [d for d in dependencies if d.get('relationship_type') == 'DYNAMIC_PROGRAM_CALL']
        logger.info(f"API: Found {len(dynamic_deps)} dynamic dependencies out of {len(dependencies)} total")
        
        # FIXED: Separate regular dependencies from dynamic calls and resolve them
        regular_dependencies = []
        dynamic_call_groups = {}
        
        for dep in dependencies:
            if dep.get('relationship_type') == 'DYNAMIC_PROGRAM_CALL':
                analysis = dep.get('analysis_details', {})
                variable_name = analysis.get('variable_name', 'UNKNOWN')
                source_program = dep.get('source_component')
                target_component = dep.get('target_component')
                
                logger.debug(f"Processing dynamic call: {source_program} -> {target_component} via {variable_name}")
                
                # FIXED: Only process if we have a real program name, not a generic variable reference
                if (target_component and 
                    not target_component.startswith('DYNAMIC_CALL_') and 
                    not target_component.startswith('UNRESOLVED_') and
                    target_component != variable_name):
                    
                    # Create individual dependency entry for each resolved program
                    resolved_dependency = {
                        'source_component': source_program,
                        'target_component': target_component,  # ACTUAL resolved program name
                        'relationship_type': 'DYNAMIC_PROGRAM_CALL',
                        'interface_type': 'COBOL',
                        'confidence_score': dep.get('confidence_score', 0.8),
                        'dependency_status': dep.get('dependency_status', 'unknown'),
                        'display_status': dep.get('dependency_status', 'unknown'),
                        'analysis_details': {
                            **analysis,
                            'is_dynamic_resolution': True,
                            'variable_name': variable_name,
                            'resolved_program': target_component,
                            'original_variable': variable_name,
                            'resolution_confirmed': True
                        },
                        'source_code_evidence': dep.get('source_code_evidence', f"Dynamic call via {variable_name} -> {target_component}"),
                        'created_at': dep.get('created_at', ''),
                        'call_type': 'dynamic',
                        'variable_used': variable_name,
                        'resolution_method': analysis.get('resolution_method', 'variable_analysis'),
                        'business_context': f"Dynamic call via {variable_name} resolves to {target_component}",
                        # Additional metadata for UI display
                        'is_resolved_dynamic': True,
                        'display_label': f"{target_component} (via {variable_name})",
                        'dynamic_resolution_info': {
                            'variable': variable_name,
                            'method': analysis.get('resolution_method', 'unknown'),
                            'confidence': analysis.get('confidence', dep.get('confidence_score', 0.8))
                        }
                    }
                    
                    regular_dependencies.append(resolved_dependency)
                    logger.debug(f"Added resolved dynamic dependency: {source_program} -> {target_component}")
                    
                    # Also track for grouping summary
                    group_key = f"{source_program}::{variable_name}"
                    if group_key not in dynamic_call_groups:
                        dynamic_call_groups[group_key] = {
                            'variable_name': variable_name,
                            'source_program': source_program,
                            'resolved_programs': [],
                            'total_programs': 0,
                            'missing_programs': 0,
                            'present_programs': 0
                        }
                    
                    group = dynamic_call_groups[group_key]
                    group['resolved_programs'].append({
                        'program_name': target_component,
                        'status': dep.get('dependency_status', 'unknown')
                    })
                    group['total_programs'] += 1
                    
                    if dep.get('dependency_status') == 'missing':
                        group['missing_programs'] += 1
                    elif dep.get('dependency_status') == 'present':
                        group['present_programs'] += 1
                        
                else:
                    # This is an unresolved dynamic call - create a placeholder entry
                    logger.warning(f"Unresolved dynamic call: {source_program} -> {target_component} via {variable_name}")
                    unresolved_dependency = {
                        'source_component': source_program,
                        'target_component': f"UNRESOLVED({variable_name})",
                        'relationship_type': 'DYNAMIC_PROGRAM_CALL',
                        'interface_type': 'COBOL',
                        'confidence_score': 0.1,
                        'dependency_status': 'unresolved',
                        'display_status': 'unresolved',
                        'analysis_details': {
                            **analysis,
                            'is_unresolved': True,
                            'variable_name': variable_name,
                            'resolution_issue': 'Could not resolve variable to specific programs',
                            'original_target': target_component
                        },
                        'source_code_evidence': f"Unresolved dynamic call: CALL {variable_name}",
                        'call_type': 'dynamic_unresolved',
                        'variable_used': variable_name,
                        'business_context': f"Dynamic call via {variable_name} - resolution incomplete",
                        'is_resolved_dynamic': False,
                        'display_label': f"Unresolved: {variable_name}"
                    }
                    regular_dependencies.append(unresolved_dependency)
            else:
                # Regular dependency - add as-is
                regular_dependencies.append(dep)
        
        # FIXED: Categorize dependencies with resolved program names
        categorized_deps = {
            'program_calls': [],
            'file_operations': [],
            'cics_operations': [],
            'dynamic_calls': [],
            'db2_operations': [],
            'unresolved_dynamic_calls': []
        }
        
        # Process all dependencies for categorization
        for dep in regular_dependencies:
            rel_type = dep.get('relationship_type', '')
            interface_type = dep.get('interface_type', '')
            analysis_details = dep.get('analysis_details', {})
            
            dep_info = {
                'source_component': dep.get('source_component'),
                'target_component': dep.get('target_component'),
                'relationship_type': rel_type,
                'interface_type': interface_type,
                'confidence_score': dep.get('confidence_score', 0),
                'display_status': dep.get('display_status', 'unknown'),
                'analysis_details': analysis_details,
                'source_code_evidence': dep.get('source_code_evidence', ''),
                'created_at': dep.get('created_at', '')
            }
            
            # FIXED: Categorize dynamic calls with resolved program names
            if rel_type == 'DYNAMIC_PROGRAM_CALL':
                if dep.get('is_resolved_dynamic', True):
                    # Resolved dynamic call - show actual program name
                    dynamic_call_info = {
                        **dep_info,
                        'call_type': 'dynamic',
                        'variable_name': dep.get('variable_used', analysis_details.get('variable_name', '')),
                        'resolution_method': dep.get('resolution_method', analysis_details.get('resolution_method', '')),
                        'is_resolved': True,
                        'display_label': dep.get('display_label', dep.get('target_component')),
                        'business_context': dep.get('business_context', ''),
                        'dynamic_resolution_info': dep.get('dynamic_resolution_info', {}),
                        # For UI tooltips and details
                        'tooltip': f"Dynamic call via {dep.get('variable_used', 'variable')} resolves to {dep.get('target_component')}",
                        'resolution_confidence': dep.get('dynamic_resolution_info', {}).get('confidence', 0.8)
                    }
                    categorized_deps['dynamic_calls'].append(dynamic_call_info)
                else:
                    # Unresolved dynamic call
                    unresolved_info = {
                        **dep_info,
                        'call_type': 'dynamic_unresolved',
                        'variable_name': dep.get('variable_used', ''),
                        'is_resolved': False,
                        'display_label': dep.get('display_label', 'Unresolved'),
                        'resolution_issue': analysis_details.get('resolution_issue', 'Unknown resolution issue')
                    }
                    categorized_deps['unresolved_dynamic_calls'].append(unresolved_info)
                    
            elif rel_type in ['PROGRAM_CALL']:
                # Static program calls
                static_call_info = {
                    **dep_info,
                    'call_type': analysis_details.get('call_type', 'static'),
                    'line_number': analysis_details.get('line_number', 0),
                    'business_context': analysis_details.get('business_context', '')
                }
                categorized_deps['program_calls'].append(static_call_info)
                
            elif 'FILE' in rel_type and interface_type == 'CICS':
                # CICS file operations
                cics_info = {
                    **dep_info,
                    'io_direction': analysis_details.get('io_direction', 'unknown'),
                    'operations': analysis_details.get('operations', []),
                    'has_layout_association': analysis_details.get('layout_resolved', False),
                    'associated_layouts': analysis_details.get('associated_layouts', [])
                }
                categorized_deps['cics_operations'].append(cics_info)
                
            elif 'FILE' in rel_type:
                # Regular file operations
                file_info = {
                    **dep_info,
                    'io_direction': analysis_details.get('io_direction', 'unknown'),
                    'operations': analysis_details.get('operations', []),
                    'file_type': analysis_details.get('file_type', 'BATCH_FILE')
                }
                categorized_deps['file_operations'].append(file_info)
                
            elif 'DB2' in rel_type or interface_type == 'DB2':
                # DB2 table operations
                db2_info = {
                    **dep_info,
                    'sql_operation': analysis_details.get('sql_operation', 'unknown'),
                    'io_direction': analysis_details.get('io_direction', 'unknown'),
                    'table_name': dep.get('target_component', ''),
                    'sql_statement': analysis_details.get('sql_snippet', '')
                }
                categorized_deps['db2_operations'].append(db2_info)
        
        # Calculate status counts
        status_counts = {
            'present': len([d for d in regular_dependencies if d.get('display_status') == 'present']),
            'missing': len([d for d in regular_dependencies if d.get('display_status') == 'missing']),
            'file': len([d for d in regular_dependencies if d.get('display_status') == 'file']),
            'unresolved': len([d for d in regular_dependencies if d.get('display_status') == 'unresolved']),
            'unknown': len([d for d in regular_dependencies if d.get('display_status') == 'unknown'])
        }
        
        # Enhanced statistics for dynamic calls
        dynamic_stats = {
            'total_dynamic_variables': len(dynamic_call_groups),
            'total_resolved_programs': sum(g['total_programs'] for g in dynamic_call_groups.values()),
            'dynamic_calls_with_missing': len([g for g in dynamic_call_groups.values() if g['missing_programs'] > 0]),
            'fully_resolved_variables': len([g for g in dynamic_call_groups.values() if g['missing_programs'] == 0]),
            'resolution_summary': [
                {
                    'variable': group['variable_name'],
                    'source_program': group['source_program'],
                    'resolved_programs': [p['program_name'] for p in group['resolved_programs']],
                    'resolution_status': 'complete' if group['missing_programs'] == 0 else 'partial'
                }
                for group in dynamic_call_groups.values()
            ]
        }
        
        logger.info(f"FIXED dependencies processing complete:")
        logger.info(f"  - Total dependencies: {len(regular_dependencies)}")
        logger.info(f"  - Dynamic calls resolved: {len(categorized_deps['dynamic_calls'])}")
        logger.info(f"  - Dynamic calls unresolved: {len(categorized_deps['unresolved_dynamic_calls'])}")
        logger.info(f"  - Static program calls: {len(categorized_deps['program_calls'])}")
        logger.info(f"  - File operations: {len(categorized_deps['file_operations'])}")
        logger.info(f"  - CICS operations: {len(categorized_deps['cics_operations'])}")
        logger.info(f"  - DB2 operations: {len(categorized_deps['db2_operations'])}")
        
        return jsonify({
            'success': True,
            'dependencies': regular_dependencies,  # FIXED: Contains individual resolved program dependencies
            'categorized_dependencies': categorized_deps,
            'status_counts': status_counts,
            'dynamic_call_statistics': dynamic_stats,
            'total_count': len(regular_dependencies),
            'enhanced_features': True,
            'dynamic_call_grouping': True,
            'dynamic_call_resolution': True,
            'shows_actual_program_names': True,  # Key indicator for frontend
            'resolution_summary': f"Resolved {len(categorized_deps['dynamic_calls'])} dynamic program calls to actual program names"
        })
        
    except Exception as e:
        logger.error(f"Error getting enhanced dependencies with resolved dynamic calls: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        })

@app.route('/api/dependencies-enhanced/<session_id>')
def get_enhanced_dependencies_api(session_id):
    """Get enhanced dependencies with status information"""
    try:
        # Get enhanced dependencies with missing status
        dependencies = analyzer.db_manager.get_enhanced_dependencies(session_id)
        
        # Get missing dependencies summary
        missing_summary = analyzer.db_manager.get_missing_dependencies_summary(session_id)
        
        # Categorize dependencies for better UI display
        categorized_deps = {
            'program_calls': [],
            'file_operations': [],
            'cics_operations': [],
            'dynamic_calls': []
        }
        
        for dep in dependencies:
            rel_type = dep.get('relationship_type', '')
            interface_type = dep.get('interface_type', '')
            analysis_details = dep.get('analysis_details', {})
            
            dep_info = {
                'source_component': dep.get('source_component'),
                'target_component': dep.get('target_component'),
                'relationship_type': rel_type,
                'interface_type': interface_type,
                'confidence_score': dep.get('confidence_score', 0),
                'display_status': dep.get('display_status', 'unknown'),
                'analysis_details': analysis_details,
                'source_code_evidence': dep.get('source_code_evidence', ''),
                'created_at': dep.get('created_at', '')
            }
            
            # Categorize for UI display
            if rel_type == 'DYNAMIC_PROGRAM_CALL':
                dep_info['call_type'] = 'dynamic'
                dep_info['variable_name'] = analysis_details.get('variable_name', '')
                dep_info['resolution_method'] = analysis_details.get('resolution_method', '')
                categorized_deps['dynamic_calls'].append(dep_info)
            elif rel_type in ['PROGRAM_CALL']:
                dep_info['call_type'] = analysis_details.get('call_type', 'static')
                categorized_deps['program_calls'].append(dep_info)
            elif 'FILE' in rel_type and interface_type == 'CICS':
                dep_info['io_direction'] = analysis_details.get('io_direction', 'unknown')
                dep_info['operations'] = analysis_details.get('operations', [])
                categorized_deps['cics_operations'].append(dep_info)
            elif 'FILE' in rel_type:
                dep_info['io_direction'] = analysis_details.get('io_direction', 'unknown')
                dep_info['operations'] = analysis_details.get('operations', [])
                categorized_deps['file_operations'].append(dep_info)
        
        return jsonify({
            'success': True,
            'dependencies': dependencies,
            'categorized_dependencies': categorized_deps,
            'missing_summary': missing_summary,
            'total_count': len(dependencies),
            'status_counts': {
                'present': len([d for d in dependencies if d.get('display_status') == 'present']),
                'missing': len([d for d in dependencies if d.get('display_status') == 'missing']),
                'file': len([d for d in dependencies if d.get('display_status') == 'file']),
                'unknown': len([d for d in dependencies if d.get('display_status') == 'unknown'])
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting enhanced dependencies: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/dynamic-call-analysis/<session_id>')
def get_dynamic_call_analysis_api(session_id):
    """Get detailed analysis of dynamic program calls"""
    try:
        # Get all components to analyze dynamic calls
        components = analyzer.db_manager.get_session_components(session_id)
        
        dynamic_call_analysis = []
        
        for component in components:
            if component.get('component_type') != 'PROGRAM':
                continue
                
            try:
                # Get component analysis
                analysis_result = json.loads(component.get('analysis_result_json', '{}'))
                program_calls = analysis_result.get('program_calls', [])
                
                # Filter for dynamic calls
                dynamic_calls = [call for call in program_calls if call.get('call_type') == 'dynamic']
                
                if dynamic_calls:
                    # Get source code for business logic analysis
                    source_data = analyzer.db_manager.get_component_source_code(
                        session_id, component.get('component_name'), max_size=100000
                    )
                    
                    program_analysis = {
                        'program_name': component.get('component_name'),
                        'dynamic_calls': dynamic_calls,
                        'total_dynamic_calls': len(dynamic_calls),
                        'variables_used': list(set(call.get('variable_name', '') for call in dynamic_calls)),
                        'resolved_programs': [],
                        'business_logic_summary': ''
                    }
                    
                    # Collect all resolved programs
                    for call in dynamic_calls:
                        resolved_programs = call.get('resolved_programs', [])
                        for resolved in resolved_programs:
                            if resolved not in program_analysis['resolved_programs']:
                                program_analysis['resolved_programs'].append(resolved)
                    
                    # Generate business logic summary
                    if source_data.get('success') and source_data.get('components'):
                        source_code = source_data['components'][0].get('source_for_chat', '')
                        program_analysis['business_logic_summary'] = analyzer._generate_dynamic_call_summary(
                            source_code, dynamic_calls
                        )
                    
                    dynamic_call_analysis.append(program_analysis)
                    
            except Exception as comp_error:
                logger.error(f"Error analyzing component {component.get('component_name')}: {str(comp_error)}")
                continue
        
        return jsonify({
            'success': True,
            'dynamic_call_analysis': dynamic_call_analysis,
            'total_programs_with_dynamic_calls': len(dynamic_call_analysis),
            'summary': {
                'total_dynamic_calls': sum(p['total_dynamic_calls'] for p in dynamic_call_analysis),
                'unique_variables': list(set(var for p in dynamic_call_analysis for var in p['variables_used'])),
                'all_resolved_programs': list(set(prog['program_name'] for p in dynamic_call_analysis 
                                                for prog in p['resolved_programs']))
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting dynamic call analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })
    
@app.route('/api/program-flow/<session_id>/<program_name>', methods=['GET'])
def analyze_program_flow_fixed(session_id, program_name):
    """FIXED: Analyze complete program flow with dependency refresh"""
    try:
        logger.info(f"Program flow analysis request: {program_name} in session {session_id[:8]}")
        
        # Check if program exists in session
        components = analyzer.db_manager.get_session_components(session_id)
        program_exists = any(c['component_name'].upper() == program_name.upper() for c in components)
        
        if not program_exists:
            available_programs = [c['component_name'] for c in components if c.get('component_type') == 'PROGRAM']
            return jsonify({
                'success': False,
                'error': f'Program {program_name} not found in session',
                'available_programs': available_programs
            })
        
        # FIXED: Force dependency refresh before analysis
        analyzer.program_flow_analyzer._refresh_dependency_status(session_id)
        
        # Analyze program flow with refreshed dependencies
        flow_analysis = analyzer.program_flow_analyzer.analyze_complete_program_flow(session_id, program_name)
        
        # Enhanced response with status breakdown
        program_chain = flow_analysis.get('program_chain', [])
        available_count = len([s for s in program_chain if not s.get('is_missing')])
        missing_count = len([s for s in program_chain if s.get('is_missing')])
        
        return jsonify({
            'success': True,
            'flow_analysis': flow_analysis,
            'program_name': program_name,
            'analysis_type': 'complete_program_flow',
            'statistics': {
                'total_programs': len(program_chain),
                'available_programs': available_count,
                'missing_programs': missing_count,
                'field_flows': len(flow_analysis.get('field_flows', [])),
                'file_operations': len(flow_analysis.get('file_operations', []))
            },
            'dependency_refresh': True  # Indicate that dependencies were refreshed
        })
        
    except Exception as e:
        logger.error(f"Error in program flow analysis: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        })

@app.route('/api/refresh-dependencies/<session_id>', methods=['POST'])
def refresh_dependencies_status(session_id):
    """Manually refresh dependency status after uploading new programs"""
    try:
        logger.info(f"Manual dependency refresh requested for session {session_id[:8]}")
        
        # Get current component count
        components = analyzer.db_manager.get_session_components(session_id)
        program_count = len([c for c in components if c.get('component_type') == 'PROGRAM'])
        
        # Refresh dependency status
        analyzer.program_flow_analyzer._refresh_dependency_status(session_id)
        
        # Get updated dependency counts
        dependencies = analyzer.db_manager.get_enhanced_dependencies(session_id)
        status_counts = {}
        for dep in dependencies:
            status = dep.get('dependency_status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return jsonify({
            'success': True,
            'message': 'Dependencies refreshed successfully',
            'program_count': program_count,
            'dependency_status_counts': status_counts,
            'total_dependencies': len(dependencies)
        })
        
    except Exception as e:
        logger.error(f"Error refreshing dependencies: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })
@app.route('/api/program-flow-enhanced/<session_id>/<program_name>', methods=['GET'])
def get_enhanced_program_flow(session_id, program_name):
    """Get enhanced program flow with detailed layout associations"""
    try:
        logger.info(f"Enhanced program flow request: {program_name} in session {session_id[:8]}")
        
        # Check if program exists
        components = analyzer.db_manager.get_session_components(session_id)
        program_exists = any(c['component_name'].upper() == program_name.upper() for c in components)
        
        if not program_exists:
            available_programs = [c['component_name'] for c in components if c.get('component_type') == 'PROGRAM']
            return jsonify({
                'success': False,
                'error': f'Program {program_name} not found in session',
                'available_programs': available_programs
            })
        
        # Use enhanced analysis method
        flow_analysis = analyzer.program_flow_analyzer.analyze_complete_program_flow_enhanced(session_id, program_name)
        
        # Create enhanced visualization data
        visualization_data = create_enhanced_visualization_data(flow_analysis, session_id)
        
        return jsonify({
            'success': True,
            'flow_analysis': flow_analysis,
            'visualization_data': visualization_data,
            'program_name': program_name,
            'enhanced_features': True,
            'layout_details_included': True,
            'field_transformation_analysis': True,
            'statistics': {
                'total_programs': len(flow_analysis.get('program_chain', [])),
                'layout_associations': len(flow_analysis.get('layout_associations', [])),
                'field_flows': len(flow_analysis.get('field_flows', [])),
                'file_operations': len(flow_analysis.get('file_operations', [])),
                'missing_programs': len(flow_analysis.get('missing_programs', []))
            }
        })
        
    except Exception as e:
        logger.error(f"Error in enhanced program flow: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def create_enhanced_visualization_data(flow_analysis, session_id):
    """Create enhanced visualization data with layout details"""
    
    visualization_data = {
        'nodes': [],
        'edges': [],
        'program_chain_summary': [],
        'layout_associations': flow_analysis.get('layout_associations', []),
        'field_flows': {},
        'missing_programs': flow_analysis.get('missing_programs', []),
        'file_operations': flow_analysis.get('file_operations', [])
    }
    
    # Create program nodes with layout information
    programs_in_flow = set()
    program_details = {}
    
    # Collect program information
    for step in flow_analysis.get('program_chain', []):
        programs_in_flow.add(step['source_program'])
        programs_in_flow.add(step['target_program'])
        
        program_details[step['source_program']] = {
            'status': 'present',
            'is_starting': step['source_program'] == flow_analysis['starting_program']
        }
        
        program_details[step['target_program']] = {
            'status': 'missing' if step.get('is_missing') else 'present',
            'is_starting': False
        }
    
    # Create program nodes with enhanced layout associations
    node_levels = assign_node_levels(flow_analysis.get('program_chain', []), flow_analysis['starting_program'])
    
    for program in programs_in_flow:
        details = program_details.get(program, {'status': 'unknown', 'is_starting': False})
        level = node_levels.get(program, 999)
        
        # Get layout associations for this program
        program_layouts = [layout for layout in flow_analysis.get('layout_associations', []) 
                          if layout.get('program_name') == program]
        
        node = {
            'id': program,
            'label': program,
            'type': 'program',
            'status': details['status'],
            'is_starting_program': details.get('is_starting', False),
            'level': level,
            'x': 100 + (level * 200),
            'y': 100 + (len([p for p, l in node_levels.items() if l == level and p != program]) * 80),
            'width': 120,
            'height': 60,
            'layout_count': len(program_layouts),
            'associated_layouts': [layout['layout_name'] for layout in program_layouts],
            'layout_details': program_layouts
        }
        
        visualization_data['nodes'].append(node)
    
    # Create program chain summary with enhanced details
    for step in flow_analysis.get('program_chain', []):
        summary_item = {
            'sequence': step.get('sequence'),
            'source_program': step.get('source_program'),
            'target_program': step.get('target_program'),
            'call_type': step.get('call_type', 'CALL'),
            'status': 'MISSING' if step.get('is_missing') else 'AVAILABLE',
            'business_context': step.get('business_context', ''),
            'variable_name': step.get('variable_name', ''),
            'confidence': step.get('confidence', 0.8),
            'data_passed_count': len(step.get('data_passed', [])),
            'layout_associations_count': len([layout for layout in flow_analysis.get('layout_associations', []) 
                                            if layout.get('program_name') == step.get('source_program')])
        }
        visualization_data['program_chain_summary'].append(summary_item)
    
    # Enhanced field flows with transformation details
    field_flows_enhanced = {}
    for field_flow in flow_analysis.get('field_flows', []):
        field_name = field_flow.get('field_name')
        if field_name:
            if field_name not in field_flows_enhanced:
                field_flows_enhanced[field_name] = []
            
            # Add transformation details to flow info
            flow_info = {
                'from': field_flow.get('source_program'),
                'to': field_flow.get('target_program'),
                'type': field_flow.get('flow_type'),
                'transformation': field_flow.get('transformation_logic', 'Direct transfer'),
                'source_layouts': field_flow.get('source_layouts', []),
                'target_layouts': field_flow.get('target_layouts', []),
                'transformation_details': field_flow.get('transformation_details', {})
            }
            field_flows_enhanced[field_name].append(flow_info)
    
    visualization_data['field_flows'] = field_flows_enhanced
    
    # Add file operations with layout connections
    enhanced_file_ops = []
    for file_op in flow_analysis.get('file_operations', []):
        enhanced_op = {
            **file_op,
            'layout_connections': file_op.get('layout_details', []),
            'fields_involved': file_op.get('fields_involved', [])
        }
        enhanced_file_ops.append(enhanced_op)
    
    visualization_data['file_operations'] = enhanced_file_ops
    
    return visualization_data

def assign_node_levels(program_chain, starting_program):
    """Assign hierarchical levels to programs for visualization"""
    node_levels = {}
    
    # Starting program at level 0
    if starting_program:
        node_levels[starting_program] = 0
    
    # Assign levels based on call sequence
    for step in sorted(program_chain, key=lambda x: x.get('sequence', 0)):
        source = step.get('source_program')
        target = step.get('target_program')
        
        if source in node_levels:
            source_level = node_levels[source]
            if target not in node_levels:
                node_levels[target] = source_level + 1
        elif source not in node_levels:
            # Assign a default level if source not yet assigned
            node_levels[source] = 0
            if target not in node_levels:
                node_levels[target] = 1
    
    return node_levels

@app.route('/api/program-flow-visualization/<session_id>/<program_name>', methods=['GET'])

def get_program_flow_visualization(session_id, program_name):
    """FINAL FIXED: Complete program flow using existing analyzed dependencies"""
    try:
        logger.info(f"FIXED program flow visualization for {program_name} using existing dependency data")
        
        # Get complete dependency data (already analyzed and stored)
        dependency_data = analyzer.db_manager.get_complete_dependency_data_for_flow(session_id)
        
        if not dependency_data['dependencies']:
            return jsonify({
                'success': False,
                'error': f'No dependencies found for session {session_id}. Upload and analyze COBOL programs first.',
                'recommendations': ['Upload COBOL programs', 'Run dependency analysis', 'Check session data']
            })
        
        # Check if starting program exists
        uploaded_programs = set(comp['component_name'].upper() for comp in dependency_data['components'])
        if program_name.upper() not in uploaded_programs:
            available_programs = [comp['component_name'] for comp in dependency_data['components'] if comp['component_type'] == 'PROGRAM']
            return jsonify({
                'success': False,
                'error': f'Program {program_name} not found in session',
                'available_programs': available_programs[:10]
            })
        
        # Build complete flow visualization
        flow_viz = build_final_flow_visualization(
            session_id, program_name, dependency_data, analyzer.db_manager
        )
        
        return jsonify({
            'success': True,
            'flow_visualization': flow_viz,
            'program_name': program_name,
            'data_source': 'existing_analyzed_dependencies',
            'statistics': {
                'total_programs_in_flow': len(flow_viz.get('program_nodes', [])),
                'missing_programs': len(flow_viz.get('missing_programs', [])),
                'file_operations': len(flow_viz.get('file_nodes', [])),
                'db2_operations': len(flow_viz.get('db2_nodes', [])),
                'dynamic_calls': len([e for e in flow_viz.get('program_edges', []) if e.get('is_dynamic')])
            }
        })
        
    except Exception as e:
        logger.error(f"Error in fixed program flow visualization: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def build_final_flow_visualization(session_id, starting_program, dependency_data, db_manager):
    """Build flow visualization using existing dependency data"""
    
    dependencies = dependency_data['dependencies']
    components = dependency_data['components']
    uploaded_programs = set(comp['component_name'].upper() for comp in components)
    
    flow_viz = {
        'starting_program': starting_program,
        'program_nodes': [],
        'file_nodes': [],
        'db2_nodes': [],
        'program_edges': [],
        'file_edges': [],
        'db2_edges': [],
        'missing_programs': []
    }
    
    # Build program hierarchy from existing dependencies
    program_hierarchy = {}
    programs_to_process = [(starting_program, 0)]
    processed = set()
    
    while programs_to_process and len(processed) < 50:
        current_program, depth = programs_to_process.pop(0)
        
        if current_program in processed or depth > 5:
            continue
            
        processed.add(current_program)
        
        program_info = {
            'program_name': current_program,
            'depth': depth,
            'is_missing': current_program.upper() not in uploaded_programs,
            'calls_to': [],
            'file_operations': [],
            'db2_operations': []
        }
        
        # Process dependencies for this program
        for dep in dependencies:
            if dep['source_component'] != current_program:
                continue
                
            target = dep['target_component']
            rel_type = dep['relationship_type']
            interface = dep['interface_type']
            analysis = dep.get('analysis_details', {})
            
            # Program calls
            if rel_type in ['PROGRAM_CALL', 'DYNAMIC_PROGRAM_CALL']:
                call_info = {
                    'target': target,
                    'call_type': analysis.get('call_type', 'CALL'),
                    'is_dynamic': rel_type == 'DYNAMIC_PROGRAM_CALL',
                    'variable_name': analysis.get('variable_name', ''),
                    'status': dep.get('dependency_status', 'unknown')
                }
                program_info['calls_to'].append(call_info)
                
                # Queue target for processing
                if target.upper() in uploaded_programs and depth < 5:
                    programs_to_process.append((target, depth + 1))
            
            # File operations
            elif 'FILE' in rel_type or interface in ['FILE_SYSTEM', 'CICS']:
                file_info = {
                    'file_name': target,
                    'operations': analysis.get('operations', []),
                    'io_direction': analysis.get('io_direction', 'UNKNOWN'),
                    'interface_type': interface,
                    'has_layout_resolution': analysis.get('layout_resolved', False)
                }
                program_info['file_operations'].append(file_info)
            
            # DB2 operations
            elif interface == 'DB2':
                db2_info = {
                    'table_name': target,
                    'sql_operation': analysis.get('sql_operation', 'UNKNOWN'),
                    'io_direction': analysis.get('io_direction', 'UNKNOWN')
                }
                program_info['db2_operations'].append(db2_info)
        
        program_hierarchy[current_program] = program_info
    
    # Calculate positions
    level_positions = {}
    for program, info in program_hierarchy.items():
        level_positions[program] = info['depth']
    
    # Create nodes and edges
    for program_name, program_info in program_hierarchy.items():
        is_missing = program_info['is_missing']
        level = level_positions.get(program_name, 0)
        
        # Program node
        program_node = {
            'id': program_name,
            'label': program_name,
            'type': 'program',
            'status': 'missing' if is_missing else 'present',
            'is_starting_program': program_name == starting_program,
            'level': level,
            'x': 100 + (level * 200),
            'y': 100 + (list(program_hierarchy.keys()).index(program_name) % 5) * 80,
            'width': 140,
            'height': 70
        }
        flow_viz['program_nodes'].append(program_node)
        
        if is_missing:
            flow_viz['missing_programs'].append({
                'program_name': program_name,
                'status': 'missing'
            })
        
        # Program call edges
        for call in program_info['calls_to']:
            edge = {
                'from': program_name,
                'to': call['target'],
                'type': 'program_call',
                'call_type': call['call_type'],
                'is_dynamic': call['is_dynamic'],
                'variable_name': call['variable_name'],
                'status': call['status'],
                'label': f"DYNAMIC via {call['variable_name']}" if call['is_dynamic'] else call['call_type']
            }
            flow_viz['program_edges'].append(edge)
        
        # File nodes and edges
        for i, file_op in enumerate(program_info['file_operations']):
            file_node_id = f"FILE_{file_op['file_name']}_{program_name}_{i}"
            
            file_node = {
                'id': file_node_id,
                'label': file_op['file_name'],
                'type': 'file',
                'interface_type': file_op['interface_type'],
                'io_direction': file_op['io_direction'],
                'operations': file_op['operations'],
                'has_layout_resolution': file_op['has_layout_resolution'],
                'x': 80 + (level * 200) + (i * 130),
                'y': 300,
                'width': 120,
                'height': 50
            }
            flow_viz['file_nodes'].append(file_node)
            
            file_edge = {
                'from': program_name,
                'to': file_node_id,
                'type': 'file_operation',
                'io_direction': file_op['io_direction'],
                'operations': file_op['operations'],
                'label': f"{file_op['io_direction']}: {', '.join(file_op['operations'])}"
            }
            flow_viz['file_edges'].append(file_edge)
        
        # DB2 nodes and edges
        for i, db2_op in enumerate(program_info['db2_operations']):
            db2_node_id = f"DB2_{db2_op['table_name']}_{program_name}_{i}"
            
            db2_node = {
                'id': db2_node_id,
                'label': db2_op['table_name'],
                'type': 'db2_table',
                'sql_operation': db2_op['sql_operation'],
                'io_direction': db2_op['io_direction'],
                'x': 80 + (level * 200) + (i * 130),
                'y': 400,
                'width': 120,
                'height': 50
            }
            flow_viz['db2_nodes'].append(db2_node)
            
            db2_edge = {
                'from': program_name,
                'to': db2_node_id,
                'type': 'db2_operation',
                'sql_operation': db2_op['sql_operation'],
                'label': f"DB2 {db2_op['sql_operation']}"
            }
            flow_viz['db2_edges'].append(db2_edge)
    
    return flow_viz

@app.route('/api/field-flow-trace/<session_id>/<field_name>', methods=['GET'])
def trace_field_flow(session_id, field_name):
    """Trace a specific field through program flows"""
    try:
        with analyzer.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT fdf.*, pft.source_program, pft.target_program, pft.call_mechanism
                FROM field_data_flow fdf
                JOIN program_flow_traces pft ON fdf.flow_id = pft.flow_id 
                WHERE fdf.session_id = ? AND UPPER(fdf.field_name) = UPPER(?)
                ORDER BY fdf.sequence_in_flow
            ''', (session_id, field_name))
            
            field_flows = [dict(row) for row in cursor.fetchall()]
            
            if not field_flows:
                return jsonify({
                    'success': False,
                    'message': f'No flow data found for field {field_name}'
                })
            
            # Build field journey
            field_journey = {
                'field_name': field_name,
                'flow_steps': [],
                'programs_involved': set(),
            }
            
            for flow in field_flows:
                step = {
                    'sequence': flow['sequence_in_flow'],
                    'source_program': flow['source_program'],
                    'target_program': flow['target_program'],
                    'flow_type': flow['flow_type'],
                    'transformation': flow['transformation_logic'],
                    'call_mechanism': flow['call_mechanism']
                }
                field_journey['flow_steps'].append(step)
                field_journey['programs_involved'].add(flow['source_program'])
                field_journey['programs_involved'].add(flow['target_program'])
            
            field_journey['programs_involved'] = list(field_journey['programs_involved'])
            
            return jsonify({
                'success': True,
                'field_journey': field_journey
            })
            
    except Exception as e:
        logger.error(f"Error tracing field flow: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/missing-program-impact/<session_id>/<program_name>', methods=['GET'])
def get_missing_program_impact(session_id, program_name):
    """Get impact analysis for a missing program"""
    try:
        with analyzer.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT pft.source_program, pft.call_mechanism, pft.business_context
                FROM program_flow_traces pft
                WHERE pft.session_id = ? AND pft.target_program = ?
            ''', (session_id, program_name))
            
            blocked_flows = [dict(row) for row in cursor.fetchall()]
        
        impact_analysis = {
            'missing_program': program_name,
            'blocked_flows': blocked_flows,
            'calling_programs': [flow['source_program'] for flow in blocked_flows],
            'impact_description': f"Missing {program_name} blocks {len(blocked_flows)} program flows",
            'recommendations': [
                f"Upload {program_name} to restore complete flow analysis",
                f"This program is called by: {', '.join(set(flow['source_program'] for flow in blocked_flows))}"
            ]
        }
        
        return jsonify({
            'success': True,
            'impact_analysis': impact_analysis
        })
        
    except Exception as e:
        logger.error(f"Error analyzing missing program impact: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


# Add this route to your main.py Flask application

@app.route('/api/layout-field-details/<session_id>/<layout_name>')
def get_layout_field_details(session_id, layout_name):
    """Get detailed field information for a specific layout"""
    try:
        with analyzer.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get layout basic information
            cursor.execute('''
                SELECT layout_name, friendly_name, business_purpose, fields_count,
                       record_classification, record_usage_description, program_name
                FROM record_layouts 
                WHERE session_id = ? AND layout_name = ?
                LIMIT 1
            ''', (session_id, layout_name))
            
            layout_info = cursor.fetchone()
            
            if not layout_info:
                return jsonify({
                    'success': False,
                    'error': f'Layout {layout_name} not found'
                })
            
            # Get all fields for this layout with detailed information
            cursor.execute('''
                SELECT fad.field_name, fad.friendly_name, fad.usage_type, 
                       fad.business_purpose, fad.mainframe_data_type, fad.oracle_data_type,
                       fad.mainframe_length, fad.oracle_length, fad.total_program_references,
                       fad.move_source_count, fad.move_target_count, fad.definition_code
                FROM field_analysis_details fad
                JOIN record_layouts rl ON fad.field_id = rl.id
                WHERE fad.session_id = ? AND rl.layout_name = ?
                ORDER BY fad.field_name
            ''', (session_id, layout_name))
            
            fields_data = cursor.fetchall()
            
            # Get usage context for this layout
            cursor.execute('''
                SELECT fad.usage_type, COUNT(*) as count
                FROM field_analysis_details fad
                JOIN record_layouts rl ON fad.field_id = rl.id
                WHERE fad.session_id = ? AND rl.layout_name = ?
                GROUP BY fad.usage_type
                ORDER BY count DESC
            ''', (session_id, layout_name))
            
            usage_stats = cursor.fetchall()
            
            # Build layout details structure
            layout_details = {
                'layout_name': layout_info[0],
                'friendly_name': layout_info[1] or layout_info[0].replace('-', ' ').title(),
                'business_purpose': layout_info[2] or f'Data structure for {layout_info[6] or "program"}',
                'fields_count': len(fields_data),
                'record_classification': layout_info[4] or 'RECORD',
                'program_name': layout_info[6] or 'Unknown',
                'usage_context': determine_layout_usage_context(usage_stats),
                'usage_statistics': [{'usage_type': row[0], 'field_count': row[1]} for row in usage_stats],
                'fields': []
            }
            
            # Process field details
            for field_row in fields_data:
                field_detail = {
                    'field_name': field_row[0],
                    'friendly_name': field_row[1] or field_row[0].replace('-', ' ').title(),
                    'usage_type': field_row[2] or 'STATIC',
                    'business_purpose': field_row[3] or f'Field {field_row[0]} processing',
                    'data_type': field_row[4] or 'UNKNOWN',
                    'oracle_data_type': field_row[5] or 'VARCHAR2(50)',
                    'mainframe_length': field_row[6] or 0,
                    'oracle_length': field_row[7] or 50,
                    'reference_count': field_row[8] or 0,
                    'move_operations': {
                        'source_count': field_row[9] or 0,
                        'target_count': field_row[10] or 0
                    },
                    'definition_code': field_row[11] or '',
                    'complexity_score': calculate_field_complexity(field_row)
                }
                layout_details['fields'].append(field_detail)
            
            return jsonify({
                'success': True,
                'layout_details': layout_details
            })
            
    except Exception as e:
        logger.error(f"Error getting layout field details for {layout_name}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def determine_layout_usage_context(usage_stats):
    """Determine the primary usage context for a layout based on field usage statistics"""
    if not usage_stats:
        return "General data structure"
    
    primary_usage = usage_stats[0][0]  # Most common usage type
    total_fields = sum(row[1] for row in usage_stats)
    primary_count = usage_stats[0][1]
    primary_percentage = (primary_count / total_fields) * 100
    
    context_map = {
        'INPUT': f"Primary input processing ({primary_percentage:.0f}% of fields)",
        'OUTPUT': f"Primary output generation ({primary_percentage:.0f}% of fields)", 
        'INPUT_OUTPUT': f"Bidirectional data processing ({primary_percentage:.0f}% of fields)",
        'DERIVED': f"Calculated field generation ({primary_percentage:.0f}% of fields)",
        'STATIC': f"Configuration and constants ({primary_percentage:.0f}% of fields)",
        'REFERENCE': f"Data lookup and validation ({primary_percentage:.0f}% of fields)"
    }
    
    base_context = context_map.get(primary_usage, f"Data processing ({primary_percentage:.0f}% {primary_usage.lower()})")
    
    # Add mixed usage note if significant secondary usage
    if len(usage_stats) > 1 and usage_stats[1][1] > total_fields * 0.2:  # >20% secondary usage
        secondary_usage = usage_stats[1][0]
        secondary_percentage = (usage_stats[1][1] / total_fields) * 100
        base_context += f", with significant {secondary_usage.lower()} usage ({secondary_percentage:.0f}%)"
    
    return base_context

def calculate_field_complexity(field_row):
    """Calculate a complexity score for a field based on its usage patterns"""
    reference_count = field_row[8] or 0
    move_source = field_row[9] or 0
    move_target = field_row[10] or 0
    
    # Simple complexity calculation
    complexity = 0
    
    # More references = more complex
    if reference_count > 10:
        complexity += 0.3
    elif reference_count > 5:
        complexity += 0.2
    elif reference_count > 0:
        complexity += 0.1
    
    # Bidirectional usage = more complex
    if move_source > 0 and move_target > 0:
        complexity += 0.4
    elif move_source > 0 or move_target > 0:
        complexity += 0.2
    
    # Field name complexity (heuristic)
    field_name = field_row[0] or ''
    if len(field_name) > 20 or '-' in field_name:
        complexity += 0.1
    
    return min(complexity, 1.0)  # Cap at 1.0

@app.route('/api/trace-layout-usage/<session_id>/<layout_name>')
def trace_layout_usage(session_id, layout_name):
    """Trace how a layout is used across programs"""
    try:
        with analyzer.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get programs that use this layout
            cursor.execute('''
                SELECT DISTINCT rl.program_name, ca.component_type, ca.business_purpose
                FROM record_layouts rl
                LEFT JOIN component_analysis ca ON (rl.program_name = ca.component_name AND ca.session_id = ?)
                WHERE rl.session_id = ? AND rl.layout_name = ?
                AND rl.program_name IS NOT NULL
            ''', (session_id, session_id, layout_name))
            
            using_programs = cursor.fetchall()
            
            # Get field usage across programs
            cursor.execute('''
                SELECT fad.program_name, fad.field_name, fad.usage_type, 
                       fad.move_source_count, fad.move_target_count, fad.total_program_references
                FROM field_analysis_details fad
                JOIN record_layouts rl ON fad.field_id = rl.id
                WHERE fad.session_id = ? AND rl.layout_name = ?
                ORDER BY fad.program_name, fad.total_program_references DESC
            ''', (session_id, layout_name))
            
            field_usage = cursor.fetchall()
            
            # Get program call relationships involving these programs
            cursor.execute('''
                SELECT dr.source_component, dr.target_component, dr.relationship_type
                FROM dependency_relationships dr
                WHERE dr.session_id = ? 
                AND (dr.source_component IN ({}) OR dr.target_component IN ({}))
                AND dr.relationship_type LIKE '%PROGRAM_CALL%'
            '''.format(
                ','.join('?' * len(using_programs)), 
                ','.join('?' * len(using_programs))
            ), tuple([session_id] + [prog[0] for prog in using_programs] * 2))
            
            program_relationships = cursor.fetchall()
            
            # Build usage trace
            usage_trace = {
                'layout_name': layout_name,
                'using_programs': [
                    {
                        'program_name': prog[0],
                        'program_type': prog[1] or 'PROGRAM',
                        'business_purpose': prog[2] or f'Program {prog[0]} processing'
                    } for prog in using_programs
                ],
                'field_usage_by_program': {},
                'program_relationships': [
                    {
                        'source': rel[0],
                        'target': rel[1], 
                        'relationship_type': rel[2]
                    } for rel in program_relationships
                ],
                'usage_flow': analyze_layout_usage_flow(using_programs, field_usage, program_relationships)
            }
            
            # Group field usage by program
            for field_usage_row in field_usage:
                program = field_usage_row[0]
                if program not in usage_trace['field_usage_by_program']:
                    usage_trace['field_usage_by_program'][program] = []
                
                usage_trace['field_usage_by_program'][program].append({
                    'field_name': field_usage_row[1],
                    'usage_type': field_usage_row[2],
                    'move_source_count': field_usage_row[3] or 0,
                    'move_target_count': field_usage_row[4] or 0,
                    'reference_count': field_usage_row[5] or 0
                })
            
            return jsonify({
                'success': True,
                'usage_trace': usage_trace
            })
            
    except Exception as e:
        logger.error(f"Error tracing layout usage: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def analyze_layout_usage_flow(using_programs, field_usage, program_relationships):
    """Analyze how the layout flows through the program chain"""
    
    # Create a simple flow analysis
    programs = [prog[0] for prog in using_programs]
    flow_steps = []
    
    # Find program call chains
    call_chains = []
    for rel in program_relationships:
        source, target, rel_type = rel
        if source in programs and target in programs:
            call_chains.append({
                'from_program': source,
                'to_program': target,
                'call_type': rel_type,
                'layout_passed': True  # Assume layout is passed in program calls
            })
    
    # Build flow narrative
    flow_narrative = []
    if call_chains:
        flow_narrative.append("Layout flows through program calls:")
        for chain in call_chains:
            flow_narrative.append(f"  {chain['from_program']} -> {chain['to_program']} ({chain['call_type']})")
    else:
        flow_narrative.append("Layout used independently in each program (no direct program calls detected)")
    
    # Analyze field modification patterns
    modification_programs = []
    for program in programs:
        program_fields = [f for f in field_usage if f[0] == program]
        
        # Count field modifications
        input_fields = len([f for f in program_fields if f[2] == 'INPUT' or f[4] > 0])  # move_target_count > 0
        output_fields = len([f for f in program_fields if f[2] == 'OUTPUT' or f[3] > 0])  # move_source_count > 0
        
        if input_fields > 0 and output_fields > 0:
            modification_programs.append(f"{program} (modifies {input_fields} fields, outputs {output_fields} fields)")
        elif input_fields > 0:
            modification_programs.append(f"{program} (reads {input_fields} fields)")
        elif output_fields > 0:
            modification_programs.append(f"{program} (populates {output_fields} fields)")
        else:
            modification_programs.append(f"{program} (references layout structure)")
    
    if modification_programs:
        flow_narrative.append("Field modification patterns:")
        flow_narrative.extend([f"  {prog}" for prog in modification_programs])
    
    return {
        'call_chains': call_chains,
        'flow_description': ' '.join(flow_narrative),
        'programs_involved': programs,
        'total_relationships': len(call_chains)
    }

@app.route('/api/program-layout-summary/<session_id>/<program_name>')
def get_program_layout_summary(session_id, program_name):
    """Get summary of all layouts used by a specific program"""
    try:
        with analyzer.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all layouts for this program with field counts and usage stats
            cursor.execute('''
                SELECT rl.layout_name, rl.friendly_name, rl.business_purpose,
                       COUNT(fad.id) as field_count,
                       SUM(CASE WHEN fad.usage_type = 'INPUT' THEN 1 ELSE 0 END) as input_fields,
                       SUM(CASE WHEN fad.usage_type = 'OUTPUT' THEN 1 ELSE 0 END) as output_fields,
                       SUM(CASE WHEN fad.usage_type = 'INPUT_OUTPUT' THEN 1 ELSE 0 END) as io_fields,
                       SUM(CASE WHEN fad.usage_type = 'STATIC' THEN 1 ELSE 0 END) as static_fields,
                       SUM(CASE WHEN fad.usage_type = 'DERIVED' THEN 1 ELSE 0 END) as derived_fields
                FROM record_layouts rl
                LEFT JOIN field_analysis_details fad ON rl.id = fad.field_id
                WHERE rl.session_id = ? AND rl.program_name = ?
                GROUP BY rl.id, rl.layout_name, rl.friendly_name, rl.business_purpose
                ORDER BY field_count DESC
            ''', (session_id, program_name))
            
            layouts = cursor.fetchall()
            
            # Get program's call relationships to understand data flow context
            cursor.execute('''
                SELECT dr.target_component, dr.relationship_type, dr.interface_type
                FROM dependency_relationships dr
                WHERE dr.session_id = ? AND dr.source_component = ?
                AND dr.relationship_type IN ('PROGRAM_CALL', 'DYNAMIC_PROGRAM_CALL')
            ''', (session_id, program_name))
            
            program_calls = cursor.fetchall()
            
            layout_summary = {
                'program_name': program_name,
                'total_layouts': len(layouts),
                'layouts': [],
                'program_calls': [
                    {
                        'target_program': call[0],
                        'call_type': call[1],
                        'interface': call[2]
                    } for call in program_calls
                ],
                'data_flow_context': generate_data_flow_context(layouts, program_calls)
            }
            
            for layout in layouts:
                layout_info = {
                    'layout_name': layout[0],
                    'friendly_name': layout[1] or layout[0].replace('-', ' ').title(),
                    'business_purpose': layout[2] or f'Data structure in {program_name}',
                    'field_count': layout[3],
                    'field_distribution': {
                        'input_fields': layout[4],
                        'output_fields': layout[5],
                        'io_fields': layout[6],
                        'static_fields': layout[7],
                        'derived_fields': layout[8]
                    },
                    'primary_usage': determine_primary_layout_usage(layout[4:9]),
                    'complexity_indicator': calculate_layout_complexity(layout[3], layout[4:9])
                }
                layout_summary['layouts'].append(layout_info)
            
            return jsonify({
                'success': True,
                'layout_summary': layout_summary
            })
            
    except Exception as e:
        logger.error(f"Error getting program layout summary: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def generate_data_flow_context(layouts, program_calls):
    """Generate context about how layouts relate to program's data flow"""
    context = []
    
    if layouts:
        total_fields = sum(layout[3] for layout in layouts)  # field_count is at index 3
        context.append(f"Program manages {total_fields} total fields across {len(layouts)} layouts")
        
        # Analyze field distribution patterns
        total_input = sum(layout[4] for layout in layouts)
        total_output = sum(layout[5] for layout in layouts)
        total_io = sum(layout[6] for layout in layouts)
        
        if total_input > total_output:
            context.append("Primarily processes input data (data consumer)")
        elif total_output > total_input:
            context.append("Primarily generates output data (data producer)")
        elif total_io > 0:
            context.append("Processes data bidirectionally (data transformer)")
        else:
            context.append("Works with static data structures")
    
    if program_calls:
        context.append(f"Calls {len(program_calls)} other programs, potentially passing layout data")
    else:
        context.append("Standalone program with no detected program calls")
    
    return '; '.join(context)

def determine_primary_layout_usage(field_counts):
    """Determine primary usage pattern for a layout based on field type distribution"""
    input_fields, output_fields, io_fields, static_fields, derived_fields = field_counts
    
    # Create usage pattern analysis
    usage_scores = {
        'input_processing': input_fields * 2 + io_fields,
        'output_generation': output_fields * 2 + io_fields,
        'data_transformation': io_fields * 3 + derived_fields,
        'configuration': static_fields * 2,
        'calculation': derived_fields * 2
    }
    
    # Find primary usage
    max_score = max(usage_scores.values())
    if max_score == 0:
        return 'structure_definition'
    
    primary_usage = max(usage_scores.keys(), key=lambda k: usage_scores[k])
    
    return primary_usage

def calculate_layout_complexity(field_count, field_type_counts):
    """Calculate complexity indicator for a layout"""
    if field_count == 0:
        return 'simple'
    
    input_fields, output_fields, io_fields, static_fields, derived_fields = field_type_counts
    
    # Calculate complexity based on multiple factors
    complexity_score = 0
    
    # Field count factor
    if field_count > 50:
        complexity_score += 0.4
    elif field_count > 20:
        complexity_score += 0.2
    elif field_count > 10:
        complexity_score += 0.1
    
    # Usage diversity factor
    usage_types_present = sum(1 for count in field_type_counts if count > 0)
    complexity_score += usage_types_present * 0.1
    
    # Bidirectional usage factor (more complex)
    if io_fields > 0:
        complexity_score += 0.2
    
    # Derived fields factor (calculations = complexity)
    if derived_fields > 0:
        complexity_score += 0.2
    
    # Return complexity category
    if complexity_score >= 0.7:
        return 'complex'
    elif complexity_score >= 0.4:
        return 'moderate'
    else:
        return 'simple'
    
if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"🚀 Mainframe Analyzer Starting")
    print(f"📡 Chat System: {analyzer.chat_system_type.upper()}")
    if analyzer.chat_system_type == "agentic_rag":
        print(f"🧠 Features: Vector Search, Query Analysis, Smart Context Retrieval")
    else:
        print(f"💬 Features: Basic Chat, Field Analysis")
    print(f"🌐 Server: http://localhost:5000")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)