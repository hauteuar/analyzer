"""
Complete Enhanced Database Manager Module with Lock Handling
Handles all SQL operations and data persistence with improved error handling
"""

import sqlite3
import json
import uuid
import datetime
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "mainframe_analyzer.db"):
        self.db_path = db_path
        self.init_executed = False
        self._lock = threading.Lock()  # Thread safety
    
    @contextmanager
    def get_connection(self, timeout: int = 30, retries: int = 3):
        """Context manager for database connections with timeout and retry logic"""
        conn = None
        attempt = 0
        
        while attempt < retries:
            try:
                # Set timeout for connection
                conn = sqlite3.connect(self.db_path, timeout=timeout)
                conn.row_factory = sqlite3.Row
                
                # Enable WAL mode for better concurrency
                conn.execute('PRAGMA journal_mode=WAL;')
                # Set busy timeout
                conn.execute(f'PRAGMA busy_timeout={timeout * 1000};')
                # Enable foreign keys
                conn.execute('PRAGMA foreign_keys=ON;')
                
                yield conn
                break
                
            except sqlite3.OperationalError as e:
                attempt += 1
                if "database is locked" in str(e).lower() and attempt < retries:
                    logger.warning(f"Database locked, retrying... (attempt {attempt}/{retries})")
                    time.sleep(0.1 * attempt)  # Exponential backoff
                    if conn:
                        try:
                            conn.close()
                        except:
                            pass
                    conn = None
                    continue
                else:
                    logger.error(f"Database error after {attempt} attempts: {str(e)}")
                    raise
            except Exception as e:
                if conn:
                    conn.rollback()
                logger.error(f"Database error: {str(e)}")
                raise
            finally:
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
    
    def initialize_database(self):
        """Initialize database schema with lock protection"""
        with self._lock:  # Ensure only one thread initializes
            if self.init_executed:
                return
                
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Session management
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS analysis_sessions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT UNIQUE NOT NULL,
                            project_name TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            status TEXT DEFAULT 'active',
                            total_components INTEGER DEFAULT 0,
                            total_fields INTEGER DEFAULT 0
                        )
                    ''')
                    
                    # LLM call tracking
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS llm_analysis_calls (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            analysis_type TEXT NOT NULL,
                            chunk_number INTEGER DEFAULT 1,
                            total_chunks INTEGER DEFAULT 1,
                            prompt_tokens INTEGER DEFAULT 0,
                            response_tokens INTEGER DEFAULT 0,
                            processing_time_ms INTEGER DEFAULT 0,
                            success BOOLEAN DEFAULT 1,
                            error_message TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
                        )
                    ''')
                    
                    # Component analysis
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS component_analysis (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            component_name TEXT NOT NULL,
                            component_type TEXT NOT NULL,
                            file_path TEXT,
                            analysis_status TEXT DEFAULT 'completed',
                            total_lines INTEGER DEFAULT 0,
                            total_fields INTEGER DEFAULT 0,
                            dependencies_count INTEGER DEFAULT 0,
                            analysis_result_json TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
                        )
                    ''')
                    
                    # Record layouts (01 levels)
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS record_layouts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            layout_name TEXT NOT NULL,
                            program_name TEXT,
                            level_number TEXT DEFAULT '01',
                            line_start INTEGER,
                            line_end INTEGER,
                            source_code TEXT,
                            fields_count INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
                        )
                    ''')
                    
                    # Field mappings
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS field_mappings (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            target_file_name TEXT NOT NULL,
                            field_name TEXT NOT NULL,
                            mainframe_data_type TEXT,
                            oracle_data_type TEXT,
                            mainframe_length INTEGER,
                            oracle_length INTEGER,
                            population_source TEXT,
                            source_record_layout TEXT,
                            business_logic_type TEXT,
                            business_logic_description TEXT,
                            derivation_logic TEXT,
                            programs_involved_json TEXT,
                            confidence_score REAL DEFAULT 0.0,
                            analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
                        )
                    ''')
                    
                    # Field details with code references
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS field_analysis_details (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            field_id INTEGER,
                            field_name TEXT NOT NULL,
                            program_name TEXT NOT NULL,
                            operation_type TEXT,
                            line_number INTEGER,
                            code_snippet TEXT,
                            usage_type TEXT,
                            source_field TEXT,
                            target_field TEXT,
                            business_purpose TEXT,
                            analysis_confidence REAL DEFAULT 0.0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id),
                            FOREIGN KEY (field_id) REFERENCES field_mappings(id)
                        )
                    ''')
                    
                    # Dependency relationships
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS dependency_relationships (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            source_component TEXT NOT NULL,
                            target_component TEXT NOT NULL,
                            relationship_type TEXT NOT NULL,
                            interface_type TEXT,
                            confidence_score REAL DEFAULT 0.0,
                            analysis_details_json TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
                        )
                    ''')
                    
                    # Chat conversations
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS chat_conversations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            conversation_id TEXT NOT NULL,
                            message_type TEXT NOT NULL,
                            message_content TEXT NOT NULL,
                            context_used_json TEXT,
                            tokens_used INTEGER DEFAULT 0,
                            processing_time_ms INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
                        )
                    ''')
                    
                    # Create indexes for performance
                    indexes = [
                        'CREATE INDEX IF NOT EXISTS idx_field_mappings_target_file ON field_mappings(target_file_name, session_id)',
                        'CREATE INDEX IF NOT EXISTS idx_dependency_source ON dependency_relationships(source_component, session_id)',
                        'CREATE INDEX IF NOT EXISTS idx_dependency_target ON dependency_relationships(target_component, session_id)',
                        'CREATE INDEX IF NOT EXISTS idx_chat_session_conv ON chat_conversations(session_id, conversation_id)',
                        'CREATE INDEX IF NOT EXISTS idx_component_session ON component_analysis(session_id, component_type)',
                        'CREATE INDEX IF NOT EXISTS idx_field_details_field ON field_analysis_details(field_name, session_id)',
                        'CREATE INDEX IF NOT EXISTS idx_record_layouts_session ON record_layouts(session_id, program_name)'
                    ]
                    
                    for index_sql in indexes:
                        cursor.execute(index_sql)
                    
                    conn.commit()
                    logger.info("Database schema initialized successfully")
                    self.init_executed = True
                    
            except Exception as e:
                logger.error(f"Failed to initialize database: {str(e)}")
                raise
    
    def create_session(self, session_id: str, project_name: str) -> bool:
        """Create new analysis session with improved error handling"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO analysis_sessions (session_id, project_name)
                    VALUES (?, ?)
                ''', (session_id, project_name))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"Created session: {session_id} for project: {project_name}")
                    return True
                else:
                    logger.info(f"Session {session_id} already exists")
                    return True  # Session already exists, which is fine
                    
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            return False
    
    def log_llm_call(self, session_id: str, analysis_type: str, chunk_number: int, 
                     total_chunks: int, prompt_tokens: int, response_tokens: int,
                     processing_time_ms: int, success: bool, error_message: str = None):
        """Log LLM call details with improved error handling"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO llm_analysis_calls 
                    (session_id, analysis_type, chunk_number, total_chunks, 
                     prompt_tokens, response_tokens, processing_time_ms, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, analysis_type, chunk_number, total_chunks,
                      prompt_tokens, response_tokens, processing_time_ms, success, error_message))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging LLM call: {str(e)}")
    
    def store_component_analysis(self, session_id: str, component_name: str, 
                               component_type: str, file_path: str, analysis_result: Dict):
        """Store component analysis results with transaction management"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Use transaction explicitly
                cursor.execute('BEGIN IMMEDIATE;')
                
                try:
                    # Store main component
                    cursor.execute('''
                        INSERT OR REPLACE INTO component_analysis 
                        (session_id, component_name, component_type, file_path, 
                         total_lines, total_fields, analysis_result_json, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (session_id, component_name, component_type, file_path,
                          analysis_result.get('total_lines', 0),
                          len(analysis_result.get('fields', [])),
                          json.dumps(analysis_result)))
                    
                    # Store record layouts if present
                    if 'record_layouts' in analysis_result:
                        for layout in analysis_result['record_layouts']:
                            self._store_record_layout_in_transaction(cursor, session_id, layout, component_name)
                    
                    conn.commit()
                    logger.info(f"Stored component analysis: {component_name}")
                    
                except Exception as e:
                    conn.rollback()
                    raise e
                    
        except Exception as e:
            logger.error(f"Error storing component analysis: {str(e)}")
            raise
    
    def _store_record_layout_in_transaction(self, cursor, session_id: str, layout_data: Dict, program_name: str):
        """Store record layout within existing transaction"""
        cursor.execute('''
            INSERT OR REPLACE INTO record_layouts 
            (session_id, layout_name, program_name, level_number, 
             line_start, line_end, source_code, fields_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, layout_data.get('name'), program_name,
              layout_data.get('level', '01'),
              layout_data.get('line_start', 0),
              layout_data.get('line_end', 0),
              layout_data.get('source_code', ''),
              len(layout_data.get('fields', []))))
        
        layout_id = cursor.lastrowid
        
        # Store field details for this layout
        for field in layout_data.get('fields', []):
            self._store_field_details_in_transaction(cursor, session_id, field, program_name, layout_id)
    
    def _store_field_details_in_transaction(self, cursor, session_id: str, field_data: Dict, 
                                          program_name: str, layout_id: int = None):
        """Store field analysis details within existing transaction"""
        cursor.execute('''
            INSERT INTO field_analysis_details 
            (session_id, field_id, field_name, program_name, operation_type,
             line_number, code_snippet, usage_type, source_field, target_field,
             business_purpose, analysis_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, layout_id, field_data.get('name'),
              program_name, field_data.get('operation_type'),
              field_data.get('line_number', 0), field_data.get('code_snippet', ''),
              field_data.get('usage', 'UNKNOWN'), field_data.get('source_field', ''),
              field_data.get('target_field', ''), field_data.get('business_purpose', ''),
              field_data.get('confidence', 0.8)))
    
    def store_record_layout(self, session_id: str, layout_data: Dict, program_name: str):
        """Store record layout (01 level) information"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('BEGIN IMMEDIATE;')
                
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO record_layouts 
                        (session_id, layout_name, program_name, level_number, 
                         line_start, line_end, source_code, fields_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (session_id, layout_data.get('name'), program_name,
                          layout_data.get('level', '01'),
                          layout_data.get('line_start', 0),
                          layout_data.get('line_end', 0),
                          layout_data.get('source_code', ''),
                          len(layout_data.get('fields', []))))
                    
                    layout_id = cursor.lastrowid
                    
                    # Store field details for this layout
                    for field in layout_data.get('fields', []):
                        self.store_field_details(session_id, field, program_name, layout_id)
                    
                    conn.commit()
                    
                except Exception as e:
                    conn.rollback()
                    raise e
                    
        except Exception as e:
            logger.error(f"Error storing record layout: {str(e)}")
    
    def store_field_details(self, session_id: str, field_data: Dict, 
                          program_name: str, layout_id: int = None):
        """Store field analysis details"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO field_analysis_details 
                    (session_id, field_id, field_name, program_name, operation_type,
                     line_number, code_snippet, usage_type, source_field, target_field,
                     business_purpose, analysis_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, layout_id, field_data.get('name'),
                      program_name, field_data.get('operation_type'),
                      field_data.get('line_number', 0), field_data.get('code_snippet', ''),
                      field_data.get('usage', 'UNKNOWN'), field_data.get('source_field', ''),
                      field_data.get('target_field', ''), field_data.get('business_purpose', ''),
                      field_data.get('confidence', 0.8)))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing field details: {str(e)}")
    
    def store_field_mappings(self, session_id: str, target_file: str, mappings: List[Dict]):
        """Store field mapping analysis results with batch processing"""
        if not mappings:
            return
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('BEGIN IMMEDIATE;')
                
                try:
                    # Use executemany for better performance
                    cursor.executemany('''
                        INSERT OR REPLACE INTO field_mappings 
                        (session_id, target_file_name, field_name, mainframe_data_type,
                         oracle_data_type, mainframe_length, oracle_length, population_source,
                         source_record_layout, business_logic_type, business_logic_description,
                         derivation_logic, programs_involved_json, confidence_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', [
                        (session_id, target_file, mapping.get('field_name'),
                         mapping.get('mainframe_data_type'), mapping.get('oracle_data_type'),
                         mapping.get('mainframe_length', 0), mapping.get('oracle_length', 0),
                         mapping.get('population_source'), mapping.get('source_record_layout'),
                         mapping.get('business_logic_type'), mapping.get('business_logic_description'),
                         mapping.get('derivation_logic'), json.dumps(mapping.get('programs_involved', [])),
                         mapping.get('confidence_score', 0.0))
                        for mapping in mappings
                    ])
                    
                    conn.commit()
                    logger.info(f"Stored {len(mappings)} field mappings for {target_file}")
                    
                except Exception as e:
                    conn.rollback()
                    raise e
                    
        except Exception as e:
            logger.error(f"Error storing field mappings: {str(e)}")
            raise
    
    def get_session_components(self, session_id: str) -> List[Dict]:
        """Get all components for session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT component_name, component_type, file_path, analysis_status,
                           total_lines, total_fields, dependencies_count, created_at
                    FROM component_analysis 
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                ''', (session_id,))
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting session components: {str(e)}")
            return []
    
    def get_record_layouts(self, session_id: str, program_name: str = None) -> List[Dict]:
        """Get record layouts for session or specific program"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if program_name:
                    cursor.execute('''
                        SELECT * FROM record_layouts 
                        WHERE session_id = ? AND program_name = ?
                        ORDER BY layout_name
                    ''', (session_id, program_name))
                else:
                    cursor.execute('''
                        SELECT * FROM record_layouts 
                        WHERE session_id = ?
                        ORDER BY program_name, layout_name
                    ''', (session_id,))
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting record layouts: {str(e)}")
            return []
    
    def get_field_matrix(self, session_id: str, record_layout: str = None, program_name: str = None) -> List[Dict]:
        """Get field matrix data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if record_layout:
                    # Get fields for specific record layout
                    cursor.execute('''
                        SELECT fad.field_name, fad.program_name, fad.operation_type,
                               fad.line_number, fad.code_snippet, fad.usage_type,
                               fad.source_field, fad.target_field, fad.business_purpose,
                               rl.layout_name as record_layout
                        FROM field_analysis_details fad
                        JOIN record_layouts rl ON fad.field_id = rl.id
                        WHERE fad.session_id = ? AND rl.layout_name = ?
                        ORDER BY fad.field_name, fad.line_number
                    ''', (session_id, record_layout))
                elif program_name:
                    # Get all fields for specific program with their record layouts
                    cursor.execute('''
                        SELECT fad.field_name, fad.program_name, fad.operation_type,
                               fad.line_number, fad.code_snippet, fad.usage_type,
                               fad.source_field, fad.target_field, fad.business_purpose,
                               rl.layout_name as record_layout, rl.level_number
                        FROM field_analysis_details fad
                        LEFT JOIN record_layouts rl ON fad.field_id = rl.id
                        WHERE fad.session_id = ? AND fad.program_name = ?
                        ORDER BY rl.layout_name, fad.field_name, fad.line_number
                    ''', (session_id, program_name))
                else:
                    # Get all field matrix data
                    cursor.execute('''
                        SELECT fad.field_name, fad.program_name, fad.operation_type,
                               fad.line_number, fad.code_snippet, fad.usage_type,
                               fad.source_field, fad.target_field, fad.business_purpose,
                               rl.layout_name as record_layout, rl.level_number
                        FROM field_analysis_details fad
                        LEFT JOIN record_layouts rl ON fad.field_id = rl.id
                        WHERE fad.session_id = ?
                        ORDER BY fad.program_name, rl.layout_name, fad.field_name
                    ''', (session_id,))
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting field matrix: {str(e)}")
            return []
    
    def get_field_mappings(self, session_id: str, target_file: str) -> List[Dict]:
        """Get field mappings for target file"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM field_mappings 
                    WHERE session_id = ? AND target_file_name = ?
                    ORDER BY field_name
                ''', (session_id, target_file))
                
                mappings = []
                for row in cursor.fetchall():
                    mapping = dict(row)
                    # Parse JSON fields
                    if mapping['programs_involved_json']:
                        try:
                            mapping['programs_involved'] = json.loads(mapping['programs_involved_json'])
                        except json.JSONDecodeError:
                            mapping['programs_involved'] = []
                    else:
                        mapping['programs_involved'] = []
                    mappings.append(mapping)
                
                return mappings
        except Exception as e:
            logger.error(f"Error getting field mappings: {str(e)}")
            return []
    
    def get_dependencies(self, session_id: str) -> List[Dict]:
        """Get dependency relationships"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM dependency_relationships 
                    WHERE session_id = ?
                    ORDER BY source_component, target_component
                ''', (session_id,))
                
                dependencies = []
                for row in cursor.fetchall():
                    dep = dict(row)
                    # Parse JSON fields
                    if dep['analysis_details_json']:
                        try:
                            dep['analysis_details'] = json.loads(dep['analysis_details_json'])
                        except json.JSONDecodeError:
                            dep['analysis_details'] = {}
                    else:
                        dep['analysis_details'] = {}
                    dependencies.append(dep)
                
                return dependencies
        except Exception as e:
            logger.error(f"Error getting dependencies: {str(e)}")
            return []
    
    def get_session_metrics(self, session_id: str) -> Dict:
        """Get session metrics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Component counts
                cursor.execute('''
                    SELECT component_type, COUNT(*) as count
                    FROM component_analysis 
                    WHERE session_id = ?
                    GROUP BY component_type
                ''', (session_id,))
                component_counts = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Field counts
                cursor.execute('''
                    SELECT COUNT(DISTINCT field_name) as total_fields
                    FROM field_analysis_details 
                    WHERE session_id = ?
                ''', (session_id,))
                total_fields_result = cursor.fetchone()
                total_fields = total_fields_result[0] if total_fields_result else 0
                
                # Lines of code
                cursor.execute('''
                    SELECT SUM(total_lines) as total_lines
                    FROM component_analysis 
                    WHERE session_id = ?
                ''', (session_id,))
                total_lines_result = cursor.fetchone()
                total_lines = total_lines_result[0] if total_lines_result and total_lines_result[0] else 0
                
                # Token usage
                cursor.execute('''
                    SELECT SUM(prompt_tokens) as total_prompt_tokens,
                           SUM(response_tokens) as total_response_tokens,
                           COUNT(*) as total_calls
                    FROM llm_analysis_calls 
                    WHERE session_id = ?
                ''', (session_id,))
                token_usage = cursor.fetchone()
                
                return {
                    'component_counts': component_counts,
                    'total_components': sum(component_counts.values()),
                    'total_fields': total_fields,
                    'total_lines': total_lines,
                    'token_usage': {
                        'total_prompt_tokens': token_usage[0] or 0,
                        'total_response_tokens': token_usage[1] or 0,
                        'total_calls': token_usage[2] or 0
                    }
                }
        except Exception as e:
            logger.error(f"Error getting session metrics: {str(e)}")
            return {}
    
    def store_chat_message(self, session_id: str, conversation_id: str, 
                          message_type: str, content: str, context_used: Dict = None,
                          tokens_used: int = 0, processing_time_ms: int = 0):
        """Store chat conversation message"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO chat_conversations 
                    (session_id, conversation_id, message_type, message_content,
                     context_used_json, tokens_used, processing_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, conversation_id, message_type, content,
                      json.dumps(context_used) if context_used else None,
                      tokens_used, processing_time_ms))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing chat message: {str(e)}")
    
    def get_chat_history(self, session_id: str, conversation_id: str, limit: int = 10) -> List[Dict]:
        """Get chat conversation history"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT message_type, message_content, context_used_json, created_at
                    FROM chat_conversations 
                    WHERE session_id = ? AND conversation_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (session_id, conversation_id, limit))
                
                history = []
                for row in cursor.fetchall():
                    message = dict(row)
                    if message['context_used_json']:
                        try:
                            message['context_used'] = json.loads(message['context_used_json'])
                        except json.JSONDecodeError:
                            message['context_used'] = {}
                    else:
                        message['context_used'] = {}
                    history.append(message)
                
                return list(reversed(history))  # Return in chronological order
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []
    
    def get_context_for_field(self, session_id: str, field_name: str) -> Dict:
        """Get context information for a specific field"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get field details
                cursor.execute('''
                    SELECT * FROM field_analysis_details 
                    WHERE session_id = ? AND field_name LIKE ?
                    ORDER BY program_name, line_number
                ''', (session_id, f'%{field_name}%'))
                
                field_details = [dict(row) for row in cursor.fetchall()]
                
                # Get field mappings
                cursor.execute('''
                    SELECT * FROM field_mappings 
                    WHERE session_id = ? AND field_name LIKE ?
                ''', (session_id, f'%{field_name}%'))
                
                field_mappings = []
                for row in cursor.fetchall():
                    mapping = dict(row)
                    if mapping['programs_involved_json']:
                        try:
                            mapping['programs_involved'] = json.loads(mapping['programs_involved_json'])
                        except json.JSONDecodeError:
                            mapping['programs_involved'] = []
                    else:
                        mapping['programs_involved'] = []
                    field_mappings.append(mapping)
                
                return {
                    'field_details': field_details,
                    'field_mappings': field_mappings
                }
        except Exception as e:
            logger.error(f"Error getting field context: {str(e)}")
            return {}
    
    def export_field_mappings(self, session_id: str) -> List[Dict]:
        """Export field mappings for the session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT target_file_name, field_name, mainframe_data_type, oracle_data_type,
                           mainframe_length, oracle_length, population_source, business_logic_type,
                           business_logic_description, programs_involved_json, confidence_score
                    FROM field_mappings 
                    WHERE session_id = ?
                    ORDER BY target_file_name, field_name
                ''', (session_id,))
                
                mappings = []
                for row in cursor.fetchall():
                    mapping = dict(row)
                    if mapping['programs_involved_json']:
                        try:
                            mapping['programs_involved'] = json.loads(mapping['programs_involved_json'])
                        except json.JSONDecodeError:
                            mapping['programs_involved'] = []
                    else:
                        mapping['programs_involved'] = []
                    mappings.append(mapping)
                
                return mappings
        except Exception as e:
            logger.error(f"Error exporting field mappings: {str(e)}")
            return []
    
    def export_components(self, session_id: str) -> List[Dict]:
        """Export component analysis for the session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT component_name, component_type, file_path, total_lines,
                           total_fields, dependencies_count, analysis_result_json
                    FROM component_analysis 
                    WHERE session_id = ?
                    ORDER BY component_name
                ''', (session_id,))
                
                components = []
                for row in cursor.fetchall():
                    component = dict(row)
                    if component['analysis_result_json']:
                        try:
                            component['analysis_result'] = json.loads(component['analysis_result_json'])
                        except json.JSONDecodeError:
                            component['analysis_result'] = {}
                    else:
                        component['analysis_result'] = {}
                    components.append(component)
                
                return components
        except Exception as e:
            logger.error(f"Error exporting components: {str(e)}")
            return []
    
    def export_dependencies(self, session_id: str) -> List[Dict]:
        """Export dependency relationships for the session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT source_component, target_component, relationship_type,
                           interface_type, confidence_score, analysis_details_json
                    FROM dependency_relationships 
                    WHERE session_id = ?
                    ORDER BY source_component, target_component
                ''', (session_id,))
                
                dependencies = []
                for row in cursor.fetchall():
                    dep = dict(row)
                    if dep['analysis_details_json']:
                        try:
                            dep['analysis_details'] = json.loads(dep['analysis_details_json'])
                        except json.JSONDecodeError:
                            dep['analysis_details'] = {}
                    else:
                        dep['analysis_details'] = {}
                    dependencies.append(dep)
                
                return dependencies
        except Exception as e:
            logger.error(f"Error exporting dependencies: {str(e)}")
            return []
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check database health and return status"""
        try:
            with self.get_connection(timeout=5) as conn:
                cursor = conn.cursor()
                
                # Check if we can read/write
                cursor.execute('SELECT 1')
                can_read = cursor.fetchone() is not None
                
                # Check WAL mode
                cursor.execute('PRAGMA journal_mode')
                journal_mode = cursor.fetchone()[0]
                
                # Check database integrity
                cursor.execute('PRAGMA integrity_check(1)')
                integrity = cursor.fetchone()[0]
                
                # Check page count and size
                cursor.execute('PRAGMA page_count')
                page_count = cursor.fetchone()[0]
                
                cursor.execute('PRAGMA page_size')
                page_size = cursor.fetchone()[0]
                
                return {
                    'status': 'healthy',
                    'can_read': can_read,
                    'journal_mode': journal_mode,
                    'integrity_check': integrity,
                    'database_size_pages': page_count,
                    'page_size': page_size,
                    'estimated_size_mb': (page_count * page_size) / (1024 * 1024),
                    'database_path': self.db_path
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'database_path': self.db_path
            }
    
    def force_unlock_database(self):
        """Force unlock database (use with caution)"""
        try:
            # Close any existing connections
            import gc
            gc.collect()
            
            # Try to connect and run a quick operation
            with self.get_connection(timeout=1) as conn:
                cursor = conn.cursor()
                cursor.execute('BEGIN IMMEDIATE;')
                cursor.execute('ROLLBACK;')
            
            logger.info("Database unlock attempted")
        except Exception as e:
            logger.error(f"Failed to unlock database: {str(e)}")
    
    def optimize_database(self):
        """Optimize database performance"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Analyze tables for query optimization
                cursor.execute('ANALYZE;')
                
                # Check if we can vacuum (not recommended during active use)
                cursor.execute('PRAGMA wal_checkpoint(TRUNCATE);')
                
                conn.commit()
                logger.info("Database optimization completed")
        except Exception as e:
            logger.error(f"Database optimization failed: {str(e)}")
    
    def get_database_size(self) -> Dict[str, int]:
        """Get database file sizes"""
        import os
        try:
            main_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            wal_size = 0
            shm_size = 0
            
            wal_path = self.db_path + '-wal'
            shm_path = self.db_path + '-shm'
            
            if os.path.exists(wal_path):
                wal_size = os.path.getsize(wal_path)
            if os.path.exists(shm_path):
                shm_size = os.path.getsize(shm_path)
            
            return {
                'main_db_size': main_size,
                'wal_size': wal_size,
                'shm_size': shm_size,
                'total_size': main_size + wal_size + shm_size,
                'main_db_size_mb': main_size / (1024 * 1024),
                'wal_size_mb': wal_size / (1024 * 1024),
                'total_size_mb': (main_size + wal_size + shm_size) / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting database size: {str(e)}")
            return {'error': str(e)}
    
    def store_dependency_relationship(self, session_id: str, source_component: str, 
                                    target_component: str, relationship_type: str, 
                                    interface_type: str = None, confidence_score: float = 0.0,
                                    analysis_details: Dict = None):
        """Store dependency relationship"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO dependency_relationships 
                    (session_id, source_component, target_component, relationship_type,
                     interface_type, confidence_score, analysis_details_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, source_component, target_component, relationship_type,
                      interface_type, confidence_score, 
                      json.dumps(analysis_details) if analysis_details else None))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing dependency relationship: {str(e)}")
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all analysis sessions"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT session_id, project_name, created_at, status,
                           total_components, total_fields
                    FROM analysis_sessions 
                    ORDER BY created_at DESC
                ''')
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting all sessions: {str(e)}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its related data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('BEGIN IMMEDIATE;')
                
                try:
                    # Delete in reverse dependency order
                    tables_to_clean = [
                        'chat_conversations',
                        'dependency_relationships',
                        'field_analysis_details',
                        'field_mappings',
                        'record_layouts',
                        'llm_analysis_calls',
                        'component_analysis',
                        'analysis_sessions'
                    ]
                    
                    for table in tables_to_clean:
                        cursor.execute(f'DELETE FROM {table} WHERE session_id = ?', (session_id,))
                    
                    conn.commit()
                    logger.info(f"Deleted session: {session_id}")
                    return True
                    
                except Exception as e:
                    conn.rollback()
                    raise e
                    
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            return False
    
    def update_session_status(self, session_id: str, status: str, 
                            total_components: int = None, total_fields: int = None) -> bool:
        """Update session status and metrics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if total_components is not None and total_fields is not None:
                    cursor.execute('''
                        UPDATE analysis_sessions 
                        SET status = ?, total_components = ?, total_fields = ?
                        WHERE session_id = ?
                    ''', (status, total_components, total_fields, session_id))
                else:
                    cursor.execute('''
                        UPDATE analysis_sessions 
                        SET status = ?
                        WHERE session_id = ?
                    ''', (status, session_id))
                
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating session status: {str(e)}")
            return False