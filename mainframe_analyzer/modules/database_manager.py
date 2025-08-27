"""
Database Manager Module - Complete Version with Enhanced Logging
Handles all SQL operations and data persistence with proper connection management
"""

import sqlite3
import json
import uuid
import datetime
import logging
import threading
import time
import os
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "mainframe_analyzer.db"):
        self.db_path = db_path
        self.init_executed = False
        self._lock = threading.Lock()  # Thread safety
        self._ensure_database_directory()
    
    def _ensure_database_directory(self):
        """Ensure database directory exists and is writable"""
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, mode=0o755)
            logger.info(f"Created database directory: {db_dir}")
        
        # Check if directory is writable
        if not os.access(db_dir, os.W_OK):
            logger.error(f"Database directory not writable: {db_dir}")
            raise PermissionError(f"Cannot write to database directory: {db_dir}")
    
    @contextmanager
    def get_connection(self):
        """Simplified connection manager without threading locks"""
        conn = None
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Database connection attempt {attempt + 1}")
                
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=10.0,  # Reduced timeout
                    isolation_level=None,
                    check_same_thread=False  # Allow cross-thread usage
                )
                conn.row_factory = sqlite3.Row
                
                # Simpler SQLite configuration
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL") 
                conn.execute("PRAGMA busy_timeout=5000")  # 5 seconds
                
                logger.debug("Database connection established")
                yield conn
                return  # Success, exit retry loop
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database locked on attempt {attempt + 1}, retrying in {retry_delay}s...")
                    if conn:
                        try:
                            conn.close()
                        except:
                            pass
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    logger.error(f"Database connection failed: {str(e)}")
                    raise
                    
            except Exception as e:
                logger.error(f"Unexpected database error: {str(e)}")
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
                raise
                
            finally:
                if conn:
                    try:
                        conn.close()
                        logger.debug("Database connection closed")
                    except:
                        pass
        
        raise sqlite3.OperationalError("Failed to establish database connection after retries")
    
    def _handle_database_lock(self, conn):
        """Handle database lock situations"""
        try:
            if conn:
                conn.close()
        except:
            pass
        
        # Wait a bit and check for lock files
        time.sleep(1)
        
        # Check for SQLite lock files and log their status
        wal_file = self.db_path + "-wal"
        shm_file = self.db_path + "-shm"
        
        for lock_file in [wal_file, shm_file]:
            if os.path.exists(lock_file):
                try:
                    file_size = os.path.getsize(lock_file)
                    logger.warning(f"Lock file exists: {lock_file} (size: {file_size} bytes)")
                except OSError as e:
                    logger.error(f"Cannot access lock file {lock_file}: {e}")
    
    def force_unlock_database(self):
        """Force unlock database by removing lock files (use with caution)"""
        logger.warning("Attempting to force unlock database - this should only be used when no other processes are using the database")
        
        lock_files = [
            self.db_path + "-wal",
            self.db_path + "-shm",
            self.db_path + "-journal"
        ]
        
        for lock_file in lock_files:
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                    logger.info(f"Removed lock file: {lock_file}")
                except OSError as e:
                    logger.error(f"Cannot remove lock file {lock_file}: {e}")
    
    def initialize_database(self):
        """Initialize database schema with complete field context support"""
        if self.init_executed:
            logger.debug("Database already initialized, skipping")
            return
        
        logger.info("Initializing database schema...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Drop existing tables to start fresh
                    cursor.execute('DROP TABLE IF EXISTS field_analysis_details')
                    cursor.execute('DROP TABLE IF EXISTS field_mappings') 
                    cursor.execute('DROP TABLE IF EXISTS record_layouts')
                    cursor.execute('DROP TABLE IF EXISTS component_analysis')
                    cursor.execute('DROP TABLE IF EXISTS dependency_relationships')
                    cursor.execute('DROP TABLE IF EXISTS chat_conversations')
                    cursor.execute('DROP TABLE IF EXISTS llm_analysis_calls')
                    cursor.execute('DROP TABLE IF EXISTS analysis_sessions')
                    
                    # Session management
                    cursor.execute('''
                        CREATE TABLE analysis_sessions (
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
                        CREATE TABLE llm_analysis_calls (
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
                    
                    # Component analysis with complete source storage
                    cursor.execute('''
                        CREATE TABLE component_analysis (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            component_name TEXT NOT NULL,
                            component_type TEXT NOT NULL,
                            file_path TEXT,
                            analysis_status TEXT DEFAULT 'completed',
                            total_lines INTEGER DEFAULT 0,
                            executable_lines INTEGER DEFAULT 0,
                            comment_lines INTEGER DEFAULT 0,
                            total_fields INTEGER DEFAULT 0,
                            dependencies_count INTEGER DEFAULT 0,
                            business_purpose TEXT,
                            complexity_score REAL DEFAULT 0.5,
                            source_content TEXT,
                            analysis_result_json TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
                        )
                    ''')
                    
                    # Record layouts with enhanced source tracking
                    cursor.execute('''
                        CREATE TABLE record_layouts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            layout_name TEXT NOT NULL,
                            friendly_name TEXT,
                            program_name TEXT,
                            level_number TEXT DEFAULT '01',
                            line_start INTEGER,
                            line_end INTEGER,
                            source_code TEXT,
                            fields_count INTEGER DEFAULT 0,
                            business_purpose TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
                        )
                    ''')
                    
                    # Enhanced field analysis with complete source code context
                    cursor.execute('''
                        CREATE TABLE field_analysis_details (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            field_id INTEGER,
                            field_name TEXT NOT NULL,
                            friendly_name TEXT,
                            program_name TEXT NOT NULL,
                            layout_name TEXT,
                            operation_type TEXT,
                            line_number INTEGER,
                            code_snippet TEXT,
                            usage_type TEXT,
                            source_field TEXT,
                            target_field TEXT,
                            business_purpose TEXT,
                            analysis_confidence REAL DEFAULT 0.0,
                            
                            -- NEW: Complete source code context fields
                            definition_line_number INTEGER,
                            definition_code TEXT,
                            program_source_content TEXT,
                            field_references_json TEXT,
                            usage_summary_json TEXT,
                            total_program_references INTEGER DEFAULT 0,
                            move_source_count INTEGER DEFAULT 0,
                            move_target_count INTEGER DEFAULT 0,
                            arithmetic_count INTEGER DEFAULT 0,
                            conditional_count INTEGER DEFAULT 0,
                            cics_count INTEGER DEFAULT 0,
                            
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id),
                            FOREIGN KEY (field_id) REFERENCES record_layouts(id)
                        )
                    ''')
                    
                    # Enhanced field mappings
                    cursor.execute('''
                        CREATE TABLE field_mappings (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            target_file_name TEXT NOT NULL,
                            field_name TEXT NOT NULL,
                            friendly_name TEXT,
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
                            
                            -- NEW: Source code evidence
                            source_code_evidence_json TEXT,
                            actual_cobol_definition TEXT,
                            usage_patterns_json TEXT,
                            
                            analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
                        )
                    ''')
                    
                    # Dependency relationships
                    cursor.execute('''
                        CREATE TABLE dependency_relationships (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            source_component TEXT NOT NULL,
                            target_component TEXT NOT NULL,
                            relationship_type TEXT NOT NULL,
                            interface_type TEXT,
                            confidence_score REAL DEFAULT 0.0,
                            analysis_details_json TEXT,
                            source_code_evidence TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
                        )
                    ''')
                    
                    # Chat conversations
                    cursor.execute('''
                        CREATE TABLE chat_conversations (
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
                    
                    # Create performance indexes
                    indexes = [
                        'CREATE INDEX idx_field_mappings_target_file ON field_mappings(target_file_name, session_id)',
                        'CREATE INDEX idx_field_details_field_name ON field_analysis_details(field_name, session_id)',
                        'CREATE INDEX idx_field_details_program ON field_analysis_details(program_name, session_id)',
                        'CREATE INDEX idx_dependency_source ON dependency_relationships(source_component, session_id)',
                        'CREATE INDEX idx_dependency_target ON dependency_relationships(target_component, session_id)',
                        'CREATE INDEX idx_chat_session_conv ON chat_conversations(session_id, conversation_id)',
                        'CREATE INDEX idx_component_session ON component_analysis(session_id, component_type)',
                        'CREATE INDEX idx_record_layouts_session ON record_layouts(session_id, program_name)',
                        'CREATE INDEX idx_record_layouts_name ON record_layouts(layout_name, session_id)'
                    ]
                    
                    for index_sql in indexes:
                        cursor.execute(index_sql)
                    
                    logger.info("Fresh database schema created successfully with enhanced field context support")
                    self.init_executed = True
                    return
                    
            except Exception as e:
                logger.error(f"Database initialization error: {str(e)}")
                raise
        
        raise sqlite3.OperationalError("Failed to initialize database after multiple attempts")

    def _analyze_field_in_program_source(self, field_name: str, program_source: str, program_name: str) -> Dict:
        """Live analysis of field usage in program source code"""
        analysis = {
            'field_name': field_name,
            'program_name': program_name,
            'references': [],
            'usage_counts': {
                'definition': 0,
                'move_source': 0,
                'move_target': 0,
                'arithmetic': 0,
                'conditional': 0,
                'cics': 0
            }
        }
        
        try:
            lines = program_source.split('\n')
            field_upper = field_name.upper()
            
            for line_idx, line in enumerate(lines, 1):
                line_stripped = line.strip()
                line_upper = line_stripped.upper()
                
                if field_upper in line_upper:
                    operation_type = 'REFERENCE'
                    
                    if 'PIC' in line_upper and any(line_upper.strip().startswith(level) for level in ['01', '02', '03', '04', '05', '77']):
                        operation_type = 'DEFINITION'
                        analysis['usage_counts']['definition'] += 1
                    elif 'MOVE' in line_upper:
                        if f' TO {field_upper}' in line_upper:
                            operation_type = 'MOVE_TARGET'
                            analysis['usage_counts']['move_target'] += 1
                        elif f'{field_upper} TO ' in line_upper:
                            operation_type = 'MOVE_SOURCE'
                            analysis['usage_counts']['move_source'] += 1
                    elif any(op in line_upper for op in ['COMPUTE', 'ADD', 'SUBTRACT']):
                        operation_type = 'ARITHMETIC'
                        analysis['usage_counts']['arithmetic'] += 1
                    elif any(op in line_upper for op in ['IF', 'WHEN', 'EVALUATE']):
                        operation_type = 'CONDITIONAL'
                        analysis['usage_counts']['conditional'] += 1
                    elif 'CICS' in line_upper:
                        operation_type = 'CICS'
                        analysis['usage_counts']['cics'] += 1
                    
                    context_start = max(0, line_idx - 3)
                    context_end = min(len(lines), line_idx + 2)
                    context_lines = lines[context_start:context_end]
                    
                    reference = {
                        'line_number': line_idx,
                        'line_content': line_stripped,
                        'operation_type': operation_type,
                        'context_lines': context_lines,
                        'context_block': '\n'.join([
                            f"{context_start + i + 1:4d}: {ctx_line}" 
                            for i, ctx_line in enumerate(context_lines)
                        ])
                    }
                    
                    analysis['references'].append(reference)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in live field analysis: {str(e)}")
            return analysis
    
    def create_session(self, session_id: str, project_name: str) -> bool:
        """Create new analysis session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO analysis_sessions (session_id, project_name)
                    VALUES (?, ?)
                ''', (session_id, project_name))
                logger.info(f"Created session: {session_id} for project: {project_name}")
                return True
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            return False
    
    def log_llm_call(self, session_id: str, analysis_type: str, chunk_number: int, 
                     total_chunks: int, prompt_tokens: int, response_tokens: int,
                     processing_time_ms: int, success: bool, error_message: str = None):
        """Log LLM call details"""
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
        except Exception as e:
            logger.error(f"Error logging LLM call: {str(e)}")
    
    def store_component_analysis(self, session_id: str, component_name: str, 
                           component_type: str, file_path: str, analysis_result: Dict):
        """Store component analysis - FIXED VERSION"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Debug logging
                logger.info(f"📥 Storing component: {component_name}")
                logger.info(f"🔍 Analysis result keys: {list(analysis_result.keys())}")
                logger.info(f"📊 Derived components count: {len(analysis_result.get('derived_components', []))}")
                
                # Check if component already exists
                cursor.execute('''
                    SELECT COUNT(*) FROM component_analysis 
                    WHERE session_id = ? AND component_name = ? AND component_type = ?
                ''', (session_id, component_name, component_type))
                
                exists = cursor.fetchone()[0] > 0
                
                # Prepare data with proper handling
                total_lines = int(analysis_result.get('total_lines', 0))
                total_fields = len(analysis_result.get('fields', []))
                
                # IMPORTANT: Preserve business_purpose from analysis_result
                business_purpose = analysis_result.get('business_purpose', '')
                if not business_purpose and analysis_result.get('llm_summary'):
                    business_purpose = analysis_result['llm_summary'].get('business_purpose', '')
                
                complexity_score = float(analysis_result.get('complexity_score', 0.5))
                if analysis_result.get('llm_summary'):
                    complexity_score = float(analysis_result['llm_summary'].get('complexity_score', complexity_score))
                
                # Store the COMPLETE analysis_result as JSON
                analysis_json = json.dumps(analysis_result, default=str, ensure_ascii=False)
                source_content = str(analysis_result.get('content', ''))
                
                logger.info(f"💾 Storing: business_purpose='{business_purpose[:50]}...', complexity={complexity_score}")
                logger.info(f"📝 JSON length: {len(analysis_json)} characters")
                
                if exists:
                    cursor.execute('''
                        UPDATE component_analysis 
                        SET total_lines = ?, total_fields = ?, business_purpose = ?, 
                            complexity_score = ?, analysis_result_json = ?, updated_at = CURRENT_TIMESTAMP,
                            source_content = ?
                        WHERE session_id = ? AND component_name = ? AND component_type = ?
                    ''', (
                        total_lines, total_fields, business_purpose[:500], complexity_score, 
                        analysis_json, source_content,
                        session_id, component_name, component_type
                    ))
                else:
                    cursor.execute('''
                        INSERT INTO component_analysis 
                        (session_id, component_name, component_type, file_path, 
                        total_lines, total_fields, business_purpose, complexity_score,
                        source_content, analysis_result_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id, component_name, component_type, file_path,
                        total_lines, total_fields, business_purpose[:500], complexity_score,
                        source_content, analysis_json
                    ))
                
                # Verify what was actually stored
                cursor.execute('''
                    SELECT business_purpose, complexity_score, LENGTH(analysis_result_json) as json_length
                    FROM component_analysis 
                    WHERE session_id = ? AND component_name = ? AND component_type = ?
                ''', (session_id, component_name, component_type))
                
                stored_data = cursor.fetchone()
                if stored_data:
                    logger.info(f"✅ Verified storage: business_purpose='{stored_data[0][:50]}...', complexity={stored_data[1]}, json_length={stored_data[2]}")
                
                logger.info(f"✅ Successfully stored component: {component_name}")
                
        except Exception as e:
            logger.error(f"❌ Error storing component {component_name}: {str(e)}")
            logger.error(f"📋 Analysis result sample: {str(analysis_result)[:200]}...")
            raise  # Re-raise to see the full error
                # Don't re-raise the exception to prevent stopping the entire process
    
    def store_record_layout(self, session_id: str, layout_data: Dict, program_name: str):
        """Simplified record layout storage"""
        layout_name = layout_data.get('name', 'UNKNOWN')
        logger.info(f"Storing record layout: {layout_name}")
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Store layout first
                cursor.execute('''
                    INSERT OR REPLACE INTO record_layouts 
                    (session_id, layout_name, program_name, level_number, 
                    line_start, line_end, source_code, fields_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id, 
                    layout_name,
                    program_name,
                    str(layout_data.get('level', '01')),
                    layout_data.get('line_start', 0),
                    layout_data.get('line_end', 0),
                    layout_data.get('source_code', ''),
                    len(layout_data.get('fields', []))
                ))
                
                layout_id = cursor.lastrowid
                logger.info(f"Stored layout {layout_name} with ID {layout_id}")
                return layout_id
                
        except Exception as e:
            logger.error(f"Error storing layout {layout_name}: {str(e)}")
            raise

    def _store_field_details_internal(self, session_id: str, field_data: Dict, 
                                    program_name: str, layout_id: int, cursor):
        """Internal method to store field details using existing cursor - SIMPLIFIED VERSION"""
        field_name = field_data.get('name', 'UNNAMED_FIELD')
        logger.debug(f"Storing field details: {field_name}")
        
        try:
            # Validate and limit data lengths to prevent database issues
            insert_data = (
                session_id, 
                layout_id, 
                field_name[:100],  # Limit field name length
                program_name[:100], 
                field_data.get('operation_type', 'DEFINITION')[:50],
                field_data.get('line_number', 0), 
                str(field_data.get('code_snippet', ''))[:1000],  # Limit code snippet
                field_data.get('usage_type', 'UNKNOWN')[:50], 
                field_data.get('source_field', '')[:100],
                field_data.get('target_field', '')[:100], 
                field_data.get('business_purpose', f"Field definition for {field_name}")[:500],
                float(field_data.get('confidence', 0.8))
            )
            
            cursor.execute('''
                INSERT INTO field_analysis_details 
                (session_id, field_id, field_name, program_name, operation_type,
                line_number, code_snippet, usage_type, source_field, target_field,
                business_purpose, analysis_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', insert_data)
            
            logger.debug(f"Successfully stored field details for: {field_name}")
            
        except Exception as e:
            logger.error(f"Error in _store_field_details_internal for field {field_name}: {str(e)}")
            logger.error(f"Insert data: {insert_data}")
            raise
            
    def _generate_field_code_snippet(self, field_data: Dict) -> str:
        """Generate a code snippet for the field definition"""
        try:
            level = field_data.get('level', '05')
            name = field_data.get('name', 'UNNAMED')
            picture = field_data.get('picture', '')
            usage = field_data.get('usage', '')
            
            snippet = f"{level} {name}"
            if picture:
                snippet += f" PIC {picture}"
            if usage:
                snippet += f" USAGE {usage}"
            
            return snippet.strip()
            
        except Exception as e:
            logger.warning(f"Error generating code snippet for field: {str(e)}")
            return field_data.get('code_snippet', f"Field: {field_data.get('name', 'UNKNOWN')}")
    
    def store_field_details(self, session_id: str, field_data: Dict, 
                          program_name: str, layout_id: int = None):
        """Store field analysis details with retry logic (public method)"""
        field_name = field_data.get('name', 'UNNAMED_FIELD')
        logger.debug(f"Public store_field_details called for: {field_name}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    self._store_field_details_internal(session_id, field_data, program_name, layout_id, cursor)
                    return  # Success, exit retry loop
                    
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database locked storing field details for {field_name}, attempt {attempt + 1}, retrying...")
                    time.sleep(1 + attempt)  # Incremental backoff
                    continue
                else:
                    logger.error(f"Error storing field details for {field_name}: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Error storing field details for {field_name}: {str(e)}")
                raise
        
        # If we get here, all retries failed
        raise sqlite3.OperationalError(f"Failed to store field details for {field_name} after {max_retries} attempts")
    
    def store_field_mappings(self, session_id: str, target_file: str, mappings: List[Dict]):
        """Store field mapping analysis results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                for mapping in mappings:
                    cursor.execute('''
                        INSERT OR REPLACE INTO field_mappings 
                        (session_id, target_file_name, field_name, mainframe_data_type,
                         oracle_data_type, mainframe_length, oracle_length, population_source,
                         source_record_layout, business_logic_type, business_logic_description,
                         derivation_logic, programs_involved_json, confidence_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (session_id, target_file, mapping.get('field_name'),
                          mapping.get('mainframe_data_type'), mapping.get('oracle_data_type'),
                          mapping.get('mainframe_length', 0), mapping.get('oracle_length', 0),
                          mapping.get('population_source'), mapping.get('source_record_layout'),
                          mapping.get('business_logic_type'), mapping.get('business_logic_description'),
                          mapping.get('derivation_logic'), json.dumps(mapping.get('programs_involved', [])),
                          mapping.get('confidence_score', 0.0)))
                
                logger.info(f"Stored {len(mappings)} field mappings for {target_file}")
        except Exception as e:
            logger.error(f"Error storing field mappings: {str(e)}")
    
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
        """Get record layouts with friendly names"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if program_name:
                    cursor.execute('''
                        SELECT id, layout_name, friendly_name, program_name, level_number,
                            line_start, line_end, source_code, fields_count, business_purpose
                        FROM record_layouts 
                        WHERE session_id = ? AND program_name = ?
                        ORDER BY layout_name
                    ''', (session_id, program_name))
                else:
                    cursor.execute('''
                        SELECT id, layout_name, friendly_name, program_name, level_number,
                            line_start, line_end, source_code, fields_count, business_purpose
                        FROM record_layouts 
                        WHERE session_id = ?
                        ORDER BY program_name, layout_name
                    ''', (session_id,))
                
                layouts = []
                for row in cursor.fetchall():
                    layout_dict = dict(row)
                    # Ensure we use the stored friendly name
                    if not layout_dict['friendly_name']:
                        layout_dict['friendly_name'] = self.generate_friendly_name(layout_dict['layout_name'], 'Record Layout')
                    layouts.append(layout_dict)
                
                return layouts
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
                        mapping['programs_involved'] = json.loads(mapping['programs_involved_json'])
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
                
                return [dict(row) for row in cursor.fetchall()]
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
                
                # Field mapping counts

                # Field counts
                cursor.execute('''
                    SELECT COUNT(DISTINCT field_name) as total_fields
                    FROM field_analysis_details 
                    WHERE session_id = ?
                ''', (session_id,))
                total_fields = cursor.fetchone()[0]
                
                # Lines of code
                cursor.execute('''
                    SELECT SUM(total_lines) as total_lines
                    FROM component_analysis 
                    WHERE session_id = ?
                ''', (session_id,))
                total_lines = cursor.fetchone()[0] or 0
                
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
                        message['context_used'] = json.loads(message['context_used_json'])
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
                
                field_mappings = [dict(row) for row in cursor.fetchall()]
                
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
                
                return [dict(row) for row in cursor.fetchall()]
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
                
                return [dict(row) for row in cursor.fetchall()]
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
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error exporting dependencies: {str(e)}")
            return []
        

        
    def debug_database_schema(self):
        """Debug method to check database schema"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(component_analysis)")
                columns = cursor.fetchall()
                logger.info("Component Analysis Table Schema:")
                for column in columns:
                    logger.info(f"  {column}")
        except Exception as e:
            logger.error(f"Error checking schema: {str(e)}")
    # Add this method to database_manager.py:

    def get_component_source_code(self, session_id: str, component_name: str = None, 
                                component_type: str = None, max_size: int = 50000) -> Dict:
        """Get source code for chat, with size-based truncation"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if component_name:
                    cursor.execute('''
                        SELECT component_name, component_type, source_content, 
                            total_lines, analysis_result_json
                        FROM component_analysis 
                        WHERE session_id = ? AND component_name = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                    ''', (session_id, component_name))
                else:
                    cursor.execute('''
                        SELECT component_name, component_type, source_content, 
                            total_lines, analysis_result_json
                        FROM component_analysis 
                        WHERE session_id = ?
                        ORDER BY total_lines ASC, created_at DESC
                    ''', (session_id,))
                
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    source_content = row_dict.get('source_content', '')
                    
                    # Determine if we should send full or partial source
                    if len(source_content) <= max_size:
                        # Send full source for small components
                        row_dict['source_strategy'] = 'full'
                        row_dict['source_for_chat'] = source_content
                    else:
                        # Send summary + key sections for large components
                        row_dict['source_strategy'] = 'partial'
                        row_dict['source_for_chat'] = self._create_source_summary(
                            source_content, row_dict.get('analysis_result_json', '{}')
                        )
                    
                    results.append(row_dict)
                
                return {
                    'success': True,
                    'components': results
                }
                
        except Exception as e:
            logger.error(f"Error getting component source code: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'components': []
            }

    def _create_source_summary(self, source_content: str, analysis_json: str) -> str:
        """Create a summary of source code for large components"""
        try:
            analysis = json.loads(analysis_json) if analysis_json else {}
            
            lines = source_content.split('\n')
            total_lines = len(lines)
            
            # Create summary with key sections
            summary_parts = [
                f"=== SOURCE CODE SUMMARY (Total: {total_lines} lines) ===",
                "",
                "BUSINESS PURPOSE:",
                analysis.get('business_purpose', 'Not analyzed'),
                "",
                "KEY SECTIONS:",
            ]
            
            # Add first 20 lines (header/identification)
            summary_parts.extend([
                "--- PROGRAM HEADER (First 20 lines) ---"
            ])
            summary_parts.extend(lines[:20])
            summary_parts.append("")
            
            # Add record layouts if available
            record_layouts = analysis.get('record_layouts', [])
            if record_layouts:
                summary_parts.extend([
                    f"--- RECORD LAYOUTS ({len(record_layouts)} found) ---"
                ])
                for layout in record_layouts[:3]:  # Show first 3 layouts
                    layout_name = layout.get('name', 'Unknown')
                    field_count = len(layout.get('fields', []))
                    summary_parts.append(f"Layout: {layout_name} ({field_count} fields)")
                    
                    # Show source code for this layout if available
                    if layout.get('source_code'):
                        summary_parts.extend(layout['source_code'].split('\n')[:10])
                    summary_parts.append("")
            
            # Add procedure division start if found
            proc_div_start = -1
            for i, line in enumerate(lines):
                if 'PROCEDURE DIVISION' in line.upper():
                    proc_div_start = i
                    break
            
            if proc_div_start >= 0:
                summary_parts.extend([
                    "--- PROCEDURE DIVISION START (Next 15 lines) ---"
                ])
                end_idx = min(proc_div_start + 15, total_lines)
                summary_parts.extend(lines[proc_div_start:end_idx])
            
            return '\n'.join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error creating source summary: {str(e)}")
            return f"Source code available ({len(source_content)} characters) but summary generation failed"