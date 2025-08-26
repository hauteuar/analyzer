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
        """Context manager for database connections with proper locking and WAL mode"""
        with self._lock:  # Thread-safe access
            conn = None
            try:
                logger.debug(f"Attempting to connect to database: {self.db_path}")
                conn = sqlite3.connect(
                    self.db_path, 
                    timeout=30.0,  # 30 second timeout for locks
                    isolation_level=None  # Autocommit mode
                )
                conn.row_factory = sqlite3.Row  # Enable dict-like access
                
                # Configure SQLite for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
                conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
                conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
                conn.execute("PRAGMA temp_store=MEMORY")  # Temp tables in memory
                conn.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
                
                logger.debug("Database connection established successfully")
                yield conn
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower():
                    logger.error(f"Database locked error: {str(e)}")
                    # Try to resolve lock by waiting and retrying
                    self._handle_database_lock(conn)
                    raise
                else:
                    logger.error(f"Database operational error: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Database error: {str(e)}")
                if conn:
                    conn.rollback()
                raise
            finally:
                if conn:
                    try:
                        conn.close()
                        logger.debug("Database connection closed")
                    except:
                        pass  # Ignore close errors
    
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
        """Initialize database schema with retry logic"""
        if self.init_executed:
            logger.debug("Database already initialized, skipping")
            return
        
        logger.info("Initializing database schema...")
        max_retries = 3
        for attempt in range(max_retries):
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
                            FOREIGN KEY (field_id) REFERENCES record_layouts(id)
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
                    
                    logger.info("Database schema initialized successfully")
                    self.init_executed = True
                    return
                    
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database locked on initialization attempt {attempt + 1}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise
            except Exception as e:
                logger.error(f"Database initialization error: {str(e)}")
                raise
        
        raise sqlite3.OperationalError("Failed to initialize database after multiple attempts")
    
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
        """Store component analysis results - FIXED VERSION"""
        logger.info(f"Storing component: {component_name} ({component_type})")
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Extract data with fallbacks for missing fields
                total_lines = 0
                total_fields = 0
                
                # Handle different data structures from component extractor
                if isinstance(analysis_result, dict):
                    # Try different field name variations
                    total_lines = (analysis_result.get('total_lines', 0) or 
                                analysis_result.get('lines', 0) or 
                                len(str(analysis_result.get('content', '')).split('\n')))
                    
                    # Count fields from different sources
                    fields = analysis_result.get('fields', [])
                    if not fields and 'record_layouts' in analysis_result:
                        # Count fields in all record layouts
                        for layout in analysis_result['record_layouts']:
                            if isinstance(layout, dict) and 'fields' in layout:
                                fields.extend(layout['fields'])
                    
                    total_fields = len(fields)
                    
                    # Add essential data that might be missing
                    if 'business_purpose' not in analysis_result and component_type == 'PROGRAM':
                        analysis_result['business_purpose'] = f'COBOL program component with {total_lines} lines'
                    
                    if 'friendly_name' not in analysis_result:
                        analysis_result['friendly_name'] = component_name.replace('_', ' ').replace('-', ' ').title()
                
                logger.debug(f"Component data: lines={total_lines}, fields={total_fields}")
                
                # Store main component with complete data
                cursor.execute('''
                    INSERT OR REPLACE INTO component_analysis 
                    (session_id, component_name, component_type, file_path, 
                    total_lines, total_fields, analysis_result_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (session_id, component_name, component_type, file_path,
                    total_lines, total_fields, json.dumps(analysis_result)))
                
                logger.info(f"Successfully stored component: {component_name}")
                
                # Store record layouts if present (skip if already stored by component extractor)
                # Replace the hanging section with this:
if component_type == 'PROGRAM' and 'record_layouts' in analysis_result:
    try:
        logger.info(f"Checking for existing layouts for {component_name}...")
        
        # Add timeout protection for the database query
        existing_layouts = []
        try:
            # Quick check with timeout
            with sqlite3.connect(self.db_path, timeout=5.0) as check_conn:
                cursor = check_conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM record_layouts 
                    WHERE session_id = ? AND program_name = ?
                ''', (session_id, component_name))
                count = cursor.fetchone()[0]
                existing_layouts = ['dummy'] if count > 0 else []
                logger.info(f"Found {count} existing layouts for {component_name}")
        except sqlite3.OperationalError as db_error:
            logger.warning(f"Database timeout checking layouts: {str(db_error)}")
            # Assume no existing layouts to avoid hanging
            existing_layouts = []
        
        if not existing_layouts:
            record_layouts = analysis_result['record_layouts']
            logger.info(f"Storing {len(record_layouts)} record layouts for {component_name}")
            
            for i, layout_data in enumerate(record_layouts, 1):
                try:
                    logger.info(f"Storing layout {i}/{len(record_layouts)}: {layout_data.get('name', 'unknown')}")
                    self.store_record_layout(session_id, layout_data, component_name)
                    logger.info(f"Successfully stored layout {i}/{len(record_layouts)}")
                except Exception as layout_error:
                    logger.error(f"Error storing layout {layout_data.get('name', 'unknown')}: {str(layout_error)}")
                    # Continue with next layout instead of failing completely
                    continue
            
            logger.info(f"Completed storing all {len(record_layouts)} layouts for {component_name}")
        else:
            logger.info(f"Layouts already exist for {component_name}, skipping storage")
            
    except Exception as layout_section_error:
        logger.error(f"Error in layout storage section for {component_name}: {str(layout_section_error)}")
        # Don't let layout storage failure block component storage
        pass

    
    def store_record_layout(self, session_id: str, layout_data: Dict, program_name: str):
        """Store record layout (01 level) information with enhanced logging and retry logic"""
        layout_name = layout_data.get('name', 'UNKNOWN')
        logger.info(f"Storing record layout: {layout_name} for program: {program_name}")
        
        # Add immediate validation
        if not layout_data:
            logger.error("Empty layout_data provided")
            return
        
        if not session_id:
            logger.error("Empty session_id provided")
            return
            
        fields = layout_data.get('fields', [])
        logger.info(f"Layout {layout_name} has {len(fields)} fields to process")
        
        max_retries = 3
        for attempt in range(max_retries):
            conn = None
            try:
                logger.debug(f"Attempt {attempt + 1} to store layout {layout_name}")
                
                # Use direct connection instead of context manager for better control
                conn = sqlite3.connect(
                    self.db_path, 
                    timeout=10.0,  # Reduced timeout
                    isolation_level=None
                )
                conn.row_factory = sqlite3.Row
                
                # Configure for faster writes
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=10000")  # 10 second timeout
                
                cursor = conn.cursor()
                
                # Log the data being inserted with validation
                logger.debug(f"Inserting record layout: {layout_name}")
                logger.debug(f"Session ID: {session_id}")
                logger.debug(f"Program name: {program_name}")
                
                # Validate required fields before insert
                insert_data = (
                    session_id, 
                    layout_data.get('name', 'UNNAMED_LAYOUT'),
                    program_name or 'UNKNOWN_PROGRAM',
                    str(layout_data.get('level', '01')),
                    layout_data.get('line_start', 0),
                    layout_data.get('line_end', 0),
                    layout_data.get('source_code', '')[:4000],  # Limit source code length
                    len(fields)
                )
                
                # Insert record layout with explicit commit
                logger.debug("Executing INSERT for record_layouts table")
                cursor.execute('''
                    INSERT OR REPLACE INTO record_layouts 
                    (session_id, layout_name, program_name, level_number, 
                    line_start, line_end, source_code, fields_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', insert_data)
                
                layout_id = cursor.lastrowid
                logger.info(f"Inserted record layout with ID: {layout_id}")
                
                # Process fields in smaller batches to avoid locks
                batch_size = 10
                field_batches = [fields[i:i + batch_size] for i in range(0, len(fields), batch_size)]
                
                logger.info(f"Processing {len(fields)} fields in {len(field_batches)} batches")
                
                for batch_idx, field_batch in enumerate(field_batches, 1):
                    logger.debug(f"Processing field batch {batch_idx}/{len(field_batches)} ({len(field_batch)} fields)")
                    
                    for field_idx, field in enumerate(field_batch, 1):
                        try:
                            field_name = field.get('name', f'UNNAMED_FIELD_{field_idx}')
                            logger.debug(f"Processing field: {field_name}")
                            
                            # Validate field data
                            field_insert_data = (
                                session_id, 
                                layout_id, 
                                field_name,
                                program_name, 
                                field.get('operation_type', 'DEFINITION'),
                                field.get('line_number', 0), 
                                str(field.get('code_snippet', ''))[:1000],  # Limit length
                                field.get('usage_type', 'UNKNOWN'), 
                                field.get('source_field', '')[:100],  # Limit length
                                field.get('target_field', '')[:100],  # Limit length
                                field.get('business_purpose', f"Field definition for {field_name}")[:500],  # Limit length
                                float(field.get('confidence', 0.8))
                            )
                            
                            cursor.execute('''
                                INSERT INTO field_analysis_details 
                                (session_id, field_id, field_name, program_name, operation_type,
                                line_number, code_snippet, usage_type, source_field, target_field,
                                business_purpose, analysis_confidence)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', field_insert_data)
                            
                            logger.debug(f"Successfully inserted field: {field_name}")
                            
                        except Exception as field_error:
                            logger.error(f"Error storing field {field.get('name', 'UNKNOWN')}: {str(field_error)}")
                            # Continue with next field instead of failing entire batch
                            continue
                    
                    # Commit after each batch
                    conn.commit()
                    logger.debug(f"Committed batch {batch_idx}")
                
                # Final commit
                conn.commit()
                logger.info(f"Successfully stored record layout: {layout_name} with {len(fields)} fields")
                return  # Success, exit retry loop
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database locked storing record layout {layout_name}, attempt {attempt + 1}, retrying in {1 + attempt}s...")
                    time.sleep(1 + attempt)  # Incremental backoff
                    continue
                else:
                    logger.error(f"SQLite error storing record layout {layout_name}: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error storing record layout {layout_name}: {str(e)}")
                logger.error(f"Layout data keys: {list(layout_data.keys())}")
                if fields:
                    logger.error(f"First field sample: {fields[0] if fields else 'No fields'}")
                raise
            finally:
                if conn:
                    try:
                        conn.close()
                        logger.debug(f"Closed database connection for layout {layout_name}")
                    except:
                        pass
        
        # If we get here, all retries failed
        raise sqlite3.OperationalError(f"Failed to store record layout {layout_name} after {max_retries} attempts")

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