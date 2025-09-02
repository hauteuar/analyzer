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
import re
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
        """Initialize database schema with record-level classification support"""
        if self.init_executed:
            logger.debug("Database already initialized, skipping")
            return
        
        logger.info("Initializing database schema...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Get all existing tables first
                    cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    """)
                    existing_tables = cursor.fetchall()
                    
                    # Drop all existing tables
                    for table in existing_tables:
                        table_name = table[0]
                        cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
                        logger.info(f"Dropped existing table: {table_name}")
                    
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
                    
                    # Component analysis
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
                    
                    # Derived components
                    cursor.execute('''
                        CREATE TABLE derived_components (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            parent_component TEXT NOT NULL,
                            component_name TEXT NOT NULL,
                            component_type TEXT NOT NULL,
                            friendly_name TEXT,
                            business_purpose TEXT,
                            line_start INTEGER,
                            line_end INTEGER,
                            fields_count INTEGER DEFAULT 0,
                            source_code TEXT,
                            metadata_json TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
                        )
                    ''')
                    
                    # Record layouts with record-level classification (UPDATED)
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
                            
                            -- Record-level classification columns
                            record_classification TEXT DEFAULT 'STATIC',
                            record_usage_description TEXT,
                            has_whole_record_operations BOOLEAN DEFAULT 0,
                            
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
                        )
                    ''')
                    
                    # Field analysis with record classification context (UPDATED)
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
                            
                            -- Complete source code context fields
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
                            
                            -- FIXED: Add field length and type columns
                            mainframe_length INTEGER DEFAULT 0,
                            oracle_length INTEGER DEFAULT 0,
                            mainframe_data_type TEXT,
                            oracle_data_type TEXT,
                            
                            -- Record-level classification context
                            record_classification TEXT DEFAULT 'STATIC',
                            inherited_from_record BOOLEAN DEFAULT 0,
                            effective_classification TEXT DEFAULT 'UNKNOWN',
                            
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id),
                            FOREIGN KEY (field_id) REFERENCES record_layouts(id)
                        )
                    ''')
                    
                    # Field mappings (unchanged)
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
                            source_code_evidence_json TEXT,
                            actual_cobol_definition TEXT,
                            usage_patterns_json TEXT,
                            analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
                        )
                    ''')
                    
                    # Dependency relationships (unchanged)
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
                    
                    # Chat conversations (unchanged)
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

                    cursor.execute('''
                        CREATE TABLE llm_component_summaries (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            component_name TEXT NOT NULL,
                            component_type TEXT NOT NULL,
                            business_purpose TEXT,
                            primary_function TEXT,
                            complexity_score REAL DEFAULT 0.5,
                            key_features_json TEXT,
                            integration_points_json TEXT,
                            data_sources_json TEXT,
                            business_domain TEXT DEFAULT 'WEALTH_MANAGEMENT',
                            raw_llm_response TEXT,
                            is_raw_response BOOLEAN DEFAULT 0,
                            friendly_name TEXT,
                            analysis_confidence REAL DEFAULT 0.8,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id),
                            UNIQUE(session_id, component_name, component_type)
                        )
                    ''')
                    
                    # Add index for faster retrieval
                    cursor.execute('''
                        CREATE INDEX idx_llm_summaries_lookup 
                        ON llm_component_summaries(session_id, component_name, component_type)
                    ''')

                    # Add these new tables for RAG system

                    # NEW: RAG query metrics table
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
                    
                    # NEW: Query patterns table
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


                    # Create performance indexes (UPDATED with new classification indexes)
                    indexes = [
                        'CREATE INDEX idx_derived_components_parent ON derived_components(session_id, parent_component)',
                        'CREATE INDEX idx_field_mappings_target_file ON field_mappings(target_file_name, session_id)',
                        'CREATE INDEX idx_field_details_field_name ON field_analysis_details(field_name, session_id)',
                        'CREATE INDEX idx_field_details_program ON field_analysis_details(program_name, session_id)',
                        'CREATE INDEX idx_dependency_source ON dependency_relationships(source_component, session_id)',
                        'CREATE INDEX idx_dependency_target ON dependency_relationships(target_component, session_id)',
                        'CREATE INDEX idx_chat_session_conv ON chat_conversations(session_id, conversation_id)',
                        'CREATE INDEX idx_component_session ON component_analysis(session_id, component_type)',
                        'CREATE INDEX idx_record_layouts_session ON record_layouts(session_id, program_name)',
                        'CREATE INDEX idx_record_layouts_name ON record_layouts(layout_name, session_id)',
                        
                        # New indexes for record classification
                        'CREATE INDEX idx_record_layouts_classification ON record_layouts(record_classification, session_id)',
                        'CREATE INDEX idx_field_record_classification ON field_analysis_details(record_classification, session_id)',
                        'CREATE INDEX idx_field_effective_classification ON field_analysis_details(effective_classification, session_id)'
                    ]

                    for index_sql in indexes:
                        try:
                            cursor.execute(index_sql)
                        except Exception as idx_error:
                            logger.warning(f"Failed to create index: {index_sql[:50]}... Error: {idx_error}")
                    self.create_friendly_name_cache_table()
                    logger.info("Database schema created successfully with record-level classification support")
                    self.init_executed = True
                    return
                    
            except Exception as e:
                logger.error(f"Database initialization error: {str(e)}")
                raise
        
        raise sqlite3.OperationalError("Failed to initialize database after multiple attempts")
    def store_derived_components(self, session_id: str, parent_component: str, derived_components: List[Dict]):
        """Store derived components in separate table"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Clear existing derived components for this parent
                cursor.execute('''
                    DELETE FROM derived_components 
                    WHERE session_id = ? AND parent_component = ?
                ''', (session_id, parent_component))
                
                logger.info(f"Storing {len(derived_components)} derived components for {parent_component}")
                
                # Insert new derived components
                for component in derived_components:
                    cursor.execute('''
                        INSERT INTO derived_components 
                        (session_id, parent_component, component_name, component_type, 
                        friendly_name, business_purpose, line_start, line_end, 
                        fields_count, source_code, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id, 
                        parent_component, 
                        component.get('name', ''),
                        component.get('type', ''),
                        component.get('friendly_name', ''),
                        component.get('business_purpose', ''),
                        component.get('line_start', 0),
                        component.get('line_end', 0),
                        len(component.get('fields', [])),
                        component.get('source_code', ''),
                        json.dumps(component, default=str)
                    ))
                
                logger.info(f"Successfully stored {len(derived_components)} derived components for {parent_component}")
                
        except Exception as e:
            logger.error(f"Error storing derived components for {parent_component}: {str(e)}")
            raise

    def get_derived_components(self, session_id: str, parent_component: str = None) -> List[Dict]:
        """Get derived components from separate table"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if parent_component:
                    cursor.execute('''
                        SELECT * FROM derived_components 
                        WHERE session_id = ? AND parent_component = ?
                        ORDER BY component_type, component_name
                    ''', (session_id, parent_component))
                else:
                    cursor.execute('''
                        SELECT * FROM derived_components 
                        WHERE session_id = ?
                        ORDER BY parent_component, component_type, component_name
                    ''', (session_id,))
                
                components = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    # Parse metadata if available
                    if row_dict['metadata_json']:
                        try:
                            metadata = json.loads(row_dict['metadata_json'])
                            row_dict['metadata'] = metadata
                        except:
                            pass
                    components.append(row_dict)
                
                return components
                
        except Exception as e:
            logger.error(f"Error getting derived components: {str(e)}")
            return []

    def get_derived_components_count(self, session_id: str, parent_component: str) -> int:
        """Get count of derived components for a parent"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM derived_components 
                    WHERE session_id = ? AND parent_component = ?
                ''', (session_id, parent_component))
                
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting derived components count: {str(e)}")
            return 0


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
    
        # Add this method to DatabaseManager to check field limits
    def check_database_limits(self):
        """Check database field limitations"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(component_analysis)")
                columns = cursor.fetchall()
                
                for column in columns:
                    if column[1] == 'analysis_result_json':
                        logger.info(f"analysis_result_json column type: {column[2]}")
                        
        except Exception as e:
            logger.error(f"Error checking database limits: {e}")

    # Changes for component_extractor.py

    def store_component_analysis(self, session_id: str, component_name: str, 
                        component_type: str, file_path: str, analysis_result: Dict):
        """Store component analysis with improved source code handling"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # CHANGE 1: Ensure source content is preserved properly
                source_content = analysis_result.get('content', '')
                if not source_content:
                    logger.warning(f"No source content for {component_name}")
                
                # CHANGE 2: Create a clean copy for JSON serialization
                clean_analysis = {}
                for key, value in analysis_result.items():
                    if key == 'content':
                        # Don't store content in JSON to save space
                        continue
                    elif key == 'derived_components':
                        clean_analysis[key] = value if isinstance(value, list) else []
                    elif isinstance(value, (str, int, float, bool, list, dict)):
                        clean_analysis[key] = value
                    else:
                        clean_analysis[key] = str(value)
                
                # CHANGE 3: Better JSON size management
                analysis_json = json.dumps(clean_analysis, ensure_ascii=False, separators=(',', ':'))
                
                # CHANGE 4: If JSON is too large, keep only essential data
                #if len(analysis_json) > 500000:  # 500KB limit
                #    logger.warning(f"Analysis JSON too large ({len(analysis_json)} chars), keeping essentials...")
                #    essential_data = {
                #        'name': analysis_result.get('name'),
                #        'type': analysis_result.get('type'),
                #        'friendly_name': analysis_result.get('friendly_name'),
                #        'business_purpose': analysis_result.get('business_purpose'),
                #        'complexity_score': analysis_result.get('complexity_score'),
                #        'total_lines': analysis_result.get('total_lines'),
                #        'total_fields': len(analysis_result.get('fields', [])),
                #        'file_operations': analysis_result.get('file_operations', [])[:10],  # Limit arrays
                #        'cics_operations': analysis_result.get('cics_operations', [])[:10],
                #        'program_calls': analysis_result.get('program_calls', [])[:10],
                #        'derived_components': analysis_result.get('derived_components', []),
                #        'divisions': analysis_result.get('divisions', [])[:5],
                
                #        'copybooks': analysis_result.get('copybooks', [])[:10]
                #    }
                #    analysis_json = json.dumps(essential_data, ensure_ascii=False)
                
                # Check if component exists
                cursor.execute('''
                    SELECT COUNT(*) FROM component_analysis 
                    WHERE session_id = ? AND component_name = ? AND component_type = ?
                ''', (session_id, component_name, component_type))
                
                exists = cursor.fetchone()[0] > 0
                
                # Prepare fields with better validation
                total_lines = int(analysis_result.get('total_lines', 0))
                total_fields = len(analysis_result.get('fields', []))
                business_purpose = str(analysis_result.get('business_purpose', ''))[:1000]  # Increased limit
                complexity_score = float(analysis_result.get('complexity_score', 0.5))
                
                # CHANGE 5: Ensure source content is stored properly (not truncated)
                if exists:
                    cursor.execute('''
                        UPDATE component_analysis 
                        SET total_lines = ?, total_fields = ?, business_purpose = ?, 
                            complexity_score = ?, analysis_result_json = ?, updated_at = CURRENT_TIMESTAMP,
                            source_content = ?
                        WHERE session_id = ? AND component_name = ? AND component_type = ?
                    ''', (
                        total_lines, total_fields, business_purpose, complexity_score, 
                        analysis_json, source_content,  # Full source content
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
                        total_lines, total_fields, business_purpose, complexity_score,
                        source_content, analysis_json  # Full source content
                    ))
                
                logger.info(f"Component storage completed: {component_name} with {len(source_content)} chars source")
                
        except Exception as e:
                logger.error(f"Error storing component {component_name}: {str(e)}")
                # Don't raise to prevent stopping the process
            
    
                        # Don't re-raise the exception to prevent stopping the entire process
    
    def store_component_analysis_with_full_source(self, session_id: str, component_name: str, 
                                           component_type: str, file_path: str, 
                                           analysis_result: Dict, full_source_content: str):
        """Store component analysis ensuring full source code is preserved"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create analysis JSON WITHOUT source content to save space
                clean_analysis = {}
                for key, value in analysis_result.items():
                    if key == 'content':
                        continue  # Don't duplicate in JSON
                    elif isinstance(value, (str, int, float, bool, list, dict)):
                        clean_analysis[key] = value
                    else:
                        clean_analysis[key] = str(value)
                
                analysis_json = json.dumps(clean_analysis, ensure_ascii=False, separators=(',', ':'))
                
                # Check if component exists
                cursor.execute('''
                    SELECT COUNT(*) FROM component_analysis 
                    WHERE session_id = ? AND component_name = ? AND component_type = ?
                ''', (session_id, component_name, component_type))
                
                exists = cursor.fetchone()[0] > 0
                
                # CRITICAL: Always store full source content separately
                if exists:
                    cursor.execute('''
                        UPDATE component_analysis 
                        SET total_lines = ?, total_fields = ?, business_purpose = ?, 
                            complexity_score = ?, analysis_result_json = ?, 
                            source_content = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE session_id = ? AND component_name = ? AND component_type = ?
                    ''', (
                        int(analysis_result.get('total_lines', 0)),
                        len(analysis_result.get('fields', [])),
                        str(analysis_result.get('business_purpose', ''))[:1000],
                        float(analysis_result.get('complexity_score', 0.5)),
                        analysis_json,
                        full_source_content,  # Full source stored here
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
                        int(analysis_result.get('total_lines', 0)),
                        len(analysis_result.get('fields', [])),
                        str(analysis_result.get('business_purpose', ''))[:1000],
                        float(analysis_result.get('complexity_score', 0.5)),
                        full_source_content,  # Full source stored here
                        analysis_json
                    ))
                
                logger.info(f"Stored component {component_name} with {len(full_source_content)} chars of source")
                
        except Exception as e:
            logger.error(f"Error storing component {component_name}: {str(e)}")
            raise



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
        """Internal method to store field details with record classification context"""
        field_name = field_data.get('name', 'UNNAMED_FIELD')
        logger.debug(f"Storing field details: {field_name}")
        
        try:
            # Validate and limit data lengths
            insert_data = (
                session_id, 
                layout_id, 
                field_name[:100],
                field_data.get('friendly_name', field_name)[:100],
                program_name[:100], 
                field_data.get('parent_layout', '')[:100],
                field_data.get('operation_type', 'DEFINITION')[:50],
                field_data.get('line_number', 0), 
                str(field_data.get('code_snippet', ''))[:1000],
                field_data.get('usage_type', 'UNKNOWN')[:50], 
                field_data.get('source_field', '')[:100],
                field_data.get('target_field', '')[:100], 
                field_data.get('business_purpose', f"Field definition for {field_name}")[:500],
                float(field_data.get('confidence', 0.8)),
                
                # Enhanced field data
                field_data.get('definition_line_number', 0),
                field_data.get('definition_code', '')[:500],
                field_data.get('program_source_content', '')[:10000],  # Limit size
                field_data.get('field_references_json', '[]')[:2000],
                field_data.get('usage_summary_json', '{}')[:1000],
                field_data.get('total_program_references', 0),
                field_data.get('move_source_count', 0),
                field_data.get('move_target_count', 0),
                field_data.get('arithmetic_count', 0),
                field_data.get('conditional_count', 0),
                field_data.get('cics_count', 0),
                
                # NEW: Record-level classification fields
                field_data.get('record_classification', 'STATIC')[:50],
                field_data.get('inherited_from_record', False),
                field_data.get('effective_classification', field_data.get('usage_type', 'UNKNOWN'))[:50]
            )
            
            cursor.execute('''
                INSERT INTO field_analysis_details 
                (session_id, field_id, field_name, friendly_name, program_name, layout_name,
                operation_type, line_number, code_snippet, usage_type, source_field, target_field,
                business_purpose, analysis_confidence,
                definition_line_number, definition_code, program_source_content,
                field_references_json, usage_summary_json, total_program_references,
                move_source_count, move_target_count, arithmetic_count, conditional_count, cics_count,
                record_classification, inherited_from_record, effective_classification)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', insert_data)
            
            logger.debug(f"Successfully stored field details for: {field_name}")
            
        except Exception as e:
            logger.error(f"Error in _store_field_details_internal for field {field_name}: {str(e)}")
            logger.error(f"Insert data length: {len(insert_data)}")
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
    
    def store_field_details_enhanced(self, session_id: str, field_data: Dict, program_name: str, layout_id: int = None):
        """Enhanced field details storage with proper lengths"""
        field_name = field_data.get('name', 'UNNAMED_FIELD')
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO field_analysis_details 
                    (session_id, field_id, field_name, friendly_name, program_name, layout_name,
                    operation_type, line_number, code_snippet, usage_type, source_field, target_field,
                    business_purpose, analysis_confidence,
                    definition_line_number, definition_code, program_source_content,
                    field_references_json, usage_summary_json, total_program_references,
                    move_source_count, move_target_count, arithmetic_count, conditional_count, cics_count,
                    mainframe_length, oracle_length, mainframe_data_type, oracle_data_type,
                    record_classification, inherited_from_record, effective_classification)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id, layout_id, field_name,
                    field_data.get('friendly_name', field_name)[:100],
                    program_name[:100], 
                    field_data.get('parent_layout', '')[:100],
                    field_data.get('operation_type', 'DEFINITION')[:50],
                    field_data.get('line_number', 0),
                    field_data.get('code_snippet', '')[:1000],
                    field_data.get('usage_type', 'STATIC')[:50],
                    field_data.get('source_field', '')[:100],
                    field_data.get('target_field', '')[:100],
                    field_data.get('business_purpose', f"Field definition for {field_name}")[:500],
                    float(field_data.get('confidence', 0.8)),
                    field_data.get('definition_line_number', 0),
                    field_data.get('definition_code', '')[:5000],
                    field_data.get('program_source_content', '')[:100000],
                    field_data.get('field_references_json', '[]')[:50000],
                    field_data.get('usage_summary_json', '{}')[:100000],
                    field_data.get('total_program_references', 0),
                    field_data.get('move_source_count', 0),
                    field_data.get('move_target_count', 0),
                    field_data.get('arithmetic_count', 0),
                    field_data.get('conditional_count', 0),
                    field_data.get('cics_count', 0),
                    # Enhanced field length and type information
                    field_data.get('mainframe_length', 0),
                    field_data.get('oracle_length', 50),
                    field_data.get('mainframe_data_type', 'UNKNOWN')[:100],
                    field_data.get('oracle_data_type', 'VARCHAR2(50)')[:100],
                    field_data.get('record_classification', 'STATIC')[:50],
                    field_data.get('inherited_from_record', False),
                    field_data.get('effective_classification', 'UNKNOWN')[:50]
                ))
                
        except Exception as e:
            logger.error(f"Error storing enhanced field details for {field_name}: {str(e)}")
            raise

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
                    self.store_field_details_enhanced(session_id, field_data, program_name, layout_id)
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
    
    def generate_friendly_name(self, technical_name: str, context: str = '', 
                            business_domain: str = 'WEALTH_MANAGEMENT') -> str:
        """Generate business-friendly names with caching and fallback logic"""
        
        # Check cache first
        cache_key = f"{technical_name}_{context}_{business_domain}"
        if hasattr(self, '_friendly_name_cache') and cache_key in self._friendly_name_cache:
            return self._friendly_name_cache[cache_key]
        
        # Initialize cache if it doesn't exist
        if not hasattr(self, '_friendly_name_cache'):
            self._friendly_name_cache = {}
        
        try:
            # Try to get from database first
            friendly_name = self._get_cached_friendly_name_from_db(technical_name, context)
            if friendly_name:
                self._friendly_name_cache[cache_key] = friendly_name
                return friendly_name
            
            # Generate new friendly name
            friendly_name = self._generate_simple_friendly_name(technical_name, context, business_domain)
            
            # Store in cache and database
            self._friendly_name_cache[cache_key] = friendly_name
            self._store_friendly_name_in_db(technical_name, context, friendly_name)
            
            return friendly_name
            
        except Exception as e:
            logger.error(f"Error generating friendly name for {technical_name}: {str(e)}")
            return self._fallback_friendly_name(technical_name, context)

    def _get_cached_friendly_name_from_db(self, technical_name: str, context: str) -> Optional[str]:
        """Retrieve cached friendly name from database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT friendly_name FROM friendly_name_cache 
                    WHERE technical_name = ? AND context = ?
                    ORDER BY created_at DESC LIMIT 1
                ''', (technical_name, context))
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            logger.debug(f"Error retrieving cached friendly name: {str(e)}")
            return None

    def _store_friendly_name_in_db(self, technical_name: str, context: str, friendly_name: str):
        """Store friendly name in database cache"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO friendly_name_cache 
                    (technical_name, context, friendly_name, created_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (technical_name, context, friendly_name))
                
        except Exception as e:
            logger.debug(f"Error storing friendly name cache: {str(e)}")

    def _generate_simple_friendly_name(self, technical_name: str, context: str = '', 
                                    business_domain: str = 'WEALTH_MANAGEMENT') -> str:
        """Generate friendly name using business domain patterns"""
        if not technical_name:
            return context.title() if context else 'Unknown Component'

        name = str(technical_name).upper().strip()

        # Remove common technical prefixes
        prefixes = ['WS-', 'LS-', 'WK-', 'FD-', 'FD_', 'TB-', 'TB_', 'SRV-', 'PRG-', 'TMS', 'TMST']
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        # Business domain mappings for wealth management
        if business_domain == 'WEALTH_MANAGEMENT':
            wm_mappings = {
                'ACCT': 'Account',
                'CUST': 'Customer', 
                'PORT': 'Portfolio',
                'POS': 'Position',
                'TXN': 'Transaction',
                'TRAN': 'Transaction',
                'BAL': 'Balance',
                'VAL': 'Valuation',
                'PERF': 'Performance',
                'RISK': 'Risk',
                'FEE': 'Fee',
                'COMM': 'Commission',
                'RPT': 'Report',
                'ALLOC': 'Allocation',
                'TRADE': 'Trade',
                'HOLD': 'Holdings',
                'CASH': 'Cash',
                'SEC': 'Security',
                'PRICE': 'Pricing',
                'STMT': 'Statement',
                'ADDR': 'Address',
                'PHONE': 'Phone',
                'EMAIL': 'Email'
            }
            
            # Apply mappings
            for tech_term, business_term in wm_mappings.items():
                if tech_term in name:
                    name = name.replace(tech_term, business_term)
        
        # Clean up formatting
        name = re.sub(r'[_\-\.]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        name = name.title()
        
        # Add context suffix if provided
        if context and context.lower() not in name.lower():
            context_suffix = context.replace('_', ' ').title()
            if context_suffix not in ['Field', 'Component', 'Program', 'File']:
                name = f"{name} {context_suffix}"
        
        if not name:
            return context.title() if context else technical_name
        
        return name

    def _fallback_friendly_name(self, technical_name: str, context: str) -> str:
        """Fallback friendly name generation"""
        if not technical_name:
            return 'Unknown'
        
        # Simple cleanup
        name = str(technical_name).replace('_', ' ').replace('-', ' ').title()
        
        if context:
            return f"{name} ({context})"
        
        return name

    def create_friendly_name_cache_table(self):
        """Create friendly name cache table - call this in initialize_database()"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS friendly_name_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        technical_name TEXT NOT NULL,
                        context TEXT NOT NULL DEFAULT '',
                        friendly_name TEXT NOT NULL,
                        business_domain TEXT DEFAULT 'WEALTH_MANAGEMENT',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(technical_name, context)
                    )
                ''')
                
                # Create index for faster lookups
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_friendly_name_lookup 
                    ON friendly_name_cache(technical_name, context)
                ''')
                
                logger.debug("Friendly name cache table created successfully")
                
        except Exception as e:
            logger.error(f"Error creating friendly name cache table: {str(e)}")

    def batch_generate_friendly_names(self, items: List[Dict], context: str = '', 
                                    business_domain: str = 'WEALTH_MANAGEMENT') -> Dict[str, str]:
        """Generate friendly names for multiple items efficiently"""
        
        friendly_names = {}
        
        for item in items[:50]:  # Limit to prevent performance issues
            try:
                if isinstance(item, dict):
                    name = item.get('name', item.get('copybook_name', 
                        item.get('file_name', item.get('program_name', 'UNKNOWN'))))
                else:
                    name = str(item)
                
                if name and name != 'UNKNOWN':
                    friendly_names[name] = self.generate_friendly_name(name, context, business_domain)
                    
            except Exception as e:
                logger.warning(f"Error generating friendly name for item {item}: {str(e)}")
                continue
        
        return friendly_names

    def update_component_friendly_names(self, session_id: str):
        """Update friendly names for all components in a session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get all components that need friendly names
                cursor.execute('''
                    SELECT id, component_name, component_type 
                    FROM component_analysis 
                    WHERE session_id = ? AND (friendly_name IS NULL OR friendly_name = '' OR friendly_name = component_name)
                ''', (session_id,))
                
                components = cursor.fetchall()
                updated_count = 0
                
                for comp_id, comp_name, comp_type in components:
                    try:
                        friendly_name = self.generate_friendly_name(comp_name, comp_type)
                        
                        cursor.execute('''
                            UPDATE component_analysis 
                            SET friendly_name = ? 
                            WHERE id = ?
                        ''', (friendly_name, comp_id))
                        
                        updated_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error updating friendly name for {comp_name}: {str(e)}")
                        continue
                
                logger.info(f"Updated {updated_count} component friendly names for session {session_id}")
                
        except Exception as e:
            logger.error(f"Error updating component friendly names: {str(e)}")


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

    # Changes for database_manager.py

    def get_component_source_code(self, session_id: str, component_name: str = None, 
                            component_type: str = None, max_size: int = 200000) -> Dict:  # Increased default
        """Get source code for chat with better size handling"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if component_name:
                    cursor.execute('''
                        SELECT component_name, component_type, source_content, 
                            total_lines, analysis_result_json, file_path
                        FROM component_analysis 
                        WHERE session_id = ? AND component_name = ?
                        ORDER BY created_at DESC LIMIT 1
                    ''', (session_id, component_name))
                else:
                    cursor.execute('''
                        SELECT component_name, component_type, source_content, 
                            total_lines, analysis_result_json, file_path
                        FROM component_analysis 
                        WHERE session_id = ? AND source_content IS NOT NULL 
                            AND LENGTH(source_content) > 100
                        ORDER BY 
                            CASE WHEN component_type = 'PROGRAM' THEN 1 ELSE 2 END,
                            total_lines ASC, created_at DESC
                        LIMIT 5
                    ''', (session_id,))
                
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    source_content = row_dict.get('source_content', '')
                    
                    # CHANGE 3: Better source strategy with more content
                    if len(source_content) <= max_size:
                        # Send full source
                        row_dict['source_strategy'] = 'full'
                        row_dict['source_for_chat'] = source_content
                    elif len(source_content) <= max_size * 3:  # Increased multiplier
                        # Send substantial portion
                        row_dict['source_strategy'] = 'substantial'  
                        row_dict['source_for_chat'] = self._create_substantial_source_summary(
                            source_content, row_dict.get('analysis_result_json', '{}'), max_size
                        )
                    else:
                        # Enhanced summary with more actual code
                        row_dict['source_strategy'] = 'enhanced_summary'
                        row_dict['source_for_chat'] = self._create_enhanced_source_summary_with_more_code(
                            source_content, row_dict.get('analysis_result_json', '{}')
                        )
                    
                    results.append(row_dict)
                
                return {
                    'success': True,
                    'components': results,
                    'source_available': len(results) > 0 and any(len(r.get('source_for_chat', '')) > 100 for r in results)
                }
                
        except Exception as e:
            logger.error(f"Error getting component source code: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'components': [],
                'source_available': False
            }
        
    def _create_substantial_source_summary(self, source_content: str, analysis_json: str, max_size: int) -> str:
        """Create substantial source summary with more actual code"""
        try:
            analysis = json.loads(analysis_json) if analysis_json else {}
            lines = source_content.split('\n')
            total_lines = len(lines)
            
            # CHANGE 4: Include more actual source code
            summary_parts = [
                f"=== SOURCE CODE EXTRACT (Total: {total_lines} lines) ===",
                "",
                "BUSINESS PURPOSE:",
                analysis.get('business_purpose', 'Not analyzed'),
                "",
            ]
            
            # CHANGE 5: Include more header lines
            summary_parts.extend([
                "--- PROGRAM HEADER (First 30 lines) ---"
            ])
            summary_parts.extend(lines[:30])  # Increased from 20
            summary_parts.append("")
            
            # Add procedure division with more context
            proc_div_start = -1
            for i, line in enumerate(lines):
                if 'PROCEDURE DIVISION' in line.upper():
                    proc_div_start = i
                    break
            
            if proc_div_start >= 0:
                summary_parts.extend([
                    "--- PROCEDURE DIVISION (First 40 lines) ---"  # Increased from 15
                ])
                end_idx = min(proc_div_start + 40, total_lines)
                summary_parts.extend(lines[proc_div_start:end_idx])
                summary_parts.append("")
            
            # CHANGE 6: Add working storage section
            ws_start = -1
            for i, line in enumerate(lines):
                if 'WORKING-STORAGE SECTION' in line.upper():
                    ws_start = i
                    break
            
            if ws_start >= 0:
                summary_parts.extend([
                    "--- WORKING STORAGE SECTION (First 30 lines) ---"
                ])
                end_idx = min(ws_start + 30, total_lines)
                summary_parts.extend(lines[ws_start:end_idx])
                summary_parts.append("")
            
            # CHANGE 7: Add record layouts with full source
            record_layouts = analysis.get('record_layouts', [])
            if record_layouts:
                summary_parts.extend([
                    f"--- RECORD LAYOUTS ({len(record_layouts)} found) ---"
                ])
                for layout in record_layouts[:2]:  # Show first 2 layouts with full source
                    layout_name = layout.get('name', 'Unknown')
                    summary_parts.append(f"Layout: {layout_name}")
                    
                    # Include the actual layout source code if available
                    if layout.get('source_code'):
                        summary_parts.extend(layout['source_code'].split('\n'))
                    summary_parts.append("")
            
            result = '\n'.join(summary_parts)
            
            # Trim if still too large
            if len(result) > max_size:
                result = result[:max_size] + "\n\n[SOURCE TRUNCATED - More code available on request]"
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating substantial source summary: {str(e)}")
            return f"Source code available ({len(source_content)} characters) but summary generation failed: {str(e)}"

    def _create_enhanced_source_summary_with_more_code(self, source_content: str, analysis_json: str) -> str:
        """Create enhanced summary with MORE actual source code"""
        try:
            analysis = json.loads(analysis_json) if analysis_json else {}
            lines = source_content.split('\n')
            total_lines = len(lines)
            
            summary_parts = [
                f"=== COMPREHENSIVE SOURCE ANALYSIS ({total_lines} lines) ===",
                "",
                "BUSINESS PURPOSE:",
                analysis.get('business_purpose', 'Business logic analysis pending'),
                "",
            ]
            
            # Include MORE header content
            summary_parts.extend([
                "--- PROGRAM HEADER (First 50 lines) ---"  # Increased
            ])
            summary_parts.extend(lines[:50])  # More lines
            summary_parts.append("")
            
            # Include working storage section
            ws_start = -1
            for i, line in enumerate(lines):
                if 'WORKING-STORAGE SECTION' in line.upper():
                    ws_start = i
                    break
            
            if ws_start >= 0:
                summary_parts.extend([
                    "--- WORKING STORAGE SECTION (First 50 lines) ---"  # Increased
                ])
                end_idx = min(ws_start + 50, total_lines)
                summary_parts.extend(lines[ws_start:end_idx])
                summary_parts.append("")
            
            # Include procedure division
            proc_div_start = -1
            for i, line in enumerate(lines):
                if 'PROCEDURE DIVISION' in line.upper():
                    proc_div_start = i
                    break
            
            if proc_div_start >= 0:
                summary_parts.extend([
                    "--- PROCEDURE DIVISION (First 60 lines) ---"  # Increased
                ])
                end_idx = min(proc_div_start + 60, total_lines)
                summary_parts.extend(lines[proc_div_start:end_idx])
                summary_parts.append("")
            
            # Include record layouts with full source
            record_layouts = analysis.get('record_layouts', [])
            if record_layouts:
                summary_parts.extend([
                    f"--- RECORD LAYOUTS ({len(record_layouts)} found) ---"
                ])
                for layout in record_layouts[:3]:  # Show first 3 layouts
                    layout_name = layout.get('name', 'Unknown')
                    summary_parts.append(f"=== LAYOUT: {layout_name} ===")
                    
                    if layout.get('source_code'):
                        summary_parts.extend(layout['source_code'].split('\n'))
                    else:
                        summary_parts.append(f"[Layout {layout_name} - source not available]")
                    summary_parts.append("")
            
            # Add business logic excerpts
            summary_parts.extend([
                "--- KEY BUSINESS LOGIC EXCERPTS ---"
            ])
            
            business_logic_lines = []
            keywords = ['PERFORM', 'IF', 'EVALUATE', 'CALL', 'MOVE', 'COMPUTE', 'EXEC CICS', 'READ', 'WRITE']
            
            for i, line in enumerate(lines):
                line_upper = line.strip().upper()
                if any(keyword in line_upper for keyword in keywords) and line.strip():
                    business_logic_lines.append(f"{i+1:4d}: {line.strip()}")
                    if len(business_logic_lines) >= 30:  # More logic lines
                        break
            
            summary_parts.extend(business_logic_lines)
            summary_parts.append("")
            summary_parts.append("[Additional source code sections available on request]")
            
            return '\n'.join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error creating enhanced source summary: {str(e)}")
            return f"Source code available ({len(source_content)} characters) - summary generation failed: {str(e)}"
        
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
    

    def store_llm_summary(self, session_id: str, component_name: str, component_type: str, llm_summary: Dict):
        """Store LLM summary in dedicated table"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Handle both structured and raw responses
                if llm_summary.get('is_raw', False):
                    cursor.execute('''
                        INSERT OR REPLACE INTO llm_component_summaries 
                        (session_id, component_name, component_type, business_purpose, 
                        raw_llm_response, is_raw_response, friendly_name, analysis_confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id, component_name, component_type,
                        llm_summary.get('business_purpose', ''),
                        llm_summary.get('raw_response', ''),
                        True,
                        llm_summary.get('friendly_name', ''),
                        llm_summary.get('analysis_confidence', 0.5)
                    ))
                else:
                    cursor.execute('''
                        INSERT OR REPLACE INTO llm_component_summaries 
                        (session_id, component_name, component_type, business_purpose, primary_function,
                        complexity_score, key_features_json, integration_points_json, data_sources_json,
                        business_domain, friendly_name, analysis_confidence, is_raw_response)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id, component_name, component_type,
                        llm_summary.get('business_purpose', ''),
                        llm_summary.get('primary_function', ''),
                        llm_summary.get('complexity_score', 0.5),
                        json.dumps(llm_summary.get('key_features', [])),
                        json.dumps(llm_summary.get('integration_points', [])),
                        json.dumps(llm_summary.get('data_sources', [])),
                        llm_summary.get('business_domain', 'WEALTH_MANAGEMENT'),
                        llm_summary.get('friendly_name', ''),
                        llm_summary.get('analysis_confidence', 0.8),
                        False
                    ))
                
                logger.info(f"Stored LLM summary for {component_name}")
                
        except Exception as e:
            logger.error(f"Error storing LLM summary for {component_name}: {str(e)}")

    def get_llm_summary(self, session_id: str, component_name: str, component_type: str = None) -> Dict:
        """Get LLM summary from dedicated table"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if component_type:
                    cursor.execute('''
                        SELECT * FROM llm_component_summaries 
                        WHERE session_id = ? AND component_name = ? AND component_type = ?
                    ''', (session_id, component_name, component_type))
                else:
                    cursor.execute('''
                        SELECT * FROM llm_component_summaries 
                        WHERE session_id = ? AND component_name = ?
                        ORDER BY created_at DESC LIMIT 1
                    ''', (session_id, component_name))
                
                result = cursor.fetchone()
                
                if result:
                    summary = dict(result)
                    
                    # Parse JSON fields
                    if summary['key_features_json']:
                        summary['key_features'] = json.loads(summary['key_features_json'])
                    if summary['integration_points_json']:
                        summary['integration_points'] = json.loads(summary['integration_points_json'])
                    if summary['data_sources_json']:
                        summary['data_sources'] = json.loads(summary['data_sources_json'])
                    
                    return summary
                
                return {}
                
        except Exception as e:
            logger.error(f"Error getting LLM summary for {component_name}: {str(e)}")
            return {}

    def get_all_llm_summaries(self, session_id: str) -> List[Dict]:
        """Get all LLM summaries for a session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM llm_component_summaries 
                    WHERE session_id = ?
                    ORDER BY component_type, component_name
                ''', (session_id,))
                
                summaries = []
                for row in cursor.fetchall():
                    summary = dict(row)
                    
                    # Parse JSON fields
                    if summary['key_features_json']:
                        summary['key_features'] = json.loads(summary['key_features_json'])
                    if summary['integration_points_json']:
                        summary['integration_points'] = json.loads(summary['integration_points_json'])
                    if summary['data_sources_json']:
                        summary['data_sources'] = json.loads(summary['data_sources_json'])
                    
                    summaries.append(summary)
                
                return summaries
                
        except Exception as e:
            logger.error(f"Error getting all LLM summaries: {str(e)}")
            return []


    def store_field_details_with_lengths(self, session_id: str, field_data: Dict, 
                                   program_name: str, layout_id: int = None):
        """Enhanced field storage with proper length calculation"""
        field_name = field_data.get('name', 'UNNAMED_FIELD')
        
        try:
            # Ensure proper field lengths are calculated
            if not field_data.get('mainframe_length') or field_data.get('mainframe_length') == 0:
                # Calculate lengths if missing
                picture = field_data.get('picture', '')
                usage = field_data.get('usage', '')
                
                if picture:
                    # Use parser's calculation method
                    mainframe_length, oracle_length, oracle_type = self._calculate_field_lengths_from_pic(
                        picture, usage
                    )
                    field_data['mainframe_length'] = mainframe_length
                    field_data['oracle_length'] = oracle_length
                    field_data['oracle_data_type'] = oracle_type
            
            # Call the existing store method
            self.store_field_details(session_id, field_data, program_name, layout_id)
            
        except Exception as e:
            logger.error(f"Error storing field details with lengths for {field_name}: {str(e)}")
            raise

    def _calculate_field_lengths_from_pic(self, picture: str, usage: str = "") -> Tuple[int, int, str]:
        """Database version of field length calculation"""
        if not picture or picture.strip() == '':
            return 1, 50, "VARCHAR2(50)"
        
        pic_upper = picture.upper().strip()
        mainframe_length = 1
        oracle_length = 50
        oracle_type = "VARCHAR2(50)"
        
        try:
            # Numeric fields
            if re.search(r'[9S]', pic_upper):
                total_digits = 0
                decimal_digits = 0
                
                # Handle 9(n) patterns
                paren_matches = re.findall(r'[9S]\((\d+)\)', pic_upper)
                for match in paren_matches:
                    total_digits += int(match)
                
                # Handle explicit 9s
                remaining_pic = re.sub(r'[9S]\(\d+\)', '', pic_upper)
                explicit_nines = len(re.findall(r'9', remaining_pic))
                total_digits += explicit_nines
                
                # Handle V (decimal)
                if 'V' in pic_upper:
                    v_parts = pic_upper.split('V', 1)
                    if len(v_parts) > 1:
                        decimal_part = v_parts[1]
                        decimal_paren = re.findall(r'9\((\d+)\)', decimal_part)
                        for match in decimal_paren:
                            decimal_digits += int(match)
                        decimal_explicit = len(re.findall(r'9', re.sub(r'9\(\d+\)', '', decimal_part)))
                        decimal_digits += decimal_explicit
                
                total_digits = max(total_digits, 1)
                
                # Calculate storage
                if usage.upper() in ['COMP-3', 'PACKED-DECIMAL']:
                    mainframe_length = max((total_digits + 1) // 2 + 1, 1)
                elif usage.upper() in ['COMP', 'BINARY']:
                    if total_digits <= 4:
                        mainframe_length = 2
                    elif total_digits <= 9:
                        mainframe_length = 4
                    else:
                        mainframe_length = 8
                else:
                    mainframe_length = total_digits + (1 if 'S' in pic_upper else 0)
                    mainframe_length = max(mainframe_length, 1)
                
                if decimal_digits > 0:
                    oracle_type = f"NUMBER({total_digits},{decimal_digits})"
                    oracle_length = total_digits + 1
                else:
                    oracle_type = f"NUMBER({total_digits})"
                    oracle_length = total_digits
            
            # Alphanumeric fields
            elif re.search(r'[XA]', pic_upper):
                paren_matches = re.findall(r'[XA]\((\d+)\)', pic_upper)
                if paren_matches:
                    mainframe_length = sum(int(match) for match in paren_matches)
                else:
                    mainframe_length = max(len(re.findall(r'[XA]', pic_upper)), 1)
                
                oracle_length = mainframe_length
                if oracle_length <= 4000:
                    oracle_type = f"VARCHAR2({oracle_length})"
                else:
                    oracle_type = "CLOB"
            
            mainframe_length = max(mainframe_length, 1)
            oracle_length = max(oracle_length, mainframe_length)
            
        except Exception as e:
            logger.warning(f"Error calculating lengths for PIC {picture}: {str(e)}")
            mainframe_length = 1
            oracle_length = 50
            oracle_type = "VARCHAR2(50)"
        
        return mainframe_length, oracle_length, oracle_type

    def get_enhanced_field_mappings(self, session_id: str, target_file: str) -> List[Dict]:
        """Get field mappings with enhanced validation and lengths"""
        try:
            with self.get_connection() as conn:
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
                    AND fad.mainframe_length > 0
                    AND fad.field_name NOT IN ('VALUE', 'VALUES', 'PIC', 'PICTURE', 'USAGE', 'COMP')
                    AND LENGTH(fad.field_name) >= 3
                    AND (UPPER(rl.layout_name) LIKE UPPER(?) 
                        OR UPPER(fad.program_name) LIKE UPPER(?)
                        OR UPPER(fad.field_name) LIKE UPPER(?))
                    ORDER BY rl.layout_name, fad.field_name
                ''', (session_id, f'%{target_file}%', f'%{target_file}%', f'%{target_file}%'))
                                
                
                mappings = []
                for row in cursor.fetchall():
                    mapping = dict(row)
                    # Normalize business logic type for consistency
                    mapping['business_logic_type'] = self._normalize_usage_type(mapping.get('usage_type', 'STATIC'))
                    mappings.append(mapping)
                
                return mappings

        except Exception as e:
            logger.error(f"Error getting enhanced field mappings: {str(e)}")
            return []

    def _normalize_usage_type(self, usage_type: str) -> str:
        """Normalize usage types for consistency across UI"""
        if not usage_type:
            return 'STATIC'
        
        normalized_mappings = {
            'MOVE_TARGET': 'INPUT',
            'MOVE_SOURCE': 'OUTPUT',
            'COMPUTED': 'DERIVED',
            'DEFINITION': 'UNUSED',
            'CONDITION': 'REFERENCE',
            'CONDITIONAL': 'REFERENCE',
            'CICS_INPUT': 'INPUT',
            'CICS_OUTPUT': 'OUTPUT',
            'STRING_MANIPULATION': 'DERIVED',
            'CALCULATED': 'DERIVED'
        }
        
        return normalized_mappings.get(usage_type.upper(), usage_type.upper())
