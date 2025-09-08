"""
Consolidated Mainframe Analyzer System
Refactored for batch processing, improved LLM analysis, and multi-user support
"""

import sqlite3
import json
import uuid
import datetime
import time
import os
import re
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import requests
import backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Component:
    """Unified component model for all mainframe artifacts"""
    id: str
    name: str
    type: str  # COBOL, JCL, COPYBOOK, PROC, etc.
    source_content: str
    file_path: str
    uploaded_by: str
    uploaded_at: datetime.datetime
    friendly_name: str = ""
    business_purpose: str = ""
    complexity_score: float = 0.0
    total_lines: int = 0
    metadata: Dict = field(default_factory=dict)
    llm_analyzed: bool = False
    dependencies: List[str] = field(default_factory=list)

@dataclass
class Dependency:
    """Unified dependency model"""
    source_component: str
    target_component: str
    relationship_type: str  # CALLS, INCLUDES, USES_FILE, CICS_LINK, etc.
    interface_type: str    # COBOL, CICS, DB2, JCL, etc.
    confidence_score: float
    analysis_details: Dict = field(default_factory=dict)
    is_dynamic: bool = False
    resolved_targets: List[str] = field(default_factory=list)

@dataclass
class FieldMapping:
    """Enhanced field mapping with layout associations"""
    field_name: str
    friendly_name: str
    source_layout: str
    target_system: str
    mainframe_type: str
    target_type: str
    business_logic: str
    transformation_rules: List[str] = field(default_factory=list)
    confidence_score: float = 0.0

@dataclass
class LLMResponse:
    success: bool
    content: str
    prompt_tokens: int = 0
    response_tokens: int = 0
    processing_time_ms: int = 0
    error_message: str = ""

@dataclass
class TokenUsage:
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    
@dataclass
class ChunkInfo:
    content: str
    chunk_number: int
    total_chunks: int
    context_overlap: str = ""
    estimated_tokens: int = 0

# ============================================================================
# Core Database Manager - Consolidated
# ============================================================================

class DatabaseManager:
    """Consolidated database manager with multi-user support"""
    
    def __init__(self, db_path: str = "mainframe_analyzer.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ensure_database_directory()
        self.initialize_database()
    
    def _ensure_database_directory(self):
        """Ensure database directory exists"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Thread-safe database connection"""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            if conn:
                conn.close()
    
    def initialize_database(self):
        """Initialize consolidated database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Components table - unified for all component types
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS components (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    source_content TEXT NOT NULL,
                    file_path TEXT,
                    uploaded_by TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    friendly_name TEXT,
                    business_purpose TEXT,
                    complexity_score REAL DEFAULT 0.0,
                    total_lines INTEGER DEFAULT 0,
                    metadata_json TEXT,
                    llm_analyzed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Dependencies table - unified for all dependency types
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dependencies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_component TEXT NOT NULL,
                    target_component TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    interface_type TEXT,
                    confidence_score REAL DEFAULT 0.0,
                    analysis_details_json TEXT,
                    is_dynamic BOOLEAN DEFAULT FALSE,
                    resolved_targets_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_component) REFERENCES components(id),
                    UNIQUE(source_component, target_component, relationship_type)
                )
            ''')
            
            # Field mappings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS field_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    friendly_name TEXT,
                    source_layout TEXT,
                    target_system TEXT,
                    mainframe_type TEXT,
                    target_type TEXT,
                    business_logic TEXT,
                    transformation_rules_json TEXT,
                    confidence_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (component_id) REFERENCES components(id)
                )
            ''')
            
            # User sessions for tracking analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    project_name TEXT,
                    selected_components_json TEXT,
                    analysis_status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # LLM analysis results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component_id TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    analysis_result_json TEXT,
                    processing_time_ms INTEGER,
                    token_usage INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (component_id) REFERENCES components(id)
                )
            ''')
            
            # Chat conversations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    context_components_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
                )
            ''')
            
            # Create indexes for performance
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_components_type ON components(type)',
                'CREATE INDEX IF NOT EXISTS idx_components_name ON components(name)',
                'CREATE INDEX IF NOT EXISTS idx_dependencies_source ON dependencies(source_component)',
                'CREATE INDEX IF NOT EXISTS idx_dependencies_target ON dependencies(target_component)',
                'CREATE INDEX IF NOT EXISTS idx_field_mappings_component ON field_mappings(component_id)',
                'CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_conversations(session_id)'
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
    
    def store_component(self, component: Component) -> bool:
        """Store a component in the database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO components 
                    (id, name, type, source_content, file_path, uploaded_by, 
                     friendly_name, business_purpose, complexity_score, total_lines, 
                     metadata_json, llm_analyzed, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    component.id, component.name, component.type, component.source_content,
                    component.file_path, component.uploaded_by, component.friendly_name,
                    component.business_purpose, component.complexity_score, component.total_lines,
                    json.dumps(component.metadata), component.llm_analyzed
                ))
                return True
        except Exception as e:
            logger.error(f"Error storing component {component.name}: {e}")
            return False
    
    def store_dependency(self, dependency: Dependency) -> bool:
        """Store a dependency relationship"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO dependencies 
                    (source_component, target_component, relationship_type, interface_type,
                     confidence_score, analysis_details_json, is_dynamic, resolved_targets_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    dependency.source_component, dependency.target_component,
                    dependency.relationship_type, dependency.interface_type,
                    dependency.confidence_score, json.dumps(dependency.analysis_details),
                    dependency.is_dynamic, json.dumps(dependency.resolved_targets)
                ))
                return True
        except Exception as e:
            logger.error(f"Error storing dependency: {e}")
            return False
    
    def get_all_components(self, component_type: str = None) -> List[Component]:
        """Get all components, optionally filtered by type"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if component_type:
                    cursor.execute('SELECT * FROM components WHERE type = ? ORDER BY name', (component_type,))
                else:
                    cursor.execute('SELECT * FROM components ORDER BY type, name')
                
                components = []
                for row in cursor.fetchall():
                    metadata = json.loads(row['metadata_json'] or '{}')
                    component = Component(
                        id=row['id'],
                        name=row['name'],
                        type=row['type'],
                        source_content=row['source_content'],
                        file_path=row['file_path'] or '',
                        uploaded_by=row['uploaded_by'] or '',
                        uploaded_at=datetime.datetime.fromisoformat(row['uploaded_at']),
                        friendly_name=row['friendly_name'] or '',
                        business_purpose=row['business_purpose'] or '',
                        complexity_score=row['complexity_score'] or 0.0,
                        total_lines=row['total_lines'] or 0,
                        metadata=metadata,
                        llm_analyzed=bool(row['llm_analyzed'])
                    )
                    components.append(component)
                
                return components
        except Exception as e:
            logger.error(f"Error getting components: {e}")
            return []


class LLMClient:
    def __init__(self, endpoint: str = "http://localhost:8100/generate"):
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.max_retries = 3
        self.rate_limit_delay = 1.0
        self.last_call_time = 0
        self.timeout = 60
        self.default_max_tokens = 2000
        self.default_temperature = 0.1
    
    def update_config(self, config: Dict):
        """Update client configuration dynamically"""
        if 'endpoint' in config:
            self.endpoint = config['endpoint']
        if 'timeout' in config:
            self.timeout = config['timeout']
        if 'retries' in config:
            self.max_retries = config['retries']
        if 'maxTokens' in config:
            self.default_max_tokens = config['maxTokens']
        if 'temperature' in config:
            self.default_temperature = config['temperature']
        
        logger.info(f"Updated LLM client config: endpoint={self.endpoint}, timeout={self.timeout}")
    
    def _apply_rate_limit(self):
        """Apply rate limiting between LLM calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.Timeout),
        max_tries=3,
        max_time=60
    )
    def call_llm(self, prompt: str, max_tokens: int = None, temperature: float = None) -> LLMResponse:
        """Call VLLM endpoint with retry logic and error handling"""
        self._apply_rate_limit()
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if temperature is None:
            temperature = self.default_temperature
        
        logger.info(f"Making LLM call to {self.endpoint}")
        logger.info(f"Request params: max_tokens={max_tokens}, temperature={temperature}")
        logger.info(f"Prompt length: {len(prompt)} characters (~{len(prompt)//4} tokens)")
        
        start_time = time.time()
        
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": ["</analysis>", "END_OF_RESPONSE"],
                "stream": False
            }
            
            response = self.session.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            if response.status_code == 200:
                logger.info(f"LLM request successful in {processing_time}ms")
                result = response.json()
                
                # Extract response content
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0].get('text', '').strip()
                elif 'text' in result:
                    content = result['text'].strip()
                elif 'response' in result:
                    content = result['response'].strip()
                else:
                    content = str(result)
                
                # Extract token usage if available
                usage = result.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', len(prompt) // 4)
                completion_tokens = usage.get('completion_tokens', len(content) // 4)
                
                logger.info(f"Token usage: {prompt_tokens} prompt + {completion_tokens} response")
                
                return LLMResponse(
                    success=True,
                    content=content,
                    prompt_tokens=prompt_tokens,
                    response_tokens=completion_tokens,
                    processing_time_ms=processing_time
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"LLM request failed: {error_msg}")
                
                return LLMResponse(
                    success=False,
                    content="",
                    processing_time_ms=processing_time,
                    error_message=error_msg
                )
                
        except requests.exceptions.Timeout:
            error_msg = f"LLM request timeout after {self.timeout}s"
            logger.error(error_msg)
            return LLMResponse(
                success=False,
                content="",
                processing_time_ms=int((time.time() - start_time) * 1000),
                error_message=error_msg
            )
            
        except requests.exceptions.ConnectionError:
            error_msg = f"Cannot connect to LLM server at {self.endpoint}"
            logger.error(error_msg)
            return LLMResponse(
                success=False,
                content="",
                processing_time_ms=int((time.time() - start_time) * 1000),
                error_message=error_msg
            )
            
        except Exception as e:
            error_msg = f"Unexpected error calling LLM: {str(e)}"
            logger.error(error_msg)
            return LLMResponse(
                success=False,
                content="",
                processing_time_ms=int((time.time() - start_time) * 1000),
                error_message=error_msg
            )
    
    def extract_json_from_response(self, response_content: str) -> Optional[Dict[Any, Any]]:
        """Extract JSON from LLM response content with enhanced parsing"""
        if not response_content:
            return None
            
        try:
            return json.loads(response_content.strip())
        except json.JSONDecodeError:
            pass
        
        # Clean the response content
        cleaned_content = response_content.strip()
        
        # Look for JSON blocks in code fences
        json_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'<json>(.*?)</json>',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, cleaned_content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    clean_match = match.strip()
                    if clean_match:
                        return json.loads(clean_match)
                except json.JSONDecodeError:
                    continue
        
        # Try to extract JSON by finding balanced braces
        try:
            start_idx = cleaned_content.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                
                for i, char in enumerate(cleaned_content[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break
                
                if brace_count == 0:
                    json_str = cleaned_content[start_idx:end_idx + 1]
                    return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        
        logger.warning(f"Could not extract JSON from LLM response. Content preview: {cleaned_content[:200]}...")
        return cleaned_content
    
    def health_check(self) -> bool:
        """Check if LLM server is healthy"""
        try:
            test_prompt = "Hello, respond with 'OK' if you can process this request."
            response = self.call_llm(test_prompt, max_tokens=10)
            return response.success and "OK" in response.content.upper()
        except Exception as e:
            logger.error(f"LLM health check failed: {str(e)}")
            return False


class TokenManager:
    def __init__(self):
        self.MAX_TOKENS_PER_CALL = 6000
        self.EFFECTIVE_CONTENT_LIMIT = 5500
        self.CHUNK_OVERLAP_TOKENS = 200
        self.CHARACTERS_PER_TOKEN = 4
        self.token_usage_cache = {}
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length"""
        return len(text) // self.CHARACTERS_PER_TOKEN
    
    def needs_chunking(self, text: str) -> bool:
        """Check if text needs to be chunked"""
        estimated_tokens = self.estimate_tokens(text)
        return estimated_tokens > self.EFFECTIVE_CONTENT_LIMIT
    
    def chunk_cobol_code(self, code: str, preserve_structure: bool = True) -> List[ChunkInfo]:
        """Intelligently chunk COBOL code preserving structure"""
        if not self.needs_chunking(code):
            logger.info("Content fits in single chunk, no chunking needed")
            return [ChunkInfo(
                content=code,
                chunk_number=1,
                total_chunks=1,
                estimated_tokens=self.estimate_tokens(code)
            )]
        
        logger.info(f"Content too large ({self.estimate_tokens(code)} tokens), starting chunking...")
        
        chunks = []
        
        if preserve_structure:
            chunks = self._chunk_by_structure(code)
        else:
            chunks = self._chunk_by_size(code)
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Add overlap between chunks
        chunks = self._add_context_overlap(chunks)
        
        return chunks
    
    def _chunk_by_structure(self, code: str) -> List[ChunkInfo]:
        """Chunk COBOL code by structural elements"""
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        division_pattern = r'^\s*(IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION'
        section_pattern = r'^\s*\w+\s+SECTION\s*\.'
        paragraph_pattern = r'^\s*[A-Z0-9][A-Z0-9\-]*\s*\.'
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_tokens = self.estimate_tokens(line)
            
            if (current_size + line_tokens > self.EFFECTIVE_CONTENT_LIMIT and 
                current_chunk and 
                (re.match(division_pattern, line, re.IGNORECASE) or
                 re.match(section_pattern, line, re.IGNORECASE) or
                 re.match(paragraph_pattern, line, re.IGNORECASE))):
                
                chunk_content = '\n'.join(current_chunk)
                chunks.append(ChunkInfo(
                    content=chunk_content,
                    chunk_number=len(chunks) + 1,
                    total_chunks=0,
                    estimated_tokens=current_size
                ))
                
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += line_tokens
            i += 1
        
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(ChunkInfo(
                content=chunk_content,
                chunk_number=len(chunks) + 1,
                total_chunks=0,
                estimated_tokens=current_size
            ))
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_size(self, text: str) -> List[ChunkInfo]:
        """Simple size-based chunking"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_tokens = self.estimate_tokens(word + " ")
            
            if current_size + word_tokens > self.EFFECTIVE_CONTENT_LIMIT and current_chunk:
                chunk_content = ' '.join(current_chunk)
                chunks.append(ChunkInfo(
                    content=chunk_content,
                    chunk_number=len(chunks) + 1,
                    total_chunks=0,
                    estimated_tokens=current_size
                ))
                
                current_chunk = []
                current_size = 0
            
            current_chunk.append(word)
            current_size += word_tokens
        
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append(ChunkInfo(
                content=chunk_content,
                chunk_number=len(chunks) + 1,
                total_chunks=0,
                estimated_tokens=current_size
            ))
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _add_context_overlap(self, chunks: List[ChunkInfo]) -> List[ChunkInfo]:
        """Add context overlap between chunks"""
        if len(chunks) <= 1:
            return chunks
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            prev_words = prev_chunk.content.split()
            
            overlap_chars = self.CHUNK_OVERLAP_TOKENS * self.CHARACTERS_PER_TOKEN
            overlap_content = ""
            
            char_count = 0
            overlap_words = []
            for word in reversed(prev_words):
                if char_count + len(word) + 1 <= overlap_chars:
                    overlap_words.insert(0, word)
                    char_count += len(word) + 1
                else:
                    break
            
            if overlap_words:
                overlap_content = ' '.join(overlap_words)
                chunks[i].context_overlap = overlap_content
        
        return chunks
    
    def get_chunked_analysis_prompt(self, chunk: ChunkInfo, analysis_type: str) -> str:
        """Generate analysis prompt for a specific chunk"""
        base_prompt = self._get_base_prompt(analysis_type)
        
        chunk_info = ""
        if chunk.total_chunks > 1:
            chunk_info = f"\nCHUNK {chunk.chunk_number} of {chunk.total_chunks}"
            if chunk.context_overlap:
                chunk_info += f"\n\nCONTEXT FROM PREVIOUS CHUNK:\n{chunk.context_overlap}\n\n"
        
        return f"{base_prompt}{chunk_info}\n\nCODE TO ANALYZE:\n{chunk.content}"
    
    def _get_base_prompt(self, analysis_type: str) -> str:
        """Get base prompt for analysis type"""
        prompts = {
            'business_focused': """
Analyze this COBOL code and provide business-focused analysis in JSON format:
{
    "business_purpose": "Specific business function",
    "primary_function": "CUSTOMER_PROCESSING|ACCOUNT_MANAGEMENT|TRANSACTION_PROCESSING|etc",
    "business_domain": "Domain description",
    "key_business_rules": ["rule1", "rule2"],
    "data_flows": [{"source": "input", "target": "output", "transformation": "desc"}],
    "integration_points": ["system1", "system2"],
    "complexity_assessment": {
        "business_complexity": 0.8,
        "technical_complexity": 0.7,
        "maintenance_risk": "LOW|MEDIUM|HIGH"
    },
    "modernization_recommendations": ["rec1", "rec2"]
}
""",
            'component_extraction': """
Extract components from this code in JSON format:
{
    "components": [
        {
            "name": "component_name",
            "type": "PROGRAM|COPYBOOK|SECTION",
            "line_start": 1,
            "line_end": 100,
            "business_purpose": "description"
        }
    ]
}
"""
        }
        
        return prompts.get(analysis_type, "Analyze the provided code.")
    
    def track_token_usage(self, session_id: str, analysis_type: str, 
                         prompt_tokens: int, response_tokens: int) -> None:
        """Track token usage for session"""
        if session_id not in self.token_usage_cache:
            self.token_usage_cache[session_id] = {}
        
        if analysis_type not in self.token_usage_cache[session_id]:
            self.token_usage_cache[session_id][analysis_type] = {
                'total_prompt_tokens': 0,
                'total_response_tokens': 0,
                'call_count': 0
            }
        
        cache = self.token_usage_cache[session_id][analysis_type]
        cache['total_prompt_tokens'] += prompt_tokens
        cache['total_response_tokens'] += response_tokens
        cache['call_count'] += 1
    
    def get_session_token_usage(self, session_id: str) -> Dict:
        """Get token usage summary for session"""
        if session_id not in self.token_usage_cache:
            return {
                'total_prompt_tokens': 0,
                'total_response_tokens': 0,
                'total_tokens': 0,
                'analysis_breakdown': {}
            }
        
        session_cache = self.token_usage_cache[session_id]
        total_prompt = sum(data['total_prompt_tokens'] for data in session_cache.values())
        total_response = sum(data['total_response_tokens'] for data in session_cache.values())
        
        return {
            'total_prompt_tokens': total_prompt,
            'total_response_tokens': total_response,
            'total_tokens': total_prompt + total_response,
            'analysis_breakdown': session_cache
        }

# ============================================================================
# Consolidated Parser - All Component Types
# ============================================================================

class UnifiedParser:
    """Unified parser for all mainframe component types"""
    
    def __init__(self):
        self.component_patterns = {
            'COBOL': {
                'identification': r'IDENTIFICATION\s+DIVISION',
                'program_id': r'PROGRAM-ID\.\s+([A-Z0-9\-]+)',
                'procedure': r'PROCEDURE\s+DIVISION',
                'working_storage': r'WORKING-STORAGE\s+SECTION'
            },
            'JCL': {
                'job_card': r'^//([A-Z0-9]+)\s+JOB',
                'exec_step': r'^//([A-Z0-9]+)\s+EXEC',
                'dd_statement': r'^//([A-Z0-9]+)\s+DD'
            },
            'COPYBOOK': {
                'level_items': r'^\s*(\d{2})\s+([A-Z0-9\-]+)',
                'pic_clause': r'PIC(?:TURE)?\s+([X9SVP\(\),\.\+\-\*\$Z]+)'
            }
        }
    
    def parse_component(self, content: str, component_type: str, file_name: str) -> Dict:
        """Parse any component type and extract relevant information"""
        if component_type.upper() == 'COBOL':
            return self._parse_cobol(content, file_name)
        elif component_type.upper() == 'JCL':
            return self._parse_jcl(content, file_name)
        elif component_type.upper() == 'COPYBOOK':
            return self._parse_copybook(content, file_name)
        else:
            return self._parse_generic(content, file_name, component_type)
    
    def _parse_cobol(self, content: str, file_name: str) -> Dict:
        """Parse COBOL program with enhanced analysis"""
        lines = content.split('\n')
        analysis = {
            'type': 'COBOL',
            'name': self._extract_program_name(content, file_name),
            'total_lines': len(lines),
            'divisions': [],
            'copybooks': [],
            'file_operations': [],
            'program_calls': [],
            'cics_operations': [],
            'db2_operations': [],
            'record_layouts': [],
            'business_comments': [],
            'complexity_indicators': {}
        }
        
        # Extract business comments first (these are valuable for LLM)
        analysis['business_comments'] = self._extract_business_comments(lines)
        
        # Extract main structural elements
        analysis['divisions'] = self._extract_divisions(lines)
        analysis['copybooks'] = self._extract_copybooks(lines)
        analysis['file_operations'] = self._extract_file_operations(lines)
        analysis['program_calls'] = self._extract_program_calls(lines)
        analysis['cics_operations'] = self._extract_cics_operations(lines)
        analysis['db2_operations'] = self._extract_db2_operations(lines)
        analysis['record_layouts'] = self._extract_record_layouts(lines)
        
        # Calculate complexity
        analysis['complexity_indicators'] = self._calculate_complexity(analysis)
        
        return analysis
    
    def _parse_jcl(self, content: str, file_name: str) -> Dict:
        """Parse JCL with step and dataset analysis"""
        lines = content.split('\n')
        analysis = {
            'type': 'JCL',
            'name': self._extract_job_name(content, file_name),
            'total_lines': len(lines),
            'job_steps': [],
            'datasets': [],
            'programs_called': [],
            'procedures': [],
            'includes': []
        }
        
        current_step = None
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//*'):
                continue
            
            # Job card
            if ' JOB ' in line and line.startswith('//'):
                job_match = re.match(r'//([A-Z0-9]+)\s+JOB', line)
                if job_match:
                    analysis['name'] = job_match.group(1)
            
            # Step cards
            elif ' EXEC ' in line and line.startswith('//'):
                if current_step:
                    analysis['job_steps'].append(current_step)
                
                step_match = re.match(r'//([A-Z0-9]+)\s+EXEC', line)
                if step_match:
                    current_step = {
                        'step_name': step_match.group(1),
                        'program': self._extract_program_from_exec(line),
                        'datasets': []
                    }
            
            # DD statements
            elif ' DD ' in line and line.startswith('//') and current_step:
                dataset_info = self._extract_dataset_info(line)
                if dataset_info:
                    current_step['datasets'].append(dataset_info)
                    analysis['datasets'].append(dataset_info)
        
        if current_step:
            analysis['job_steps'].append(current_step)
        
        return analysis
    
    def _parse_copybook(self, content: str, file_name: str) -> Dict:
        """Parse copybook with field structure analysis"""
        lines = content.split('\n')
        analysis = {
            'type': 'COPYBOOK',
            'name': file_name.split('.')[0] if '.' in file_name else file_name,
            'total_lines': len(lines),
            'record_structures': [],
            'field_definitions': [],
            'constants': [],
            'redefines': []
        }
        
        # Extract field structures
        current_record = None
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            
            # Level number pattern
            level_match = re.match(r'^\s*(\d{2})\s+([A-Z0-9\-]+)(?:\s+(.*))?', line)
            if level_match:
                level = int(level_match.group(1))
                field_name = level_match.group(2)
                definition = level_match.group(3) or ''
                
                field_info = {
                    'level': level,
                    'name': field_name,
                    'line_number': line_num,
                    'definition': definition,
                    'picture': self._extract_picture(definition),
                    'usage': self._extract_usage(definition),
                    'value': self._extract_value(definition),
                    'occurs': self._extract_occurs(definition),
                    'redefines': self._extract_redefines(definition)
                }
                
                if level == 1:
                    if current_record:
                        analysis['record_structures'].append(current_record)
                    current_record = {
                        'name': field_name,
                        'line_start': line_num,
                        'fields': []
                    }
                elif current_record:
                    current_record['fields'].append(field_info)
                
                analysis['field_definitions'].append(field_info)
        
        if current_record:
            analysis['record_structures'].append(current_record)
        
        return analysis
    
    def _extract_business_comments(self, lines: List[str]) -> List[str]:
        """Extract business-relevant comments from source"""
        business_comments = []
        
        for line in lines:
            if len(line) > 6 and line[6] in ['*', '/', 'C', 'c']:
                comment = line[7:].strip()
                # Filter for business-relevant comments
                if (len(comment) > 20 and 
                    any(keyword in comment.upper() for keyword in 
                        ['PURPOSE', 'FUNCTION', 'PROCESS', 'BUSINESS', 'CUSTOMER', 
                         'ACCOUNT', 'TRANSACTION', 'CALCULATE', 'VALIDATE'])):
                    business_comments.append(comment)
        
        return business_comments[:20]  # Limit for token management
    
    def _extract_program_calls(self, lines: List[str]) -> List[Dict]:
        """Extract both static and dynamic program calls"""
        calls = []
        
        for line_num, line in enumerate(lines, 1):
            line_upper = line.upper().strip()
            
            # Static calls
            static_match = re.search(r'CALL\s+[\'"]([A-Z0-9\-]+)[\'"]', line_upper)
            if static_match:
                calls.append({
                    'type': 'static',
                    'program': static_match.group(1),
                    'line_number': line_num,
                    'method': 'CALL'
                })
            
            # Dynamic calls
            dynamic_match = re.search(r'CALL\s+([A-Z0-9\-]+)', line_upper)
            if dynamic_match and not static_match:
                variable_name = dynamic_match.group(1)
                # Try to resolve the variable
                resolved_values = self._resolve_dynamic_variable(variable_name, lines)
                calls.append({
                    'type': 'dynamic',
                    'variable': variable_name,
                    'resolved_programs': resolved_values,
                    'line_number': line_num,
                    'method': 'CALL'
                })
            
            # CICS calls
            cics_match = re.search(r'EXEC\s+CICS\s+(LINK|XCTL)\s+PROGRAM\s*\(\s*([A-Z0-9\-\'\"]+)\s*\)', line_upper)
            if cics_match:
                method = cics_match.group(1)
                program = cics_match.group(2).strip('\'"')
                
                if program.startswith(("'", '"')):
                    call_type = 'static'
                    actual_program = program.strip('\'"')
                else:
                    call_type = 'dynamic'
                    resolved_values = self._resolve_dynamic_variable(program, lines)
                    actual_program = resolved_values
                
                calls.append({
                    'type': call_type,
                    'program': actual_program,
                    'line_number': line_num,
                    'method': f'CICS_{method}'
                })
        
        return calls
    
    def _resolve_dynamic_variable(self, variable_name: str, lines: List[str]) -> List[str]:
        """Resolve dynamic variable to possible program names"""
        resolved_values = []
        
        # Look for MOVE statements and group structures
        for line in lines:
            line_upper = line.upper().strip()
            
            # Direct MOVE to variable
            move_match = re.search(rf'MOVE\s+[\'"]([A-Z0-9\-]+)[\'"]\s+TO\s+{re.escape(variable_name)}', line_upper)
            if move_match:
                resolved_values.append(move_match.group(1))
            
            # Group structure with prefix
            group_match = re.search(rf'MOVE\s+[\'"]([A-Z0-9\-]+)[\'"]\s+TO\s+([A-Z0-9\-]*{re.escape(variable_name)}[A-Z0-9\-]*)', line_upper)
            if group_match:
                prefix = group_match.group(1)
                # Look for suffix patterns
                suffix_patterns = self._find_suffix_patterns(variable_name, lines)
                for suffix in suffix_patterns:
                    resolved_values.append(f"{prefix}{suffix}")
        
        return list(set(resolved_values))  # Remove duplicates
    
    def _find_suffix_patterns(self, variable_name: str, lines: List[str]) -> List[str]:
        """Find suffix patterns for dynamic program construction"""
        suffixes = []
        
        # Look for related field moves in group structures
        for line in lines:
            line_upper = line.upper().strip()
            
            # Look for patterns like MOVE 'P4' TO HOLD-TRANX
            if f'HOLD-{variable_name}' in line_upper or f'{variable_name}-' in line_upper:
                suffix_match = re.search(r'MOVE\s+[\'"]([A-Z0-9\-]+)[\'"]', line_upper)
                if suffix_match:
                    suffixes.append(suffix_match.group(1))
        
        return suffixes

# ============================================================================
# Batch Component Processor
# ============================================================================

class BatchComponentProcessor:
    """Handles batch upload and processing of components"""
    
    def __init__(self, db_manager: DatabaseManager, parser: UnifiedParser):
        self.db_manager = db_manager
        self.parser = parser
        self.processing_status = {}
    
    def process_batch_upload(self, files: List[Dict], user_id: str) -> Dict:
        """Process multiple files in batch"""
        batch_id = str(uuid.uuid4())
        self.processing_status[batch_id] = {
            'total_files': len(files),
            'processed': 0,
            'failed': 0,
            'components': [],
            'errors': []
        }
        
        for file_data in files:
            try:
                component = self._process_single_file(file_data, user_id)
                if component:
                    self.db_manager.store_component(component)
                    self.processing_status[batch_id]['components'].append(component.id)
                    self.processing_status[batch_id]['processed'] += 1
                else:
                    self.processing_status[batch_id]['failed'] += 1
            except Exception as e:
                self.processing_status[batch_id]['failed'] += 1
                self.processing_status[batch_id]['errors'].append(str(e))
                logger.error(f"Error processing file {file_data.get('name', 'unknown')}: {e}")
        
        # Extract dependencies after all components are loaded
        self._extract_batch_dependencies(self.processing_status[batch_id]['components'])
        
        return {
            'batch_id': batch_id,
            'status': self.processing_status[batch_id]
        }
    
    def _process_single_file(self, file_data: Dict, user_id: str) -> Optional[Component]:
        """Process a single file into a Component"""
        try:
            content = file_data['content']
            file_name = file_data['name']
            component_type = self._detect_component_type(content, file_name)
            
            # Parse the component
            parsed_data = self.parser.parse_component(content, component_type, file_name)
            
            # Create component
            component = Component(
                id=str(uuid.uuid4()),
                name=parsed_data['name'],
                type=component_type,
                source_content=content,
                file_path=file_name,
                uploaded_by=user_id,
                uploaded_at=datetime.datetime.now(),
                total_lines=parsed_data['total_lines'],
                metadata=parsed_data
            )
            
            return component
            
        except Exception as e:
            logger.error(f"Error processing file {file_data.get('name', 'unknown')}: {e}")
            return None
    
    def _detect_component_type(self, content: str, file_name: str) -> str:
        """Detect component type from content and filename"""
        # Check file extension first
        ext_map = {
            '.cbl': 'COBOL', '.cob': 'COBOL', '.cobol': 'COBOL',
            '.jcl': 'JCL', '.job': 'JCL',
            '.cpy': 'COPYBOOK', '.copy': 'COPYBOOK',
            '.prc': 'PROC', '.proc': 'PROC'
        }
        
        for ext, comp_type in ext_map.items():
            if file_name.lower().endswith(ext):
                return comp_type
        
        # Check content patterns
        content_upper = content.upper()
        
        if 'IDENTIFICATION DIVISION' in content_upper:
            return 'COBOL'
        elif content.strip().startswith('//') and ' JOB ' in content_upper:
            return 'JCL'
        elif re.search(r'^\s*\d{2}\s+[A-Z0-9\-]+', content, re.MULTILINE):
            return 'COPYBOOK'
        
        return 'UNKNOWN'
    
    def _extract_batch_dependencies(self, component_ids: List[str]):
        """Extract dependencies for all components in batch"""
        components = {}
        
        # Load all components
        for comp_id in component_ids:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM components WHERE id = ?', (comp_id,))
                row = cursor.fetchone()
                if row:
                    components[comp_id] = {
                        'id': row['id'],
                        'name': row['name'],
                        'type': row['type'],
                        'metadata': json.loads(row['metadata_json'] or '{}')
                    }
        
        # Extract dependencies
        for comp_id, comp_data in components.items():
            dependencies = self._extract_component_dependencies(comp_data, components)
            for dep in dependencies:
                self.db_manager.store_dependency(dep)

    def _extract_component_dependencies(self, component: Dict, all_components: Dict) -> List[Dependency]:
        """Extract dependencies for a single component"""
        dependencies = []
        metadata = component['metadata']
        
        # Program calls
        for call in metadata.get('program_calls', []):
            if call['type'] == 'static':
                # Look for target in uploaded components
                target_comp = self._find_component_by_name(call['program'], all_components)
                dependencies.append(Dependency(
                    source_component=component['id'],
                    target_component=target_comp['id'] if target_comp else call['program'],
                    relationship_type='CALLS',
                    interface_type='COBOL',
                    confidence_score=0.95,
                    analysis_details={'method': call['method'], 'line_number': call['line_number']},
                    is_dynamic=False
                ))
            else:  # dynamic
                # Handle dynamic calls with resolved targets
                resolved_targets = call.get('resolved_programs', [])
                for target in resolved_targets:
                    target_comp = self._find_component_by_name(target, all_components)
                    dependencies.append(Dependency(
                        source_component=component['id'],
                        target_component=target_comp['id'] if target_comp else target,
                        relationship_type='CALLS_DYNAMIC',
                        interface_type='COBOL',
                        confidence_score=0.7,
                        analysis_details={
                            'method': call['method'],
                            'variable': call['variable'],
                            'line_number': call['line_number']
                        },
                        is_dynamic=True,
                        resolved_targets=resolved_targets
                    ))
        
        # Copybook includes
        for copybook in metadata.get('copybooks', []):
            target_comp = self._find_component_by_name(copybook, all_components)
            dependencies.append(Dependency(
                source_component=component['id'],
                target_component=target_comp['id'] if target_comp else copybook,
                relationship_type='INCLUDES',
                interface_type='COPYBOOK',
                confidence_score=0.98
            ))
        
        # File operations
        for file_op in metadata.get('file_operations', []):
            dependencies.append(Dependency(
                source_component=component['id'],
                target_component=file_op['file_name'],
                relationship_type='USES_FILE',
                interface_type='FILE_SYSTEM',
                confidence_score=0.9,
                analysis_details={
                    'operation': file_op.get('operation', 'UNKNOWN'),
                    'io_direction': file_op.get('io_direction', 'UNKNOWN')
                }
            ))
        
        # CICS operations
        for cics_op in metadata.get('cics_operations', []):
            dependencies.append(Dependency(
                source_component=component['id'],
                target_component=cics_op['file_name'],
                relationship_type='CICS_FILE',
                interface_type='CICS',
                confidence_score=0.9,
                analysis_details={
                    'operation': cics_op.get('operation', 'UNKNOWN'),
                    'layout': cics_op.get('layout_name', '')
                }
            ))
        
        # DB2 operations  
        for db2_op in metadata.get('db2_operations', []):
            for table in db2_op.get('tables', []):
                dependencies.append(Dependency(
                    source_component=component['id'],
                    target_component=table,
                    relationship_type='DB2_TABLE',
                    interface_type='DB2',
                    confidence_score=0.95,
                    analysis_details={
                        'operation': db2_op.get('operation_type', 'UNKNOWN'),
                        'sql_snippet': db2_op.get('sql', '')[:100]
                    }
                ))
        
        # JCL program calls
        if component['type'] == 'JCL':
            for step in metadata.get('job_steps', []):
                if step.get('program'):
                    target_comp = self._find_component_by_name(step['program'], all_components)
                    dependencies.append(Dependency(
                        source_component=component['id'],
                        target_component=target_comp['id'] if target_comp else step['program'],
                        relationship_type='JCL_EXECUTES',
                        interface_type='JCL',
                        confidence_score=0.98,
                        analysis_details={'step_name': step['step_name']}
                    ))
        
        return dependencies
    
    def _find_component_by_name(self, name: str, components: Dict) -> Optional[Dict]:
        """Find component by name in the loaded components"""
        for comp_id, comp_data in components.items():
            if comp_data['name'].upper() == name.upper():
                return comp_data
        return None

# ============================================================================
# Enhanced LLM Analysis Engine
# ============================================================================

class LLMAnalysisEngine:
    """Enhanced LLM analysis focusing on business logic and comments"""
    
    def __init__(self, llm_client, db_manager: DatabaseManager):
        self.llm_client = llm_client
        self.db_manager = db_manager
    
    def analyze_component_batch(self, component_ids: List[str], session_id: str) -> Dict:
        """Analyze multiple components focusing on business logic"""
        results = {}
        
        for comp_id in component_ids:
            try:
                component = self._get_component_by_id(comp_id)
                if component:
                    analysis = self._analyze_single_component(component, session_id)
                    results[comp_id] = analysis
                    
                    # Update component with analysis
                    self._update_component_analysis(comp_id, analysis)
                    
            except Exception as e:
                logger.error(f"Error analyzing component {comp_id}: {e}")
                results[comp_id] = {'error': str(e)}
        
        return results
    
    def _analyze_single_component(self, component: Component, session_id: str) -> Dict:
        """Analyze single component with enhanced business focus"""
        metadata = component.metadata
        
        # Prepare focused content for LLM
        analysis_content = self._prepare_analysis_content(component)
        
        if component.type == 'COBOL':
            return self._analyze_cobol_business_logic(component, analysis_content, session_id)
        elif component.type == 'JCL':
            return self._analyze_jcl_workflow(component, analysis_content, session_id)
        elif component.type == 'COPYBOOK':
            return self._analyze_copybook_structure(component, analysis_content, session_id)
        else:
            return self._analyze_generic_component(component, analysis_content, session_id)
    
    def _prepare_analysis_content(self, component: Component) -> str:
        """Prepare focused content for LLM analysis"""
        metadata = component.metadata
        content_parts = []
        
        # Add business comments first (most valuable)
        business_comments = metadata.get('business_comments', [])
        if business_comments:
            content_parts.append("=== BUSINESS COMMENTS ===")
            content_parts.extend(business_comments)
            content_parts.append("")
        
        if component.type == 'COBOL':
            # For COBOL, focus on procedure division, not working storage
            lines = component.source_content.split('\n')
            
            # Find procedure division
            proc_start = -1
            for i, line in enumerate(lines):
                if 'PROCEDURE DIVISION' in line.upper():
                    proc_start = i
                    break
            
            if proc_start > -1:
                content_parts.append("=== PROCEDURE DIVISION ===")
                # Take meaningful portion of procedure division
                proc_lines = lines[proc_start:proc_start + 100]  # First 100 lines
                content_parts.extend(proc_lines)
            else:
                # Fallback to first 100 lines
                content_parts.append("=== SOURCE CODE EXCERPT ===")
                content_parts.extend(lines[:100])
        
        elif component.type == 'JCL':
            # For JCL, include step analysis
            content_parts.append("=== JCL STEPS ===")
            for step in metadata.get('job_steps', []):
                content_parts.append(f"Step: {step['step_name']}")
                content_parts.append(f"Program: {step.get('program', 'N/A')}")
                content_parts.append("")
            
            # Include actual JCL source
            content_parts.append("=== JCL SOURCE ===")
            content_parts.extend(component.source_content.split('\n')[:50])
        
        elif component.type == 'COPYBOOK':
            # For copybooks, focus on record structures
            content_parts.append("=== RECORD STRUCTURES ===")
            for record in metadata.get('record_structures', []):
                content_parts.append(f"Record: {record['name']}")
                content_parts.append(f"Fields: {len(record.get('fields', []))}")
                content_parts.append("")
        
        return '\n'.join(content_parts)
    
    def _analyze_cobol_business_logic(self, component: Component, content: str, session_id: str) -> Dict:
        """Analyze COBOL program focusing on business logic"""
        metadata = component.metadata
        
        prompt = f"""
Analyze this COBOL program focusing on its business purpose and logic.

Program: {component.name}
Lines of Code: {component.total_lines}

Key Metrics:
- Program Calls: {len(metadata.get('program_calls', []))}
- File Operations: {len(metadata.get('file_operations', []))}
- CICS Operations: {len(metadata.get('cics_operations', []))}
- DB2 Operations: {len(metadata.get('db2_operations', []))}
- Record Layouts: {len(metadata.get('record_layouts', []))}

{content}

Based on the business comments and procedure division logic, provide a JSON analysis:
{{
    "business_purpose": "Specific business function this program performs",
    "primary_function": "CUSTOMER_PROCESSING|ACCOUNT_MANAGEMENT|TRANSACTION_PROCESSING|PORTFOLIO_MANAGEMENT|REPORTING|BATCH_PROCESSING|ONLINE_TRANSACTION|DATA_CONVERSION|VALIDATION|FILE_MAINTENANCE",
    "business_domain": "Specific domain (Customer, Account, Transaction, etc.)",
    "key_business_rules": ["rule1", "rule2", "rule3"],
    "data_flows": [
        {{"source": "input_source", "target": "output_target", "transformation": "description"}}
    ],
    "integration_points": ["system1", "system2"],
    "complexity_assessment": {{
        "business_complexity": 0.8,
        "technical_complexity": 0.7,
        "maintenance_risk": "LOW|MEDIUM|HIGH"
    }},
    "modernization_recommendations": ["recommendation1", "recommendation2"]
}}

Focus on BUSINESS VALUE and PURPOSE, not technical implementation details.
"""
        
        try:
            response = self.llm_client.call_llm(prompt, max_tokens=1500, temperature=0.2)
            if response.success:
                    analysis = self.llm_client.extract_json_from_response(response.content)
                    # Track token usage
                    self.token_manager.track_token_usage(
                        session_id, 'cobol_business',
                        response.prompt_tokens, response.response_tokens
                    )
            else:
                return {'error': response.error_message}
                
        except Exception as e:
            logger.error(f"Error in COBOL analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_jcl_workflow(self, component: Component, content: str, session_id: str) -> Dict:
        """Analyze JCL workflow and dependencies"""
        metadata = component.metadata
        
        prompt = f"""
Analyze this JCL job focusing on its business workflow and data processing.

Job: {component.name}
Steps: {len(metadata.get('job_steps', []))}
Programs Called: {len(metadata.get('programs_called', []))}

{content}

Provide JSON analysis:
{{
    "business_purpose": "What business process this job supports",
    "workflow_type": "BATCH_PROCESSING|DATA_LOAD|REPORT_GENERATION|FILE_TRANSFER|BACKUP|VALIDATION",
    "processing_sequence": ["step1_purpose", "step2_purpose"],
    "data_dependencies": ["input_data1", "input_data2"],
    "output_deliverables": ["output1", "output2"],
    "scheduling_requirements": "Daily|Weekly|Monthly|On-Demand",
    "business_impact": "What happens if this job fails",
    "modernization_path": "Cloud migration recommendations"
}}
"""
        
        try:
            response = self.llm_client.call_llm(prompt, max_tokens=1000, temperature=0.2)
            
            if response.success:
                analysis = self._extract_json_from_response(response.content)
                analysis['llm_analysis_type'] = 'workflow_focused'
                return analysis
            else:
                return {'error': response.error_message}
                
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_copybook_structure(self, component: Component, content: str, session_id: str) -> Dict:
        """Analyze copybook data structures"""
        metadata = component.metadata
        
        prompt = f"""
Analyze this copybook focusing on its data structure and business meaning.

Copybook: {component.name}
Record Structures: {len(metadata.get('record_structures', []))}
Total Fields: {len(metadata.get('field_definitions', []))}

{content}

Provide JSON analysis:
{{
    "business_purpose": "What business data this copybook represents",
    "data_domain": "CUSTOMER|ACCOUNT|TRANSACTION|PRODUCT|REFERENCE|CONTROL",
    "record_types": [
        {{"name": "record_name", "purpose": "business_purpose", "field_count": 10}}
    ],
    "key_data_elements": ["element1", "element2"],
    "data_relationships": "How this relates to other data structures",
    "usage_patterns": "How this copybook is typically used",
    "modernization_mapping": "Target modern data structure recommendations"
}}
"""
        
        try:
            response = self.llm_client.call_llm(prompt, max_tokens=800, temperature=0.2)
            
            if response.success:
                analysis = self._extract_json_from_response(response.content)
                analysis['llm_analysis_type'] = 'data_structure_focused'
                return analysis
            else:
                return {'error': response.error_message}
                
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_json_from_response(self, content: str) -> Dict:
        """Extract JSON from LLM response"""
        try:
            # Find JSON in response
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx+1]
                return json.loads(json_str)
            else:
                # Fallback to raw content
                return {
                    'business_purpose': content[:500],
                    'analysis_type': 'raw_response',
                    'raw_content': content
                }
        except json.JSONDecodeError:
            return {
                'business_purpose': content[:500],
                'analysis_type': 'parse_failed',
                'raw_content': content
            }
    
    def _get_component_by_id(self, comp_id: str) -> Optional[Component]:
        """Get component by ID from database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM components WHERE id = ?', (comp_id,))
                row = cursor.fetchone()
                
                if row:
                    metadata = json.loads(row['metadata_json'] or '{}')
                    return Component(
                        id=row['id'],
                        name=row['name'],
                        type=row['type'],
                        source_content=row['source_content'],
                        file_path=row['file_path'] or '',
                        uploaded_by=row['uploaded_by'] or '',
                        uploaded_at=datetime.datetime.fromisoformat(row['uploaded_at']),
                        friendly_name=row['friendly_name'] or '',
                        business_purpose=row['business_purpose'] or '',
                        complexity_score=row['complexity_score'] or 0.0,
                        total_lines=row['total_lines'] or 0,
                        metadata=metadata,
                        llm_analyzed=bool(row['llm_analyzed'])
                    )
                return None
        except Exception as e:
            logger.error(f"Error getting component {comp_id}: {e}")
            return None
    
    def _update_component_analysis(self, comp_id: str, analysis: Dict):
        """Update component with LLM analysis results"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Store analysis in separate table
                cursor.execute('''
                    INSERT INTO llm_analysis 
                    (component_id, analysis_type, analysis_result_json, processing_time_ms, token_usage)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    comp_id,
                    analysis.get('llm_analysis_type', 'general'),
                    json.dumps(analysis),
                    analysis.get('processing_time_ms', 0),
                    analysis.get('tokens_used', 0)
                ))
                
                # Update component with key analysis results
                cursor.execute('''
                    UPDATE components 
                    SET business_purpose = ?, complexity_score = ?, llm_analyzed = TRUE, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (
                    analysis.get('business_purpose', '')[:1000],  # Limit length
                    analysis.get('complexity_assessment', {}).get('business_complexity', 0.5),
                    comp_id
                ))
                
        except Exception as e:
            logger.error(f"Error updating component analysis: {e}")

# ============================================================================
# Agentic RAG Chat System
# ============================================================================

class AgenticChatSystem:
    """Enhanced chat system with RAG capabilities"""
    
    def __init__(self, db_manager: DatabaseManager, llm_client):
        self.db_manager = db_manager
        self.llm_client = llm_client
        self.context_window_size = 8000  # tokens
    
    def process_chat_query(self, session_id: str, query: str, user_id: str = None) -> Dict:
        """Process chat query with RAG-enhanced context"""
        try:
            # Analyze query to identify relevant components
            relevant_components = self._identify_relevant_components(query)
            
            # Build context from relevant components
            context = self._build_chat_context(relevant_components, query)
            
            # Generate response
            response = self._generate_chat_response(query, context, session_id)
            
            # Store conversation
            self._store_chat_message(session_id, query, response, relevant_components)
            
            return {
                'success': True,
                'response': response,
                'components_used': [comp['name'] for comp in relevant_components],
                'context_tokens': len(context.split()) * 1.3  # Rough token estimate
            }
            
        except Exception as e:
            logger.error(f"Error processing chat query: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': 'I encountered an error processing your query. Please try again.'
            }
    
    def _identify_relevant_components(self, query: str) -> List[Dict]:
        """Identify components relevant to the query"""
        relevant_components = []
        query_upper = query.upper()
        
        # Extract potential component names from query
        potential_names = re.findall(r'\b[A-Z][A-Z0-9\-]{2,}\b', query_upper)
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Search by explicit names
                for name in potential_names:
                    cursor.execute('''
                        SELECT id, name, type, business_purpose, llm_analyzed 
                        FROM components 
                        WHERE UPPER(name) = ? OR UPPER(friendly_name) LIKE ?
                    ''', (name, f'%{name}%'))
                    
                    for row in cursor.fetchall():
                        relevant_components.append(dict(row))
                
                # If no explicit matches, search by content keywords
                if not relevant_components:
                    keywords = ['CUSTOMER', 'ACCOUNT', 'TRANSACTION', 'PROCESS', 'FILE', 'PROGRAM']
                    query_keywords = [kw for kw in keywords if kw in query_upper]
                    
                    if query_keywords:
                        keyword_condition = ' OR '.join(['UPPER(business_purpose) LIKE ?' for _ in query_keywords])
                        keyword_params = [f'%{kw}%' for kw in query_keywords]
                        
                        cursor.execute(f'''
                            SELECT id, name, type, business_purpose, llm_analyzed 
                            FROM components 
                            WHERE llm_analyzed = TRUE AND ({keyword_condition})
                            ORDER BY complexity_score DESC
                            LIMIT 5
                        ''', keyword_params)
                        
                        for row in cursor.fetchall():
                            relevant_components.append(dict(row))
                
                # Fallback to most recently analyzed components
                if not relevant_components:
                    cursor.execute('''
                        SELECT id, name, type, business_purpose, llm_analyzed 
                        FROM components 
                        WHERE llm_analyzed = TRUE 
                        ORDER BY updated_at DESC 
                        LIMIT 3
                    ''')
                    
                    for row in cursor.fetchall():
                        relevant_components.append(dict(row))
        
        except Exception as e:
            logger.error(f"Error identifying relevant components: {e}")
        
        return relevant_components[:5]  # Limit to top 5
    
    def _build_chat_context(self, components: List[Dict], query: str) -> str:
        """Build context for chat from relevant components"""
        context_parts = []
        
        context_parts.append("=== MAINFRAME CODEBASE CONTEXT ===")
        context_parts.append(f"Available Components: {len(components)}")
        context_parts.append("")
        
        for comp in components:
            context_parts.append(f"Component: {comp['name']} ({comp['type']})")
            
            if comp['business_purpose']:
                context_parts.append(f"Purpose: {comp['business_purpose']}")
            
            # Get additional context from analysis
            if comp['llm_analyzed']:
                analysis_context = self._get_component_analysis_context(comp['id'])
                if analysis_context:
                    context_parts.append(f"Analysis: {analysis_context}")
            
            # Get dependencies
            dependencies = self._get_component_dependencies_summary(comp['id'])
            if dependencies:
                context_parts.append(f"Dependencies: {dependencies}")
            
            context_parts.append("")
        
        # Add dependency overview
        context_parts.append("=== DEPENDENCY OVERVIEW ===")
        dependency_summary = self._get_overall_dependency_summary()
        context_parts.append(dependency_summary)
        
        return '\n'.join(context_parts)
    
    def _get_component_analysis_context(self, comp_id: str) -> str:
        """Get analysis context for a component"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT analysis_result_json 
                    FROM llm_analysis 
                    WHERE component_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT 1
                ''', (comp_id,))
                
                row = cursor.fetchone()
                if row:
                    analysis = json.loads(row['analysis_result_json'])
                    
                    # Extract key points for context
                    key_points = []
                    if 'primary_function' in analysis:
                        key_points.append(f"Function: {analysis['primary_function']}")
                    if 'key_business_rules' in analysis:
                        rules = analysis['key_business_rules'][:3]  # Top 3
                        key_points.append(f"Rules: {', '.join(rules)}")
                    if 'integration_points' in analysis:
                        integrations = analysis['integration_points'][:3]
                        key_points.append(f"Integrations: {', '.join(integrations)}")
                    
                    return '; '.join(key_points)
        except Exception as e:
            logger.error(f"Error getting analysis context: {e}")
        
        return ""
    
    def _get_component_dependencies_summary(self, comp_id: str) -> str:
        """Get dependencies summary for a component"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get outbound dependencies
                cursor.execute('''
                    SELECT target_component, relationship_type 
                    FROM dependencies 
                    WHERE source_component = ?
                    LIMIT 5
                ''', (comp_id,))
                
                deps = []
                for row in cursor.fetchall():
                    deps.append(f"{row['relationship_type']}:{row['target_component']}")
                
                return ', '.join(deps) if deps else "No dependencies"
                
        except Exception as e:
            logger.error(f"Error getting dependencies: {e}")
            return ""
    
    def _get_overall_dependency_summary(self) -> str:
        """Get overall system dependency summary"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get dependency type counts
                cursor.execute('''
                    SELECT relationship_type, COUNT(*) as count 
                    FROM dependencies 
                    GROUP BY relationship_type 
                    ORDER BY count DESC
                ''')
                
                summary_parts = []
                for row in cursor.fetchall():
                    summary_parts.append(f"{row['relationship_type']}: {row['count']}")
                
                return '; '.join(summary_parts[:5])  # Top 5 types
                
        except Exception as e:
            logger.error(f"Error getting dependency summary: {e}")
            return "Dependency analysis not available"
    
    def _generate_chat_response(self, query: str, context: str, session_id: str) -> str:
        """Generate chat response using LLM with context"""
        prompt = f"""
You are a mainframe systems analyst with deep knowledge of COBOL, JCL, and related technologies.
Use the provided codebase context to answer the user's question accurately and helpfully.

CODEBASE CONTEXT:
{context}

USER QUESTION: {query}

Provide a clear, accurate response based on the codebase context. If the question is about:
- Specific programs: Explain their purpose and business logic
- Dependencies: Describe the relationships between components
- Data flow: Explain how data moves through the system
- Modernization: Suggest approaches based on the current architecture
- Business impact: Relate technical details to business value

Be specific and reference the actual components when relevant.
"""
        
        try:
            response = self.llm_client.call_llm(prompt, max_tokens=1000, temperature=0.3)
            
            if response.success:
                return response.content
            else:
                return f"I encountered an error generating a response: {response.error_message}"
                
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return "I'm experiencing technical difficulties. Please try your question again."
    
    def _store_chat_message(self, session_id: str, query: str, response: str, components: List[Dict]):
        """Store chat conversation"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Store user message
                cursor.execute('''
                    INSERT INTO chat_conversations 
                    (session_id, message_type, content, context_components_json)
                    VALUES (?, ?, ?, ?)
                ''', (
                    session_id, 'user', query, 
                    json.dumps([comp['name'] for comp in components])
                ))
                
                # Store assistant response
                cursor.execute('''
                    INSERT INTO chat_conversations 
                    (session_id, message_type, content, context_components_json)
                    VALUES (?, ?, ?, ?)
                ''', (
                    session_id, 'assistant', response,
                    json.dumps([comp['name'] for comp in components])
                ))
                
        except Exception as e:
            logger.error(f"Error storing chat message: {e}")

# ============================================================================
# Enhanced Flask Application
# ============================================================================

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Mock LLM client for development
class LLMAnalysisEngine:
        """Enhanced LLM analysis focusing on business logic and comments"""
    
        def __init__(self, llm_client: LLMClient, db_manager: DatabaseManager, token_manager: TokenManager = None):
            self.llm_client = llm_client
            self.db_manager = db_manager
            self.token_manager = token_manager or TokenManager()

def create_enhanced_app():
    """Create enhanced Flask application"""
    app = Flask(__name__)
    CORS(app)
    
    # Initialize core components
    db_manager = DatabaseManager()
    parser = UnifiedParser()
    batch_processor = BatchComponentProcessor(db_manager, parser)
    
    # Initialize REAL LLM client and token manager (REPLACE THE MOCK)
    llm_client = LLMClient()  # Real LLM client
    token_manager = TokenManager()
    llm_engine = LLMAnalysisEngine(llm_client, db_manager, token_manager)
    chat_system = AgenticChatSystem(db_manager, llm_client)

    # ADD THESE NEW ENDPOINTS:
    
    @app.route('/api/llm/config', methods=['POST'])
    def update_llm_config():
        """Update LLM client configuration"""
        try:
            config = request.json
            llm_client.update_config(config)
            
            return jsonify({
                'success': True,
                'message': 'LLM configuration updated',
                'config': {
                    'endpoint': llm_client.endpoint,
                    'timeout': llm_client.timeout,
                    'max_tokens': llm_client.default_max_tokens,
                    'temperature': llm_client.default_temperature
                }
            })
        except Exception as e:
            logger.error(f"Error updating LLM config: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/llm/health', methods=['GET'])
    def check_llm_health():
        """Check LLM server health"""
        try:
            is_healthy = llm_client.health_check()
            return jsonify({
                'success': True,
                'healthy': is_healthy,
                'endpoint': llm_client.endpoint
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'healthy': False,
                'error': str(e)
            })
    
    @app.route('/api/usage/<session_id>', methods=['GET'])
    def get_token_usage(session_id):
        """Get token usage for a session"""
        try:
            usage = token_manager.get_session_token_usage(session_id)
            return jsonify({
                'success': True,
                'session_id': session_id,
                'token_usage': usage
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500


  
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/components', methods=['GET'])
    def get_all_components():
        """Get all components in the system"""
        component_type = request.args.get('type')
        components = db_manager.get_all_components(component_type)
        
        return jsonify({
            'success': True,
            'components': [
                {
                    'id': comp.id,
                    'name': comp.name,
                    'type': comp.type,
                    'friendly_name': comp.friendly_name,
                    'business_purpose': comp.business_purpose,
                    'complexity_score': comp.complexity_score,
                    'total_lines': comp.total_lines,
                    'llm_analyzed': comp.llm_analyzed,
                    'uploaded_by': comp.uploaded_by,
                    'uploaded_at': comp.uploaded_at.isoformat()
                }
                for comp in components
            ],
            'total_count': len(components)
        })
    
    @app.route('/api/upload/batch', methods=['POST'])
    def batch_upload():
        """Handle batch file upload"""
        try:
            data = request.json
            files = data.get('files', [])
            user_id = data.get('user_id', 'anonymous')
            
            if not files:
                return jsonify({
                    'success': False,
                    'error': 'No files provided'
                }), 400
            
            result = batch_processor.process_batch_upload(files, user_id)
            
            return jsonify({
                'success': True,
                'batch_id': result['batch_id'],
                'status': result['status']
            })
            
        except Exception as e:
            logger.error(f"Error in batch upload: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/analyze/batch', methods=['POST'])
    def analyze_components_batch():
        """Analyze selected components with LLM"""
        try:
            data = request.json
            component_ids = data.get('component_ids', [])
            session_id = data.get('session_id', str(uuid.uuid4()))
            
            if not component_ids:
                return jsonify({
                    'success': False,
                    'error': 'No components selected'
                }), 400
            
            # CHECK LLM HEALTH FIRST
            if not llm_client.health_check():
                return jsonify({
                    'success': False,
                    'error': 'LLM server is not available'
                }), 503
            
            # Create user session
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO user_sessions 
                    (session_id, selected_components_json, analysis_status)
                    VALUES (?, ?, ?)
                ''', (session_id, json.dumps(component_ids), 'processing'))
            
            # Start analysis
            results = llm_engine.analyze_component_batch(component_ids, session_id)
            
            # GET TOKEN USAGE
            token_usage = token_manager.get_session_token_usage(session_id)
            
            # Update session status
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE user_sessions 
                    SET analysis_status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                ''', ('completed', session_id))
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'analysis_results': results,
                'components_analyzed': len(component_ids),
                'token_usage': token_usage  # ADD THIS
            })
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/chat', methods=['POST'])
    def chat_with_codebase():
        """Chat with the codebase using RAG"""
        try:
            data = request.json
            query = data.get('message', '')
            session_id = data.get('session_id', str(uuid.uuid4()))
            user_id = data.get('user_id', 'anonymous')
            
            if not query:
                return jsonify({
                    'success': False,
                    'error': 'No message provided'
                }), 400
            
            result = chat_system.process_chat_query(session_id, query, user_id)
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'response': 'I encountered an error processing your query.'
            }), 500
    
    @app.route('/api/dependencies/<component_id>', methods=['GET'])
    def get_component_dependencies(component_id):
        """Get dependencies for a specific component"""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get outbound dependencies
                cursor.execute('''
                    SELECT d.*, c.name as target_name, c.type as target_type
                    FROM dependencies d
                    LEFT JOIN components c ON d.target_component = c.id
                    WHERE d.source_component = ?
                    ORDER BY d.relationship_type, d.target_component
                ''', (component_id,))
                
                outbound = []
                for row in cursor.fetchall():
                    dep_dict = dict(row)
                    if dep_dict['analysis_details_json']:
                        dep_dict['analysis_details'] = json.loads(dep_dict['analysis_details_json'])
                    if dep_dict['resolved_targets_json']:
                        dep_dict['resolved_targets'] = json.loads(dep_dict['resolved_targets_json'])
                    outbound.append(dep_dict)
                
                # Get inbound dependencies
                cursor.execute('''
                    SELECT d.*, c.name as source_name, c.type as source_type
                    FROM dependencies d
                    LEFT JOIN components c ON d.source_component = c.id
                    WHERE d.target_component = ?
                    ORDER BY d.relationship_type, d.source_component
                ''', (component_id,))
                
                inbound = []
                for row in cursor.fetchall():
                    dep_dict = dict(row)
                    if dep_dict['analysis_details_json']:
                        dep_dict['analysis_details'] = json.loads(dep_dict['analysis_details_json'])
                    inbound.append(dep_dict)
            
            return jsonify({
                'success': True,
                'component_id': component_id,
                'outbound_dependencies': outbound,
                'inbound_dependencies': inbound,
                'total_outbound': len(outbound),
                'total_inbound': len(inbound)
            })
            
        except Exception as e:
            logger.error(f"Error getting dependencies: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/components/<component_id>/source', methods=['GET'])
    def get_component_source(component_id):
        """Get source code for a component"""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT name, type, source_content, metadata_json 
                    FROM components 
                    WHERE id = ?
                ''', (component_id,))
                
                row = cursor.fetchone()
                if not row:
                    return jsonify({
                        'success': False,
                        'error': 'Component not found'
                    }), 404
                
                metadata = json.loads(row['metadata_json'] or '{}')
                
                return jsonify({
                    'success': True,
                    'component': {
                        'name': row['name'],
                        'type': row['type'],
                        'source_content': row['source_content'],
                        'metadata': metadata
                    }
                })
                
        except Exception as e:
            logger.error(f"Error getting component source: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/analysis/<component_id>', methods=['GET'])
    def get_component_analysis(component_id):
        """Get LLM analysis for a component"""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT analysis_type, analysis_result_json, created_at, token_usage
                    FROM llm_analysis 
                    WHERE component_id = ?
                    ORDER BY created_at DESC
                ''', (component_id,))
                
                analyses = []
                for row in cursor.fetchall():
                    analysis = {
                        'type': row['analysis_type'],
                        'result': json.loads(row['analysis_result_json']),
                        'created_at': row['created_at'],
                        'token_usage': row['token_usage']
                    }
                    analyses.append(analysis)
                
                return jsonify({
                    'success': True,
                    'component_id': component_id,
                    'analyses': analyses
                })
                
        except Exception as e:
            logger.error(f"Error getting component analysis: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/sessions/<session_id>/chat-history', methods=['GET'])
    def get_chat_history(session_id):
        """Get chat history for a session"""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT message_type, content, context_components_json, created_at
                    FROM chat_conversations 
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                ''', (session_id,))
                
                messages = []
                for row in cursor.fetchall():
                    message = {
                        'type': row['message_type'],
                        'content': row['content'],
                        'context_components': json.loads(row['context_components_json'] or '[]'),
                        'timestamp': row['created_at']
                    }
                    messages.append(message)
                
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'messages': messages
                })
                
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/system/stats', methods=['GET'])
    def get_system_stats():
        """Get overall system statistics"""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Component counts by type
                cursor.execute('''
                    SELECT type, COUNT(*) as count 
                    FROM components 
                    GROUP BY type
                ''')
                component_stats = dict(cursor.fetchall())
                
                # Analysis statistics
                cursor.execute('''
                    SELECT COUNT(*) as total, 
                           SUM(CASE WHEN llm_analyzed = 1 THEN 1 ELSE 0 END) as analyzed
                    FROM components
                ''')
                analysis_stats = dict(cursor.fetchone())
                
                # Dependency statistics
                cursor.execute('''
                    SELECT relationship_type, COUNT(*) as count 
                    FROM dependencies 
                    GROUP BY relationship_type
                ''')
                dependency_stats = dict(cursor.fetchall())
                
                # Recent activity
                cursor.execute('''
                    SELECT COUNT(*) as recent_uploads
                    FROM components 
                    WHERE uploaded_at > datetime('now', '-24 hours')
                ''')
                recent_activity = cursor.fetchone()['recent_uploads']
                
                return jsonify({
                    'success': True,
                    'stats': {
                        'components': component_stats,
                        'analysis': analysis_stats,
                        'dependencies': dependency_stats,
                        'recent_activity': recent_activity
                    }
                })
                
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/search', methods=['GET'])
    def search_components():
        """Search components by name, type, or content"""
        try:
            query = request.args.get('q', '')
            component_type = request.args.get('type', '')
            limit = int(request.args.get('limit', 20))
            
            if not query:
                return jsonify({
                    'success': False,
                    'error': 'Search query required'
                }), 400
            
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                sql_conditions = []
                params = []
                
                # Text search
                sql_conditions.append('''(
                    UPPER(name) LIKE UPPER(?) OR 
                    UPPER(friendly_name) LIKE UPPER(?) OR 
                    UPPER(business_purpose) LIKE UPPER(?)
                )''')
                search_param = f'%{query}%'
                params.extend([search_param, search_param, search_param])
                
                # Type filter
                if component_type:
                    sql_conditions.append('UPPER(type) = UPPER(?)')
                    params.append(component_type)
                
                sql = f'''
                    SELECT id, name, type, friendly_name, business_purpose, 
                           complexity_score, total_lines, llm_analyzed
                    FROM components 
                    WHERE {' AND '.join(sql_conditions)}
                    ORDER BY 
                        CASE WHEN UPPER(name) = UPPER(?) THEN 1 ELSE 2 END,
                        complexity_score DESC,
                        name
                    LIMIT ?
                '''
                params.extend([query, limit])
                
                cursor.execute(sql, params)
                
                results = []
                for row in cursor.fetchall():
                    results.append(dict(row))
                
                return jsonify({
                    'success': True,
                    'query': query,
                    'results': results,
                    'total_found': len(results)
                })
                
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return app

# ============================================================================
# Utility Functions for Parser Enhancement
# ============================================================================

def extract_program_name(content: str, file_name: str) -> str:
    """Extract program name from COBOL content or filename"""
    # Try to find PROGRAM-ID
    prog_id_match = re.search(r'PROGRAM-ID\.\s+([A-Z0-9\-]+)', content.upper())
    if prog_id_match:
        return prog_id_match.group(1)
    
    # Fallback to filename
    return file_name.split('.')[0] if '.' in file_name else file_name

def extract_job_name(content: str, file_name: str) -> str:
    """Extract job name from JCL content"""
    job_match = re.search(r'^//([A-Z0-9]+)\s+JOB', content, re.MULTILINE)
    if job_match:
        return job_match.group(1)
    
    return file_name.split('.')[0] if '.' in file_name else file_name

def calculate_complexity(analysis: Dict) -> Dict:
    """Calculate complexity indicators for a component"""
    complexity = {
        'cyclomatic_complexity': 0,
        'data_complexity': 0,
        'integration_complexity': 0,
        'overall_score': 0.0
    }
    
    # Calculate based on various factors
    program_calls = len(analysis.get('program_calls', []))
    file_operations = len(analysis.get('file_operations', []))
    cics_operations = len(analysis.get('cics_operations', []))
    db2_operations = len(analysis.get('db2_operations', []))
    record_layouts = len(analysis.get('record_layouts', []))
    
    # Integration complexity
    complexity['integration_complexity'] = program_calls + file_operations + cics_operations + db2_operations
    
    # Data complexity
    complexity['data_complexity'] = record_layouts
    
    # Overall score (normalized 0-1)
    total_elements = complexity['integration_complexity'] + complexity['data_complexity']
    complexity['overall_score'] = min(1.0, total_elements / 50.0)  # Normalize to 50 elements = 1.0
    
    return complexity

# ============================================================================
# Enhanced Parser Helper Methods
# ============================================================================

class UnifiedParser:
    """Adding the missing helper methods"""
    
    def _extract_divisions(self, lines: List[str]) -> List[Dict]:
        """Extract COBOL divisions"""
        divisions = []
        for i, line in enumerate(lines):
            div_match = re.search(r'(IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION', line.upper())
            if div_match:
                divisions.append({
                    'name': div_match.group(1),
                    'line_number': i + 1
                })
        return divisions
    
    def _extract_copybooks(self, lines: List[str]) -> List[str]:
        """Extract copybook names"""
        copybooks = []
        for line in lines:
            copy_match = re.search(r'COPY\s+([A-Z0-9\-]+)', line.upper())
            if copy_match:
                copybooks.append(copy_match.group(1))
        return list(set(copybooks))
    
    def _extract_file_operations(self, lines: List[str]) -> List[Dict]:
        """Extract file operations"""
        operations = []
        for i, line in enumerate(lines):
            line_upper = line.upper()
            
            # FD statements
            fd_match = re.search(r'FD\s+([A-Z0-9\-]+)', line_upper)
            if fd_match:
                operations.append({
                    'operation': 'FD',
                    'file_name': fd_match.group(1),
                    'line_number': i + 1,
                    'io_direction': 'DECLARATION'
                })
            
            # File I/O operations
            io_patterns = [
                (r'READ\s+([A-Z0-9\-]+)', 'READ', 'INPUT'),
                (r'WRITE\s+([A-Z0-9\-]+)', 'WRITE', 'OUTPUT'),
                (r'OPEN\s+INPUT\s+([A-Z0-9\-]+)', 'OPEN', 'INPUT'),
                (r'OPEN\s+OUTPUT\s+([A-Z0-9\-]+)', 'OPEN', 'OUTPUT')
            ]
            
            for pattern, op, direction in io_patterns:
                match = re.search(pattern, line_upper)
                if match:
                    operations.append({
                        'operation': op,
                        'file_name': match.group(1),
                        'line_number': i + 1,
                        'io_direction': direction
                    })
        
        return operations
    
    def _extract_cics_operations(self, lines: List[str]) -> List[Dict]:
        """Extract CICS operations with layout associations"""
        operations = []
        for i, line in enumerate(lines):
            if 'EXEC CICS' in line.upper():
                # Get the complete CICS command (may span multiple lines)
                cics_command = self._get_complete_cics_command(lines, i)
                cics_op = self._parse_cics_command(cics_command, i + 1)
                if cics_op:
                    operations.append(cics_op)
        return operations
    
    def _get_complete_cics_command(self, lines: List[str], start_idx: int) -> str:
        """Get complete CICS command across multiple lines"""
        command_parts = []
        for i in range(start_idx, min(len(lines), start_idx + 10)):
            line = lines[i].strip()
            command_parts.append(line)
            if 'END-EXEC' in line.upper():
                break
        return ' '.join(command_parts)
    
    def _parse_cics_command(self, command: str, line_number: int) -> Optional[Dict]:
        """Parse CICS command to extract operation details"""
        command_upper = command.upper()
        
        # Extract operation type
        op_match = re.search(r'EXEC\s+CICS\s+(\w+)', command_upper)
        if not op_match:
            return None
        
        operation = op_match.group(1)
        
        # Extract file/dataset name
        file_match = re.search(r'(?:FILE|DATASET)\s*\(\s*[\'"]?([A-Z0-9\-]+)[\'"]?\s*\)', command_upper)
        file_name = file_match.group(1) if file_match else None
        
        # Extract layout from INTO/FROM clauses
        layout_match = re.search(r'(?:INTO|FROM)\s*\(\s*([A-Z0-9\-]+)\s*\)', command_upper)
        layout_name = layout_match.group(1) if layout_match else None
        
        return {
            'operation': operation,
            'file_name': file_name,
            'layout_name': layout_name,
            'line_number': line_number,
            'has_layout_association': bool(layout_name)
        }
    
    def _extract_db2_operations(self, lines: List[str]) -> List[Dict]:
        """Extract DB2 SQL operations"""
        operations = []
        in_sql = False
        sql_buffer = []
        start_line = 0
        
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            
            if 'EXEC SQL' in line_upper:
                in_sql = True
                sql_buffer = [line.strip()]
                start_line = i + 1
            elif in_sql:
                sql_buffer.append(line.strip())
                if 'END-EXEC' in line_upper:
                    sql_statement = ' '.join(sql_buffer)
                    operation = self._analyze_sql_statement(sql_statement, start_line)
                    if operation:
                        operations.append(operation)
                    in_sql = False
                    sql_buffer = []
        
        return operations
    
    def _analyze_sql_statement(self, sql: str, line_number: int) -> Optional[Dict]:
        """Analyze SQL statement to extract table operations"""
        sql_upper = sql.upper()
        
        # Determine operation type
        if 'SELECT' in sql_upper:
            operation_type = 'SELECT'
        elif 'INSERT' in sql_upper:
            operation_type = 'INSERT'
        elif 'UPDATE' in sql_upper:
            operation_type = 'UPDATE'
        elif 'DELETE' in sql_upper:
            operation_type = 'DELETE'
        else:
            operation_type = 'OTHER'
        
        # Extract table names
        tables = self._extract_table_names(sql_upper)
        
        return {
            'operation_type': operation_type,
            'sql': sql,
            'tables': tables,
            'line_number': line_number
        } if tables else None
    
    def _extract_table_names(self, sql: str) -> List[str]:
        """Extract table names from SQL statement"""
        tables = []
        
        # FROM clause
        from_match = re.search(r'FROM\s+([A-Z0-9_\.]+)', sql)
        if from_match:
            tables.append(from_match.group(1).split('.')[0])
        
        # UPDATE table
        update_match = re.search(r'UPDATE\s+([A-Z0-9_\.]+)', sql)
        if update_match:
            tables.append(update_match.group(1).split('.')[0])
        
        # INSERT INTO table
        insert_match = re.search(r'INSERT\s+INTO\s+([A-Z0-9_\.]+)', sql)
        if insert_match:
            tables.append(insert_match.group(1).split('.')[0])
        
        return list(set(tables))
    
    def _extract_record_layouts(self, lines: List[str]) -> List[Dict]:
        """Extract record layout structures"""
        layouts = []
        current_layout = None
        
        for i, line in enumerate(lines):
            # Look for 01 level items
            level_match = re.match(r'^\s*01\s+([A-Z0-9\-]+)', line.strip().upper())
            if level_match:
                if current_layout:
                    layouts.append(current_layout)
                
                current_layout = {
                    'name': level_match.group(1),
                    'line_start': i + 1,
                    'fields': []
                }
            elif current_layout and re.match(r'^\s*\d{2}\s+', line.strip()):
                # Add field to current layout
                field_info = self._parse_field_definition(line, i + 1)
                if field_info:
                    current_layout['fields'].append(field_info)
        
        if current_layout:
            layouts.append(current_layout)
        
        return layouts
    
    def _parse_field_definition(self, line: str, line_number: int) -> Optional[Dict]:
        """Parse individual field definition"""
        level_match = re.match(r'^\s*(\d{2})\s+([A-Z0-9\-]+)(?:\s+(.*))?', line.strip().upper())
        if not level_match:
            return None
        
        level = int(level_match.group(1))
        name = level_match.group(2)
        definition = level_match.group(3) or ''
        
        return {
            'level': level,
            'name': name,
            'line_number': line_number,
            'picture': self._extract_picture(definition),
            'usage': self._extract_usage(definition),
            'value': self._extract_value(definition),
            'occurs': self._extract_occurs(definition),
            'redefines': self._extract_redefines(definition)
        }
    
    def _extract_picture(self, definition: str) -> str:
        """Extract PIC clause from field definition"""
        pic_match = re.search(r'PIC(?:TURE)?\s+([X9SVP\(\),\.\+\-\*\$Z]+)', definition.upper())
        return pic_match.group(1) if pic_match else ''
    
    def _extract_usage(self, definition: str) -> str:
        """Extract USAGE clause"""
        usage_match = re.search(r'USAGE\s+(COMP|COMP-3|DISPLAY|BINARY|PACKED-DECIMAL)', definition.upper())
        return usage_match.group(1) if usage_match else ''
    
    def _extract_value(self, definition: str) -> str:
        """Extract VALUE clause"""
        value_match = re.search(r'VALUE\s+(["\'].*?["\']|\S+)', definition.upper())
        return value_match.group(1).strip('"\'') if value_match else ''
    
    def _extract_occurs(self, definition: str) -> int:
        """Extract OCCURS clause"""
        occurs_match = re.search(r'OCCURS\s+(\d+)', definition.upper())
        return int(occurs_match.group(1)) if occurs_match else 0
    
    def _extract_redefines(self, definition: str) -> str:
        """Extract REDEFINES clause"""
        redefines_match = re.search(r'REDEFINES\s+([A-Z0-9\-]+)', definition.upper())
        return redefines_match.group(1) if redefines_match else ''
    
    def _extract_program_from_exec(self, line: str) -> str:
        """Extract program name from EXEC statement"""
        prog_match = re.search(r'PGM=([A-Z0-9\-]+)', line.upper())
        return prog_match.group(1) if prog_match else ''
    
    def _extract_dataset_info(self, line: str) -> Optional[Dict]:
        """Extract dataset information from DD statement"""
        dd_match = re.match(r'//([A-Z0-9]+)\s+DD', line.upper())
        if not dd_match:
            return None
        
        dd_name = dd_match.group(1)
        
        # Extract DSN
        dsn_match = re.search(r'DSN=([A-Z0-9\.\(\)]+)', line.upper())
        dsn = dsn_match.group(1) if dsn_match else None
        
        return {
            'dd_name': dd_name,
            'dsn': dsn,
            'line': line.strip()
        }
    
    def _parse_generic(self, content: str, file_name: str, component_type: str) -> Dict:
        """Parse generic component type"""
        lines = content.split('\n')
        return {
            'type': component_type,
            'name': file_name.split('.')[0] if '.' in file_name else file_name,
            'total_lines': len(lines),
            'content_preview': '\n'.join(lines[:20])  # First 20 lines
        }

if __name__ == '__main__':
    app = create_enhanced_app()
    print("="*60)
    print("Enhanced Mainframe Analyzer Starting")
    print("Features:")
    print("- Batch component upload and processing")
    print("- Enhanced LLM analysis focusing on business logic")
    print("- Agentic RAG-enabled chat system")
    print("- Comprehensive dependency analysis")
    print("- Multi-user support")
    print("- Dynamic call resolution")
    print("Server: http://localhost:6000")
    print("="*60)
    app.run(host='0.0.0.0', port=6000, debug=True)