"""
Token Management Module for LLM Calls
Handles chunking, token estimation, and context management
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

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

class TokenManager:
    def __init__(self):
        self.MAX_TOKENS_PER_CALL = 6000
        self.EFFECTIVE_CONTENT_LIMIT = 5500  # Reserve 500 for system prompts
        self.CHUNK_OVERLAP_TOKENS = 200
        self.CHARACTERS_PER_TOKEN = 4  # Rough estimation
        self.token_usage_cache = {}
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length"""
        return len(text) // self.CHARACTERS_PER_TOKEN
    
    def needs_chunking(self, text: str) -> bool:
        """Check if text needs to be chunked"""
        estimated_tokens = self.estimate_tokens(text)
        return estimated_tokens > self.EFFECTIVE_CONTENT_LIMIT
    
    def chunk_cobol_code(self, code: str, preserve_structure: bool = True) -> List[ChunkInfo]:
        """
        Intelligently chunk COBOL code preserving structure
        """
        if not self.needs_chunking(code):
            return [ChunkInfo(
                content=code,
                chunk_number=1,
                total_chunks=1,
                estimated_tokens=self.estimate_tokens(code)
            )]
        
        chunks = []
        
        if preserve_structure:
            chunks = self._chunk_by_structure(code)
        else:
            chunks = self._chunk_by_size(code)
        
        # Add overlap between chunks
        chunks = self._add_context_overlap(chunks)
        
        return chunks
    
    def _chunk_by_structure(self, code: str) -> List[ChunkInfo]:
        """Chunk COBOL code by structural elements"""
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        # COBOL structural patterns
        division_pattern = r'^\s*(IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION'
        section_pattern = r'^\s*\w+\s+SECTION\s*\.'
        paragraph_pattern = r'^\s*[A-Z0-9][A-Z0-9\-]*\s*\.'
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_tokens = self.estimate_tokens(line)
            
            # Check if we need to start a new chunk
            if (current_size + line_tokens > self.EFFECTIVE_CONTENT_LIMIT and 
                current_chunk and 
                (re.match(division_pattern, line, re.IGNORECASE) or
                 re.match(section_pattern, line, re.IGNORECASE) or
                 re.match(paragraph_pattern, line, re.IGNORECASE))):
                
                # Finalize current chunk
                chunk_content = '\n'.join(current_chunk)
                chunks.append(ChunkInfo(
                    content=chunk_content,
                    chunk_number=len(chunks) + 1,
                    total_chunks=0,  # Will be set later
                    estimated_tokens=current_size
                ))
                
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += line_tokens
            i += 1
        
        # Add final chunk if there's remaining content
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(ChunkInfo(
                content=chunk_content,
                chunk_number=len(chunks) + 1,
                total_chunks=0,
                estimated_tokens=current_size
            ))
        
        # Set total chunks count
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
                # Finalize current chunk
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
        
        # Add final chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append(ChunkInfo(
                content=chunk_content,
                chunk_number=len(chunks) + 1,
                total_chunks=0,
                estimated_tokens=current_size
            ))
        
        # Set total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _add_context_overlap(self, chunks: List[ChunkInfo]) -> List[ChunkInfo]:
        """Add context overlap between chunks"""
        if len(chunks) <= 1:
            return chunks
        
        for i in range(1, len(chunks)):
            # Get overlap from previous chunk
            prev_chunk = chunks[i-1]
            prev_words = prev_chunk.content.split()
            
            # Calculate overlap size
            overlap_chars = self.CHUNK_OVERLAP_TOKENS * self.CHARACTERS_PER_TOKEN
            overlap_content = ""
            
            # Get last N words from previous chunk
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
            'component_extraction': """
Analyze this COBOL code and extract components with the following JSON structure:
{
    "components": [
        {
            "name": "friendly_component_name",
            "type": "PROGRAM|COPYBOOK|RECORD_LAYOUT|PARAGRAPH|SECTION",
            "line_start": number,
            "line_end": number,
            "fields": [],
            "file_operations": [],
            "program_calls": [],
            "copybooks": [],
            "business_logic": []
        }
    ],
    "record_layouts": [
        {
            "name": "friendly_name",
            "level": "01",
            "fields": [
                {
                    "name": "field_name",
                    "level": "05",
                    "data_type": "PIC X(10)",
                    "usage": "INPUT|OUTPUT|DERIVED|REFERENCE|STATIC|UNUSED",
                    "source_program": "program_name",
                    "business_purpose": "description"
                }
            ]
        }
    ]
}
""",
            'field_mapping': """
Analyze field mappings and data flow for the specified target file:
Return JSON with field mappings including business logic classification:
{
    "field_mappings": [
        {
            "field_name": "friendly_field_name",
            "mainframe_data_type": "COBOL type",
            "oracle_data_type": "Oracle equivalent",
            "population_source": "source description",
            "business_logic_type": "MOVE|DERIVED|CONDITIONAL|CALCULATED|STRING_MANIPULATION|UNUSED",
            "business_logic_description": "detailed description",
            "programs_involved": ["program1", "program2"],
            "confidence_score": 0.95
        }
    ]
}
""",
            'field_analysis': """
Analyze field usage and relationships in COBOL code:
Identify how fields are populated, used, and their business purpose.
Return detailed field analysis with code references.
"""
        }
        
        return prompts.get(analysis_type, "Analyze the provided COBOL code.")
    
    def consolidate_chunk_results(self, results: List[Dict], analysis_type: str) -> Dict:
        """Consolidate results from multiple chunks"""
        if not results:
            return {}
        
        if len(results) == 1:
            return results[0]
        
        # Consolidation logic based on analysis type
        if analysis_type == 'component_extraction':
            return self._consolidate_component_results(results)
        elif analysis_type == 'field_mapping':
            return self._consolidate_field_mapping_results(results)
        else:
            return self._consolidate_generic_results(results)
    
    def _consolidate_component_results(self, results: List[Dict]) -> Dict:
        """Consolidate component extraction results"""
        consolidated = {
            'components': [],
            'record_layouts': []
        }
        
        seen_components = set()
        seen_layouts = set()
        
        for result in results:
            # Merge components
            if 'components' in result:
                for component in result['components']:
                    comp_key = f"{component.get('name', '')}_{component.get('type', '')}"
                    if comp_key not in seen_components:
                        consolidated['components'].append(component)
                        seen_components.add(comp_key)
            
            # Merge record layouts
            if 'record_layouts' in result:
                for layout in result['record_layouts']:
                    layout_key = layout.get('name', '')
                    if layout_key not in seen_layouts:
                        consolidated['record_layouts'].append(layout)
                        seen_layouts.add(layout_key)
        
        return consolidated
    
    def _consolidate_field_mapping_results(self, results: List[Dict]) -> Dict:
        """Consolidate field mapping results"""
        consolidated = {'field_mappings': []}
        field_map = {}
        
        for result in results:
            if 'field_mappings' in result:
                for mapping in result['field_mappings']:
                    field_name = mapping.get('field_name', '')
                    if field_name in field_map:
                        # Merge programs involved
                        existing = field_map[field_name]
                        programs = set(existing.get('programs_involved', []))
                        programs.update(mapping.get('programs_involved', []))
                        existing['programs_involved'] = list(programs)
                        
                        # Update confidence score (average)
                        existing_conf = existing.get('confidence_score', 0)
                        new_conf = mapping.get('confidence_score', 0)
                        existing['confidence_score'] = (existing_conf + new_conf) / 2
                    else:
                        field_map[field_name] = mapping
        
        consolidated['field_mappings'] = list(field_map.values())
        return consolidated
    
    def _consolidate_generic_results(self, results: List[Dict]) -> Dict:
        """Generic result consolidation"""
        consolidated = {}
        for result in results:
            for key, value in result.items():
                if key not in consolidated:
                    consolidated[key] = []
                if isinstance(value, list):
                    consolidated[key].extend(value)
                else:
                    consolidated[key].append(value)
        return consolidated
    
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