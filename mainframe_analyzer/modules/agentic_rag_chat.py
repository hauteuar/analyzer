"""
Complete Rewritten Agentic RAG Chat Manager
Clean architecture with improved query processing and specialized handlers
"""

import re
import json
import logging
import traceback
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

# Vector search imports with fallback
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Core Data Structures
class QueryType(Enum):
    FIELD_ANALYSIS = "field_analysis"
    PROGRAM_CALLS = "program_calls"
    DYNAMIC_CALLS = "dynamic_calls"
    FILE_OPERATIONS = "file_operations"
    BUSINESS_LOGIC = "business_logic"
    PROGRAM_STRUCTURE = "program_structure"
    DEPENDENCIES = "dependencies"
    GENERAL = "general"

class AnalysisComplexity(Enum):
    SIMPLE = "simple"           # Basic queries, direct answers
    ENHANCED = "enhanced"       # Complex field/program analysis
    COMPREHENSIVE = "comprehensive"  # Multi-faceted business logic

@dataclass
class QueryAnalysis:
    query_type: QueryType
    complexity: AnalysisComplexity
    entities: List[str]
    keywords: List[str]
    confidence: float
    requires_specialized_handler: bool = False
    context_requirements: Dict[str, Any] = None

@dataclass
class RetrievedContext:
    source_code: str
    metadata: Dict[str, Any]
    relevance_score: float
    retrieval_method: str
    component_name: str

class QueryClassifier:
    """Improved query classification with clear patterns"""
    
    def __init__(self):
        self.query_patterns = {
            QueryType.FIELD_ANALYSIS: {
                'keywords': ['field', 'variable', 'what is', 'tell me about', 'how is', 'structure of'],
                'patterns': [
                    r'what\s+is\s+([A-Z][A-Z0-9\-]+)',
                    r'tell\s+me\s+about\s+([A-Z][A-Z0-9\-]+)',
                    r'how\s+is\s+([A-Z][A-Z0-9\-]+)\s+used',
                    r'field\s+([A-Z][A-Z0-9\-]+)',
                    r'structure\s+of\s+([A-Z][A-Z0-9\-]+)'
                ]
            },
            QueryType.PROGRAM_CALLS: {
                'keywords': ['call', 'calls', 'program', 'xctl', 'link'],
                'patterns': [
                    r'what\s+programs?\s+(?:does\s+)?(?:it\s+)?calls?',
                    r'program\s+calls?',
                    r'calls?\s+what\s+programs?',
                    r'xctl\s+|link\s+',
                    r'calls?\s+([A-Z][A-Z0-9\-]+)'
                ]
            },
            QueryType.DYNAMIC_CALLS: {
                'keywords': ['dynamic', 'variable', 'runtime', 'conditions', 'different programs'],
                'patterns': [
                    r'dynamic\s+call',
                    r'variable\s+program',
                    r'runtime\s+call',
                    r'different\s+programs',
                    r'conditions?\s+.*call',
                    r'how.*calls?\s+.*program',
                    r'routing\s+logic'
                ]
            },
            QueryType.FILE_OPERATIONS: {
                'keywords': ['file', 'read', 'write', 'cics', 'dataset'],
                'patterns': [
                    r'file\s+operations?',
                    r'reads?\s+.*file',
                    r'writes?\s+.*file',
                    r'cics\s+(?:read|write)',
                    r'dataset'
                ]
            },
            QueryType.BUSINESS_LOGIC: {
                'keywords': ['business', 'logic', 'calculation', 'rule', 'process', 'how does'],
                'patterns': [
                    r'business\s+logic',
                    r'how\s+does\s+.*work',
                    r'calculation',
                    r'business\s+rule',
                    r'process\s+flow'
                ]
            }
        }
    
    def classify_query(self, message: str, entities: List[str]) -> QueryAnalysis:
        """Classify query with improved logic"""
        message_lower = message.lower()
        
        # Score each query type
        scores = {}
        matched_patterns = {}
        
        for query_type, config in self.query_patterns.items():
            score = 0
            patterns_found = []
            
            # Keyword matching
            for keyword in config['keywords']:
                if keyword in message_lower:
                    score += 1
            
            # Pattern matching with entity extraction
            for pattern in config['patterns']:
                matches = re.findall(pattern, message_lower)
                if matches:
                    score += 2
                    patterns_found.extend(matches)
            
            scores[query_type] = score
            matched_patterns[query_type] = patterns_found
        
        # Determine primary query type
        primary_type = max(scores, key=scores.get) if any(scores.values()) else QueryType.GENERAL
        primary_score = scores[primary_type]
        
        # Determine complexity
        complexity = self._determine_complexity(message, entities, primary_type, primary_score)
        
        # Extract keywords and entities
        all_keywords = []
        for config in self.query_patterns.values():
            all_keywords.extend([kw for kw in config['keywords'] if kw in message_lower])
        
        confidence = min(primary_score / 5.0, 1.0) if primary_score > 0 else 0.3
        
        return QueryAnalysis(
            query_type=primary_type,
            complexity=complexity,
            entities=entities,
            keywords=list(set(all_keywords)),
            confidence=confidence,
            requires_specialized_handler=complexity in [AnalysisComplexity.ENHANCED, AnalysisComplexity.COMPREHENSIVE],
            context_requirements=self._build_context_requirements(primary_type, complexity)
        )
    
    def _determine_complexity(self, message: str, entities: List[str], 
                             query_type: QueryType, score: int) -> AnalysisComplexity:
        """Determine query complexity"""
        
        # Complex indicators
        complex_indicators = [
            'how', 'why', 'explain', 'business logic', 'conditions', 
            'different values', 'routing', 'flow', 'process'
        ]
        
        enhanced_indicators = [
            'dynamic', 'variable', 'group field', 'structure', 
            'comprehensive', 'detailed'
        ]
        
        message_lower = message.lower()
        
        if any(indicator in message_lower for indicator in enhanced_indicators):
            return AnalysisComplexity.COMPREHENSIVE
        elif (len(entities) > 0 and 
              any(indicator in message_lower for indicator in complex_indicators)):
            return AnalysisComplexity.ENHANCED
        elif score >= 3 or len(entities) > 1:
            return AnalysisComplexity.ENHANCED
        else:
            return AnalysisComplexity.SIMPLE
    
    def _build_context_requirements(self, query_type: QueryType, 
                                   complexity: AnalysisComplexity) -> Dict[str, Any]:
        """Build context requirements based on query type and complexity"""
        
        base_requirements = {
            'max_contexts': 3,
            'min_relevance': 0.3,
            'source_code_needed': True,
            'metadata_needed': True
        }
        
        if complexity == AnalysisComplexity.COMPREHENSIVE:
            base_requirements.update({
                'max_contexts': 5,
                'min_relevance': 0.2,
                'cross_program_analysis': True,
                'business_logic_extraction': True
            })
        elif complexity == AnalysisComplexity.ENHANCED:
            base_requirements.update({
                'max_contexts': 4,
                'min_relevance': 0.25,
                'detailed_analysis': True
            })
        
        # Query-specific requirements
        type_requirements = {
            QueryType.FIELD_ANALYSIS: {
                'field_definitions_needed': True,
                'field_usage_analysis': True
            },
            QueryType.PROGRAM_CALLS: {
                'call_analysis_needed': True,
                'dependency_tracking': True
            },
            QueryType.DYNAMIC_CALLS: {
                'variable_resolution': True,
                'business_logic_extraction': True,
                'multi_line_analysis': True
            },
            QueryType.FILE_OPERATIONS: {
                'file_tracking': True,
                'io_direction_analysis': True
            }
        }
        
        base_requirements.update(type_requirements.get(query_type, {}))
        return base_requirements

class ContextRetriever:
    """Improved context retrieval with specialized strategies"""
    
    def __init__(self, db_manager, vector_store=None):
        self.db_manager = db_manager
        self.vector_store = vector_store
    
    def retrieve_contexts(self, session_id: str, query_analysis: QueryAnalysis) -> List[RetrievedContext]:
        """Main context retrieval method"""
        
        all_contexts = []
        
        # Strategy 1: Entity-based retrieval
        if query_analysis.entities:
            entity_contexts = self._retrieve_entity_contexts(session_id, query_analysis.entities)
            all_contexts.extend(entity_contexts)
        
        # Strategy 2: Query type specific retrieval
        type_contexts = self._retrieve_by_query_type(session_id, query_analysis)
        all_contexts.extend(type_contexts)
        
        # Strategy 3: Vector search if available
        if VECTOR_SEARCH_AVAILABLE and self.vector_store:
            vector_contexts = self._retrieve_vector_contexts(session_id, query_analysis)
            all_contexts.extend(vector_contexts)
        
        # Strategy 4: Dependency-based retrieval for complex queries
        if query_analysis.complexity == AnalysisComplexity.COMPREHENSIVE:
            dependency_contexts = self._retrieve_dependency_contexts(session_id, query_analysis)
            all_contexts.extend(dependency_contexts)
        
        # Deduplicate and rank
        unique_contexts = self._deduplicate_contexts(all_contexts)
        ranked_contexts = self._rank_contexts(unique_contexts, query_analysis)
        
        # Apply filters
        requirements = query_analysis.context_requirements
        max_contexts = requirements.get('max_contexts', 3)
        min_relevance = requirements.get('min_relevance', 0.3)
        
        filtered_contexts = [
            ctx for ctx in ranked_contexts 
            if ctx.relevance_score >= min_relevance
        ][:max_contexts]
        
        return filtered_contexts
    
    def _retrieve_entity_contexts(self, session_id: str, entities: List[str]) -> List[RetrievedContext]:
        """Retrieve contexts for specific entities"""
        contexts = []
        
        for entity in entities:
            try:
                # Get components containing this entity
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT component_name, source_content, analysis_result_json
                        FROM component_analysis 
                        WHERE session_id = ? AND (
                            source_content LIKE ? OR 
                            analysis_result_json LIKE ?
                        )
                        LIMIT 10
                    ''', (session_id, f'%{entity}%', f'%{entity}%'))
                    
                    for row in cursor.fetchall():
                        component_name, source_content, analysis_json = row
                        
                        if source_content and entity.upper() in source_content.upper():
                            contexts.append(RetrievedContext(
                                source_code=source_content,
                                metadata={
                                    'component_name': component_name,
                                    'component_type': 'PROGRAM',
                                    'entity_match': entity,
                                    'source': 'entity_search'
                                },
                                relevance_score=0.9,
                                retrieval_method='entity_based',
                                component_name=component_name
                            ))
            
            except Exception as e:
                logger.error(f"Error retrieving entity contexts for {entity}: {str(e)}")
        
        return contexts
    
    def _retrieve_by_query_type(self, session_id: str, query_analysis: QueryAnalysis) -> List[RetrievedContext]:
        """Retrieve contexts based on query type"""
        
        if query_analysis.query_type == QueryType.PROGRAM_CALLS:
            return self._retrieve_program_call_contexts(session_id, query_analysis)
        elif query_analysis.query_type == QueryType.FILE_OPERATIONS:
            return self._retrieve_file_operation_contexts(session_id, query_analysis)
        elif query_analysis.query_type == QueryType.DYNAMIC_CALLS:
            return self._retrieve_dynamic_call_contexts(session_id, query_analysis)
        else:
            return self._retrieve_general_contexts(session_id, query_analysis)
    
    def _retrieve_program_call_contexts(self, session_id: str, query_analysis: QueryAnalysis) -> List[RetrievedContext]:
        """Retrieve contexts for program call queries"""
        contexts = []
        
        try:
            # Get components with program calls
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT ca.component_name, ca.source_content
                    FROM component_analysis ca
                    LEFT JOIN dependencies d ON ca.component_name = d.source_component
                    WHERE ca.session_id = ? AND (
                        ca.source_content LIKE '%CALL%' OR
                        ca.source_content LIKE '%XCTL%' OR
                        ca.source_content LIKE '%LINK%' OR
                        d.relationship_type LIKE '%PROGRAM%'
                    )
                    LIMIT 15
                ''', (session_id,))
                
                for component_name, source_content in cursor.fetchall():
                    if source_content:
                        contexts.append(RetrievedContext(
                            source_code=source_content,
                            metadata={
                                'component_name': component_name,
                                'component_type': 'PROGRAM',
                                'source': 'program_calls'
                            },
                            relevance_score=0.8,
                            retrieval_method='program_calls',
                            component_name=component_name
                        ))
        
        except Exception as e:
            logger.error(f"Error retrieving program call contexts: {str(e)}")
        
        return contexts
    
    def _retrieve_dynamic_call_contexts(self, session_id: str, query_analysis: QueryAnalysis) -> List[RetrievedContext]:
        """Retrieve contexts for dynamic call queries"""
        contexts = []
        
        try:
            # Look for dynamic call patterns
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT component_name, source_content
                    FROM component_analysis 
                    WHERE session_id = ? AND (
                        source_content LIKE '%PROGRAM(%' OR
                        source_content LIKE '%EXEC CICS%' OR
                        source_content LIKE '%MOVE%TO%'
                    )
                    LIMIT 10
                ''', (session_id,))
                
                for component_name, source_content in cursor.fetchall():
                    if self._has_dynamic_call_patterns(source_content):
                        contexts.append(RetrievedContext(
                            source_code=source_content,
                            metadata={
                                'component_name': component_name,
                                'component_type': 'PROGRAM',
                                'source': 'dynamic_calls',
                                'analysis_type': 'dynamic_call_logic'
                            },
                            relevance_score=0.95,
                            retrieval_method='dynamic_call_analysis',
                            component_name=component_name
                        ))
        
        except Exception as e:
            logger.error(f"Error retrieving dynamic call contexts: {str(e)}")
        
        return contexts
    
    def _has_dynamic_call_patterns(self, source_code: str) -> bool:
        """Check if source code has dynamic call patterns"""
        source_upper = source_code.upper()
        
        patterns = [
            r'EXEC\s+CICS\s+(?:XCTL|LINK)\s+PROGRAM\s*\(\s*[A-Z0-9\-]+\s*\)',
            r'MOVE\s+.*\s+TO\s+[A-Z0-9\-]+.*(?:XCTL|LINK)',
            r'01\s+[A-Z0-9\-]+.*\n.*05\s+FILLER.*VALUE',
            r'05\s+HOLD-[A-Z0-9\-]+'
        ]
        
        return any(re.search(pattern, source_upper, re.MULTILINE | re.DOTALL) for pattern in patterns)
    
    def _retrieve_file_operation_contexts(self, session_id: str, query_analysis: QueryAnalysis) -> List[RetrievedContext]:
        """Retrieve contexts for file operation queries"""
        contexts = []
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT component_name, source_content
                    FROM component_analysis 
                    WHERE session_id = ? AND (
                        source_content LIKE '%READ%' OR
                        source_content LIKE '%WRITE%' OR
                        source_content LIKE '%CICS%' OR
                        source_content LIKE '%FILE%'
                    )
                    LIMIT 10
                ''', (session_id,))
                
                for component_name, source_content in cursor.fetchall():
                    if source_content:
                        contexts.append(RetrievedContext(
                            source_code=source_content,
                            metadata={
                                'component_name': component_name,
                                'component_type': 'PROGRAM',
                                'source': 'file_operations'
                            },
                            relevance_score=0.75,
                            retrieval_method='file_operations',
                            component_name=component_name
                        ))
        
        except Exception as e:
            logger.error(f"Error retrieving file operation contexts: {str(e)}")
        
        return contexts
    
    def _retrieve_general_contexts(self, session_id: str, query_analysis: QueryAnalysis) -> List[RetrievedContext]:
        """Retrieve general contexts"""
        contexts = []
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT component_name, source_content, component_type
                    FROM component_analysis 
                    WHERE session_id = ? AND source_content IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 8
                ''', (session_id,))
                
                for component_name, source_content, component_type in cursor.fetchall():
                    contexts.append(RetrievedContext(
                        source_code=source_content,
                        metadata={
                            'component_name': component_name,
                            'component_type': component_type or 'PROGRAM',
                            'source': 'general'
                        },
                        relevance_score=0.5,
                        retrieval_method='general',
                        component_name=component_name
                    ))
        
        except Exception as e:
            logger.error(f"Error retrieving general contexts: {str(e)}")
        
        return contexts
    
    def _retrieve_vector_contexts(self, session_id: str, query_analysis: QueryAnalysis) -> List[RetrievedContext]:
        """Retrieve contexts using vector search"""
        if not self.vector_store or not self.vector_store.index_built:
            return []
        
        # Build search query
        search_terms = query_analysis.entities + query_analysis.keywords
        search_query = ' '.join(search_terms)
        
        try:
            results = self.vector_store.semantic_search(search_query, top_k=3)
            
            contexts = []
            for result in results:
                contexts.append(RetrievedContext(
                    source_code=result['source_code'],
                    metadata=result['metadata'],
                    relevance_score=result['similarity'],
                    retrieval_method='vector_search',
                    component_name=result['metadata'].get('component_name', 'Unknown')
                ))
            
            return contexts
        
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
    def _retrieve_dependency_contexts(self, session_id: str, query_analysis: QueryAnalysis) -> List[RetrievedContext]:
        """Retrieve dependency-related contexts"""
        contexts = []
        
        try:
            # Get components with dependencies
            dependencies = self.db_manager.get_dependencies(session_id)
            
            component_names = set()
            for dep in dependencies[:10]:  # Limit to avoid too many
                component_names.add(dep.get('source_component'))
                component_names.add(dep.get('target_component'))
            
            # Get source for these components
            for component_name in list(component_names)[:5]:
                try:
                    source_data = self.db_manager.get_component_source_code(
                        session_id, component_name, max_size=300000
                    )
                    
                    if source_data.get('success') and source_data.get('components'):
                        for comp in source_data['components']:
                            contexts.append(RetrievedContext(
                                source_code=comp.get('source_for_chat', ''),
                                metadata={
                                    'component_name': comp.get('component_name'),
                                    'component_type': comp.get('component_type'),
                                    'source': 'dependencies'
                                },
                                relevance_score=0.6,
                                retrieval_method='dependency_based',
                                component_name=comp.get('component_name', 'Unknown')
                            ))
                
                except Exception as e:
                    logger.warning(f"Error getting source for {component_name}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error retrieving dependency contexts: {str(e)}")
        
        return contexts
    
    def _deduplicate_contexts(self, contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        """Remove duplicate contexts"""
        seen_components = set()
        unique_contexts = []
        
        for context in contexts:
            component_name = context.component_name
            if component_name not in seen_components:
                seen_components.add(component_name)
                unique_contexts.append(context)
        
        return unique_contexts
    
    def _rank_contexts(self, contexts: List[RetrievedContext], 
                      query_analysis: QueryAnalysis) -> List[RetrievedContext]:
        """Rank contexts by relevance"""
        
        # Boost relevance based on query type matching
        for context in contexts:
            if query_analysis.query_type == QueryType.DYNAMIC_CALLS:
                if context.retrieval_method == 'dynamic_call_analysis':
                    context.relevance_score *= 1.3
            elif query_analysis.query_type == QueryType.PROGRAM_CALLS:
                if context.retrieval_method == 'program_calls':
                    context.relevance_score *= 1.2
        
        # Sort by relevance score
        return sorted(contexts, key=lambda x: x.relevance_score, reverse=True)

class SpecializedHandlerRegistry:
    """Registry for specialized query handlers"""
    
    def __init__(self, db_manager, llm_client):
        self.db_manager = db_manager
        self.llm_client = llm_client
        self.handlers = {}
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all specialized handlers"""
        self.handlers[QueryType.DYNAMIC_CALLS] = DynamicCallHandler(self.db_manager, self.llm_client)
        self.handlers[QueryType.PROGRAM_CALLS] = ProgramCallHandler(self.db_manager, self.llm_client)
        self.handlers[QueryType.FILE_OPERATIONS] = FileOperationHandler(self.db_manager, self.llm_client)
        self.handlers[QueryType.FIELD_ANALYSIS] = FieldAnalysisHandler(self.db_manager, self.llm_client)
    
    def get_handler(self, query_type: QueryType):
        """Get handler for query type"""
        return self.handlers.get(query_type)

class DynamicCallHandler:
    """Fixed dynamic call handler with no repetition"""
    
    def __init__(self, db_manager, llm_client):
        self.db_manager = db_manager
        self.llm_client = llm_client
    
    def handle(self, session_id: str, message: str, query_analysis: QueryAnalysis, 
               contexts: List[RetrievedContext]) -> str:
        """Handle dynamic call queries without repetition"""
        
        if not contexts:
            return "No dynamic call patterns found in the analyzed code."
        
        # Process only unique components to avoid repetition
        processed_components = set()
        all_analyses = []
        
        for context in contexts:
            component_name = context.component_name
            
            # Skip if already processed
            if component_name in processed_components:
                continue
            
            processed_components.add(component_name)
            
            # Extract dynamic calls from this component
            dynamic_calls = self._extract_dynamic_calls_concise(
                context.source_code, component_name, query_analysis.entities
            )
            
            if dynamic_calls:
                all_analyses.extend(dynamic_calls)
        
        if not all_analyses:
            return "No dynamic program calls found in the analyzed components."
        
        # Build single comprehensive response
        response_parts = [
            "**Dynamic Program Call Analysis**",
            ""
        ]
        
        # Group by program to avoid repetition
        by_program = {}
        for analysis in all_analyses:
            prog_name = analysis['program']
            if prog_name not in by_program:
                by_program[prog_name] = []
            by_program[prog_name].append(analysis)
        
        for program_name, analyses in by_program.items():
            response_parts.extend([
                f"**Program: {program_name}**",
                ""
            ])
            
            # Combine analyses for this program
            combined_analysis = self._combine_program_analyses(analyses)
            response_parts.extend(combined_analysis)
            response_parts.append("")
        
        return '\n'.join(response_parts)
    
    def _extract_dynamic_calls_concise(self, source_code: str, component_name: str, 
                                     entities: List[str]) -> List[Dict]:
        """Extract dynamic calls with concise analysis"""
        
        dynamic_calls = []
        lines = source_code.split('\n')
        
        # Find CICS dynamic calls
        for i, line in enumerate(lines):
            line_clean = self._extract_program_area(line)
            if not line_clean:
                continue
            
            line_upper = line_clean.upper()
            
            # Look for EXEC CICS with PROGRAM(variable)
            if 'EXEC CICS' in line_upper and 'PROGRAM(' in line_upper:
                program_match = re.search(r'PROGRAM\s*\(\s*([A-Z0-9\-]+)\s*\)', line_upper)
                if program_match:
                    variable = program_match.group(1)
                    
                    # Check if it's a variable (not literal)
                    if not (variable.startswith("'") or variable.startswith('"')):
                        call_type = 'XCTL' if 'XCTL' in line_upper else 'LINK'
                        
                        # Get variable analysis (simplified)
                        var_analysis = self._analyze_variable_simple(lines, variable, i)
                        
                        if var_analysis:
                            dynamic_calls.append({
                                'program': component_name,
                                'call_type': call_type,
                                'variable': variable,
                                'line': i + 1,
                                'analysis': var_analysis
                            })
        
        return dynamic_calls
    
    def _analyze_variable_simple(self, lines: List[str], variable: str, call_line: int) -> Optional[Dict]:
        """Simplified variable analysis"""
        
        var_upper = variable.upper()
        
        # Look for group structure around the call line
        for i in range(max(0, call_line - 50), min(len(lines), call_line + 10)):
            line_clean = self._extract_program_area(lines[i])
            if not line_clean:
                continue
            
            line_upper = line_clean.upper()
            
            # Look for level 01 definition
            if re.match(rf'^\s*01\s+{re.escape(var_upper)}\b', line_upper):
                # Found group definition, extract structure
                group_structure = self._extract_group_structure_simple(lines, i)
                if group_structure:
                    return {
                        'type': 'group_field',
                        'structure': group_structure,
                        'line_definition': i + 1
                    }
        
        return None
    
    def _extract_group_structure_simple(self, lines: List[str], start_line: int) -> Optional[Dict]:
        """Extract simplified group structure"""
        
        structure = {
            'filler_values': [],
            'dynamic_fields': []
        }
        
        # Look ahead for child fields
        for i in range(start_line + 1, min(len(lines), start_line + 20)):
            line_clean = self._extract_program_area(lines[i])
            if not line_clean:
                continue
            
            line_upper = line_clean.upper()
            
            # Stop at next 01 level or division
            if re.match(r'^\s*01\s+', line_upper) or 'DIVISION' in line_upper:
                break
            
            # Look for 05 level fields
            level_match = re.match(r'^\s*05\s+(.+)', line_upper)
            if level_match:
                field_content = level_match.group(1)
                
                if field_content.startswith('FILLER'):
                    # Extract VALUE
                    value_match = re.search(r"VALUE\s+['\"]([^'\"]*)['\"]", field_content)
                    if value_match:
                        structure['filler_values'].append(value_match.group(1))
                else:
                    # Dynamic field
                    name_match = re.match(r'^([A-Z][A-Z0-9\-]*)', field_content)
                    if name_match:
                        field_name = name_match.group(1)
                        structure['dynamic_fields'].append(field_name)
        
        return structure if structure['filler_values'] or structure['dynamic_fields'] else None
    
    def _combine_program_analyses(self, analyses: List[Dict]) -> List[str]:
        """Combine multiple analyses for a single program"""
        
        combined = []
        
        for analysis in analyses:
            call_info = analysis['analysis']
            
            combined.extend([
                f"**{analysis['call_type']} Call (Line {analysis['line']})**",
                f"• Variable: {analysis['variable']}"
            ])
            
            if call_info and call_info['type'] == 'group_field':
                structure = call_info['structure']
                
                if structure['filler_values']:
                    combined.append(f"• Prefix: '{structure['filler_values'][0]}'")
                
                if structure['dynamic_fields']:
                    combined.append(f"• Dynamic part: {', '.join(structure['dynamic_fields'])}")
                
                # Show constructed programs
                if structure['filler_values'] and structure['dynamic_fields']:
                    prefix = structure['filler_values'][0]
                    combined.append("• Constructs programs like:")
                    for field in structure['dynamic_fields'][:2]:  # Limit examples
                        combined.append(f"  - {prefix} + {field} values")
            
            combined.append("")
        
        return combined
    
    def _extract_program_area(self, line: str) -> str:
        """Extract COBOL program area"""
        if not line or len(line) < 8:
            return ""
        
        indicator = line[6] if len(line) > 6 else ' '
        if indicator in ['*', '/', 'C', 'c', 'D', 'd']:
            return ""
        
        if len(line) <= 72:
            return line[7:].rstrip('\n')
        return line[7:72]
    
class ProgramCallHandler:
    """Handler for general program call queries"""
    
    def __init__(self, db_manager, llm_client):
        self.db_manager = db_manager
        self.llm_client = llm_client
    
    def handle(self, session_id: str, message: str, query_analysis: QueryAnalysis, 
               contexts: List[RetrievedContext]) -> str:
        """Handle program call queries"""
        
        response_parts = [
            "**Program Call Analysis**",
            ""
        ]
        
        found_calls = False
        
        for context in contexts:
            call_analysis = self._extract_all_program_calls(
                context.source_code, context.component_name
            )
            
            if call_analysis['has_calls']:
                found_calls = True
                response_parts.extend([
                    f"**Program: {context.component_name}**",
                    ""
                ])
                response_parts.extend(call_analysis['analysis'])
        
        if not found_calls:
            return "No program calls found in the analyzed code."
        
        return '\n'.join(response_parts)
    
    def _extract_all_program_calls(self, source_code: str, component_name: str) -> Dict:
        """Extract all types of program calls"""
        
        lines = source_code.split('\n')
        analysis = []
        has_calls = False
        
        static_calls = []
        dynamic_calls = []
        
        for i, line in enumerate(lines):
            line_clean = self._extract_program_area(line)
            if not line_clean:
                continue
            
            line_upper = line_clean.upper()
            
            # Static CALL statements
            call_match = re.search(r"CALL\s+['\"]([A-Z0-9\-]+)['\"]", line_upper)
            if call_match:
                has_calls = True
                static_calls.append({
                    'type': 'COBOL CALL',
                    'target': call_match.group(1),
                    'line': i + 1,
                    'statement': line.strip()
                })
            
            # Static CICS calls
            cics_static = re.search(r"EXEC\s+CICS\s+(XCTL|LINK)\s+PROGRAM\s*\(\s*['\"]([A-Z0-9\-]+)['\"]\s*\)", line_upper)
            if cics_static:
                has_calls = True
                static_calls.append({
                    'type': f'CICS {cics_static.group(1)}',
                    'target': cics_static.group(2),
                    'line': i + 1,
                    'statement': line.strip()
                })
            
            # Dynamic CICS calls
            cics_dynamic = re.search(r"EXEC\s+CICS\s+(XCTL|LINK)\s+PROGRAM\s*\(\s*([A-Z0-9\-]+)\s*\)", line_upper)
            if cics_dynamic:
                variable = cics_dynamic.group(2)
                if not (variable.startswith("'") or variable.startswith('"')):
                    has_calls = True
                    dynamic_calls.append({
                        'type': f'CICS {cics_dynamic.group(1)} (Dynamic)',
                        'variable': variable,
                        'line': i + 1,
                        'statement': line.strip()
                    })
        
        # Format analysis
        if static_calls:
            analysis.extend([
                "**Static Program Calls:**",
                ""
            ])
            for call in static_calls:
                analysis.append(f"• Line {call['line']}: {call['type']} → **{call['target']}**")
                analysis.append(f"  `{call['statement']}`")
            analysis.append("")
        
        if dynamic_calls:
            analysis.extend([
                "**Dynamic Program Calls:**",
                ""
            ])
            for call in dynamic_calls:
                analysis.append(f"• Line {call['line']}: {call['type']} via **{call['variable']}**")
                analysis.append(f"  `{call['statement']}`")
            analysis.append("")
        
        if static_calls or dynamic_calls:
            total = len(static_calls) + len(dynamic_calls)
            analysis.extend([
                f"**Summary**: {total} program call(s) found",
                f"• Static calls: {len(static_calls)}",
                f"• Dynamic calls: {len(dynamic_calls)}",
                ""
            ])
        
        return {
            'has_calls': has_calls,
            'analysis': analysis
        }
    
    def _extract_program_area(self, line: str) -> str:
        """Extract COBOL program area"""
        if not line or len(line) < 8:
            return ""
        
        indicator = line[6] if len(line) > 6 else ' '
        if indicator in ['*', '/', 'C', 'c', 'D', 'd']:
            return ""
        
        if len(line) <= 72:
            return line[7:].rstrip('\n')
        return line[7:72]

class FileOperationHandler:
    """Handler for file operation queries"""
    
    def __init__(self, db_manager, llm_client):
        self.db_manager = db_manager
        self.llm_client = llm_client
    
    def handle(self, session_id: str, message: str, query_analysis: QueryAnalysis, 
               contexts: List[RetrievedContext]) -> str:
        """Handle file operation queries"""
        
        response_parts = [
            "**File Operations Analysis**",
            ""
        ]
        
        found_operations = False
        
        for context in contexts:
            file_analysis = self._extract_file_operations(
                context.source_code, context.component_name
            )
            
            if file_analysis['has_operations']:
                found_operations = True
                response_parts.extend([
                    f"**Program: {context.component_name}**",
                    ""
                ])
                response_parts.extend(file_analysis['analysis'])
        
        if not found_operations:
            return "No file operations found in the analyzed code."
        
        return '\n'.join(response_parts)
    
    def _extract_file_operations(self, source_code: str, component_name: str) -> Dict:
        """Extract file operations from source code"""
        
        lines = source_code.split('\n')
        analysis = []
        has_operations = False
        
        file_ops = []
        cics_ops = []
        
        for i, line in enumerate(lines):
            line_clean = self._extract_program_area(line)
            if not line_clean:
                continue
            
            line_upper = line_clean.upper()
            
            # Regular file operations
            file_patterns = [
                (r'READ\s+([A-Z][A-Z0-9\-]{2,})', 'READ'),
                (r'WRITE\s+([A-Z][A-Z0-9\-]{2,})', 'WRITE'),
                (r'OPEN\s+(?:INPUT|OUTPUT|I-O)\s+([A-Z][A-Z0-9\-]{2,})', 'OPEN'),
                (r'CLOSE\s+([A-Z][A-Z0-9\-]{2,})', 'CLOSE')
            ]
            
            for pattern, op_type in file_patterns:
                matches = re.findall(pattern, line_upper)
                for file_name in matches:
                    has_operations = True
                    file_ops.append({
                        'operation': op_type,
                        'file': file_name,
                        'line': i + 1,
                        'statement': line.strip()
                    })
            
            # CICS file operations
            cics_pattern = r'EXEC\s+CICS\s+(READ|WRITE|REWRITE|DELETE)\s+.*?(?:FILE|DATASET)\s*\(\s*[\'"]?([A-Z0-9\-]{3,})[\'"]?\s*\)'
            cics_matches = re.findall(cics_pattern, line_upper)
            for op_type, file_name in cics_matches:
                has_operations = True
                cics_ops.append({
                    'operation': f'CICS {op_type}',
                    'file': file_name,
                    'line': i + 1,
                    'statement': line.strip()
                })
        
        # Format analysis
        if file_ops:
            analysis.extend([
                "**File Operations:**",
                ""
            ])
            for op in file_ops:
                analysis.append(f"• Line {op['line']}: {op['operation']} {op['file']}")
            analysis.append("")
        
        if cics_ops:
            analysis.extend([
                "**CICS File Operations:**",
                ""
            ])
            for op in cics_ops:
                analysis.append(f"• Line {op['line']}: {op['operation']} {op['file']}")
            analysis.append("")
        
        if file_ops or cics_ops:
            total = len(file_ops) + len(cics_ops)
            analysis.extend([
                f"**Summary**: {total} file operation(s) found",
                ""
            ])
        
        return {
            'has_operations': has_operations,
            'analysis': analysis
        }
    
    def _extract_program_area(self, line: str) -> str:
        """Extract COBOL program area"""
        if not line or len(line) < 8:
            return ""
        
        indicator = line[6] if len(line) > 6 else ' '
        if indicator in ['*', '/', 'C', 'c', 'D', 'd']:
            return ""
        
        if len(line) <= 72:
            return line[7:].rstrip('\n')
        return line[7:72]

class FieldAnalysisHandler:
    """Handler for field analysis queries"""
    
    def __init__(self, db_manager, llm_client):
        self.db_manager = db_manager
        self.llm_client = llm_client
    
    def handle(self, session_id: str, message: str, query_analysis: QueryAnalysis, 
               contexts: List[RetrievedContext]) -> str:
        """Handle field analysis queries"""
        
        if not query_analysis.entities:
            return "No field names identified in your query."
        
        field_name = query_analysis.entities[0]
        
        try:
            # Get field context from database
            field_context = self.db_manager.get_context_for_field(session_id, field_name)
            field_details = field_context.get('field_details', [])
            
            if not field_details:
                return f"Field '{field_name}' not found in the analyzed code."
            
            detail = field_details[0]
            
            response_parts = [
                f"**Field Analysis: {field_name}**",
                "",
                f"**Definition**: {detail.get('definition_code', 'Not found')}",
                f"**Data Type**: {detail.get('mainframe_data_type', 'Unknown')}",
                f"**Length**: {detail.get('mainframe_length', 'Unknown')} characters",
                f"**Program**: {detail.get('program_name', 'Unknown')}",
                ""
            ]
            
            if detail.get('business_purpose'):
                response_parts.append(f"**Business Purpose**: {detail['business_purpose']}")
                response_parts.append("")
            
            # Usage information
            ref_count = detail.get('total_program_references', 0)
            if ref_count > 0:
                response_parts.append(f"**Usage**: Referenced {ref_count} times in the program")
            
            return '\n'.join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in field analysis: {str(e)}")
            return f"Error analyzing field '{field_name}': {str(e)}"

class ResponseGenerator:
    """Enhanced response generation with source code optimization"""
    
    def __init__(self, llm_client, handler_registry):
        self.llm_client = llm_client
        self.handler_registry = handler_registry
    
    def generate_response(self, session_id: str, message: str, query_analysis: QueryAnalysis, 
                         contexts: List[RetrievedContext]) -> str:
        """Generate response with optimized source code handling"""
        
        # Route to specialized handler if needed
        if query_analysis.requires_specialized_handler:
            handler = self.handler_registry.get_handler(query_analysis.query_type)
            if handler:
                try:
                    return handler.handle(session_id, message, query_analysis, contexts)
                except Exception as e:
                    logger.error(f"Specialized handler failed: {str(e)}")
        
        # General LLM-based response generation with optimized context
        if not contexts:
            return self._generate_no_context_response(message, query_analysis)
        
        # FIXED: Optimize contexts before sending to LLM
        optimized_contexts = self._optimize_contexts_for_llm(contexts, query_analysis)
        
        prompt = self._build_optimized_prompt(message, query_analysis, optimized_contexts)
        
        response = self.llm_client.call_llm(
            prompt, 
            max_tokens=self._get_max_tokens(query_analysis.complexity),
            temperature=self._get_temperature(query_analysis.query_type)
        )
        
        if response.success:
            return self._post_process_response(response.content, query_analysis, contexts)
        else:
            return f"Error generating response: {response.error_message}"
    
    def _optimize_contexts_for_llm(self, contexts: List[RetrievedContext], 
                                  query_analysis: QueryAnalysis) -> List[Dict]:
        """Optimize contexts to reduce source code size and avoid repetition"""
        
        optimized = []
        seen_components = set()
        total_chars = 0
        max_total_chars = 3000  # REDUCED: Maximum total characters for all contexts
        max_per_context = 800   # REDUCED: Maximum per context
        
        for context in contexts:
            component_name = context.component_name
            
            # Skip if we've already seen this component
            if component_name in seen_components:
                continue
            
            seen_components.add(component_name)
            
            # Extract relevant snippets instead of full source
            relevant_snippets = self._extract_relevant_snippets(
                context.source_code, query_analysis, max_per_context
            )
            
            if not relevant_snippets:
                continue
            
            snippet_text = '\n'.join(relevant_snippets)
            
            # Check if adding this would exceed total limit
            if total_chars + len(snippet_text) > max_total_chars:
                # Take only what fits
                remaining_chars = max_total_chars - total_chars
                if remaining_chars > 200:  # Only if meaningful amount left
                    snippet_text = snippet_text[:remaining_chars] + "..."
                else:
                    break
            
            optimized.append({
                'component_name': component_name,
                'source_snippet': snippet_text,
                'relevance_score': context.relevance_score,
                'retrieval_method': context.retrieval_method,
                'snippet_count': len(relevant_snippets)
            })
            
            total_chars += len(snippet_text)
            
            # Stop if we've reached limit
            if total_chars >= max_total_chars:
                break
        
        logger.info(f"Optimized {len(contexts)} contexts to {len(optimized)} contexts, "
                   f"total chars: {total_chars}")
        
        return optimized
    
    def _extract_relevant_snippets(self, source_code: str, query_analysis: QueryAnalysis, 
                                  max_chars: int) -> List[str]:
        """Extract relevant code snippets based on query"""
        
        if not source_code:
            return []
        
        lines = source_code.split('\n')
        snippets = []
        current_snippet = []
        snippet_chars = 0
        
        # Keywords to look for based on query type
        keywords = self._get_search_keywords(query_analysis)
        
        # Entity-specific search
        entities = [e.upper() for e in query_analysis.entities]
        
        i = 0
        while i < len(lines) and snippet_chars < max_chars:
            line = lines[i]
            line_clean = self._extract_program_area(line)
            if not line_clean:
                i += 1
                continue
            
            line_upper = line_clean.upper()
            
            # Check if line contains relevant content
            is_relevant = (
                any(keyword in line_upper for keyword in keywords) or
                any(entity in line_upper for entity in entities) or
                self._is_structurally_important(line_upper)
            )
            
            if is_relevant:
                # Collect context around this line
                context_start = max(0, i - 2)
                context_end = min(len(lines), i + 3)
                
                context_lines = []
                for j in range(context_start, context_end):
                    context_line = self._extract_program_area(lines[j])
                    if context_line:
                        context_lines.append(f"{j+1:4d}: {context_line}")
                
                snippet_text = '\n'.join(context_lines)
                
                # Check if adding this snippet would exceed limit
                if snippet_chars + len(snippet_text) + 50 > max_chars:
                    break
                
                snippets.append(f"--- Lines {context_start+1}-{context_end} ---")
                snippets.append(snippet_text)
                snippets.append("")  # Separator
                
                snippet_chars += len(snippet_text) + 50
                
                # Skip ahead to avoid overlapping contexts
                i = context_end
            else:
                i += 1
        
        return snippets
    
    def _get_search_keywords(self, query_analysis: QueryAnalysis) -> List[str]:
        """Get keywords to search for based on query type"""
        
        keyword_map = {
            QueryType.PROGRAM_CALLS: ['CALL', 'XCTL', 'LINK', 'EXEC CICS'],
            QueryType.DYNAMIC_CALLS: ['PROGRAM(', 'MOVE', 'TO', 'FILLER', 'VALUE'],
            QueryType.FILE_OPERATIONS: ['READ', 'WRITE', 'CICS', 'DATASET', 'FILE'],
            QueryType.FIELD_ANALYSIS: ['PIC', 'VALUE', 'MOVE', 'LEVEL'],
            QueryType.BUSINESS_LOGIC: ['IF', 'COMPUTE', 'PERFORM', 'EVALUATE']
        }
        
        base_keywords = keyword_map.get(query_analysis.query_type, [])
        base_keywords.extend(query_analysis.keywords)
        
        return [kw.upper() for kw in base_keywords]
    
    def _is_structurally_important(self, line: str) -> bool:
        """Check if line is structurally important (field definitions, etc.)"""
        
        # Level numbers (01, 05, etc.)
        if re.match(r'^\s*\d{2}\s+', line):
            return True
        
        # Division/section headers
        if any(word in line for word in ['DIVISION', 'SECTION']):
            return True
        
        # Important COBOL constructs
        important_constructs = [
            'WORKING-STORAGE', 'LINKAGE', 'PROCEDURE', 
            'FD ', 'SELECT', 'COPY'
        ]
        
        return any(construct in line for construct in important_constructs)
    
    def _extract_program_area(self, line: str) -> str:
        """Extract COBOL program area (columns 8-72)"""
        if not line or len(line) < 8:
            return ""
        
        indicator = line[6] if len(line) > 6 else ' '
        if indicator in ['*', '/', 'C', 'c', 'D', 'd']:
            return ""
        
        if len(line) <= 72:
            return line[7:].rstrip('\n')
        return line[7:72]
    
    def _build_optimized_prompt(self, message: str, query_analysis: QueryAnalysis, 
                               optimized_contexts: List[Dict]) -> str:
        """Build optimized prompt with reduced source code"""
        
        prompt_parts = [
            "You are an expert COBOL analyst for a wealth management system.",
            f"Query Type: {query_analysis.query_type.value}",
            f"Entities: {', '.join(query_analysis.entities) if query_analysis.entities else 'None'}",
            "",
            f"USER QUESTION: {message}",
            ""
        ]
        
        # Add concise type-specific instructions
        instruction_map = {
            QueryType.FIELD_ANALYSIS: "Analyze the field definition, structure, and business usage.",
            QueryType.PROGRAM_CALLS: "Identify all program calls (static and dynamic) with their purposes.",
            QueryType.DYNAMIC_CALLS: "Explain the dynamic call logic and how program names are constructed.",
            QueryType.FILE_OPERATIONS: "Describe file operations and data flow patterns.",
            QueryType.BUSINESS_LOGIC: "Explain the business rules and processing logic."
        }
        
        instruction = instruction_map.get(query_analysis.query_type, 
                                        "Provide a comprehensive analysis of the code.")
        prompt_parts.extend([instruction, ""])
        
        # Add optimized contexts
        prompt_parts.append("RELEVANT CODE SNIPPETS:")
        prompt_parts.append("")
        
        for i, context in enumerate(optimized_contexts, 1):
            prompt_parts.extend([
                f"=== CONTEXT {i}: {context['component_name']} ===",
                f"Method: {context['retrieval_method']} | Relevance: {context['relevance_score']:.2f}",
                f"Snippets: {context['snippet_count']}",
                "",
                context['source_snippet'],
                "",
                "=" * 50,
                ""
            ])
        
        prompt_parts.extend([
            "Provide a direct, focused answer. Reference specific line numbers when possible.",
            "Avoid repeating information. Focus on what directly answers the user's question."
        ])
        
        return '\n'.join(prompt_parts)
        
    def _build_general_prompt(self, message: str, query_analysis: QueryAnalysis, 
                             contexts: List[RetrievedContext]) -> str:
        """Build prompt for general LLM response"""
        
        prompt_parts = [
            "You are an expert COBOL analyst for a wealth management system.",
            f"Query Type: {query_analysis.query_type.value}",
            f"Complexity: {query_analysis.complexity.value}",
            f"Entities: {', '.join(query_analysis.entities) if query_analysis.entities else 'None'}",
            "",
            f"USER QUESTION: {message}",
            ""
        ]
        
        # Add type-specific instructions
        if query_analysis.query_type == QueryType.FIELD_ANALYSIS:
            prompt_parts.extend([
                "Focus on field definitions, usage, and business purpose.",
                "Reference specific line numbers when possible.",
                ""
            ])
        elif query_analysis.query_type == QueryType.PROGRAM_CALLS:
            prompt_parts.extend([
                "Analyze program call patterns and relationships.",
                "Distinguish between static and dynamic calls.",
                ""
            ])
        
        # Add contexts
        prompt_parts.append("RELEVANT SOURCE CODE:")
        prompt_parts.append("")
        
        for i, context in enumerate(contexts, 1):
            prompt_parts.extend([
                f"=== CONTEXT {i}: {context.component_name} ===",
                f"Retrieval Method: {context.retrieval_method}",
                f"Relevance: {context.relevance_score:.2f}",
                "",
                context.source_code[:2000],  # Limit source code length
                "",
                "=" * 60,
                ""
            ])
        
        prompt_parts.extend([
            "Provide a comprehensive, accurate response that directly answers the question.",
            "Use specific code references when possible.",
            "Focus on business meaning and practical implications."
        ])
        
        return '\n'.join(prompt_parts)
    
    def _get_max_tokens(self, complexity: AnalysisComplexity) -> int:
        """Get max tokens based on complexity"""
        complexity_tokens = {
            AnalysisComplexity.SIMPLE: 800,
            AnalysisComplexity.ENHANCED: 1200,
            AnalysisComplexity.COMPREHENSIVE: 1800
        }
        return complexity_tokens.get(complexity, 1000)
    
    def _get_temperature(self, query_type: QueryType) -> float:
        """Get temperature based on query type"""
        if query_type in [QueryType.FIELD_ANALYSIS, QueryType.PROGRAM_CALLS]:
            return 0.1  # More factual
        else:
            return 0.3  # More creative
    
    def _post_process_response(self, response: str, query_analysis: QueryAnalysis, 
                              contexts: List[RetrievedContext]) -> str:
        """Post-process the response"""
        
        # Add confidence indicator
        if query_analysis.confidence < 0.6:
            response += f"\n\n*Analysis confidence: {query_analysis.confidence:.0%}*"
        
        # Add context summary
        response += f"\n\n*Based on {len(contexts)} code context(s)*"
        
        return response
    
    def _generate_no_context_response(self, message: str, query_analysis: QueryAnalysis) -> str:
        """Generate response when no contexts found"""
        return (
            f"I couldn't find specific code contexts for your question: {message}\n\n"
            f"This might be because:\n"
            f"- The mentioned components haven't been analyzed yet\n"
            f"- The code doesn't contain the specific elements you're asking about\n"
            f"- More context is needed to understand your question\n\n"
            f"Try providing more specific field names, program names, or uploading additional COBOL files."
        )

class RewrittenAgenticRAGChatManager:
    """Complete rewritten Agentic RAG Chat Manager with clean architecture"""
    
    def __init__(self, llm_client, db_manager, vector_store=None):
        self.llm_client = llm_client
        self.db_manager = db_manager
        self.vector_store = vector_store
        
        # Initialize components
        self.query_classifier = QueryClassifier()
        self.context_retriever = ContextRetriever(db_manager, vector_store)
        self.handler_registry = SpecializedHandlerRegistry(db_manager, llm_client)
        self.response_generator = ResponseGenerator(llm_client, self.handler_registry)
        
        # Performance tracking
        self.performance_metrics = {
            'queries_processed': 0,
            'avg_response_time': 0,
            'context_hit_rate': 0,
            'specialized_handler_usage': 0
        }
        
        self._initialized_sessions = set()
        logger.info("Rewritten Agentic RAG Chat Manager initialized")
    
    def process_query(self, session_id: str, message: str, conversation_id: str) -> str:
        """Main query processing method"""
        start_time = time.time()
        
        try:
            # Initialize session if needed
            if session_id not in self._initialized_sessions:
                self._initialize_session(session_id)
            
            # Extract entities
            entities = self._extract_entities(message)
            
            # Classify query
            query_analysis = self.query_classifier.classify_query(message, entities)
            logger.info(f"Query classified as: {query_analysis.query_type.value} "
                       f"(complexity: {query_analysis.complexity.value}, confidence: {query_analysis.confidence:.2f})")
            
            # Retrieve contexts
            contexts = self.context_retriever.retrieve_contexts(session_id, query_analysis)
            logger.info(f"Retrieved {len(contexts)} contexts")
            
            # Generate response
            response = self.response_generator.generate_response(
                session_id, message, query_analysis, contexts
            )
            
            # Update metrics and log
            processing_time = time.time() - start_time
            self._update_metrics(query_analysis, contexts, processing_time)
            self._log_conversation(session_id, conversation_id, message, response, processing_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"I encountered an error processing your request: {str(e)}"
    
    def _initialize_session(self, session_id: str):
        """Initialize session components"""
        try:
            # Build vector index if available
            if self.vector_store:
                self.vector_store.build_index(session_id)
            
            self._initialized_sessions.add(session_id)
            logger.info(f"Session {session_id} initialized")
            
        except Exception as e:
            logger.error(f"Error initializing session: {str(e)}")
    
    def _extract_entities(self, message: str) -> List[str]:
        """Extract COBOL entities from message"""
        entities = []
        
        # Pattern for COBOL identifiers
        patterns = [
            r'\b([A-Z][A-Z0-9\-]{2,20})\b',
            r'field\s+([A-Za-z][A-Za-z0-9\-_]{2,})',
            r'what\s+is\s+([A-Z][A-Z0-9\-_]{2,})',
            r'tell\s+me\s+about\s+([A-Z][A-Z0-9\-_]{2,})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                clean_name = match.upper().replace('_', '-')
                if self._is_valid_cobol_entity(clean_name):
                    entities.append(clean_name)
        
        return list(set(entities))
    
    def _is_valid_cobol_entity(self, name: str) -> bool:
        """Validate COBOL entity name"""
        if len(name) < 3 or len(name) > 30:
            return False
        
        # Must start with letter
        if not re.match(r'^[A-Z]', name):
            return False
        
        # Valid characters
        if not re.match(r'^[A-Z0-9\-]+', name):
            return False
        
        # Exclude common English words and COBOL keywords
        excluded = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'WHAT', 'WHERE', 'HOW', 'WHEN',
                   'PIC', 'PICTURE', 'VALUE', 'USAGE', 'OCCURS', 'MOVE', 'COMPUTE'}
        
        return name not in excluded
    
    def _update_metrics(self, query_analysis: QueryAnalysis, contexts: List[RetrievedContext], 
                       processing_time: float):
        """Update performance metrics"""
        self.performance_metrics['queries_processed'] += 1
        
        # Update average response time
        current_avg = self.performance_metrics['avg_response_time']
        new_avg = ((current_avg * (self.performance_metrics['queries_processed'] - 1)) + 
                  processing_time) / self.performance_metrics['queries_processed']
        self.performance_metrics['avg_response_time'] = new_avg
        
        # Update context hit rate
        context_found = len(contexts) > 0
        hit_rate = self.performance_metrics['context_hit_rate']
        new_hit_rate = ((hit_rate * (self.performance_metrics['queries_processed'] - 1)) + 
                       (1 if context_found else 0)) / self.performance_metrics['queries_processed']
        self.performance_metrics['context_hit_rate'] = new_hit_rate
        
        # Track specialized handler usage
        if query_analysis.requires_specialized_handler:
            self.performance_metrics['specialized_handler_usage'] += 1
    
    def _log_conversation(self, session_id: str, conversation_id: str, message: str, 
                         response: str, processing_time: float):
        """Log conversation"""
        try:
            self.db_manager.store_chat_message(
                session_id, conversation_id, 'user', message,
                tokens_used=0, processing_time_ms=int(processing_time * 1000)
            )
            
            self.db_manager.store_chat_message(
                session_id, conversation_id, 'assistant', response,
                tokens_used=0, processing_time_ms=int(processing_time * 1000)
            )
            
        except Exception as e:
            logger.error(f"Error logging conversation: {str(e)}")
    
    def get_system_health(self) -> Dict:
        """Get system health information"""
        return {
            'system_type': 'rewritten_agentic_rag',
            'status': 'healthy',
            'vector_search_available': VECTOR_SEARCH_AVAILABLE,
            'initialized_sessions': len(self._initialized_sessions),
            'performance_metrics': self.performance_metrics,
            'features': [
                'Enhanced Query Classification',
                'Specialized Handler Registry', 
                'Multi-Strategy Context Retrieval',
                'Business Logic Analysis',
                'Dynamic Call Analysis',
                'Performance Monitoring'
            ]
        }

# Factory function for creating the rewritten RAG manager
def create_rewritten_agentic_rag_manager(llm_client, db_manager, vector_store=None):
    """Factory function to create the rewritten Agentic RAG Chat Manager"""
    return RewrittenAgenticRAGChatManager(llm_client, db_manager, vector_store)

# Utility class for vector store (keeping existing functionality)
class VectorStore:
    """Vector store for semantic search (compatible with existing implementation)"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.vectorizer = None
        self.vectors = None
        self.documents = []
        self.metadata = []
        self.index_built = False
        
        if VECTOR_SEARCH_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.8,
                min_df=2
            )
    
    def build_index(self, session_id: str):
        """Build vector index from session components"""
        try:
            components = self.db_manager.get_session_components(session_id)
            documents = []
            metadata = []
            
            for comp in components:
                try:
                    source_data = self.db_manager.get_component_source_code(
                        session_id, comp['component_name'], max_size=500000
                    )
                    
                    if source_data.get('success') and source_data.get('components'):
                        for source_comp in source_data['components']:
                            source_code = source_comp.get('source_for_chat', '')
                            if len(source_code) > 100:
                                processed_code = self._preprocess_code(source_code)
                                documents.append(processed_code)
                                metadata.append({
                                    'component_name': source_comp.get('component_name'),
                                    'component_type': source_comp.get('component_type'),
                                    'total_lines': source_comp.get('total_lines', 0),
                                    'original_source': source_code
                                })
                except Exception as e:
                    logger.error(f"Error processing component {comp['component_name']}: {e}")
                    continue
            
            if documents and VECTOR_SEARCH_AVAILABLE:
                self.vectors = self.vectorizer.fit_transform(documents)
                self.documents = documents
                self.metadata = metadata
                self.index_built = True
                logger.info(f"Built vector index with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error building vector index: {str(e)}")
    
    def semantic_search(self, query: str, top_k: int = 5, min_similarity: float = 0.1) -> List[Dict]:
        """Perform semantic search"""
        if not self.index_built:
            return []
        
        try:
            processed_query = self._preprocess_code(query)
            
            if VECTOR_SEARCH_AVAILABLE and self.vectors is not None:
                query_vector = self.vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.vectors)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                results = []
                for idx in top_indices:
                    similarity = similarities[idx]
                    if similarity >= min_similarity:
                        results.append({
                            'similarity': float(similarity),
                            'metadata': self.metadata[idx],
                            'document': self.documents[idx],
                            'source_code': self.metadata[idx]['original_source']
                        })
                return results
            else:
                return self._keyword_search(query, top_k)
                
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def _preprocess_code(self, source_code: str) -> str:
        """Preprocess COBOL code for better search"""
        lines = source_code.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            
            line = line.upper()
            processed_lines.append(line)
        
        return ' '.join(processed_lines)
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback keyword search"""
        keywords = query.upper().split()
        results = []
        
        for i, doc in enumerate(self.documents):
            score = 0.0
            doc_upper = doc.upper()
            
            for keyword in keywords:
                if keyword in doc_upper:
                    score += doc_upper.count(keyword) / len(doc_upper)
            
            if score > 0:
                results.append({
                    'similarity': score,
                    'metadata': self.metadata[i],
                    'document': doc,
                    'source_code': self.metadata[i]['original_source']
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]