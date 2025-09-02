"""
Complete Agentic RAG Chat Manager with Vector Store and Caching
Integrates all RAG capabilities with fallback to basic chat manager
"""

import re
import json
import logging
import traceback
import time
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

# Attempt to import vector search dependencies
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    logging.warning("Vector search libraries not available - falling back to keyword search")

logger = logging.getLogger(__name__)

# Query Analysis Classes
class QueryType(Enum):
    FIELD_ANALYSIS = "field_analysis"
    PROGRAM_STRUCTURE = "program_structure"
    BUSINESS_LOGIC = "business_logic"
    DATA_FLOW = "data_flow"
    DEPENDENCIES = "dependencies"
    CODE_SEARCH = "code_search"
    GENERAL = "general"

class RetrievalStrategy(Enum):
    PRECISE_FIELD = "precise_field"
    SEMANTIC_CODE = "semantic_code"
    STRUCTURAL = "structural"
    DEPENDENCY = "dependency"
    BUSINESS_CONTEXT = "business_context"
    VECTOR_SEARCH = "vector_search"

@dataclass
class QueryPlan:
    query_type: QueryType
    entities: List[str]
    retrieval_strategies: List[RetrievalStrategy]
    context_requirements: Dict[str, Any]
    confidence: float

@dataclass
class RetrievedContext:
    source_code: str
    metadata: Dict[str, Any]
    relevance_score: float
    retrieval_method: str

class VectorStore:
    """Vector store for semantic search of COBOL code"""
    
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
            logger.info("Vector store initialized with TF-IDF")
        else:
            logger.warning("Vector store initialized without vector search capabilities")
    
    def build_index(self, session_id: str):
        """Build vector index from all source code in session"""
        try:
            logger.info("Building vector index for semantic search...")
            
            # Get all components with source code
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
                                processed_code = self._preprocess_cobol_code(source_code)
                                documents.append(processed_code)
                                metadata.append({
                                    'component_name': source_comp.get('component_name'),
                                    'component_type': source_comp.get('component_type'),
                                    'source_strategy': source_comp.get('source_strategy'),
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
            else:
                # Fallback to simple keyword indexing
                self.documents = documents
                self.metadata = metadata
                self.index_built = True
                logger.info(f"Built keyword index with {len(documents)} documents")
                
        except Exception as e:
            logger.error(f"Error building vector index: {str(e)}")
    
    def _preprocess_cobol_code(self, source_code: str) -> str:
        """Preprocess COBOL code for better semantic search"""
        lines = source_code.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            
            line = line.upper()
            
            # Normalize COBOL constructs
            replacements = {
                'PIC X(': 'TEXT_FIELD ',
                'PIC 9(': 'NUMERIC_FIELD ',
                'PIC S9(': 'SIGNED_NUMERIC_FIELD ',
                'MOVE ': 'ASSIGN ',
                'COMPUTE ': 'CALCULATE ',
                'EXEC CICS': 'TRANSACTION ',
                'CALL ': 'INVOKE ',
                'PERFORM ': 'EXECUTE ',
            }
            
            for old, new in replacements.items():
                line = line.replace(old, new)
            
            processed_lines.append(line)
        
        return ' '.join(processed_lines)
    
    def semantic_search(self, query: str, top_k: int = 5, min_similarity: float = 0.1) -> List[Dict]:
        """Perform semantic search on the code base"""
        if not self.index_built:
            return []
        
        try:
            processed_query = self._preprocess_cobol_code(query)
            
            if VECTOR_SEARCH_AVAILABLE and self.vectors is not None:
                # Use TF-IDF vector similarity
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
                # Fallback to keyword matching
                return self._keyword_search(query, top_k)
                
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback keyword-based search"""
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

class QueryAnalyzer:
    """Analyzes user queries to determine retrieval strategy"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
        # Field name extraction patterns
        self.field_patterns = [
            r'\b([A-Z][A-Z0-9\-]{2,20})\b',
            r'field\s+([A-Za-z][A-Za-z0-9\-_]{2,})',
            r'about\s+([A-Z][A-Z0-9\-_]{2,})',
            r'([A-Z][A-Z0-9\-_]{3,})\s+field',
            r'tell\s+me\s+about\s+([A-Z][A-Z0-9\-_]{2,})',
            r'what\s+is\s+([A-Z][A-Z0-9\-_]{2,})',
            r'how\s+is\s+([A-Z][A-Z0-9\-_]{2,})',
            r'show\s+([A-Z][A-Z0-9\-_]{2,})'
        ]
        
        self.cobol_keywords = {
            'MOVE', 'TO', 'FROM', 'PIC', 'PICTURE', 'VALUE', 'OCCURS',
            'USAGE', 'COMP', 'BINARY', 'DISPLAY', 'COMPUTE', 'ADD',
            'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'IF', 'THEN', 'ELSE',
            'PERFORM', 'UNTIL', 'VARYING', 'WHEN', 'EVALUATE'
        }
    
    def analyze_query(self, message: str, session_id: str) -> QueryPlan:
        """Analyze user query and determine retrieval strategy"""
        entities = self._extract_entities(message)
        query_type = self._classify_query_type(message, entities)
        retrieval_strategies = self._determine_retrieval_strategies(query_type, message, entities)
        context_requirements = self._determine_context_requirements(query_type, message, entities)
        confidence = self._calculate_confidence(query_type, entities, message)
        
        return QueryPlan(
            query_type=query_type,
            entities=entities,
            retrieval_strategies=retrieval_strategies,
            context_requirements=context_requirements,
            confidence=confidence
        )
    
    def _extract_entities(self, message: str) -> List[str]:
        """Extract COBOL entities from message"""
        entities = set()
        
        for pattern in self.field_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                clean_name = match.upper().replace('_', '-')
                if self._is_likely_cobol_entity(clean_name):
                    entities.add(clean_name)
        
        return list(entities)
    
    def _is_likely_cobol_entity(self, name: str) -> bool:
        """Determine if a name is likely a COBOL entity"""
        if len(name) < 3 or len(name) > 30:
            return False
        
        english_words = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'WHAT', 'WHERE', 'HOW', 'WHEN'}
        if name in english_words or name in self.cobol_keywords:
            return False
        
        return (
            '-' in name or
            len(name) >= 6 or
            re.match(r'^[A-Z]{2,}[0-9]', name) or
            any(pattern in name for pattern in ['CUST', 'ACCT', 'TXN', 'BAL', 'FEE'])
        )
    
    def _classify_query_type(self, message: str, entities: List[str]) -> QueryType:
        """Enhanced query classification"""
        message_lower = message.lower()
        
        # Field analysis patterns
        if any(pattern in message_lower for pattern in [
            'field', 'what is', 'tell me about', 'how is', 'where is', 'show me'
        ]) and entities and not any(prog_word in message_lower for prog_word in ['call', 'program', 'file']):
            return QueryType.FIELD_ANALYSIS
        
        # Program structure patterns
        elif any(pattern in message_lower for pattern in [
            'program structure', 'layout', 'record', 'data structure', 'working storage'
        ]):
            return QueryType.PROGRAM_STRUCTURE
            
        # Business logic patterns
        elif any(pattern in message_lower for pattern in [
            'business logic', 'calculation', 'compute', 'rule', 'process', 'workflow', 'how does'
        ]):
            return QueryType.BUSINESS_LOGIC
            
        # Data flow patterns
        elif any(pattern in message_lower for pattern in [
            'data flow', 'move', 'copy', 'transfer', 'populate', 'source', 'target'
        ]):
            return QueryType.DATA_FLOW
            
        # Dependencies patterns (enhanced)
        elif any(pattern in message_lower for pattern in [
            'call', 'calls', 'program call', 'depend', 'link', 'relationship', 'connect',
            'file', 'read', 'write', 'cics', 'uses', 'accesses'
        ]):
            return QueryType.DEPENDENCIES
            
        # Code search patterns
        elif any(pattern in message_lower for pattern in [
            'find', 'search', 'locate', 'contains'
        ]):
            return QueryType.CODE_SEARCH
            
        else:
            return QueryType.GENERAL
    
    def _determine_retrieval_strategies(self, query_type: QueryType, message: str, entities: List[str]) -> List[RetrievalStrategy]:
        """Determine which retrieval strategies to use"""
        strategies = []
        
        if query_type == QueryType.FIELD_ANALYSIS:
            strategies = [RetrievalStrategy.PRECISE_FIELD, RetrievalStrategy.SEMANTIC_CODE]
        elif query_type == QueryType.PROGRAM_STRUCTURE:
            strategies = [RetrievalStrategy.STRUCTURAL, RetrievalStrategy.SEMANTIC_CODE]
        elif query_type == QueryType.BUSINESS_LOGIC:
            strategies = [RetrievalStrategy.SEMANTIC_CODE, RetrievalStrategy.BUSINESS_CONTEXT]
        elif query_type == QueryType.DATA_FLOW:
            strategies = [RetrievalStrategy.SEMANTIC_CODE, RetrievalStrategy.DEPENDENCY]
        elif query_type == QueryType.DEPENDENCIES:
            strategies = [RetrievalStrategy.DEPENDENCY, RetrievalStrategy.STRUCTURAL]
        elif query_type == QueryType.CODE_SEARCH:
            strategies = [RetrievalStrategy.VECTOR_SEARCH, RetrievalStrategy.SEMANTIC_CODE]
        else:  # GENERAL
            strategies = [RetrievalStrategy.BUSINESS_CONTEXT, RetrievalStrategy.STRUCTURAL]
        
        # Always add vector search if available and relevant
        if VECTOR_SEARCH_AVAILABLE and RetrievalStrategy.VECTOR_SEARCH not in strategies:
            if any(word in message.lower() for word in ['find', 'search', 'how', 'calculate']):
                strategies.insert(0, RetrievalStrategy.VECTOR_SEARCH)
        
        return strategies
    
    def _determine_context_requirements(self, query_type: QueryType, message: str, entities: List[str]) -> Dict[str, Any]:
        """Determine what context is needed"""
        requirements = {
            'source_code_needed': True,
            'metadata_needed': True,
            'max_contexts': 3,
            'min_relevance': 0.3
        }
        
        if query_type == QueryType.FIELD_ANALYSIS:
            requirements.update({
                'field_definitions_needed': True,
                'field_usage_needed': True,
                'max_contexts': 2,
                'min_relevance': 0.5
            })
        elif query_type == QueryType.PROGRAM_STRUCTURE:
            requirements.update({
                'record_layouts_needed': True,
                'divisions_needed': True,
                'max_contexts': 3,
                'min_relevance': 0.4
            })
        elif query_type == QueryType.BUSINESS_LOGIC:
            requirements.update({
                'procedure_division_needed': True,
                'calculations_needed': True,
                'max_contexts': 4,
                'min_relevance': 0.3
            })
        
        return requirements
    
    def _calculate_confidence(self, query_type: QueryType, entities: List[str], message: str) -> float:
        """Calculate confidence in the query analysis"""
        confidence = 0.5
        
        if entities:
            confidence += 0.2 * min(len(entities), 3)
        
        if query_type != QueryType.GENERAL:
            confidence += 0.2
        
        if len(message.split()) > 5:
            confidence += 0.1
            
        return min(confidence, 1.0)

class ContextRetriever:
    """Retrieves relevant context based on query plan"""
    
    def __init__(self, db_manager, vector_store=None):
        self.db_manager = db_manager
        self.vector_store = vector_store
    
    def retrieve_contexts(self, session_id: str, query_plan: QueryPlan) -> List[RetrievedContext]:
        """Execute retrieval strategies to gather context"""
        all_contexts = []
        
        for strategy in query_plan.retrieval_strategies:
            try:
                contexts = self._execute_strategy(session_id, strategy, query_plan)
                all_contexts.extend(contexts)
            except Exception as e:
                logger.error(f"Error executing strategy {strategy}: {str(e)}")
                continue
        
        # Deduplicate and rank contexts
        unique_contexts = self._deduplicate_contexts(all_contexts)
        ranked_contexts = self._rank_contexts(unique_contexts, query_plan)
        
        max_contexts = query_plan.context_requirements.get('max_contexts', 3)
        min_relevance = query_plan.context_requirements.get('min_relevance', 0.3)
        
        filtered_contexts = [
            ctx for ctx in ranked_contexts 
            if ctx.relevance_score >= min_relevance
        ][:max_contexts]
        
        return filtered_contexts
    
    def _execute_strategy(self, session_id: str, strategy: RetrievalStrategy, query_plan: QueryPlan) -> List[RetrievedContext]:
        """Enhanced strategy execution with comprehensive support"""
        contexts = []
        
        if strategy == RetrievalStrategy.PRECISE_FIELD:
            contexts = self._retrieve_field_contexts(session_id, query_plan.entities)
        elif strategy == RetrievalStrategy.SEMANTIC_CODE:
            contexts = self._retrieve_semantic_code_contexts(session_id, query_plan)
        elif strategy == RetrievalStrategy.STRUCTURAL:
            contexts = self._retrieve_structural_contexts(session_id, query_plan)
        elif strategy == RetrievalStrategy.DEPENDENCY:
            contexts = self._retrieve_dependency_contexts(session_id, query_plan)
        elif strategy == RetrievalStrategy.BUSINESS_CONTEXT:
            contexts = self._retrieve_business_contexts(session_id, query_plan)
        elif strategy == RetrievalStrategy.VECTOR_SEARCH:
            contexts = self._retrieve_vector_contexts(session_id, query_plan)
        
        return contexts
    
    def _retrieve_vector_contexts(self, session_id: str, query_plan: QueryPlan) -> List[RetrievedContext]:
        """Use vector store for semantic search"""
        contexts = []
        
        if not self.vector_store or not self.vector_store.index_built:
            return contexts
        
        try:
            # Build search query from entities and query type
            search_terms = list(query_plan.entities)
            
            type_terms = {
                QueryType.FIELD_ANALYSIS: ['field', 'picture', 'definition', 'data'],
                QueryType.BUSINESS_LOGIC: ['compute', 'calculate', 'if', 'logic', 'rule'],
                QueryType.DATA_FLOW: ['move', 'copy', 'transfer', 'read', 'write'],
                QueryType.DEPENDENCIES: ['call', 'invoke', 'link', 'program'],
                QueryType.PROGRAM_STRUCTURE: ['section', 'division', 'layout', 'structure']
            }
            
            search_terms.extend(type_terms.get(query_plan.query_type, []))
            search_query = ' '.join(search_terms)
            
            results = self.vector_store.semantic_search(search_query, top_k=5, min_similarity=0.15)
            
            for result in results:
                contexts.append(RetrievedContext(
                    source_code=result['source_code'],
                    metadata=result['metadata'],
                    relevance_score=result['similarity'],
                    retrieval_method="vector_search"
                ))
        
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
        
        return contexts
    
    def _retrieve_field_contexts(self, session_id: str, entities: List[str]) -> List[RetrievedContext]:
        """Enhanced field context retrieval with better program matching"""
        contexts = []
        
        for entity in entities:
            try:
                field_context = self.db_manager.get_context_for_field(session_id, entity)
                field_details = field_context.get('field_details', [])
                
                if field_details:
                    # Get contexts from ALL programs that have this field, not just the first one
                    programs_with_field = set(detail.get('program_name') for detail in field_details)
                    
                    for program_name in programs_with_field:
                        if program_name:
                            source_data = self.db_manager.get_component_source_code(
                                session_id, program_name, max_size=500000
                            )
                            
                            if source_data.get('success') and source_data.get('components'):
                                for comp in source_data['components']:
                                    source_code = comp.get('source_for_chat', '')
                                    
                                    if source_code and entity.upper() in source_code.upper():
                                        # Get field details specific to this program
                                        program_field_details = [
                                            detail for detail in field_details 
                                            if detail.get('program_name') == program_name
                                        ]
                                        
                                        if program_field_details:
                                            field_specific_context = self._extract_field_specific_context(
                                                source_code, entity, program_field_details[0]
                                            )
                                            
                                            contexts.append(RetrievedContext(
                                                source_code=field_specific_context,
                                                metadata={
                                                    'component_name': comp.get('component_name'),
                                                    'component_type': comp.get('component_type'),
                                                    'field_name': entity,
                                                    'field_details': program_field_details[0],
                                                    'source_strategy': comp.get('source_strategy'),
                                                    'program_match': program_name,  # Track which program this is for
                                                    'total_source_lines': len(source_code.split('\n'))
                                                },
                                                relevance_score=0.95,
                                                retrieval_method=f"precise_field_{entity}_in_{program_name}"
                                            ))
            except Exception as e:
                logger.error(f"Error retrieving field context for {entity}: {str(e)}")
                continue
        
        return contexts

    
    def _extract_field_specific_context(self, source_code: str, field_name: str, field_details: Dict) -> str:
        """Extract field-specific context with surrounding code"""
        lines = source_code.split('\n')
        field_upper = field_name.upper()
        relevant_lines = []
        
        # Find all lines containing the field
        field_line_numbers = []
        for i, line in enumerate(lines):
            if field_upper in line.upper():
                field_line_numbers.append(i)
        
        # Build context around each occurrence
        context_parts = [
            f"=== FIELD ANALYSIS: {field_name} ===",
            f"Field found at {len(field_line_numbers)} locations in source code",
            ""
        ]
        
        # Add field definition context
        def_line = field_details.get('definition_line_number', 0)
        if def_line > 0:
            start_idx = max(0, def_line - 10)
            end_idx = min(len(lines), def_line + 5)
            context_parts.extend([
                f"--- FIELD DEFINITION (Lines {start_idx+1}-{end_idx}) ---"
            ])
            for i in range(start_idx, end_idx):
                marker = ">>> " if i == def_line - 1 else "    "
                context_parts.append(f"{marker}{i+1:4d}: {lines[i]}")
            context_parts.append("")
        
        # Add usage contexts (first 3 occurrences)
        for i, line_num in enumerate(field_line_numbers[:3]):
            start_idx = max(0, line_num - 3)
            end_idx = min(len(lines), line_num + 4)
            
            context_parts.extend([
                f"--- USAGE CONTEXT {i+1} (Lines {start_idx+1}-{end_idx}) ---"
            ])
            for j in range(start_idx, end_idx):
                marker = ">>> " if j == line_num else "    "
                context_parts.append(f"{marker}{j+1:4d}: {lines[j]}")
            context_parts.append("")
        
        # Add field references summary
        field_refs_json = field_details.get('field_references_json', '[]')
        try:
            field_refs = json.loads(field_refs_json) if field_refs_json else []
            if field_refs:
                context_parts.extend([
                    f"--- FIELD REFERENCES SUMMARY ({len(field_refs)} total) ---"
                ])
                for ref in field_refs[:5]:  # First 5 references
                    context_parts.append(f"Line {ref.get('line_number')}: {ref.get('operation_type')} - {ref.get('line_content')}")
                context_parts.append("")
        except:
            pass
        
        return '\n'.join(context_parts)
    
    def _retrieve_semantic_code_contexts(self, session_id: str, query_plan: QueryPlan) -> List[RetrievedContext]:
        """Enhanced semantic code context retrieval with entity-specific matching"""
        contexts = []
        
        try:
            components = self.db_manager.get_session_components(session_id)
            
            # If query has specific entities, prioritize components that contain those entities
            if query_plan.entities:
                scored_components = []
                for comp in components:
                    score = 0
                    comp_name = comp.get('component_name', '').upper()
                    
                    # Higher score for components mentioned in query entities
                    for entity in query_plan.entities:
                        entity_upper = entity.upper()
                        if entity_upper in comp_name:
                            score += 10
                        elif any(part in comp_name for part in entity_upper.split('-')):
                            score += 5
                    
                    scored_components.append((comp, score))
                
                # Sort by score and take top components
                scored_components.sort(key=lambda x: x[1], reverse=True)
                components = [comp for comp, score in scored_components[:8]]  # Top 8 components
            
            for comp in components:
                source_data = self.db_manager.get_component_source_code(
                    session_id, comp.get('component_name'), max_size=500000
                )
                
                if source_data.get('success') and source_data.get('components'):
                    for source_comp in source_data['components']:
                        source_code = source_comp.get('source_for_chat', '')
                        if source_code:
                            # Enhanced relevance calculation
                            relevance = self._calculate_enhanced_semantic_relevance(
                                source_code, query_plan.entities, query_plan.query_type, source_comp
                            )
                            
                            if relevance > 0.3:
                                contexts.append(RetrievedContext(
                                    source_code=source_code,
                                    metadata={
                                        'component_name': source_comp.get('component_name'),
                                        'component_type': source_comp.get('component_type'),
                                        'total_lines': source_comp.get('total_lines', 0),
                                        'source_strategy': source_comp.get('source_strategy'),
                                        'entity_matches': [e for e in query_plan.entities 
                                                        if e.upper() in source_code.upper()]
                                    },
                                    relevance_score=relevance,
                                    retrieval_method="semantic_code_enhanced"
                                ))
                                
                            # Break after finding sufficient contexts to avoid overwhelming
                            if len(contexts) >= 5:
                                break
                
                if len(contexts) >= 5:
                    break
                    
        except Exception as e:
            logger.error(f"Error in enhanced semantic code retrieval: {str(e)}")
        
        return contexts

    def _calculate_enhanced_semantic_relevance(self, source_code: str, entities: List[str], 
                                         query_type: QueryType, source_comp: Dict) -> float:
        """Enhanced semantic relevance calculation with entity-specific matching"""
        relevance = 0.0
        source_upper = source_code.upper()
        comp_name = source_comp.get('component_name', '').upper()
        
        # Higher weight for entity matches
        entity_matches = 0
        for entity in entities:
            entity_upper = entity.upper()
            
            # Exact match in source code
            if entity_upper in source_upper:
                entity_matches += 1
                relevance += 0.4
                
            # Partial match in component name
            if entity_upper in comp_name or any(part in comp_name for part in entity_upper.split('-')):
                relevance += 0.3
        
        # Boost relevance for components with multiple entity matches
        if entity_matches > 1:
            relevance += 0.2
        
        # Query type specific relevance
        if query_type == QueryType.FIELD_ANALYSIS:
            if any(pattern in source_upper for pattern in ['PIC ', 'MOVE ', 'COMPUTE ']):
                relevance += 0.2
        elif query_type == QueryType.BUSINESS_LOGIC:
            if any(pattern in source_upper for pattern in ['IF ', 'EVALUATE ', 'PERFORM ']):
                relevance += 0.2
        elif query_type == QueryType.DATA_FLOW:
            if any(pattern in source_upper for pattern in ['MOVE ', 'READ ', 'WRITE ']):
                relevance += 0.2
        elif query_type == QueryType.DEPENDENCIES:
            if any(pattern in source_upper for pattern in ['CALL ', 'EXEC CICS']):
                relevance += 0.2
        
        # Component type relevance
        comp_type = source_comp.get('component_type', '')
        if comp_type == 'PROGRAM' and query_type != QueryType.FIELD_ANALYSIS:
            relevance += 0.1
        elif comp_type == 'RECORD_LAYOUT' and query_type == QueryType.FIELD_ANALYSIS:
            relevance += 0.2
        
        return min(relevance, 1.0)


    def _retrieve_structural_contexts(self, session_id: str, query_plan: QueryPlan) -> List[RetrievedContext]:
        """Retrieve structural information (layouts, divisions, etc.)"""
        contexts = []
        
        try:
            layouts = self.db_manager.get_record_layouts(session_id)
            
            for layout in layouts[:3]:
                program_name = layout.get('program_name')
                if program_name:
                    source_data = self.db_manager.get_component_source_code(
                        session_id, program_name, max_size=500000
                    )
                    
                    if source_data.get('success') and source_data.get('components'):
                        for comp in source_data['components']:
                            contexts.append(RetrievedContext(
                                source_code=comp.get('source_for_chat', ''),
                                metadata={
                                    'component_name': comp.get('component_name'),
                                    'component_type': comp.get('component_type'),
                                    'layout_name': layout.get('layout_name'),
                                    'layout_fields': layout.get('fields_count', 0),
                                    'business_purpose': layout.get('business_purpose')
                                },
                                relevance_score=0.6,
                                retrieval_method="structural"
                            ))
                            
        except Exception as e:
            logger.error(f"Error in structural retrieval: {str(e)}")
        
        return contexts
    
    def _retrieve_dependency_contexts(self, session_id: str, query_plan: QueryPlan) -> List[RetrievedContext]:
        """Enhanced dependency retrieval for program calls and file usage"""
        contexts = []
        
        try:
            # Get all dependencies from database
            dependencies = self.db_manager.get_dependencies(session_id)
            
            # Filter dependencies based on query entities and type
            relevant_deps = []
            query_entities_upper = [entity.upper() for entity in query_plan.entities]
            
            for dep in dependencies:
                source_upper = dep.get('source_component', '').upper()
                target_upper = dep.get('target_component', '').upper()
                
                # Check if any entities match source or target
                entity_match = any(
                    entity in source_upper or entity in target_upper 
                    for entity in query_entities_upper
                )
                
                # Also check for query type specific dependencies
                rel_type = dep.get('relationship_type', '')
                query_type_match = False
                
                if query_plan.query_type == QueryType.DEPENDENCIES:
                    query_type_match = True
                elif 'program' in ' '.join(query_plan.entities).lower():
                    query_type_match = 'PROGRAM' in rel_type or 'CALL' in rel_type
                elif 'file' in ' '.join(query_plan.entities).lower():
                    query_type_match = 'FILE' in rel_type or 'CICS' in rel_type
                
                if entity_match or query_type_match:
                    relevant_deps.append(dep)
            
            if not relevant_deps:
                logger.info(f"No relevant dependencies found for entities: {query_plan.entities}")
                return contexts
            
            # Get source code for programs involved in dependencies
            involved_programs = set()
            for dep in relevant_deps[:10]:  # Limit to top 10
                involved_programs.add(dep.get('source_component'))
                if dep.get('relationship_type') in ['PROGRAM_CALL']:
                    involved_programs.add(dep.get('target_component'))
            
            # Retrieve source code for involved programs
            for program in list(involved_programs)[:5]:  # Top 5 programs
                try:
                    source_data = self.db_manager.get_component_source_code(
                        session_id, program, max_size=500000
                    )
                    
                    if source_data.get('success') and source_data.get('components'):
                        for comp in source_data['components']:
                            # Create dependency-specific context
                            dep_context = self._create_dependency_context(
                                comp, relevant_deps, query_plan.entities
                            )
                            
                            contexts.append(RetrievedContext(
                                source_code=dep_context,
                                metadata={
                                    'component_name': comp.get('component_name'),
                                    'component_type': comp.get('component_type'),
                                    'dependencies': [dep for dep in relevant_deps 
                                                if dep.get('source_component') == program or 
                                                    dep.get('target_component') == program],
                                    'total_dependencies': len(relevant_deps),
                                    'source_strategy': comp.get('source_strategy')
                                },
                                relevance_score=0.8,
                                retrieval_method="dependency_analysis"
                            ))
                except Exception as e:
                    logger.error(f"Error retrieving dependency context for {program}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in dependency retrieval: {str(e)}")
        
        return contexts
    
    def _create_dependency_context(self, component: Dict, relevant_deps: List[Dict], entities: List[str]) -> str:
        """Create specialized context for dependency analysis"""
        source_code = component.get('source_for_chat', '')
        comp_name = component.get('component_name', 'Unknown')
        
        context_parts = [
            f"=== DEPENDENCY ANALYSIS: {comp_name} ===",
            ""
        ]
        
        # Categorize dependencies
        program_calls = [d for d in relevant_deps if 'PROGRAM' in d.get('relationship_type', '') or 'CALL' in d.get('relationship_type', '')]
        file_deps = [d for d in relevant_deps if 'FILE' in d.get('relationship_type', '')]
        cics_deps = [d for d in relevant_deps if 'CICS' in d.get('relationship_type', '')]
        
        # Add dependency summary
        if program_calls:
            context_parts.extend([
                f"PROGRAM CALLS ({len(program_calls)}):",
                ""
            ])
            for dep in program_calls[:5]:
                context_parts.append(f"• Calls: {dep.get('target_component')} ({dep.get('relationship_type')})")
                # Try to extract line info from analysis details
                try:
                    details = json.loads(dep.get('analysis_details_json', '{}'))
                    if details.get('line_number'):
                        context_parts.append(f"  Line {details['line_number']}")
                except:
                    pass
            context_parts.append("")
        
        if file_deps:
            context_parts.extend([
                f"FILE OPERATIONS ({len(file_deps)}):",
                ""
            ])
            for dep in file_deps[:5]:
                io_type = "reads from" if "INPUT" in dep.get('relationship_type', '') else "writes to" if "OUTPUT" in dep.get('relationship_type', '') else "accesses"
                context_parts.append(f"• {io_type}: {dep.get('target_component')} ({dep.get('relationship_type')})")
            context_parts.append("")
        
        if cics_deps:
            context_parts.extend([
                f"CICS OPERATIONS ({len(cics_deps)}):",
                ""
            ])
            for dep in cics_deps[:5]:
                context_parts.append(f"• CICS {dep.get('target_component')} ({dep.get('relationship_type')})")
            context_parts.append("")
        
        # Add relevant source code sections
        if source_code:
            context_parts.extend([
                "=== RELEVANT SOURCE CODE SECTIONS ===",
                ""
            ])
            
            # Extract sections related to the query entities
            lines = source_code.split('\n')
            relevant_sections = []
            
            for entity in entities:
                entity_upper = entity.upper()
                for i, line in enumerate(lines):
                    line_upper = line.upper()
                    if entity_upper in line_upper:
                        # Add context around this line
                        start_idx = max(0, i - 3)
                        end_idx = min(len(lines), i + 4)
                        
                        section = []
                        for j in range(start_idx, end_idx):
                            marker = ">>> " if j == i else "    "
                            section.append(f"{marker}{j+1:4d}: {lines[j]}")
                        
                        relevant_sections.append({
                            'entity': entity,
                            'line_number': i + 1,
                            'section': '\n'.join(section)
                        })
                        
                        if len(relevant_sections) >= 5:  # Limit sections
                            break
                
                if len(relevant_sections) >= 5:
                    break
            
            # Add the relevant sections to context
            for section_info in relevant_sections:
                context_parts.extend([
                    f"--- {section_info['entity']} at Line {section_info['line_number']} ---",
                    section_info['section'],
                    ""
                ])
        
        return '\n'.join(context_parts)
    
    def _retrieve_business_contexts(self, session_id: str, query_plan: QueryPlan) -> List[RetrievedContext]:
        """Retrieve business context information"""
        contexts = []
        
        try:
            components = self.db_manager.get_session_components(session_id)
            business_components = [
                comp for comp in components 
                if comp.get('business_purpose') and len(comp.get('business_purpose', '')) > 20
            ][:3]
            
            for comp in business_components:
                source_data = self.db_manager.get_component_source_code(
                    session_id, comp.get('component_name'), max_size=500000
                )
                
                if source_data.get('success') and source_data.get('components'):
                    for source_comp in source_data['components']:
                        contexts.append(RetrievedContext(
                            source_code=source_comp.get('source_for_chat', ''),
                            metadata={
                                'component_name': source_comp.get('component_name'),
                                'component_type': source_comp.get('component_type'),
                                'business_purpose': comp.get('business_purpose')
                            },
                            relevance_score=0.5,
                            retrieval_method="business_context"
                        ))
                        
        except Exception as e:
            logger.error(f"Error in business context retrieval: {str(e)}")
        
        return contexts
    
    def _calculate_semantic_relevance(self, source_code: str, entities: List[str], query_type: QueryType) -> float:
        """Calculate semantic relevance of source code to query"""
        relevance = 0.0
        source_upper = source_code.upper()
        
        for entity in entities:
            if entity.upper() in source_upper:
                relevance += 0.3
        
        if query_type == QueryType.FIELD_ANALYSIS:
            if any(pattern in source_upper for pattern in ['PIC ', 'MOVE ', 'COMPUTE ']):
                relevance += 0.2
        elif query_type == QueryType.BUSINESS_LOGIC:
            if any(pattern in source_upper for pattern in ['IF ', 'EVALUATE ', 'PERFORM ']):
                relevance += 0.2
        elif query_type == QueryType.DATA_FLOW:
            if any(pattern in source_upper for pattern in ['MOVE ', 'READ ', 'WRITE ']):
                relevance += 0.2
        elif query_type == QueryType.DEPENDENCIES:
            if any(pattern in source_upper for pattern in ['CALL ', 'EXEC CICS']):
                relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _deduplicate_contexts(self, contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        """Remove duplicate contexts"""
        seen_sources = set()
        unique_contexts = []
        
        for context in contexts:
            source_hash = hash(context.source_code[:1000])
            if source_hash not in seen_sources:
                seen_sources.add(source_hash)
                unique_contexts.append(context)
        
        return unique_contexts
    
    def _rank_contexts(self, contexts: List[RetrievedContext], query_plan: QueryPlan) -> List[RetrievedContext]:
        """Rank contexts by relevance"""
        return sorted(contexts, key=lambda x: x.relevance_score, reverse=True)

class ResponseGenerator:
    """Generates responses using retrieved context"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def generate_response(self, message: str, query_plan: QueryPlan, contexts: List[RetrievedContext]) -> str:
        """Generate response using query plan and retrieved contexts"""
        
        if not contexts:
            return self._generate_fallback_response(message, query_plan)
        
        # Build context-aware prompt
        prompt = self._build_context_prompt(message, query_plan, contexts)
        
        # Call LLM with appropriate parameters
        max_tokens, temperature = self._get_generation_parameters(query_plan.query_type)
        
        response = self.llm_client.call_llm(prompt, max_tokens=max_tokens, temperature=temperature)
        
        if response.success:
            return self._post_process_response(response.content, query_plan, contexts)
        else:
            return f"I encountered an error generating the response: {response.error_message}"
    
    def _build_context_prompt(self, message: str, query_plan: QueryPlan, contexts: List[RetrievedContext]) -> str:
        """Build a context-aware prompt for the LLM"""
        
        prompt_parts = [
            "You are an expert COBOL analyst for a wealth management system.",
            f"Query Type: {query_plan.query_type.value}",
            f"Confidence: {query_plan.confidence:.2f}",
            f"Entities: {', '.join(query_plan.entities) if query_plan.entities else 'None'}",
            "",
            f"USER QUESTION: {message}",
            ""
        ]
        
        # Add context-specific instructions
        if query_plan.query_type == QueryType.FIELD_ANALYSIS:
            prompt_parts.extend([
                "INSTRUCTIONS:",
                "- Analyze the specific field(s) mentioned in the user's question",
                "- Explain the field's definition, usage, and business purpose",
                "- Reference specific line numbers from the source code",
                "- Describe how the field fits into the business process",
                ""
            ])
        elif query_plan.query_type == QueryType.BUSINESS_LOGIC:
            prompt_parts.extend([
                "INSTRUCTIONS:",
                "- Focus on the business rules and logic in the code",
                "- Explain the decision-making processes and calculations",
                "- Describe the business impact and purpose",
                ""
            ])
        elif query_plan.query_type == QueryType.DATA_FLOW:
            prompt_parts.extend([
                "INSTRUCTIONS:",
                "- Trace how data moves through the system",
                "- Identify input sources and output destinations",
                "- Explain data transformations and processing",
                ""
            ])
        
        # Add retrieved contexts
        prompt_parts.append("RELEVANT SOURCE CODE CONTEXTS:")
        prompt_parts.append("")
        
        for i, context in enumerate(contexts, 1):
            metadata = context.metadata
            prompt_parts.extend([
                f"=== CONTEXT {i} ===",
                f"Component: {metadata.get('component_name', 'Unknown')}",
                f"Relevance: {context.relevance_score:.2f}",
                f"Method: {context.retrieval_method}",
                f"Strategy: {metadata.get('source_strategy', 'unknown')}",
                ""
            ])
            
            # Add metadata-specific info
            if 'field_name' in metadata:
                prompt_parts.append(f"Field: {metadata['field_name']}")
            if 'business_purpose' in metadata:
                prompt_parts.append(f"Business Purpose: {metadata['business_purpose']}")
            
            prompt_parts.extend([
                "",
                "SOURCE CODE:",
                context.source_code,
                "",
                "=" * 80,
                ""
            ])
        
        prompt_parts.extend([
            "",
            "Generate a comprehensive, accurate response that directly answers the user's question.",
            "Use specific code references and line numbers when possible.",
            "Focus on business meaning and practical implications.",
            "If the context is insufficient, clearly state what information is missing."
        ])
        
        return '\n'.join(prompt_parts)
    
    def _get_generation_parameters(self, query_type: QueryType) -> Tuple[int, float]:
        """Get appropriate generation parameters for query type"""
        if query_type == QueryType.FIELD_ANALYSIS:
            return 1200, 0.1  # Detailed, factual
        elif query_type == QueryType.BUSINESS_LOGIC:
            return 1500, 0.2  # Explanatory
        elif query_type == QueryType.DEPENDENCIES:
            return 1000, 0.1  # Structured
        else:
            return 1200, 0.3  # More creative for general queries
    
    def _post_process_response(self, response: str, query_plan: QueryPlan, contexts: List[RetrievedContext]) -> str:
        """Post-process the generated response"""
        
        # Add confidence indicator if low
        if query_plan.confidence < 0.6:
            response += f"\n\n*Note: This analysis is based on limited context (confidence: {query_plan.confidence:.0%}). " \
                       f"Consider providing more specific information for a more detailed analysis.*"
        
        # Add context summary
        if contexts:
            response += f"\n\n**Analysis based on {len(contexts)} code context(s)**"
        
        return response
    
    def _generate_fallback_response(self, message: str, query_plan: QueryPlan) -> str:
        """Generate fallback response when no contexts are found"""
        return (
            f"I couldn't find specific code contexts to answer your question about: {message}\n\n"
            f"This might be because:\n"
            f"- The mentioned fields/programs haven't been analyzed yet\n"
            f"- The code doesn't contain the specific elements you're asking about\n"
            f"- More context is needed to understand your question\n\n"
            f"Try:\n"
            f"- Uploading more COBOL files for analysis\n"
            f"- Being more specific about field names or program names\n"
            f"- Asking about general program structure or available components"
        )

class AgenticRAGChatManager:
    """Main Agentic RAG Chat Manager with all advanced features"""
    
    def __init__(self, llm_client, db_manager, fallback_chat_manager=None):
        self.llm_client = llm_client
        self.db_manager = db_manager
        self.fallback_chat_manager = fallback_chat_manager
        
        # Initialize components
        self.vector_store = VectorStore(db_manager)
        self.query_analyzer = QueryAnalyzer(llm_client)
        self.context_retriever = ContextRetriever(db_manager, self.vector_store)
        self.response_generator = ResponseGenerator(llm_client)
        
        # Performance tracking
        self.performance_metrics = {
            'queries_processed': 0,
            'avg_response_time': 0,
            'context_hit_rate': 0,
            'user_satisfaction': 0
        }
        
        # Cache and session tracking
        self._response_cache = {}
        self._initialized_sessions = set()
        self._query_patterns = {}
        
        logger.info(f"Agentic RAG Chat Manager initialized with vector search: {VECTOR_SEARCH_AVAILABLE}")
    
    def initialize_session(self, session_id: str):
        """Initialize the RAG system for a new session"""
        try:
            logger.info(f"Initializing Agentic RAG for session {session_id}")
            
            # Build vector index if enabled
            if self.vector_store:
                self.vector_store.build_index(session_id)
            
            # Pre-warm caches and analyze session content
            self._analyze_session_content(session_id)
            
            # Initialize query pattern tracking
            if session_id not in self._query_patterns:
                self._query_patterns[session_id] = []
            
            self._initialized_sessions.add(session_id)
            logger.info("Agentic RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
    
    def process_query_with_full_features(self, session_id: str, message: str, conversation_id: str) -> Dict:
        """Main entry point for enhanced RAG processing with full feature set"""
        start_time = time.time()
        
        try:
            # Ensure session is initialized
            if session_id not in self._initialized_sessions:
                self.initialize_session(session_id)
            
            # Step 1: Analyze query and create retrieval plan
            query_plan = self.query_analyzer.analyze_query(message, session_id)
            logger.info(f"Query analysis: {query_plan.query_type.value}, entities: {query_plan.entities}, confidence: {query_plan.confidence:.2f}")
            
            # Step 2: Check for cached responses
            cached_response = self._check_response_cache(session_id, message, query_plan)
            if cached_response:
                logger.info("Returning cached response")
                return self._format_rag_response(
                    cached_response['response'], 
                    query_plan, 
                    [], 
                    time.time() - start_time, 
                    cached=True
                )
            
            # Step 3: Execute retrieval plan
            retrieved_contexts = self.context_retriever.retrieve_contexts(session_id, query_plan)
            logger.info(f"Retrieved {len(retrieved_contexts)} contexts using strategies: {[ctx.retrieval_method for ctx in retrieved_contexts]}")
            
            # Step 4: Route to specialized handler if needed
            routed_response = self._route_specialized_query(session_id, message, query_plan, retrieved_contexts)
            if routed_response:
                return self._format_rag_response(
                    routed_response, 
                    query_plan, 
                    retrieved_contexts, 
                    time.time() - start_time, 
                    routed=True
                )
            
            # Step 5: Generate response using retrieved context
            response = self.response_generator.generate_response(message, query_plan, retrieved_contexts)
            
            # Step 6: Post-process and cache
            processing_time = time.time() - start_time
            self._cache_response(session_id, message, query_plan, response)
            self._log_conversation_with_metrics(session_id, conversation_id, message, response, query_plan, retrieved_contexts, processing_time)
            
            return self._format_rag_response(response, query_plan, retrieved_contexts, processing_time)
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            
            # Fallback to basic chat manager if available
            if self.fallback_chat_manager:
                try:
                    logger.info("Falling back to basic chat manager")
                    fallback_response = self.fallback_chat_manager.process_query(session_id, message, conversation_id)
                    return {
                        'response': fallback_response,
                        'query_plan': {'query_type': 'fallback'},
                        'contexts_used': 0,
                        'processing_time': time.time() - start_time,
                        'cached': False,
                        'routed': False,
                        'fallback_used': True
                    }
                except Exception as fallback_error:
                    logger.error(f"Fallback chat manager also failed: {str(fallback_error)}")
            
            return {
                'response': f"I encountered an error analyzing your request: {str(e)}",
                'query_plan': {},
                'contexts_used': 0,
                'processing_time': time.time() - start_time,
                'cached': False,
                'routed': False,
                'error': str(e)
            }
    
    def process_query(self, session_id: str, message: str, conversation_id: str) -> str:
        """Backward compatibility method that returns just the response string"""
        result = self.process_query_with_full_features(session_id, message, conversation_id)
        return result.get('response', 'No response generated')
    
    def get_system_health(self) -> Dict:
        """Get comprehensive system health information"""
        try:
            health = {
                'system_type': 'agentic_rag',
                'status': 'healthy',
                'vector_search': VECTOR_SEARCH_AVAILABLE,
                'features': [
                    'Query Analysis',
                    'Context Retrieval', 
                    'Response Generation',
                    'Performance Caching',
                    'Specialized Query Routing'
                ],
                'performance_metrics': self.performance_metrics,
                'initialized_sessions': len(self._initialized_sessions),
                'query_cache_size': sum(len(cache) for cache in self._response_cache.values())
            }
            
            if VECTOR_SEARCH_AVAILABLE and self.vector_store and self.vector_store.index_built:
                health['features'].append('Vector Search')
                health['vector_documents'] = len(self.vector_store.documents)
            
            return health
            
        except Exception as e:
            return {
                'system_type': 'agentic_rag',
                'status': 'error',
                'error': str(e)
            }
    
    def _format_rag_response(self, response: str, query_plan: QueryPlan, 
                            contexts: List, processing_time: float, 
                            cached: bool = False, routed: bool = False) -> Dict:
        """Format RAG response in consistent structure"""
        return {
            'response': response,
            'query_plan': {
                'query_type': query_plan.query_type.value,
                'entities': query_plan.entities,
                'confidence': query_plan.confidence,
                'strategies': [s.value for s in query_plan.retrieval_strategies]
            },
            'contexts_used': len(contexts),
            'processing_time': processing_time,
            'cached': cached,
            'routed': routed,
            'retrieval_methods': [ctx.retrieval_method for ctx in contexts] if contexts else []
        }
    
    def _check_response_cache(self, session_id: str, message: str, query_plan: QueryPlan) -> Optional[Dict]:
        """Check for cached responses to similar queries"""
        cache_key = hashlib.md5(f"{message.lower()}_{sorted(query_plan.entities)}".encode()).hexdigest()
        
        session_cache = self._response_cache.get(session_id, {})
        if cache_key in session_cache:
            cached_item = session_cache[cache_key]
            # Check if cache is still fresh (within 1 hour)
            if time.time() - cached_item['timestamp'] < 3600:
                return cached_item
        
        return None
    
    def _cache_response(self, session_id: str, message: str, query_plan: QueryPlan, response: str):
        """Cache response for future use"""
        if session_id not in self._response_cache:
            self._response_cache[session_id] = {}
        
        cache_key = hashlib.md5(f"{message.lower()}_{sorted(query_plan.entities)}".encode()).hexdigest()
        
        # Limit cache size per session
        if len(self._response_cache[session_id]) > 50:
            # Remove oldest entries
            sorted_items = sorted(
                self._response_cache[session_id].items(),
                key=lambda x: x[1]['timestamp']
            )
            for key, _ in sorted_items[:10]:  # Remove oldest 10
                del self._response_cache[session_id][key]
        
        self._response_cache[session_id][cache_key] = {
            'response': response,
            'query_plan': query_plan,
            'timestamp': time.time()
        }
    
    # Replace the _route_specialized_query method in ResponseGenerator class

    def _route_specialized_query(self, session_id: str, message: str, 
                            query_plan: QueryPlan, contexts: List) -> Optional[str]:
        """Enhanced query routing with better program-specific context matching"""
        
        if not contexts:
            return None
        
        try:
            # Enhanced LLM routing with program-specific context
            routing_decision = self._llm_determine_response_strategy_enhanced(message, query_plan, contexts)
            
            if routing_decision.get('route_to_specialized'):
                handler_type = routing_decision.get('handler_type')
                
                if handler_type == 'field_analysis':
                    return self._handle_field_analysis_query(session_id, message, query_plan, contexts)
                elif handler_type == 'program_overview':
                    return self._handle_program_overview_query_enhanced(session_id, message, query_plan, contexts)
                elif handler_type == 'dependencies':
                    return self._handle_dependencies_query(session_id, message, query_plan, contexts)
                elif handler_type == 'business_logic':
                    return self._handle_business_logic_query(session_id, message, query_plan, contexts)
                elif handler_type == 'file_operations':
                    return self._handle_file_operations_query(session_id, message, query_plan, contexts)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in enhanced query routing: {str(e)}")
            return None

    def _llm_determine_response_strategy_enhanced(self, message: str, query_plan: QueryPlan, contexts: List) -> Dict:
        """Enhanced LLM routing with program-specific context awareness"""
        
        context_summary = self._build_enhanced_context_summary(contexts, query_plan.entities)
        
        routing_prompt = f"""
    You are analyzing a user query about COBOL programs to determine the best response strategy.

    User Query: "{message}"
    Query Type: {query_plan.query_type.value}
    Entities Found: {query_plan.entities}
    Confidence: {query_plan.confidence:.2f}

    Available Context Summary:
    {context_summary}

    IMPORTANT: Pay attention to which specific program(s) the user is asking about based on the entities found and context available.

    Response Strategy Options:
    1. "field_analysis" - User wants field information (definitions, usage, properties)
    2. "program_overview" - User wants program functionality, purpose, or general info
    3. "dependencies" - User wants program calls, file usage, system interactions
    4. "business_logic" - User wants calculations, rules, processing logic
    5. "file_operations" - User wants file I/O operations
    6. "general" - Use standard generation

    Return JSON:
    {{
        "route_to_specialized": true/false,
        "handler_type": "field_analysis|program_overview|dependencies|business_logic|file_operations|general",
        "target_programs": ["program1", "program2"],
        "confidence": 0.8,
        "reasoning": "Explanation including which program(s) this relates to"
    }}
    """
        
        try:
            response = self.llm_client.call_llm(routing_prompt, max_tokens=400, temperature=0.2)
            
            if response.success:
                routing_result = self.llm_client.extract_json_from_response(response.content)
                
                if isinstance(routing_result, dict):
                    logger.info(f"Enhanced routing: {routing_result.get('handler_type')} for programs: {routing_result.get('target_programs', [])}")
                    return routing_result
            
            return {"route_to_specialized": False, "handler_type": "general"}
            
        except Exception as e:
            logger.error(f"Error in enhanced LLM routing: {str(e)}")
            return {"route_to_specialized": False, "handler_type": "general"}
        
    def _build_enhanced_context_summary(self, contexts: List, entities: List[str]) -> str:
        """Build enhanced context summary with program-specific matching"""
    
        if not contexts:
            return "No context available"
        
        summary_parts = []
        program_contexts = {}
        
        # Group contexts by program
        for context in contexts:
            metadata = context.metadata
            comp_name = metadata.get('component_name', 'Unknown')
            program_match = metadata.get('program_match', comp_name)
            
            if program_match not in program_contexts:
                program_contexts[program_match] = []
            program_contexts[program_match].append(context)
        
        # Build summary for each program
        for program_name, prog_contexts in program_contexts.items():
            summary_parts.append(f"=== PROGRAM: {program_name} ===")
            
            # Check which entities match this program
            matching_entities = []
            for context in prog_contexts:
                entity_matches = context.metadata.get('entity_matches', [])
                matching_entities.extend(entity_matches)
            
            if matching_entities:
                summary_parts.append(f"Matches entities: {', '.join(set(matching_entities))}")
            
            # Add context details
            for i, context in enumerate(prog_contexts[:2], 1):  # Max 2 contexts per program
                metadata = context.metadata
                
                summary_parts.append(f"Context {i}:")
                summary_parts.append(f"- Type: {metadata.get('component_type', 'Unknown')}")
                
                if 'field_name' in metadata:
                    summary_parts.append(f"- Field: {metadata['field_name']}")
                if 'total_lines' in metadata:
                    summary_parts.append(f"- Lines: {metadata['total_lines']}")
                
                # Source preview
                source_preview = context.source_code[:150].replace('\n', ' ').strip()
                summary_parts.append(f"- Preview: {source_preview}...")
            
            summary_parts.append("")
        
        return '\n'.join(summary_parts)

    def _build_context_summary_for_routing(self, contexts: List) -> str:
        """Build a concise context summary for routing decisions"""
        
        if not contexts:
            return "No context available"
        
        summary_parts = []
        
        for i, context in enumerate(contexts[:2], 1):  # Only first 2 contexts for routing
            metadata = context.metadata
            
            summary_parts.append(f"Context {i}:")
            summary_parts.append(f"- Component: {metadata.get('component_name', 'Unknown')}")
            summary_parts.append(f"- Type: {metadata.get('component_type', 'Unknown')}")
            
            # Add specific context info
            if 'field_name' in metadata:
                summary_parts.append(f"- Field: {metadata['field_name']}")
            if 'dependencies' in metadata:
                dep_count = len(metadata['dependencies'])
                summary_parts.append(f"- Dependencies: {dep_count} found")
            if 'business_purpose' in metadata:
                purpose = metadata['business_purpose'][:100]
                summary_parts.append(f"- Purpose: {purpose}")
            
            # Source code preview
            source_preview = context.source_code[:200].replace('\n', ' ').strip()
            summary_parts.append(f"- Source Preview: {source_preview}...")
            summary_parts.append("")
        
        return '\n'.join(summary_parts)

    def _handle_program_overview_query_enhanced(self, session_id: str, message: str, 
                                           query_plan: QueryPlan, contexts: List) -> str:
        """Enhanced program overview with program-specific context matching"""
        
        if not contexts:
            return "I don't have enough context to explain what this program does."
        
        try:
            # Group contexts by program to handle multi-program queries
            program_contexts = {}
            for context in contexts:
                comp_name = context.metadata.get('component_name', 'Unknown')
                program_match = context.metadata.get('program_match', comp_name)
                
                if program_match not in program_contexts:
                    program_contexts[program_match] = []
                program_contexts[program_match].append(context)
            
            # If user mentioned specific entities, focus on programs that contain them
            target_programs = []
            if query_plan.entities:
                for entity in query_plan.entities:
                    entity_upper = entity.upper()
                    for prog_name in program_contexts.keys():
                        if (entity_upper in prog_name.upper() or 
                            any(entity_upper in ctx.source_code.upper() for ctx in program_contexts[prog_name])):
                            if prog_name not in target_programs:
                                target_programs.append(prog_name)
            
            # If no specific programs found, use all available
            if not target_programs:
                target_programs = list(program_contexts.keys())
            
            response_parts = []
            
            for program_name in target_programs[:2]:  # Limit to 2 programs to avoid overwhelming
                if program_name in program_contexts:
                    prog_contexts = program_contexts[program_name]
                    
                    overview_prompt = self._build_program_overview_prompt_enhanced(
                        message, prog_contexts, program_name, query_plan.entities
                    )
                    
                    response = self.llm_client.call_llm(overview_prompt, max_tokens=1200, temperature=0.3)
                    
                    if response.success and response.content:
                        cleaned_response = response.content.strip()
                        response_parts.append(f"**{program_name}:**\n{cleaned_response}")
                    else:
                        response_parts.append(f"**{program_name}:**\n{self._generate_fallback_program_overview(prog_contexts)}")
            
            return '\n\n'.join(response_parts) if response_parts else "No program information available."
            
        except Exception as e:
            logger.error(f"Error in enhanced program overview handler: {str(e)}")
            return f"Error analyzing program overview: {str(e)}"

    def _build_program_overview_prompt_enhanced(self, message: str, contexts: List, 
                                          program_name: str, entities: List[str]) -> str:
        """Build enhanced program overview prompt with entity context"""
        
        prompt_parts = [
            f"You are explaining what the COBOL program '{program_name}' does in business terms.",
            f'User asked: "{message}"',
            ""
        ]
        
        if entities:
            matching_entities = []
            for context in contexts:
                entity_matches = context.metadata.get('entity_matches', [])
                matching_entities.extend(entity_matches)
            
            if matching_entities:
                prompt_parts.extend([
                    f"User is specifically asking about: {', '.join(set(matching_entities))}",
                    ""
                ])
        
        prompt_parts.extend([
            "Based on the source code analysis below, explain:",
            "1. What this program's main business purpose is",
            "2. What key operations it performs",
            "3. What systems or files it interacts with", 
            "4. How it fits into the wealth management process",
            "",
            "Keep it business-focused and conversational.",
            ""
        ])
        
        # Add context information
        for i, context in enumerate(contexts, 1):
            metadata = context.metadata
            
            prompt_parts.extend([
                f"=== SOURCE ANALYSIS {i}: {program_name} ===",
                ""
            ])
            
            if 'field_name' in metadata:
                prompt_parts.append(f"Field Context: {metadata['field_name']}")
            
            if metadata.get('total_lines'):
                prompt_parts.append(f"Program Size: {metadata['total_lines']} lines")
            
            prompt_parts.extend([
                "",
                "Source Code:",
                context.source_code,
                "",
                "=" * 60,
                ""
            ])
        
        return '\n'.join(prompt_parts)

    def _post_process_program_overview(self, response: str, contexts: List) -> str:
        """Post-process program overview response"""
        
        # Add technical summary at the end
        technical_summary = []
        
        for context in contexts:
            metadata = context.metadata
            component_name = metadata.get('component_name')
            
            if component_name:
                technical_summary.append(f"**Technical Details for {component_name}:**")
                
                if 'dependencies' in metadata:
                    deps = metadata['dependencies']
                    dep_summary = []
                    
                    program_calls = [d for d in deps if 'PROGRAM' in d.get('relationship_type', '')]
                    if program_calls:
                        dep_summary.append(f"{len(program_calls)} program call(s)")
                    
                    file_ops = [d for d in deps if 'FILE' in d.get('relationship_type', '')]
                    if file_ops:
                        dep_summary.append(f"{len(file_ops)} file operation(s)")
                    
                    if dep_summary:
                        technical_summary.append(f"• Dependencies: {', '.join(dep_summary)}")
                
                technical_summary.append("")
        
        if technical_summary:
            return response + "\n\n" + '\n'.join(technical_summary)
        else:
            return response

    def _generate_fallback_program_overview(self, contexts: List) -> str:
        """Generate fallback program overview when LLM fails"""
        
        if not contexts:
            return "No program analysis available."
        
        context = contexts[0]
        metadata = context.metadata
        component_name = metadata.get('component_name', 'Unknown Program')
        
        overview_parts = [
            f"**Program Overview: {component_name}**",
            ""
        ]
        
        # Add business purpose if available
        if 'business_purpose' in metadata:
            overview_parts.append(f"**Purpose**: {metadata['business_purpose']}")
            overview_parts.append("")
        
        # Add dependency information
        if 'dependencies' in metadata:
            dependencies = metadata['dependencies']
            
            program_calls = [d for d in dependencies if 'PROGRAM' in d.get('relationship_type', '')]
            file_ops = [d for d in dependencies if 'FILE' in d.get('relationship_type', '')]
            
            if program_calls:
                overview_parts.append(f"**Program Interactions**: Calls {len(program_calls)} other programs")
            
            if file_ops:
                overview_parts.append(f"**File Operations**: Processes {len(file_ops)} files")
        
        if not overview_parts or len(overview_parts) <= 3:
            overview_parts.append("This appears to be a COBOL program with business logic processing capabilities.")
        
        return '\n'.join(overview_parts)

    def _handle_field_analysis_query(self, session_id: str, message: str, 
                                    query_plan: QueryPlan, contexts: List) -> str:
        """Enhanced field analysis handler"""
        
        if not query_plan.entities:
            return "I couldn't identify specific field names in your question."
        
        field_name = query_plan.entities[0]  # Primary field
        
        # Use existing field definition logic but enhanced
        return self._handle_field_definition_query(session_id, field_name, contexts)

    def _handle_dependencies_query(self, session_id: str, message: str, 
                                query_plan: QueryPlan, contexts: List) -> str:
        """Handle dependency-related queries"""
        
        # Check what type of dependencies the user is asking about
        message_lower = message.lower()
        
        if 'program' in message_lower or 'call' in message_lower:
            return self._handle_program_calls_query(session_id, message, query_plan, contexts)
        elif 'file' in message_lower:
            return self._handle_file_usage_query(session_id, message, query_plan, contexts)
        elif 'cics' in message_lower:
            return self._handle_cics_operations_query(session_id, message, query_plan, contexts)
        else:
            # General dependencies overview
            return self._handle_general_dependencies_query(session_id, message, query_plan, contexts)

    def _handle_file_operations_query(self, session_id: str, message: str, 
                               query_plan: QueryPlan, contexts: List) -> str:
        """Handle file operations specific queries (READ, WRITE, file processing)"""
        try:
            if not contexts:
                return "I couldn't find file operations information in the analyzed code."
            
            response_parts = [
                "**File Operations Analysis**",
                ""
            ]
            
            all_file_operations = []
            
            for context in contexts:
                metadata = context.metadata
                component_name = metadata.get('component_name', 'Unknown')
                source_code = context.source_code
                dependencies = metadata.get('dependencies', [])
                
                # Extract file dependencies
                file_deps = [dep for dep in dependencies 
                            if 'FILE' in dep.get('relationship_type', '') and 
                            'CICS' not in dep.get('relationship_type', '')]
                
                if file_deps or self._has_file_operations_in_source(source_code):
                    response_parts.extend([
                        f"**In {component_name}:**",
                        ""
                    ])
                    
                    # Analyze file dependencies
                    if file_deps:
                        input_files = [d for d in file_deps if 'INPUT' in d.get('relationship_type', '')]
                        output_files = [d for d in file_deps if 'OUTPUT' in d.get('relationship_type', '')]
                        io_files = [d for d in file_deps if 'INPUT_OUTPUT' in d.get('relationship_type', '')]
                        
                        if input_files:
                            response_parts.append("**Input File Operations:**")
                            for dep in input_files:
                                file_name = dep.get('target_component')
                                response_parts.append(f"• **READ** operations on: `{file_name}`")
                                # Try to get line information
                                try:
                                    details = json.loads(dep.get('analysis_details_json', '{}'))
                                    if details.get('line_number'):
                                        response_parts.append(f"  - Referenced at line {details['line_number']}")
                                except:
                                    pass
                            response_parts.append("")
                        
                        if output_files:
                            response_parts.append("**Output File Operations:**")
                            for dep in output_files:
                                file_name = dep.get('target_component')
                                response_parts.append(f"• **WRITE** operations on: `{file_name}`")
                                try:
                                    details = json.loads(dep.get('analysis_details_json', '{}'))
                                    if details.get('line_number'):
                                        response_parts.append(f"  - Referenced at line {details['line_number']}")
                                except:
                                    pass
                            response_parts.append("")
                        
                        if io_files:
                            response_parts.append("**Input/Output File Operations:**")
                            for dep in io_files:
                                file_name = dep.get('target_component')
                                response_parts.append(f"• **READ/WRITE** operations on: `{file_name}`")
                            response_parts.append("")
                    
                    # Analyze source code for file operations patterns
                    file_operations = self._extract_file_operations_from_source(source_code)
                    if file_operations:
                        response_parts.append("**File Operation Statements:**")
                        for op in file_operations[:5]:  # Show first 5
                            response_parts.append(f"• Line {op['line_number']}: `{op['statement']}`")
                            response_parts.append(f"  - Operation: **{op['operation_type']}**")
                            if op.get('file_name'):
                                response_parts.append(f"  - File: `{op['file_name']}`")
                        response_parts.append("")
                    
                    all_file_operations.extend(file_deps)
            
            if not all_file_operations and not any(self._has_file_operations_in_source(ctx.source_code) for ctx in contexts):
                return "No file operations found in the analyzed components."
            
            # Add file operations summary
            read_ops = len([f for f in all_file_operations if 'INPUT' in f.get('relationship_type', '')])
            write_ops = len([f for f in all_file_operations if 'OUTPUT' in f.get('relationship_type', '')])
            
            response_parts.extend([
                "**File Operations Summary:**",
                f"• **Read Operations**: {read_ops}",
                f"• **Write Operations**: {write_ops}",
                f"• **Total File Dependencies**: {len(all_file_operations)}",
                ""
            ])
            
            # Add file processing patterns if found
            processing_patterns = self._identify_file_processing_patterns(contexts)
            if processing_patterns:
                response_parts.extend([
                    "**File Processing Patterns:**",
                    ""
                ])
                for pattern in processing_patterns:
                    response_parts.append(f"• **{pattern['type']}**: {pattern['description']}")
                response_parts.append("")
            
            return '\n'.join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in file operations handler: {str(e)}")
            return f"Error analyzing file operations: {str(e)}"

    def _has_file_operations_in_source(self, source_code: str) -> bool:
        """Check if source code contains file operation statements"""
        file_op_patterns = [
            r'\bREAD\s+[\w\-]+',
            r'\bWRITE\s+[\w\-]+',
            r'\bOPEN\s+(INPUT|OUTPUT|I-O)\s+[\w\-]+',
            r'\bCLOSE\s+[\w\-]+',
            r'\bREWRITE\s+[\w\-]+',
            r'\bDELETE\s+[\w\-]+'
        ]
        
        source_upper = source_code.upper()
        return any(re.search(pattern, source_upper) for pattern in file_op_patterns)

    def _extract_file_operations_from_source(self, source_code: str) -> List[Dict]:
        """Extract file operations from source code with line numbers"""
        operations = []
        lines = source_code.split('\n')
        
        file_op_patterns = {
            r'\b(READ)\s+([\w\-]+)': 'READ',
            r'\b(WRITE)\s+([\w\-]+)': 'WRITE', 
            r'\b(OPEN)\s+(INPUT|OUTPUT|I-O)\s+([\w\-]+)': 'OPEN',
            r'\b(CLOSE)\s+([\w\-]+)': 'CLOSE',
            r'\b(REWRITE)\s+([\w\-]+)': 'REWRITE',
            r'\b(DELETE)\s+([\w\-]+)': 'DELETE'
        }
        
        for line_num, line in enumerate(lines, 1):
            line_upper = line.upper().strip()
            if not line_upper or line_upper.startswith('*'):
                continue
                
            for pattern, op_type in file_op_patterns.items():
                match = re.search(pattern, line_upper)
                if match:
                    # Extract file name (usually the last captured group)
                    file_name = None
                    if len(match.groups()) >= 2:
                        file_name = match.groups()[-1]  # Last group is usually the file name
                    
                    operations.append({
                        'line_number': line_num,
                        'statement': line.strip(),
                        'operation_type': op_type,
                        'file_name': file_name
                    })
        
        return operations

    def _identify_file_processing_patterns(self, contexts: List) -> List[Dict]:
        """Identify common file processing patterns"""
        patterns = []
        
        for context in contexts:
            source_code = context.source_code.upper()
            
            # Sequential file processing
            if ('READ' in source_code and 'AT END' in source_code and 
                'PERFORM' in source_code):
                patterns.append({
                    'type': 'Sequential File Processing',
                    'description': 'Reads file sequentially with end-of-file handling'
                })
            
            # File update pattern
            if ('READ' in source_code and 'REWRITE' in source_code):
                patterns.append({
                    'type': 'File Update Processing',
                    'description': 'Reads and updates existing file records'
                })
            
            # Master file processing
            if ('SORT' in source_code or 'MERGE' in source_code):
                patterns.append({
                    'type': 'Sorted File Processing',
                    'description': 'Processes files with sorting or merging operations'
                })
            
            # Report generation
            if ('WRITE' in source_code and 'REPORT' in source_code):
                patterns.append({
                    'type': 'Report Generation',
                    'description': 'Generates output reports from data files'
                })
        
        return patterns

    def _handle_business_logic_query(self, session_id: str, message: str, 
                                query_plan: QueryPlan, contexts: List) -> str:
        """Handle business logic specific queries"""
        try:
            if not contexts:
                return "I couldn't find business logic information in the analyzed code."
            
            response_parts = [
                "**Business Logic Analysis**",
                ""
            ]
            
            all_business_logic = []
            
            for context in contexts:
                metadata = context.metadata
                component_name = metadata.get('component_name', 'Unknown')
                source_code = context.source_code
                
                # Extract business logic patterns
                business_logic = self._extract_business_logic_patterns(source_code)
                
                if business_logic:
                    response_parts.extend([
                        f"**In {component_name}:**",
                        ""
                    ])
                    
                    # Categorize business logic
                    calculations = [bl for bl in business_logic if bl['category'] == 'calculation']
                    validations = [bl for bl in business_logic if bl['category'] == 'validation']
                    decisions = [bl for bl in business_logic if bl['category'] == 'decision']
                    
                    if calculations:
                        response_parts.append("**Calculations:**")
                        for calc in calculations[:3]:
                            response_parts.append(f"• Line {calc['line_number']}: {calc['description']}")
                            response_parts.append(f"  ```cobol\n  {calc['code_snippet']}\n  ```")
                        response_parts.append("")
                    
                    if validations:
                        response_parts.append("**Validations:**")
                        for val in validations[:3]:
                            response_parts.append(f"• Line {val['line_number']}: {val['description']}")
                            response_parts.append(f"  ```cobol\n  {val['code_snippet']}\n  ```")
                        response_parts.append("")
                    
                    if decisions:
                        response_parts.append("**Decision Logic:**")
                        for dec in decisions[:3]:
                            response_parts.append(f"• Line {dec['line_number']}: {dec['description']}")
                            response_parts.append(f"  ```cobol\n  {dec['code_snippet']}\n  ```")
                        response_parts.append("")
                    
                    all_business_logic.extend(business_logic)
            
            if not all_business_logic:
                return "No specific business logic patterns found in the analyzed components."
            
            # Add business logic summary
            calc_count = len([bl for bl in all_business_logic if bl['category'] == 'calculation'])
            val_count = len([bl for bl in all_business_logic if bl['category'] == 'validation'])
            dec_count = len([bl for bl in all_business_logic if bl['category'] == 'decision'])
            
            response_parts.extend([
                "**Business Logic Summary:**",
                f"• **Calculations**: {calc_count}",
                f"• **Validations**: {val_count}",
                f"• **Decision Points**: {dec_count}",
                f"• **Total Logic Patterns**: {len(all_business_logic)}",
                ""
            ])
            
            # Add business context if available
            business_purposes = [ctx.metadata.get('business_purpose') for ctx in contexts 
                            if ctx.metadata.get('business_purpose')]
            
            if business_purposes:
                response_parts.extend([
                    "**Business Context:**",
                    ""
                ])
                for purpose in business_purposes[:2]:
                    response_parts.append(f"• {purpose}")
                response_parts.append("")
            
            return '\n'.join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in business logic handler: {str(e)}")
            return f"Error analyzing business logic: {str(e)}"

    def _extract_business_logic_patterns(self, source_code: str) -> List[Dict]:
        """Extract business logic patterns from source code"""
        patterns = []
        lines = source_code.split('\n')
        
        # Business logic patterns
        logic_patterns = {
            # Calculations
            r'COMPUTE\s+([\w\-]+)\s*=': ('calculation', 'Computation assignment'),
            r'ADD\s+([\w\-]+)\s+TO\s+([\w\-]+)': ('calculation', 'Addition operation'),
            r'SUBTRACT\s+([\w\-]+)\s+FROM\s+([\w\-]+)': ('calculation', 'Subtraction operation'),
            r'MULTIPLY\s+([\w\-]+)\s+BY\s+([\w\-]+)': ('calculation', 'Multiplication operation'),
            r'DIVIDE\s+([\w\-]+)\s+BY\s+([\w\-]+)': ('calculation', 'Division operation'),
            
            # Validations  
            r'IF\s+([\w\-]+)\s*(=|>|<|NOT)': ('validation', 'Conditional validation'),
            r'EVALUATE\s+([\w\-]+)': ('validation', 'Multi-way validation'),
            r'IF\s+([\w\-]+)\s+IS\s+(NUMERIC|ALPHABETIC)': ('validation', 'Data type validation'),
            
            # Decision logic
            r'IF\s+.*\s+THEN': ('decision', 'Conditional decision'),
            r'WHEN\s+': ('decision', 'Case-based decision'),
            r'PERFORM\s+([\w\-]+)\s+UNTIL': ('decision', 'Loop decision')
        }
        
        for line_num, line in enumerate(lines, 1):
            line_upper = line.upper().strip()
            if not line_upper or line_upper.startswith('*'):
                continue
                
            for pattern, (category, description) in logic_patterns.items():
                if re.search(pattern, line_upper):
                    patterns.append({
                        'line_number': line_num,
                        'category': category,
                        'description': description,
                        'code_snippet': line.strip(),
                        'pattern': pattern
                    })
        
        return patterns


    def _handle_general_dependencies_query(self, session_id: str, message: str, 
                                        query_plan: QueryPlan, contexts: List) -> str:
        """Handle general dependency queries"""
        
        if not contexts:
            return "I couldn't find dependency information in the analyzed code."
        
        response_parts = [
            "**Program Dependencies Overview**",
            ""
        ]
        
        total_deps = 0
        
        for context in contexts:
            metadata = context.metadata
            dependencies = metadata.get('dependencies', [])
            
            if dependencies:
                component_name = metadata.get('component_name', 'Unknown')
                response_parts.append(f"**{component_name}:**")
                
                # Categorize dependencies
                program_calls = [d for d in dependencies if 'PROGRAM' in d.get('relationship_type', '')]
                file_ops = [d for d in dependencies if 'FILE' in d.get('relationship_type', '')]
                cics_ops = [d for d in dependencies if 'CICS' in d.get('relationship_type', '')]
                
                if program_calls:
                    programs = [d.get('target_component') for d in program_calls]
                    response_parts.append(f"• Calls: {', '.join(programs)}")
                
                if file_ops:
                    files = [d.get('target_component') for d in file_ops]
                    response_parts.append(f"• Files: {', '.join(files)}")
                
                if cics_ops:
                    cics_files = [d.get('target_component') for d in cics_ops]
                    response_parts.append(f"• CICS: {', '.join(cics_files)}")
                
                response_parts.append("")
                total_deps += len(dependencies)
        
        if total_deps == 0:
            return "No external dependencies found in the analyzed components."
        
        response_parts.append(f"**Total Dependencies**: {total_deps}")
        
        return '\n'.join(response_parts)
    def _handle_program_calls_query(self, session_id: str, message: str, 
                                query_plan: QueryPlan, contexts: List) -> str:
        """Handle program calls queries"""
        try:
            if not contexts:
                return "I couldn't find program call information in the analyzed code."
            
            response_parts = [
                "**Program Calls Analysis**",
                ""
            ]
            
            all_program_calls = []
            
            for context in contexts:
                metadata = context.metadata
                dependencies = metadata.get('dependencies', [])
                
                # Extract program call dependencies
                program_calls = [dep for dep in dependencies 
                            if 'PROGRAM' in dep.get('relationship_type', '') or 
                                'CALL' in dep.get('relationship_type', '')]
                
                if program_calls:
                    component_name = metadata.get('component_name', 'Unknown')
                    response_parts.extend([
                        f"**In {component_name}:**",
                        ""
                    ])
                    
                    for call in program_calls:
                        target_prog = call.get('target_component')
                        rel_type = call.get('relationship_type')
                        
                        # Extract line information
                        try:
                            details = json.loads(call.get('analysis_details_json', '{}'))
                            line_info = f" (Line {details.get('line_number', 'Unknown')})" if details.get('line_number') else ""
                        except:
                            line_info = ""
                        
                        response_parts.append(f"• **{rel_type}**: {target_prog}{line_info}")
                        
                        all_program_calls.append({
                            'source': component_name,
                            'target': target_prog,
                            'type': rel_type
                        })
                    
                    response_parts.append("")
            
            if not all_program_calls:
                return "No program calls found in the analyzed components."
            
            # Add summary
            response_parts.extend([
                f"**Summary**: Found {len(all_program_calls)} program call(s) across {len(contexts)} component(s).",
                ""
            ])
            
            # Add source code evidence if available
            for context in contexts:
                source_lines = context.source_code.split('\n')
                call_lines = [line for line in source_lines 
                            if 'CALL' in line.upper() or 'LINK' in line.upper() or 'XCTL' in line.upper()]
                
                if call_lines:
                    response_parts.extend([
                        "**Source Code Examples:**",
                        ""
                    ])
                    for line in call_lines[:3]:  # Show first 3
                        response_parts.append(f"```cobol\n{line.strip()}\n```")
                    response_parts.append("")
                    break
            
            return '\n'.join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in program calls handler: {str(e)}")
            return f"Error analyzing program calls: {str(e)}"
    
    def _handle_file_usage_query(self, session_id: str, message: str, 
                            query_plan: QueryPlan, contexts: List) -> str:
        """Handle file usage queries"""
        try:
            if not contexts:
                return "I couldn't find file usage information in the analyzed code."
            
            response_parts = [
                "**File Usage Analysis**",
                ""
            ]
            
            all_file_ops = []
            
            for context in contexts:
                metadata = context.metadata
                dependencies = metadata.get('dependencies', [])
                
                # Extract file dependencies
                file_deps = [dep for dep in dependencies 
                            if 'FILE' in dep.get('relationship_type', '')]
                
                if file_deps:
                    component_name = metadata.get('component_name', 'Unknown')
                    response_parts.extend([
                        f"**In {component_name}:**",
                        ""
                    ])
                    
                    # Group by file type
                    input_files = [d for d in file_deps if 'INPUT' in d.get('relationship_type', '')]
                    output_files = [d for d in file_deps if 'OUTPUT' in d.get('relationship_type', '')]
                    io_files = [d for d in file_deps if 'INPUT_OUTPUT' in d.get('relationship_type', '')]
                    
                    if input_files:
                        response_parts.append("**Input Files:**")
                        for dep in input_files:
                            response_parts.append(f"• Reads from: {dep.get('target_component')}")
                        response_parts.append("")
                    
                    if output_files:
                        response_parts.append("**Output Files:**")
                        for dep in output_files:
                            response_parts.append(f"• Writes to: {dep.get('target_component')}")
                        response_parts.append("")
                    
                    if io_files:
                        response_parts.append("**Input/Output Files:**")
                        for dep in io_files:
                            response_parts.append(f"• Processes: {dep.get('target_component')}")
                        response_parts.append("")
                    
                    all_file_ops.extend(file_deps)
            
            if not all_file_ops:
                return "No file operations found in the analyzed components."
            
            # Add summary with file operation breakdown
            input_count = len([f for f in all_file_ops if 'INPUT' in f.get('relationship_type', '')])
            output_count = len([f for f in all_file_ops if 'OUTPUT' in f.get('relationship_type', '')])
            
            response_parts.extend([
                f"**Summary**: Found {len(all_file_ops)} file operation(s) - {input_count} input, {output_count} output",
                ""
            ])
            
            return '\n'.join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in file usage handler: {str(e)}")
            return f"Error analyzing file usage: {str(e)}"

    def _handle_field_definition_query(self, session_id: str, field_name: str, contexts: List) -> str:
        """Specialized handler for field definition queries"""
        try:
            field_context = self.db_manager.get_context_for_field(session_id, field_name)
            field_details = field_context.get('field_details', [])
            
            if not field_details:
                return f"I couldn't find detailed information about field '{field_name}' in the analyzed code."
            
            primary_detail = field_details[0]
            
            response_parts = [
                f"**Field Analysis: {field_name}**",
                "",
                f"**Definition**: {primary_detail.get('definition_code', 'Not found')}",
                f"**Data Type**: {primary_detail.get('mainframe_data_type', 'Unknown')}",
                f"**Length**: {primary_detail.get('mainframe_length', 'Unknown')} characters",
                f"**Program**: {primary_detail.get('program_name', 'Unknown')}",
                f"**Usage**: {primary_detail.get('usage_type', 'Unknown')}",
                ""
            ]
            
            if primary_detail.get('business_purpose'):
                response_parts.extend([
                    f"**Business Purpose**: {primary_detail['business_purpose']}",
                    ""
                ])
            
            ref_count = primary_detail.get('total_program_references', 0)
            if ref_count > 0:
                response_parts.append(f"**Usage**: Referenced {ref_count} times across programs")
            
            return '\n'.join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in field definition handler: {str(e)}")
            return f"Error analyzing field '{field_name}': {str(e)}"
    
    def _handle_calculation_query(self, session_id: str, message: str, contexts: List) -> str:
        """Specialized handler for calculation/business logic queries"""
        
        calculation_sections = []
        
        for context in contexts:
            source_code = context.source_code
            
            # Find calculation patterns
            calc_patterns = [
                r'COMPUTE\s+[\w\-]+.*',
                r'ADD\s+[\w\-]+\s+TO\s+[\w\-]+.*',
                r'MULTIPLY\s+[\w\-]+\s+BY\s+[\w\-]+.*',
                r'IF\s+.*\s+THEN.*'
            ]
            
            found_calculations = []
            for line_num, line in enumerate(source_code.split('\n'), 1):
                for pattern in calc_patterns:
                    if re.search(pattern, line.upper()):
                        found_calculations.append(f"Line {line_num}: {line.strip()}")
            
            if found_calculations:
                calculation_sections.append({
                    'component': context.metadata.get('component_name', 'Unknown'),
                    'calculations': found_calculations[:5]
                })
        
        if not calculation_sections:
            return "I couldn't find specific calculation logic in the analyzed code. The business logic might be implemented differently or require additional context."
        
        response_parts = [
            "**Business Logic Analysis**",
            ""
        ]
        
        for section in calculation_sections:
            response_parts.extend([
                f"**In {section['component']}:**",
                ""
            ])
            
            for calc in section['calculations']:
                response_parts.append(f"• {calc}")
            
            response_parts.append("")
        
        return '\n'.join(response_parts)
    
    def _handle_cics_operations_query(self, session_id: str, message: str, 
                                 query_plan: QueryPlan, contexts: List) -> str:
        """Handle CICS operations queries"""
        try:
            if not contexts:
                return "I couldn't find CICS operations in the analyzed code."
            
            response_parts = [
                "**CICS Operations Analysis**",
                ""
            ]
            
            all_cics_ops = []
            
            for context in contexts:
                metadata = context.metadata
                dependencies = metadata.get('dependencies', [])
                
                # Extract CICS dependencies
                cics_deps = [dep for dep in dependencies 
                            if 'CICS' in dep.get('relationship_type', '')]
                
                if cics_deps:
                    component_name = metadata.get('component_name', 'Unknown')
                    response_parts.extend([
                        f"**In {component_name}:**",
                        ""
                    ])
                    
                    for dep in cics_deps:
                        file_name = dep.get('target_component')
                        rel_type = dep.get('relationship_type')
                        
                        # Determine operation type
                        if 'INPUT' in rel_type:
                            op_desc = f"Reads from CICS file: {file_name}"
                        elif 'OUTPUT' in rel_type:
                            op_desc = f"Writes to CICS file: {file_name}"
                        else:
                            op_desc = f"Accesses CICS file: {file_name}"
                        
                        response_parts.append(f"• {op_desc}")
                        all_cics_ops.append(dep)
                    
                    response_parts.append("")
            
            if not all_cics_ops:
                return "No CICS operations found in the analyzed components."
            
            # Add CICS-specific summary
            response_parts.extend([
                f"**Summary**: Found {len(all_cics_ops)} CICS operation(s) across online transaction processing.",
                "These operations indicate this is an online transaction program that interacts with CICS files.",
                ""
            ])
            
            return '\n'.join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in CICS operations handler: {str(e)}")
            return f"Error analyzing CICS operations: {str(e)}"
    
    def _log_conversation_with_metrics(self, session_id: str, conversation_id: str, 
                                     message: str, response: str, query_plan: QueryPlan,
                                     contexts: List, processing_time: float):
        """Enhanced conversation logging with RAG metrics"""
        try:
            # Update performance metrics
            self.performance_metrics['queries_processed'] += 1
            
            # Update averages
            current_avg = self.performance_metrics['avg_response_time']
            new_avg = ((current_avg * (self.performance_metrics['queries_processed'] - 1)) + processing_time) / self.performance_metrics['queries_processed']
            self.performance_metrics['avg_response_time'] = new_avg
            
            # Update context hit rate
            context_found = len(contexts) > 0
            hit_rate = self.performance_metrics['context_hit_rate']
            new_hit_rate = ((hit_rate * (self.performance_metrics['queries_processed'] - 1)) + (1 if context_found else 0)) / self.performance_metrics['queries_processed']
            self.performance_metrics['context_hit_rate'] = new_hit_rate
            
            # Store basic conversation
            self.db_manager.store_chat_message(
                session_id, conversation_id, 'user', message,
                tokens_used=0, processing_time_ms=int(processing_time * 1000)
            )
            
            self.db_manager.store_chat_message(
                session_id, conversation_id, 'assistant', response,
                context_used={
                    'query_type': query_plan.query_type.value,
                    'entities': query_plan.entities,
                    'contexts_count': len(contexts),
                    'retrieval_methods': [ctx.retrieval_method for ctx in contexts]
                },
                tokens_used=0, processing_time_ms=int(processing_time * 1000)
            )
            
            # Store RAG-specific metrics if table exists
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO rag_query_metrics 
                        (session_id, query, query_type, entities_found, contexts_retrieved,
                         confidence_score, processing_time_ms, response_length, retrieval_methods)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id, message, query_plan.query_type.value,
                        len(query_plan.entities), len(contexts), query_plan.confidence,
                        int(processing_time * 1000), len(response),
                        json.dumps([ctx.retrieval_method for ctx in contexts])
                    ))
            except Exception as e:
                logger.debug(f"Could not store RAG metrics: {e}")
            
        except Exception as e:
            logger.error(f"Error logging conversation with metrics: {str(e)}")
    
    def _analyze_session_content(self, session_id: str):
        """Analyze session content to optimize retrieval"""
        try:
            metrics = self.db_manager.get_session_metrics(session_id)
            components = self.db_manager.get_session_components(session_id)
            
            logger.info(f"Session content: {metrics.get('total_components', 0)} components, "
                       f"{metrics.get('total_fields', 0)} fields, "
                       f"{metrics.get('total_lines', 0)} lines")
            
            # Identify high-value components for priority retrieval
            self._high_value_components = []
            for comp in components:
                if (comp.get('business_purpose') and 
                    len(comp.get('business_purpose', '')) > 50 and
                    comp.get('total_lines', 0) > 100):
                    self._high_value_components.append(comp['component_name'])
            
            logger.info(f"High-value components identified: {len(self._high_value_components)}")
            
        except Exception as e:
            logger.error(f"Error analyzing session content: {str(e)}")
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def clear_cache(self, session_id: str = None):
        """Clear response cache"""
        if session_id:
            self._response_cache.pop(session_id, None)
            logger.info(f"Cleared cache for session {session_id}")
        else:
            self._response_cache.clear()
            logger.info("Cleared all caches")

# Factory function for easy integration
def create_agentic_rag_chat_manager(llm_client, db_manager, fallback_chat_manager=None):
    """Factory function to create Agentic RAG Chat Manager"""
    return AgenticRAGChatManager(llm_client, db_manager, fallback_chat_manager)