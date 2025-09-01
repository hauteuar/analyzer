"""
Advanced RAG Features: Caching, Optimization, and Intelligent Routing
"""

import time
import hashlib
from functools import lru_cache
from typing import Dict, List, Optional, Any
import threading
from collections import defaultdict, deque

class IntelligentContextCache:
    """Smart caching system for frequently accessed contexts"""
    
    def __init__(self, max_size: int = 200, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.RLock()
        
    def get(self, cache_key: str) -> Optional[RetrievedContext]:
        """Get context from cache with LRU and TTL"""
        with self.lock:
            if cache_key not in self.cache:
                return None
                
            # Check TTL
            if time.time() - self.access_times[cache_key] > self.ttl_seconds:
                self._remove_key(cache_key)
                return None
            
            # Update access
            self.access_times[cache_key] = time.time()
            self.access_counts[cache_key] += 1
            
            return self.cache[cache_key]
    
    def put(self, cache_key: str, context: RetrievedContext):
        """Store context in cache with intelligent eviction"""
        with self.lock:
            # Evict if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_least_valuable()
            
            self.cache[cache_key] = context
            self.access_times[cache_key] = time.time()
            self.access_counts[cache_key] = 1
    
    def _evict_least_valuable(self):
        """Evict least valuable items based on access patterns"""
        if not self.cache:
            return
            
        # Calculate value scores (access_count / age)
        current_time = time.time()
        value_scores = {}
        
        for key in self.cache:
            age = current_time - self.access_times[key]
            access_count = self.access_counts[key]
            value_scores[key] = access_count / (age + 1)  # +1 to avoid division by zero
        
        # Remove lowest value items
        sorted_items = sorted(value_scores.items(), key=lambda x: x[1])
        items_to_remove = max(1, len(self.cache) // 10)  # Remove 10% or at least 1
        
        for key, _ in sorted_items[:items_to_remove]:
            self._remove_key(key)
    
    def _remove_key(self, key: str):
        """Remove key from all data structures"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'total_accesses': sum(self.access_counts.values()),
                'unique_keys': len(self.access_counts),
                'avg_access_count': sum(self.access_counts.values()) / len(self.access_counts) if self.access_counts else 0
            }

class QueryRouterAgent:
    """Intelligent agent that routes queries to specialized handlers"""
    
    def __init__(self, llm_client, db_manager):
        self.llm_client = llm_client
        self.db_manager = db_manager
        self.specialized_handlers = {}
        self.query_history = deque(maxlen=100)  # Track recent queries
        
    def register_handler(self, query_pattern: str, handler_class):
        """Register specialized handler for specific query patterns"""
        self.specialized_handlers[query_pattern] = handler_class
    
    def route_query(self, session_id: str, message: str, query_plan: QueryPlan) -> Optional[str]:
        """Route query to specialized handler if applicable"""
        
        try:
            # Check for specialized routing patterns
            message_lower = message.lower()
            
            # Field calculation queries
            if any(term in message_lower for term in ['calculate', 'computed', 'derived', 'formula']):
                if query_plan.query_type == QueryType.FIELD_ANALYSIS:
                    return self._handle_field_calculation_query(session_id, message, query_plan)
            
            # Performance analysis queries
            elif any(term in message_lower for term in ['performance', 'slow', 'optimize', 'bottleneck']):
                return self._handle_performance_query(session_id, message, query_plan)
            
            # Migration-specific queries
            elif any(term in message_lower for term in ['migrate', 'convert', 'modernize', 'oracle']):
                return self._handle_migration_query(session_id, message, query_plan)
            
            # Business process queries
            elif any(term in message_lower for term in ['process', 'workflow', 'business', 'what does']):
                return self._handle_business_process_query(session_id, message, query_plan)
            
            return None  # No specialized routing needed
            
        except Exception as e:
            logger.error(f"Error in query routing: {str(e)}")
            return None
    
    def _handle_field_calculation_query(self, session_id: str, message: str, query_plan: QueryPlan) -> str:
        """Specialized handler for field calculation queries"""
        
        try:
            # Get field calculation contexts
            calculation_contexts = []
            
            for entity in query_plan.entities:
                # Get field with arithmetic operations
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT field_name, program_name, field_references_json, 
                               arithmetic_count, definition_code, business_purpose
                        FROM field_analysis_details 
                        WHERE session_id = ? AND field_name LIKE ? AND arithmetic_count > 0
                        ORDER BY arithmetic_count DESC
                    ''', (session_id, f'%{entity}%'))
                    
                    calc_fields = cursor.fetchall()
                    
                    for field_row in calc_fields:
                        field_data = dict(field_row)
                        
                        # Get source code with calculation logic
                        source_data = self.db_manager.get_component_source_code(
                            session_id, field_data['program_name'], max_size=100000
                        )
                        
                        if source_data.get('success'):
                            for comp in source_data['components']:
                                source_code = comp.get('source_for_chat', '')
                                
                                # Extract calculation-specific sections
                                calc_sections = self._extract_calculation_sections(source_code, entity)
                                
                                calculation_contexts.append({
                                    'field_name': field_data['field_name'],
                                    'program_name': field_data['program_name'],
                                    'calculation_sections': calc_sections,
                                    'arithmetic_count': field_data['arithmetic_count'],
                                    'definition_code': field_data['definition_code'],
                                    'business_purpose': field_data['business_purpose']
                                })
            
            # Generate specialized calculation response
            if calculation_contexts:
                return self._generate_calculation_response(message, calculation_contexts)
            else:
                return None  # Fall back to standard processing
                
        except Exception as e:
            logger.error(f"Error in field calculation handler: {str(e)}")
            return None
    
    def _extract_calculation_sections(self, source_code: str, field_name: str) -> List[str]:
        """Extract sections of code that show field calculations"""
        calc_sections = []
        lines = source_code.split('\n')
        field_upper = field_name.upper()
        
        for i, line in enumerate(lines):
            line_upper = line.strip().upper()
            
            # Look for calculation patterns involving the field
            if (field_upper in line_upper and 
                any(calc_word in line_upper for calc_word in ['COMPUTE', 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE'])):
                
                # Get context around calculation
                start_idx = max(0, i - 2)
                end_idx = min(len(lines), i + 3)
                
                context_block = '\n'.join([
                    f"{start_idx + j + 1:4d}: {lines[start_idx + j]}"
                    for j in range(end_idx - start_idx)
                ])
                
                calc_sections.append(context_block)
        
        return calc_sections
    
    def _generate_calculation_response(self, message: str, calc_contexts: List[Dict]) -> str:
        """Generate specialized response for calculation queries"""
        
        prompt = f"""
You are a COBOL calculation analysis expert. The user asked: "{message}"

Analyze these field calculations and provide a detailed explanation:

"""
        
        for i, ctx in enumerate(calc_contexts, 1):
            prompt += f"""
CALCULATION CONTEXT {i}:
Field: {ctx['field_name']}
Program: {ctx['program_name']}
Arithmetic Operations: {ctx['arithmetic_count']}
Definition: {ctx['definition_code']}
Business Purpose: {ctx['business_purpose']}

Calculation Code Sections:
{chr(10).join(ctx['calculation_sections'])}

"""
        
        prompt += """
Please provide:
1. Detailed explanation of how the field is calculated
2. Business logic behind the calculations
3. Input fields and their sources
4. Step-by-step calculation process
5. Any business rules or validations applied

Focus on the mathematical and business logic aspects.
"""
        
        try:
            response = self.llm_client.call_llm(prompt, max_tokens=1500, temperature=0.1)
            if response.success:
                return response.content
        except Exception as e:
            logger.error(f"Error generating calculation response: {str(e)}")
        
        return None

class ContextQualityAnalyzer:
    """Analyze and improve the quality of retrieved contexts"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def analyze_context_quality(self, contexts: List[RetrievedContext], query_plan: QueryPlan) -> Dict:
        """Analyze the quality of retrieved contexts"""
        
        analysis = {
            'total_contexts': len(contexts),
            'avg_relevance': sum(ctx.relevance_score for ctx in contexts) / len(contexts) if contexts else 0,
            'context_coverage': {},
            'missing_elements': [],
            'quality_score': 0.0
        }
        
        try:
            # Analyze coverage for query requirements
            requirements = query_plan.context_requirements
            
            # Check if we have enough contexts
            min_contexts = requirements.get('min_contexts', 1)
            max_contexts = requirements.get('max_contexts', 5)
            
            if len(contexts) < min_contexts:
                analysis['missing_elements'].append(f"Insufficient contexts: {len(contexts)} < {min_contexts}")
            
            # Check relevance threshold
            min_relevance = requirements.get('min_relevance', 0.3)
            low_relevance_count = sum(1 for ctx in contexts if ctx.relevance_score < min_relevance)
            
            if low_relevance_count > 0:
                analysis['missing_elements'].append(f"{low_relevance_count} contexts below relevance threshold")
            
            # Analyze source code coverage
            total_source_length = sum(len(ctx.source_code) for ctx in contexts)
            analysis['context_coverage']['total_source_length'] = total_source_length
            
            if total_source_length < 1000:
                analysis['missing_elements'].append("Limited source code content")
            
            # Check for entity coverage
            entities_covered = set()
            for ctx in contexts:
                for entity in query_plan.entities:
                    if entity.upper() in ctx.source_code.upper():
                        entities_covered.add(entity)
            
            entity_coverage_rate = len(entities_covered) / len(query_plan.entities) if query_plan.entities else 1.0
            analysis['context_coverage']['entity_coverage_rate'] = entity_coverage_rate
            
            if entity_coverage_rate < 0.5:
                analysis['missing_elements'].append(f"Low entity coverage: {entity_coverage_rate:.1%}")
            
            # Calculate overall quality score
            quality_factors = [
                min(len(contexts) / max_contexts, 1.0),  # Context count factor
                analysis['avg_relevance'],  # Relevance factor
                entity_coverage_rate,  # Entity coverage factor
                min(total_source_length / 10000, 1.0)  # Source code factor
            ]
            
            analysis['quality_score'] = sum(quality_factors) / len(quality_factors)
            
        except Exception as e:
            logger.error(f"Error analyzing context quality: {str(e)}")
            analysis['error'] = str(e)
        
        return analysis

class AdaptiveResponseGenerator(ResponseGenerator):
    """Response generator that adapts based on context quality and user feedback"""
    
    def __init__(self, llm_client):
        super().__init__(llm_client)
        self.response_patterns = {}
        self.user_feedback = {}
        
    def generate_response(self, message: str, query_plan: QueryPlan, contexts: List[RetrievedContext]) -> str:
        """Generate adaptive response based on context quality"""
        
        # Analyze context quality
        quality_analyzer = ContextQualityAnalyzer(None)
        quality_analysis = quality_analyzer.analyze_context_quality(contexts, query_plan)
        
        # Adapt response strategy based on quality
        if quality_analysis['quality_score'] < 0.4:
            return self._generate_low_quality_response(message, query_plan, contexts, quality_analysis)
        elif quality_analysis['quality_score'] > 0.8:
            return self._generate_high_quality_response(message, query_plan, contexts)
        else:
            return self._generate_standard_response(message, query_plan, contexts)
    
    def _generate_low_quality_response(self, message: str, query_plan: QueryPlan, 
                                     contexts: List[RetrievedContext], quality_analysis: Dict) -> str:
        """Generate response when context quality is low"""
        
        missing_elements = quality_analysis.get('missing_elements', [])
        
        if contexts:
            # Try to answer with available context but acknowledge limitations
            response = super().generate_response(message, query_plan, contexts)
            
            # Add quality disclaimer
            response += f"\n\n**Note**: This analysis is based on limited context. "
            response += f"Missing elements: {', '.join(missing_elements)}"
            response += f"\n\nFor a more complete analysis, consider:"
            response += f"\n• Uploading additional related COBOL files"
            response += f"\n• Providing more specific field or program names"
            response += f"\n• Asking about components that have been fully analyzed"
            
            return response
        else:
            return self._generate_no_context_response(message, query_plan)
    
    def _generate_high_quality_response(self, message: str, query_plan: QueryPlan, contexts: List[RetrievedContext]) -> str:
        """Generate comprehensive response when context quality is high"""
        
        # Build enhanced prompt with more detailed instructions
        prompt = self._build_comprehensive_prompt(message, query_plan, contexts)
        
        # Use higher token limit for detailed response
        max_tokens = 2500
        temperature = 0.1 if query_plan.query_type == QueryType.FIELD_ANALYSIS else 0.2
        
        response = self.llm_client.call_llm(prompt, max_tokens=max_tokens, temperature=temperature)
        
        if response.success:
            # Add confidence indicator for high-quality responses
            result = response.content
            result += f"\n\n**High Confidence Analysis** based on {len(contexts)} comprehensive code contexts"
            return result
        else:
            return super().generate_response(message, query_plan, contexts)
    
    def _build_comprehensive_prompt(self, message: str, query_plan: QueryPlan, contexts: List[RetrievedContext]) -> str:
        """Build comprehensive prompt for high-quality responses"""
        
        prompt_parts = [
            "You are a senior COBOL architect and business analyst for a wealth management system.",
            "The user has a detailed question and you have comprehensive source code context.",
            "",
            f"USER QUESTION: {message}",
            f"QUERY TYPE: {query_plan.query_type.value}",
            f"ENTITIES: {', '.join(query_plan.entities)}",
            f"CONFIDENCE: {query_plan.confidence:.1%}",
            ""
        ]
        
        # Add detailed analysis instructions based on query type
        if query_plan.query_type == QueryType.FIELD_ANALYSIS:
            prompt_parts.extend([
                "ANALYSIS REQUIREMENTS:",
                "1. Field Definition Analysis:",
                "   - Exact PIC clause and storage requirements",
                "   - Data type and length implications",
                "   - Business meaning and purpose",
                "",
                "2. Usage Pattern Analysis:",
                "   - How the field receives data (input sources)",
                "   - How the field provides data (output targets)",
                "   - Calculation logic if applicable",
                "   - Business rules and validations",
                "",
                "3. Context Analysis:",
                "   - Position in record layout",
                "   - Relationship to other fields",
                "   - Integration points with other systems",
                "",
                "4. Migration Implications:",
                "   - Oracle data type mapping",
                "   - Potential data conversion issues",
                "   - Business logic preservation requirements",
                ""
            ])
        
        # Add all available source code contexts
        prompt_parts.append("COMPREHENSIVE SOURCE CODE CONTEXTS:")
        prompt_parts.append("")
        
        for i, context in enumerate(contexts, 1):
            metadata = context.metadata
            prompt_parts.extend([
                f"=== SOURCE CONTEXT {i} (Relevance: {context.relevance_score:.2f}) ===",
                f"Component: {metadata.get('component_name')} ({metadata.get('component_type')})",
                f"Lines: {metadata.get('total_lines', 0)}",
                f"Retrieval Method: {context.retrieval_method}",
                ""
            ])
            
            if metadata.get('field_details'):
                field_details = metadata['field_details']
                prompt_parts.extend([
                    "FIELD DETAILS:",
                    f"  Definition: {field_details.get('definition_code', 'N/A')}",
                    f"  Usage Type: {field_details.get('usage_type', 'N/A')}",
                    f"  References: {field_details.get('total_program_references', 0)}",
                    ""
                ])
            
            prompt_parts.extend([
                "FULL SOURCE CODE:",
                context.source_code,
                "",
                "=" * 80,
                ""
            ])
        
        prompt_parts.extend([
            "",
            "INSTRUCTIONS:",
            "- Provide a comprehensive, detailed analysis using ALL available source code",
            "- Reference specific line numbers and code examples",
            "- Explain business impact and technical implications",
            "- Include concrete code snippets that demonstrate your points",
            "- Structure your response with clear sections",
            "- Focus on actionable insights for modernization efforts"
        ])
        
        return '\n'.join(prompt_parts)
    
    def _generate_no_context_response(self, message: str, query_plan: QueryPlan) -> str:
        """Generate helpful response when no context is available"""
        
        entity_hint = ""
        if query_plan.entities:
            entity_hint = f" for '{', '.join(query_plan.entities)}'"
        
        return (
            f"I couldn't find specific code context{entity_hint} to answer your question.\n\n"
            f"This might be because:\n"
            f"• The mentioned components haven't been analyzed yet\n"
            f"• The field/program names might be slightly different\n"
            f"• More COBOL files need to be uploaded\n\n"
            f"Suggestions:\n"
            f"• Try asking 'What components are available?'\n"
            f"• Upload more related COBOL files\n"
            f"• Be more specific with exact field/program names\n"
            f"• Ask about general program structure first"
        )

class SmartContextPrioritizer:
    """Prioritize contexts based on user intent and historical patterns"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.priority_weights = {
            'recency': 0.2,
            'relevance': 0.4,
            'completeness': 0.2,
            'user_preference': 0.2
        }
    
    def prioritize_contexts(self, contexts: List[RetrievedContext], 
                          query_plan: QueryPlan, session_history: List[Dict]) -> List[RetrievedContext]:
        """Intelligently prioritize contexts based on multiple factors"""
        
        scored_contexts = []
        
        for context in contexts:
            score = self._calculate_priority_score(context, query_plan, session_history)
            scored_contexts.append((score, context))
        
        # Sort by score and return contexts
        scored_contexts.sort(key=lambda x: x[0], reverse=True)
        return [context for score, context in scored_contexts]
    
    def _calculate_priority_score(self, context: RetrievedContext, 
                                query_plan: QueryPlan, session_history: List[Dict]) -> float:
        """Calculate priority score for a context"""
        
        score = 0.0
        
        # Base relevance score
        score += context.relevance_score * self.priority_weights['relevance']
        
        # Completeness score (more complete source code is better)
        source_length = len(context.source_code)
        completeness = min(source_length / 10000, 1.0)  # Normalize to 10KB
        score += completeness * self.priority_weights['completeness']
        
        # User preference (favor components user has asked about before)
        component_name = context.metadata.get('component_name', '')
        historical_interest = self._get_historical_interest(component_name, session_history)
        score += historical_interest * self.priority_weights['user_preference']
        
        # Recency (favor recently accessed contexts)
        recency = self._get_recency_score(context, session_history)
        score += recency * self.priority_weights['recency']
        
        return score
    
    def _get_historical_interest(self, component_name: str, session_history: List[Dict]) -> float:
        """Calculate historical user interest in a component"""
        if not component_name or not session_history:
            return 0.0
        
        mentions = sum(1 for chat in session_history 
                      if component_name.lower() in chat.get('message', '').lower())
        
        return min(mentions / 10, 1.0)  # Normalize to max 10 mentions
    
    def _get_recency_score(self, context: RetrievedContext, session_history: List[Dict]) -> float:
        """Calculate recency score based on last access"""
        # Implementation depends on how you track context access
        return 0.5  # Default neutral score

class RAGSystemAnalytics:
    """Analytics and monitoring for the RAG system"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def generate_session_analytics(self, session_id: str) -> Dict:
        """Generate comprehensive analytics for a session"""
        
        analytics = {
            'session_overview': {},
            'query_patterns': {},
            'context_effectiveness': {},
            'user_satisfaction': {},
            'optimization_suggestions': []
        }
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Session overview
                cursor.execute('''
                    SELECT COUNT(*) as total_queries, 
                           AVG(processing_time_ms) as avg_processing_time,
                           AVG(confidence_score) as avg_confidence,
                           AVG(contexts_retrieved) as avg_contexts
                    FROM rag_query_metrics 
                    WHERE session_id = ?
                ''', (session_id,))
                
                overview = cursor.fetchone()
                if overview:
                    analytics['session_overview'] = dict(overview)
                
                # Query type distribution
                cursor.execute('''
                    SELECT query_type, COUNT(*) as count
                    FROM rag_query_metrics 
                    WHERE session_id = ?
                    GROUP BY query_type
                    ORDER BY count DESC
                ''', (session_id,))
                
                analytics['query_patterns']['type_distribution'] = dict(cursor.fetchall())
                
                # Context effectiveness
                cursor.execute('''
                    SELECT retrieval_methods, AVG(confidence_score) as effectiveness
                    FROM rag_query_metrics 
                    WHERE session_id = ? AND retrieval_methods IS NOT NULL
                    GROUP BY retrieval_methods
                ''', (session_id,))
                
                analytics['context_effectiveness'] = dict(cursor.fetchall())
                
                # Generate optimization suggestions
                analytics['optimization_suggestions'] = self._generate_optimization_suggestions(analytics)
                
        except Exception as e:
            logger.error(f"Error generating session analytics: {str(e)}")
            analytics['error'] = str(e)
        
        return analytics
    
    def _generate_optimization_suggestions(self, analytics: Dict) -> List[str]:
        """Generate optimization suggestions based on analytics"""
        suggestions = []
        
        overview = analytics.get('session_overview', {})
        query_patterns = analytics.get('query_patterns', {})
        
        # Processing time suggestions
        avg_time = overview.get('avg_processing_time', 0)
        if avg_time > 5000:  # > 5 seconds
            suggestions.append("Consider reducing context size or enabling caching for faster responses")
        
        # Confidence suggestions
        avg_confidence = overview.get('avg_confidence', 0)
        if avg_confidence < 0.6:
            suggestions.append("Low confidence scores suggest more source code context is needed")
        
        # Query pattern suggestions
        type_dist = query_patterns.get('type_distribution', {})
        if type_dist.get('field_analysis', 0) > 10:
            suggestions.append("Many field queries detected - consider pre-indexing field definitions")
        
        if not suggestions:
            suggestions.append("System performance is optimal")
        
        return suggestions

# Complete implementation example
class ProductionRAGSystem:
    """Production-ready RAG system with all features"""
    
    def __init__(self, llm_client, db_manager, config=None):
        self.config = config or RAGConfig()
        
        # Core components
        self.vector_store = VectorStore(db_manager) if self.config.ENABLE_VECTOR_SEARCH else None
        self.query_analyzer = AdvancedQueryAnalyzer(llm_client)
        self.context_retriever = EnhancedContextRetriever(db_manager, self.vector_store)
        self.response_generator = AdaptiveResponseGenerator(llm_client)
        self.query_router = QueryRouterAgent(llm_client, db_manager)
        self.context_cache = IntelligentContextCache()
        self.analytics = RAGSystemAnalytics(db_manager)
        
        # Performance tracking
        self.start_time = time.time()
        self.total_queries = 0
        self.successful_queries = 0
        
    def process_query_with_full_features(self, session_id: str, message: str, conversation_id: str) -> Dict:
        """Process query with all RAG features enabled"""
        
        query_start = time.time()
        self.total_queries += 1
        
        try:
            # Step 1: Query Analysis
            query_plan = self.query_analyzer.analyze_query(message, session_id)
            
            # Step 2: Check cache first
            cache_key = self._generate_cache_key(message, query_plan)
            cached_response = self.context_cache.get(cache_key)
            
            if cached_response and self.config.ENABLE_QUERY_CACHING:
                return {
                    "response": cached_response.source_code,  # Assuming cached response format
                    "cached": True,
                    "processing_time": time.time() - query_start
                }
            
            # Step 3: Check for specialized routing
            specialized_response = self.query_router.route_query(session_id, message, query_plan)
            
            if specialized_response:
                self.successful_queries += 1
                return {
                    "response": specialized_response,
                    "routed": True,
                    "processing_time": time.time() - query_start
                }
            
            # Step 4: Standard RAG pipeline
            retrieved_contexts = self.context_retriever.retrieve_contexts(session_id, query_plan)
            
            # Step 5: Generate response
            response = self.response_generator.generate_response(message, query_plan, retrieved_contexts)
            
            # Step 6: Cache if appropriate
            if self.config.ENABLE_QUERY_CACHING and query_plan.confidence > 0.7:
                self.context_cache.put(cache_key, RetrievedContext(
                    source_code=response,
                    metadata={'cached_at': time.time()},
                    relevance_score=query_plan.confidence,
                    retrieval_method="cached"
                ))
            
            self.successful_queries += 1
            processing_time = time.time() - query_start
            
            return {
                "response": response,
                "query_plan": {
                    "type": query_plan.query_type.value,
                    "entities": query_plan.entities,
                    "confidence": query_plan.confidence
                },
                "contexts_used": len(retrieved_contexts),
                "processing_time": processing_time,
                "cached": False,
                "routed": False
            }
            
        except Exception as e:
            logger.error(f"Error in full-featured query processing: {str(e)}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "error": True,
                "processing_time": time.time() - query_start
            }
    
    def _generate_cache_key(self, message: str, query_plan: QueryPlan) -> str:
        """Generate cache key for query"""
        content = f"{message}_{query_plan.query_type.value}_{sorted(query_plan.entities)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_system_health(self) -> Dict:
        """Get comprehensive system health metrics"""
        
        uptime = time.time() - self.start_time
        success_rate = self.successful_queries / self.total_queries if self.total_queries > 0 else 0
        
        return {
            'uptime_seconds': uptime,
            'total_queries': self.total_queries,
            'successful_queries': self.successful_queries,
            'success_rate': success_rate,
            'cache_stats': self.context_cache.get_stats() if self.context_cache else {},
            'vector_store_ready': self.vector_store is not None,
            'config': {
                'vector_search_enabled': self.config.ENABLE_VECTOR_SEARCH,
                'caching_enabled': self.config.ENABLE_QUERY_CACHING,
                'max_contexts': self.config.MAX_CONTEXTS_PER_QUERY
            }
        }

# Complete usage example with error handling
def deploy_rag_system():
    """Complete deployment script for RAG system"""
    
    try:
        # 1. Setup environment
        if not setup_rag_environment():
            logger.error("Failed to setup RAG environment")
            return None
        
        # 2. Initialize components
        llm_client = LLMClient()
        db_manager = DatabaseManager()
        db_manager.initialize_database()
        
        # 3. Update database for RAG
        update_database_for_rag(db_manager)
        
        # 4. Create production RAG system
        rag_system = ProductionRAGSystem(llm_client, db_manager, RAGConfig())
        
        # 5. Test system
        logger.info("Testing RAG system...")
        # Run basic health check
        health = rag_system.get_system_health()
        logger.info(f"System health: {health}")
        
        logger.info("RAG system deployed successfully")
        return rag_system
        
    except Exception as e:
        logger.error(f"Failed to deploy RAG system: {str(e)}")
        return None

# Usage in your main application
"""
# Initialize once at startup
rag_system = deploy_rag_system()

# For each file upload
rag_system.initialize_session(session_id)

# For each chat query
result = rag_system.process_query_with_full_features(
    session_id, user_message, conversation_id
)

# Monitor performance
health = rag_system.get_system_health()
analytics = rag_system.analytics.generate_session_analytics(session_id)
"""