"""
Chat Manager Module
Handles intelligent chat with context awareness for fields and record layouts
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatManager:
    def __init__(self, llm_client, token_manager, db_manager):
        self.llm_client = llm_client
        self.token_manager = token_manager
        self.db_manager = db_manager
        
        # Context allocation - 60% for context, 40% for conversation
        self.context_token_budget = int(self.token_manager.EFFECTIVE_CONTENT_LIMIT * 0.6)
        self.conversation_token_budget = int(self.token_manager.EFFECTIVE_CONTENT_LIMIT * 0.4)
        
        # Patterns for detecting query types
        self.field_query_patterns = [
            r'\bfield\s+(\w+[-\w]*)',
            r'\b([A-Z]+[-A-Z0-9]*)\s+field\b',
            r'\bwhat\s+is\s+(\w+[-\w]*)',
            r'\btell\s+me\s+about\s+(\w+[-\w]*)',
            r'\bshow\s+(\w+[-\w]*)'
        ]
        
        self.record_layout_patterns = [
            r'\brecord\s+(\w+[-\w]*)',
            r'\blayout\s+(\w+[-\w]*)',
            r'\bstructure\s+(\w+[-\w]*)',
            r'\b(\w+[-\w]*)\s+record\b',
            r'\b(\w+[-\w]*)\s+layout\b'
        ]
        
        self.program_query_patterns = [
            r'\bprogram\s+(\w+[-\w]*)',
            r'\b(\w+[-\w]*)\s+program\b',
            r'\bmodule\s+(\w+[-\w]*)',
            r'\bcomponent\s+(\w+[-\w]*)'
        ]
    
    # EMERGENCY DEBUG VERSION - Replace process_query method in ChatManager

    def process_query(self, session_id: str, message: str, conversation_id: str) -> Dict:
        """Process chat query with enhanced error handling"""
        logger.info(f"Processing chat query: {message[:100]}...")
        
        try:
            # Step 1: Analyze query
            logger.debug("Step 1: Analyzing query...")
            query_analysis = self._analyze_query(message)
            logger.debug(f"Query analysis: {query_analysis}")
            
            # Step 2: Build context with error protection
            logger.debug("Step 2: Building context...")
            try:
                context = self._build_context_safe(session_id, query_analysis)
                logger.debug(f"Context built successfully with {len(context)} sections")
            except Exception as context_error:
                logger.error(f"Error building context: {str(context_error)}")
                # Use minimal context as fallback
                context = {'error': 'Context building failed', 'field_details': [], 'components': []}
            
            # Step 3: Get conversation history
            logger.debug("Step 3: Getting conversation history...")
            try:
                conversation_history = self.db_manager.get_chat_history(session_id, conversation_id, limit=3)
            except Exception as history_error:
                logger.error(f"Error getting history: {str(history_error)}")
                conversation_history = []
            
            # Step 4: Create prompt
            logger.debug("Step 4: Creating chat prompt...")
            try:
                chat_prompt = self._create_chat_prompt_safe(message, context, conversation_history, query_analysis)
            except Exception as prompt_error:
                logger.error(f"Error creating prompt: {str(prompt_error)}")
                # Fallback to simple prompt
                chat_prompt = f"User question: {message}\n\nPlease provide a helpful response about mainframe code analysis."
            
            # Step 5: Call LLM
            logger.debug("Step 5: Calling LLM...")
            start_time = datetime.now()
            response = self.llm_client.call_llm(chat_prompt, max_tokens=1000, temperature=0.3)
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Step 6: Store conversation
            logger.debug("Step 6: Storing conversation...")
            try:
                self.db_manager.store_chat_message(
                    session_id, conversation_id, 'user', message, 
                    context, response.prompt_tokens if response.success else 0, processing_time
                )
            except Exception as store_error:
                logger.error(f"Error storing user message: {str(store_error)}")
            
            if response.success:
                try:
                    self.db_manager.store_chat_message(
                        session_id, conversation_id, 'assistant', response.content,
                        None, response.response_tokens, processing_time
                    )
                except Exception as store_error:
                    logger.error(f"Error storing assistant message: {str(store_error)}")
                
                return {
                    'response': response.content,
                    'context_used': context,
                    'query_type': query_analysis['type'],
                    'entities_found': query_analysis['entities'],
                    'tokens_used': response.prompt_tokens + response.response_tokens,
                    'processing_time_ms': processing_time
                }
            else:
                return {
                    'response': f"I encountered an error processing your question: {response.error_message}",
                    'error': True
                }
                
        except Exception as e:
            logger.error(f"Error processing chat query: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            return {
                'response': "I encountered an unexpected error. Please try again with a simpler question.",
                'error': True,
                'error_detail': str(e)
            }

    def _build_context_safe(self, session_id: str, query_analysis: Dict) -> Dict:
        """Safe context building with proper error handling"""
        context = {
            'field_details': [],
            'field_mappings': [],
            'record_layouts': [],
            'components': [],
            'source_code_snippets': []
        }
        
        try:
            entities = query_analysis.get('entities', [])
            query_type = query_analysis.get('type', 'GENERAL')
            
            logger.debug(f"Building context for {len(entities)} entities, query type: {query_type}")
            
            # Handle field queries
            if 'FIELD' in query_type and entities:
                for entity in entities[:3]:  # Limit to 3 entities
                    try:
                        logger.debug(f"Getting field context for: {entity}")
                        field_context = self.db_manager.get_context_for_field(session_id, entity)
                        
                        if field_context and isinstance(field_context, dict):
                            field_details = field_context.get('field_details', [])
                            field_mappings = field_context.get('field_mappings', [])
                            
                            context['field_details'].extend(field_details)
                            context['field_mappings'].extend(field_mappings)
                            
                            logger.debug(f"Added {len(field_details)} field details for {entity}")
                            
                    except Exception as field_error:
                        logger.error(f"Error processing field entity {entity}: {str(field_error)}")
                        continue
            
            # Handle layout queries
            if 'LAYOUT' in query_type and entities:
                try:
                    logger.debug("Getting record layouts...")
                    all_layouts = self.db_manager.get_record_layouts(session_id)
                    
                    for entity in entities[:3]:
                        matching_layouts = []
                        for layout in all_layouts:
                            # Safe field access with fallback
                            layout_name = layout.get('layout_name') or layout.get('name', '')
                            if layout_name and entity.upper() in layout_name.upper():
                                matching_layouts.append(layout)
                        
                        context['record_layouts'].extend(matching_layouts)
                        logger.debug(f"Found {len(matching_layouts)} matching layouts for {entity}")
                    
                except Exception as layout_error:
                    logger.error(f"Error processing layout queries: {str(layout_error)}")
            
            # Handle program queries
            if 'PROGRAM' in query_type and entities:
                try:
                    logger.debug("Getting components...")
                    all_components = self.db_manager.get_session_components(session_id)
                    
                    for entity in entities[:3]:
                        matching_components = []
                        for component in all_components:
                            # Safe field access with fallback
                            comp_name = component.get('component_name') or component.get('name', '')
                            if comp_name and entity.upper() in comp_name.upper():
                                # Create safe component dict
                                safe_component = {
                                    'name': comp_name,
                                    'component_name': comp_name,  # Provide both for compatibility
                                    'type': component.get('component_type') or component.get('type', 'Unknown'),
                                    'component_type': component.get('component_type') or component.get('type', 'Unknown'),
                                    'total_lines': component.get('total_lines', 0),
                                    'analysis_result_json': component.get('analysis_result_json', '{}')
                                }
                                matching_components.append(safe_component)
                        
                        context['components'].extend(matching_components)
                        logger.debug(f"Found {len(matching_components)} matching components for {entity}")
                    
                except Exception as program_error:
                    logger.error(f"Error processing program queries: {str(program_error)}")
            
            # Get general session context if no specific entities
            if not entities:
                try:
                    session_metrics = self.db_manager.get_session_metrics(session_id)
                    context['session_summary'] = session_metrics
                    
                    # Get a few recent components for context
                    recent_components = self.db_manager.get_session_components(session_id)
                    if recent_components:
                        context['recent_components'] = recent_components[:3]
                    
                except Exception as general_error:
                    logger.error(f"Error building general context: {str(general_error)}")
            
            logger.debug(f"Context building completed successfully")
            return context
            
        except Exception as e:
            logger.error(f"Error in _build_context_safe: {str(e)}")
            return context

    def _create_chat_prompt_safe(self, message: str, context: Dict, 
                            conversation_history: List[Dict], query_analysis: Dict) -> str:
        """Create chat prompt with safe field access"""
        
        system_prompt = """You are a mainframe code analysis assistant. Help users understand COBOL programs, record layouts, and field relationships.

    Guidelines:
    - Reference specific code examples when available
    - Explain both technical and business aspects
    - Use clear, professional language
    - Focus on practical insights for mainframe developers
    """
        
        # Build context section safely
        context_prompt = "\nAVAILABLE CONTEXT:\n"
        
        # Add field details
        field_details = context.get('field_details', [])
        if field_details:
            context_prompt += f"\nFIELD INFORMATION ({len(field_details)} fields):\n"
            for field in field_details[:3]:  # Limit to prevent token overflow
                field_name = field.get('field_name', 'Unknown')
                usage_type = field.get('usage_type', 'Unknown')
                program_name = field.get('program_name', 'Unknown')
                business_purpose = field.get('business_purpose', 'No description')
                
                context_prompt += f"- Field: {field_name}\n"
                context_prompt += f"  Usage: {usage_type} in {program_name}\n"
                context_prompt += f"  Purpose: {business_purpose}\n\n"
        
        # Add component information
        components = context.get('components', [])
        if components:
            context_prompt += f"\nCOMPONENT INFORMATION ({len(components)} components):\n"
            for comp in components[:2]:
                comp_name = comp.get('name') or comp.get('component_name', 'Unknown')
                comp_type = comp.get('type') or comp.get('component_type', 'Unknown')
                total_lines = comp.get('total_lines', 0)
                
                context_prompt += f"- Component: {comp_name} ({comp_type})\n"
                context_prompt += f"  Lines: {total_lines}\n\n"
        
        # Add session summary
        session_summary = context.get('session_summary', {})
        if session_summary:
            context_prompt += f"\nSESSION OVERVIEW:\n"
            context_prompt += f"- Total Components: {session_summary.get('total_components', 0)}\n"
            context_prompt += f"- Total Fields: {session_summary.get('total_fields', 0)}\n\n"
        
        # Add conversation history
        if conversation_history:
            context_prompt += "\nRECENT CONVERSATION:\n"
            for msg in conversation_history[-2:]:  # Last 2 messages
                role = "User" if msg['message_type'] == 'user' else "Assistant"
                content = msg['message_content'][:150]  # Limit length
                context_prompt += f"{role}: {content}\n"
        
        # Final prompt
        final_prompt = system_prompt + context_prompt
        final_prompt += f"\nUSER QUESTION: {message}\n\n"
        final_prompt += "Please provide a helpful response based on the available context."
        
        return final_prompt
    
    def _analyze_query(self, message: str) -> Dict:
        """Analyze query to determine type and extract entities"""
        message_lower = message.lower()
        
        analysis = {
            'type': 'GENERAL',
            'entities': [],
            'intent': 'unknown',
            'keywords': []
        }
        
        # Extract field names
        field_matches = []
        for pattern in self.field_query_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            field_matches.extend(matches)
        
        if field_matches:
            analysis['type'] = 'FIELD_QUERY'
            analysis['entities'].extend(field_matches)
            analysis['intent'] = 'field_information'
        
        # Extract record layout names
        layout_matches = []
        for pattern in self.record_layout_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            layout_matches.extend(matches)
        
        if layout_matches:
            if analysis['type'] == 'FIELD_QUERY':
                analysis['type'] = 'FIELD_AND_LAYOUT_QUERY'
            else:
                analysis['type'] = 'LAYOUT_QUERY'
            analysis['entities'].extend(layout_matches)
            analysis['intent'] = 'layout_information'
        
        # Extract program names
        program_matches = []
        for pattern in self.program_query_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            program_matches.extend(matches)
        
        if program_matches:
            analysis['type'] = f"{analysis['type']}_WITH_PROGRAM" if analysis['type'] != 'GENERAL' else 'PROGRAM_QUERY'
            analysis['entities'].extend(program_matches)
            analysis['intent'] = 'program_information'
        
        # Detect specific intents
        if any(word in message_lower for word in ['usage', 'used', 'populate', 'where']):
            analysis['intent'] = 'usage_analysis'
        elif any(word in message_lower for word in ['type', 'data type', 'format']):
            analysis['intent'] = 'data_type_info'
        elif any(word in message_lower for word in ['business', 'purpose', 'logic']):
            analysis['intent'] = 'business_logic'
        elif any(word in message_lower for word in ['convert', 'oracle', 'migration']):
            analysis['intent'] = 'conversion_info'
        elif any(word in message_lower for word in ['relationship', 'dependency', 'flow']):
            analysis['intent'] = 'relationship_analysis'
        
        # Extract keywords
        important_words = re.findall(r'\b[A-Z][A-Z0-9\-]*\b|\b\w{4,}\b', message)
        analysis['keywords'] = list(set(important_words))
        
        return analysis
    
    def _build_context(self, session_id: str, query_analysis: Dict) -> Dict:
        """Enhanced context building with actual source code"""
        context = {
            'field_details': [],
            'field_mappings': [],
            'record_layouts': [],
            'components': [],
            'source_code_snippets': [],
            'business_logic': []
        }
        
        try:
            entities = query_analysis['entities']
            query_type = query_analysis['type']
            
            if 'FIELD' in query_type and entities:
                # Get comprehensive field context with source code
                for entity in entities:
                    field_context = self.db_manager.get_context_for_field(session_id, entity)
                    if field_context:
                        context['field_details'].extend(field_context.get('field_details', []))
                        context['field_mappings'].extend(field_context.get('field_mappings', []))
                        
                        # Get actual source code snippets for fields
                        source_snippets = self._get_field_source_code(session_id, entity)
                        context['source_code_snippets'].extend(source_snippets)
            
            if 'LAYOUT' in query_type and entities:
                # Get layout context with full source code
                for entity in entities:
                    layouts = self.db_manager.get_record_layouts(session_id)
                    matching_layouts = [l for l in layouts if entity.upper() in l['layout_name'].upper()]
                    context['record_layouts'].extend(matching_layouts)
                    
                    # Get source code for each layout
                    for layout in matching_layouts:
                        if layout.get('source_code'):
                            context['source_code_snippets'].append({
                                'type': 'RECORD_LAYOUT',
                                'name': layout['layout_name'],
                                'source': layout['source_code'],
                                'line_start': layout.get('line_start', 0),
                                'line_end': layout.get('line_end', 0)
                            })
            
            if 'PROGRAM' in query_type and entities:
                # Get program context with source code
                for entity in entities:
                    components = self.db_manager.get_session_components(session_id)
                    matching_components = [c for c in components if entity.upper() in c['component_name'].upper()]
                    context['components'].extend(matching_components)
                    
                    # Get program source code snippets
                    for component in matching_components:
                        try:
                            analysis_result = json.loads(component.get('analysis_result_json', '{}'))
                            if analysis_result.get('content'):
                                # Get relevant code snippets (first 50 lines for context)
                                content_lines = analysis_result['content'].split('\n')
                                preview_lines = content_lines[:50]
                                context['source_code_snippets'].append({
                                    'type': 'PROGRAM',
                                    'name': component['component_name'],
                                    'source': '\n'.join(preview_lines),
                                    'total_lines': len(content_lines)
                                })
                        except:
                            continue
            
            # If no specific entities, get general context with examples
            if not entities and query_analysis['intent'] != 'unknown':
                context = self._build_general_context_enhanced(session_id, query_analysis)
            
            return context
            
        except Exception as e:
            logger.error(f"Error building enhanced context: {str(e)}")
            return context
    
    def _build_general_context(self, session_id: str, query_analysis: Dict) -> Dict:
        """Build general context for queries without specific entities"""
        context = {
            'session_summary': {},
            'recent_components': [],
            'sample_fields': []
        }
        
        try:
            # Get session metrics for overview
            context['session_summary'] = self.db_manager.get_session_metrics(session_id)
            
            # Get recent components
            components = self.db_manager.get_session_components(session_id)
            context['recent_components'] = components[:5]  # Latest 5
            
            # Get sample fields for general field questions
            if query_analysis['intent'] in ['field_information', 'usage_analysis', 'data_type_info']:
                sample_fields = self.db_manager.get_field_matrix(session_id)
                context['sample_fields'] = sample_fields[:10]  # First 10
            
        except Exception as e:
            logger.error(f"Error building general context: {str(e)}")
        
        return context
    
    def _limit_context_size(self, context: Dict) -> Dict:
        """Limit context size to fit within token budget"""
        try:
            context_str = json.dumps(context, default=str)
            estimated_tokens = self.token_manager.estimate_tokens(context_str)
            
            if estimated_tokens <= self.context_token_budget:
                return context
            
            # Reduce context size by priority
            reduction_order = [
                ('dependencies', 0.5),
                ('field_details', 0.7),
                ('components', 0.7),
                ('field_mappings', 0.8),
                ('record_layouts', 0.9)
            ]
            
            for field_name, keep_ratio in reduction_order:
                if field_name in context and context[field_name]:
                    original_size = len(context[field_name])
                    new_size = int(original_size * keep_ratio)
                    context[field_name] = context[field_name][:new_size]
                
                # Check if we're now within budget
                context_str = json.dumps(context, default=str)
                estimated_tokens = self.token_manager.estimate_tokens(context_str)
                if estimated_tokens <= self.context_token_budget:
                    break
            
        except Exception as e:
            logger.error(f"Error limiting context size: {str(e)}")
        
        return context
    
    def _reduce_context_size(self, context: Dict) -> Dict:
        """Emergency context size reduction"""
        try:
            # Keep only the most essential information
            reduced_context = {}
            
            if context.get('field_details'):
                reduced_context['field_details'] = context['field_details'][:3]
            
            if context.get('field_mappings'):
                reduced_context['field_mappings'] = context['field_mappings'][:3]
            
            if context.get('record_layouts'):
                reduced_context['record_layouts'] = context['record_layouts'][:2]
            
            return reduced_context
            
        except Exception as e:
            logger.error(f"Error reducing context size: {str(e)}")
            return {}
    
    def _get_field_source_code(self, session_id: str, field_name: str) -> List[Dict]:
        """Get actual source code snippets where field is used"""
        snippets = []
        
        try:
            # Get field details with code snippets
            field_details = self.db_manager.get_context_for_field(session_id, field_name)
            
            for detail in field_details.get('field_details', []):
                if detail.get('code_snippet'):
                    snippets.append({
                        'type': 'FIELD_USAGE',
                        'field_name': detail['field_name'],
                        'program': detail.get('program_name', 'Unknown'),
                        'operation': detail.get('operation_type', 'Unknown'),
                        'line_number': detail.get('line_number', 0),
                        'source': detail['code_snippet'],
                        'business_purpose': detail.get('business_purpose', '')
                    })
        
        except Exception as e:
            logger.error(f"Error getting field source code: {str(e)}")
        
        return snippets

    def _create_chat_prompt(self, message: str, context: Dict, 
                               conversation_history: List[Dict], query_analysis: Dict) -> str:
        """Enhanced chat prompt with actual source code context"""
        
        system_prompt = """You are an expert mainframe code analyst assistant. You help users understand COBOL programs, field mappings, record layouts, and data relationships.

    Key guidelines:
    - Provide accurate, helpful information based on the context provided
    - Always reference specific source code when available
    - Explain business logic and data relationships clearly
    - Use technical language appropriate for mainframe developers
    - When discussing field usage, show actual COBOL code examples
    - Explain both the technical and business aspects of the code
    """
        
        # Enhanced context section with source code
        context_prompt = "\nCONTEXT INFORMATION:\n"
        
        if context.get('source_code_snippets'):
            context_prompt += "\nSOURCE CODE EXAMPLES:\n"
            for snippet in context['source_code_snippets'][:5]:
                context_prompt += f"\n--- {snippet['type']}: {snippet['name']} ---\n"
                if snippet.get('line_number'):
                    context_prompt += f"Line {snippet['line_number']}: "
                context_prompt += f"{snippet['source']}\n"
                if snippet.get('business_purpose'):
                    context_prompt += f"Purpose: {snippet['business_purpose']}\n"
        
        if context.get('field_details'):
            context_prompt += "\nFIELD ANALYSIS:\n"
            for field in context['field_details'][:5]:
                context_prompt += f"- {field['field_name']} ({field.get('friendly_name', 'N/A')})\n"
                context_prompt += f"  Usage: {field.get('usage_type', 'Unknown')} in {field.get('program_name', 'N/A')}\n"
                context_prompt += f"  Operation: {field.get('operation_type', 'N/A')}\n"
                if field.get('code_snippet'):
                    context_prompt += f"  Code: {field['code_snippet']}\n"
                context_prompt += f"  Purpose: {field.get('business_purpose', 'N/A')}\n\n"
        
        if context.get('field_mappings'):
            context_prompt += "\nFIELD MAPPINGS:\n"
            for mapping in context['field_mappings'][:3]:
                context_prompt += f"- {mapping['field_name']}: {mapping.get('mainframe_data_type', 'N/A')} → {mapping.get('oracle_data_type', 'N/A')}\n"
                context_prompt += f"  Business Logic: {mapping.get('business_logic_description', 'N/A')}\n"
                context_prompt += f"  Population: {mapping.get('population_source', 'N/A')}\n\n"
        
        if context.get('record_layouts'):
            context_prompt += "\nRECORD LAYOUTS:\n"
            for layout in context['record_layouts'][:3]:
                context_prompt += f"- {layout['layout_name']} (Level {layout.get('level_number', '01')})\n"
                context_prompt += f"  Program: {layout.get('program_name', 'N/A')}\n"
                context_prompt += f"  Fields: {layout.get('fields_count', 0)}\n"
                if layout.get('source_code'):
                    # Show first few lines of layout source
                    source_lines = layout['source_code'].split('\n')[:3]
                    context_prompt += f"  Source: {chr(10).join(source_lines)}...\n\n"
        
        # Add conversation history
        history_prompt = "\nRECENT CONVERSATION:\n"
        for msg in conversation_history[-3:]:
            role = "User" if msg['message_type'] == 'user' else "Assistant"
            content = msg['message_content'][:200] + "..." if len(msg['message_content']) > 200 else msg['message_content']
            history_prompt += f"{role}: {content}\n"
        
        # Query analysis
        query_info = f"\nQUERY ANALYSIS:\n"
        query_info += f"Type: {query_analysis['type']}\n"
        query_info += f"Intent: {query_analysis['intent']}\n"
        if query_analysis['entities']:
            query_info += f"Entities: {', '.join(query_analysis['entities'])}\n"
        
        # Construct final prompt
        final_prompt = system_prompt + context_prompt + history_prompt + query_info
        final_prompt += f"\nUSER QUESTION: {message}\n\n"
        final_prompt += "Please provide a detailed response based on the source code and context above. Include specific code examples when relevant."
        
        return final_prompt