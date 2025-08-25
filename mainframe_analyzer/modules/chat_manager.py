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
    
    def process_query(self, session_id: str, message: str, conversation_id: str) -> Dict:
        """Process chat query with intelligent context"""
        try:
            # Analyze query to determine type and extract entities
            query_analysis = self._analyze_query(message)
            
            # Build relevant context based on query
            context = self._build_context(session_id, query_analysis)
            
            # Get conversation history
            conversation_history = self.db_manager.get_chat_history(session_id, conversation_id, limit=3)
            
            # Create chat prompt with context
            chat_prompt = self._create_chat_prompt(message, context, conversation_history, query_analysis)
            
            # Check token limits and adjust if necessary
            if self.token_manager.estimate_tokens(chat_prompt) > self.token_manager.EFFECTIVE_CONTENT_LIMIT:
                context = self._reduce_context_size(context)
                chat_prompt = self._create_chat_prompt(message, context, conversation_history, query_analysis)
            
            # Call LLM
            start_time = datetime.now()
            response = self.llm_client.call_llm(chat_prompt, max_tokens=1000, temperature=0.3)
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Store conversation
            self.db_manager.store_chat_message(
                session_id, conversation_id, 'user', message, 
                context, response.prompt_tokens, processing_time
            )
            
            if response.success:
                self.db_manager.store_chat_message(
                    session_id, conversation_id, 'assistant', response.content,
                    None, response.response_tokens, processing_time
                )
                
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
                    'response': f"I'm sorry, I encountered an error processing your question: {response.error_message}",
                    'error': True
                }
                
        except Exception as e:
            logger.error(f"Error processing chat query: {str(e)}")
            return {
                'response': "I'm sorry, I encountered an unexpected error. Please try again.",
                'error': True
            }
    
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
        """Build relevant context based on query analysis"""
        context = {
            'field_details': [],
            'field_mappings': [],
            'record_layouts': [],
            'components': [],
            'dependencies': []
        }
        
        try:
            entities = query_analysis['entities']
            query_type = query_analysis['type']
            
            if 'FIELD' in query_type and entities:
                # Get field context
                for entity in entities:
                    field_context = self.db_manager.get_context_for_field(session_id, entity)
                    if field_context:
                        context['field_details'].extend(field_context.get('field_details', []))
                        context['field_mappings'].extend(field_context.get('field_mappings', []))
            
            if 'LAYOUT' in query_type and entities:
                # Get layout context
                for entity in entities:
                    layouts = self.db_manager.get_record_layouts(session_id)
                    matching_layouts = [l for l in layouts if entity.upper() in l['layout_name'].upper()]
                    context['record_layouts'].extend(matching_layouts)
                    
                    if matching_layouts:
                        # Get field matrix for these layouts
                        for layout in matching_layouts:
                            field_matrix = self.db_manager.get_field_matrix(session_id, layout['layout_name'])
                            context['field_details'].extend(field_matrix)
            
            if 'PROGRAM' in query_type and entities:
                # Get program context
                for entity in entities:
                    components = self.db_manager.get_session_components(session_id)
                    matching_components = [c for c in components if entity.upper() in c['component_name'].upper()]
                    context['components'].extend(matching_components)
            
            # If no specific entities found, get general context based on intent
            if not entities and query_analysis['intent'] != 'unknown':
                context = self._build_general_context(session_id, query_analysis)
            
            # Limit context size to budget
            context = self._limit_context_size(context)
            
        except Exception as e:
            logger.error(f"Error building context: {str(e)}")
        
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
    
    def _create_chat_prompt(self, message: str, context: Dict, 
                          conversation_history: List[Dict], query_analysis: Dict) -> str:
        """Create chat prompt with context and history"""
        
        # Base system prompt
        system_prompt = """You are an expert mainframe code analyst assistant. You help users understand COBOL programs, field mappings, record layouts, and data relationships.

Key guidelines:
- Provide accurate, helpful information based on the context provided
- Use friendly, technical language appropriate for mainframe developers
- Reference specific field names, programs, and layouts when available in context
- Explain business logic and data relationships clearly
- If asked about conversions, provide both COBOL and Oracle equivalents
- Always base your responses on the provided context data

"""
        
        # Add context information
        context_prompt = "\nCONTEXT INFORMATION:\n"
        
        if context.get('field_details'):
            context_prompt += "\nField Analysis Details:\n"
            for field in context['field_details'][:5]:  # Limit to 5 for prompt size
                context_prompt += f"- {field['field_name']} ({field.get('friendly_name', 'N/A')}): {field.get('usage_type', 'Unknown')} usage in {field.get('program_name', 'N/A')}\n"
                if field.get('code_snippet'):
                    context_prompt += f"  Code: {field['code_snippet'][:100]}...\n"
        
        if context.get('field_mappings'):
            context_prompt += "\nField Mappings:\n"
            for mapping in context['field_mappings'][:3]:
                context_prompt += f"- {mapping['field_name']}: {mapping.get('mainframe_data_type', 'N/A')} -> {mapping.get('oracle_data_type', 'N/A')}\n"
                context_prompt += f"  Business Logic: {mapping.get('business_logic_type', 'N/A')} - {mapping.get('business_logic_description', 'N/A')}\n"
        
        if context.get('record_layouts'):
            context_prompt += "\nRecord Layouts:\n"
            for layout in context['record_layouts'][:3]:
                context_prompt += f"- {layout['layout_name']} ({layout.get('friendly_name', 'N/A')}): Level {layout.get('level_number', '01')} in {layout.get('program_name', 'N/A')}\n"
        
        if context.get('components'):
            context_prompt += "\nProgram Components:\n"
            for component in context['components'][:3]:
                context_prompt += f"- {component['component_name']} ({component.get('component_type', 'N/A')}): {component.get('total_lines', 0)} lines\n"
        
        if context.get('session_summary'):
            summary = context['session_summary']
            context_prompt += f"\nSession Overview:\n"
            context_prompt += f"- Total Components: {summary.get('total_components', 0)}\n"
            context_prompt += f"- Total Fields: {summary.get('total_fields', 0)}\n"
            context_prompt += f"- Component Types: {', '.join(summary.get('component_counts', {}).keys())}\n"
        
        # Add conversation history
        history_prompt = "\nCONVERSATION HISTORY:\n"
        for msg in conversation_history[-3:]:  # Last 3 messages
            role = "User" if msg['message_type'] == 'user' else "Assistant"
            content = msg['message_content'][:200] + "..." if len(msg['message_content']) > 200 else msg['message_content']
            history_prompt += f"{role}: {content}\n"
        
        # Add current query information
        query_info = f"\nCURRENT QUERY ANALYSIS:\n"
        query_info += f"Query Type: {query_analysis['type']}\n"
        query_info += f"Intent: {query_analysis['intent']}\n"
        if query_analysis['entities']:
            query_info += f"Entities Found: {', '.join(query_analysis['entities'])}\n"
        
        # Construct final prompt
        final_prompt = system_prompt + context_prompt + history_prompt + query_info
        final_prompt += f"\nUSER QUESTION: {message}\n\n"
        final_prompt += "Please provide a helpful response based on the context and conversation history above."
        
        return final_prompt