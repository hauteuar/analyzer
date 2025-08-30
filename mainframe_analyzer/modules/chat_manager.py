"""
Complete Chat Manager Module
Handles intelligent chat with full source code context for mainframe analysis
"""

import re
import json
import logging
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatManager:
    def __init__(self, llm_client, token_manager, db_manager):
        self.llm_client = llm_client
        self.token_manager = token_manager
        self.db_manager = db_manager
        
        # Field name extraction patterns
        self.field_patterns = [
            r'\b([A-Z][A-Z0-9\-]{2,})\b',                    # Standard COBOL fields
            r'field\s+([A-Za-z][A-Za-z0-9\-_]+)',            # "field CUSTOMER-NAME"
            r'about\s+([A-Za-z][A-Za-z0-9\-_]+)',            # "about ACCOUNT-NO"
            r'([A-Za-z][A-Za-z0-9\-_]+)\s+field',            # "CUSTOMER-NAME field"
            r'tell\s+me\s+about\s+([A-Za-z][A-Za-z0-9\-_]+)',# "tell me about FIELD-NAME"
            r'what\s+is\s+([A-Za-z][A-Za-z0-9\-_]+)',        # "what is FIELD-NAME"
            r'how\s+is\s+([A-Za-z][A-Za-z0-9\-_]+)',         # "how is FIELD-NAME"
            r'where\s+is\s+([A-Za-z][A-Za-z0-9\-_]+)',       # "where is FIELD-NAME"
            r'show\s+([A-Za-z][A-Za-z0-9\-_]+)'              # "show FIELD-NAME"
        ]
        
        # COBOL keywords to exclude
        self.cobol_keywords = {
            'MOVE', 'TO', 'FROM', 'PIC', 'PICTURE', 'VALUE', 'OCCURS', 'REDEFINES',
            'USAGE', 'COMP', 'BINARY', 'PACKED', 'DISPLAY', 'COMPUTE', 'ADD',
            'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'IF', 'THEN', 'ELSE', 'END',
            'PERFORM', 'UNTIL', 'VARYING', 'WHEN', 'EVALUATE', 'ACCEPT', 'DISPLAY'
        }
    
    # In chat_manager.py, update the process_query method:

    def process_query(self, session_id: str, message: str, conversation_id: str) -> str:
        """Process chat query with enhanced source code context"""
        try:
            # Get relevant context including source code
            context = self._get_enhanced_context(session_id, message)
            
            # Build comprehensive prompt with source code
            prompt = self._build_enhanced_prompt(message, context)
            
            # Call LLM (with chunking if prompt is very large)
            response = self._call_llm_with_optional_chunking(session_id, message, prompt, context)
            
            # Log the call
            self.db_manager.log_llm_call(
                session_id, 'chat_query', 1, 1,
                response.prompt_tokens, response.response_tokens, 
                response.processing_time_ms, response.success, response.error_message
            )
            
            if response.success:
                # Store conversation
                self.db_manager.store_chat_message(
                    session_id, conversation_id, 'user', message,
                    context_used={'components_referenced': len(context.get('components', []))},
                    tokens_used=response.prompt_tokens
                )
                
                self.db_manager.store_chat_message(
                    session_id, conversation_id, 'assistant', response.content,
                    tokens_used=response.response_tokens,
                    processing_time_ms=response.processing_time_ms
                )
                
                return response.content
            else:
                return f"I encountered an error: {response.error_message}"
                
        except Exception as e:
            logger.error(f"Error processing chat query: {str(e)}")
            return "I'm sorry, I encountered an error processing your request."

    def _call_llm_with_optional_chunking(self, session_id: str, message: str, prompt: str, context: Dict):
        """Call the LLM directly or via chunked analysis if the prompt is too large.

        Strategy:
        - If estimated tokens for the prompt <= CHAT_TOKEN_LIMIT, call LLM directly.
        - Otherwise, chunk the combined component source using TokenManager, call the LLM for each chunk,
          collect chunk-level analyses, then synthesize a final response with a final LLM call.
        """
        try:
            CHAT_TOKEN_LIMIT = 6000

            estimated = 0
            try:
                estimated = self.token_manager.estimate_tokens(prompt)
            except Exception:
                # Fallback estimation
                estimated = len(prompt) // getattr(self.token_manager, 'CHARACTERS_PER_TOKEN', 4)

            # If prompt is small enough, do a normal call
            if estimated <= CHAT_TOKEN_LIMIT:
                return self.llm_client.call_llm(prompt, max_tokens=2000, temperature=0.1)

            # Otherwise, perform chunked analysis on component source
            components = context.get('components', [])
            if not components:
                # No component source to chunk; fallback to direct call (may fail for very large prompts)
                return self.llm_client.call_llm(prompt, max_tokens=2000, temperature=0.1)

            # Build a combined source string from available components
            combined_source_parts = [c.get('source_for_chat', '') for c in components if c.get('source_for_chat')]
            combined_source = '\n\n'.join(combined_source_parts)
            if not combined_source.strip():
                return self.llm_client.call_llm(prompt, max_tokens=2000, temperature=0.1)

            # Chunk the combined source while preserving structure when possible
            try:
                chunks = self.token_manager.chunk_cobol_code(combined_source, preserve_structure=True)
            except Exception:
                # Fallback to naive chunking if token manager fails
                chunks = [type('C', (), {'content': combined_source, 'chunk_number': 1, 'total_chunks': 1})()]

            chunk_responses = []
            for chunk in chunks:
                chunk_prompt = (
                    f"You are a COBOL/mainframe expert. The user asked: \"{message}\"\n\n"
                    f"Analyze the following code chunk and return any information relevant to answering the user's question."
                    f" Provide concise findings, code locations, and potential impacts.\n\nCODE CHUNK:\n{chunk.content}"
                )

                resp = self.llm_client.call_llm(chunk_prompt, max_tokens=800, temperature=0.2)
                # Log chunk call
                try:
                    self.db_manager.log_llm_call(
                        session_id, 'chat_chunk', getattr(chunk, 'chunk_number', 1), getattr(chunk, 'total_chunks', 1),
                        resp.prompt_tokens, resp.response_tokens, resp.processing_time_ms, resp.success, resp.error_message
                    )
                except Exception:
                    logger.debug('Failed to log chunk LLM call', exc_info=True)

                if resp.success and resp.content:
                    chunk_responses.append((getattr(chunk, 'chunk_number', 1), resp.content))

            # If no chunk produced useful content, fallback
            if not chunk_responses:
                return self.llm_client.call_llm(prompt, max_tokens=2000, temperature=0.1)

            # Sort chunk responses by chunk number and assemble
            chunk_responses.sort(key=lambda x: x[0])
            assembled = "\n\n".join([f"--- CHUNK {num} RESPONSE ---\n{content}" for num, content in chunk_responses])

            # Synthesize final answer from assembled chunk responses
            synth_prompt = (
                f"You are a COBOL/mainframe expert. The user asked: \"{message}\"\n\n"
                "Below are analyses derived from different chunks of the program source. Use these to produce a single, concise, accurate answer to the user's question. If the analyses conflict, reconcile them and highlight uncertainty. Provide concrete code references where possible.\n\n"
                f"CHUNK ANALYSES:\n{assembled}\n\nFINAL ANSWER:\n"
            )

            final_resp = self.llm_client.call_llm(synth_prompt, max_tokens=2000, temperature=0.1)
            # Log synthesis call
            try:
                self.db_manager.log_llm_call(
                    session_id, 'chat_synthesis', 1, len(chunks),
                    final_resp.prompt_tokens, final_resp.response_tokens, final_resp.processing_time_ms, final_resp.success, final_resp.error_message
                )
            except Exception:
                logger.debug('Failed to log synthesis LLM call', exc_info=True)

            return final_resp

        except Exception as e:
            logger.error(f"Error in chunked LLM call: {e}")
            # Final fallback
            return self.llm_client.call_llm(prompt, max_tokens=2000, temperature=0.1)

    def _get_enhanced_context(self, session_id: str, message: str) -> Dict:
        """Get enhanced context including source code based on query"""
        context = {
            'components': [],
            'record_layouts': [],
            'field_mappings': [],
            'dependencies': [],
            'program_calls': [],
            'source_code_included': False
        }
        
        try:
            # First, check if the user mentions a field; if so, prefer components that contain that field
            field_names = self._extract_field_names(message)
            seen_components = set()

            if field_names:
                # For each extracted field, fetch its DB context and then the program/component that contains it
                for fname in field_names[:2]:
                    try:
                        fctx = self.db_manager.get_context_for_field(session_id, fname)
                        field_details = fctx.get('field_details', []) if fctx else []

                        for fd in field_details:
                            prog = fd.get('program_name')
                            if not prog:
                                continue

                            # Avoid duplicate component fetches
                            if prog in seen_components:
                                continue
                            seen_components.add(prog)

                            source_data = self.db_manager.get_component_source_code(
                                session_id, prog, max_size=30000
                            )
                            if source_data.get('success') and source_data.get('components'):
                                # Append components returned for this program
                                for comp in source_data['components']:
                                    if comp.get('component_name') not in [c.get('component_name') for c in context['components']]:
                                        context['components'].append(comp)
                                        context['source_code_included'] = True
                    except Exception as e:
                        logger.debug(f"Error fetching components for field {fname}: {e}")

            # If no field-based components were attached, fall back to explicit component mention or general sample
            if not context['components']:
                # Check if user is asking about specific component
                component_mentioned = self._extract_component_name_from_message(message)
                
                if component_mentioned:
                    # Get specific component with source code
                    source_data = self.db_manager.get_component_source_code(
                        session_id, component_mentioned, max_size=30000
                    )
                    if source_data['success'] and source_data['components']:
                        context['components'] = source_data['components']
                        context['source_code_included'] = True
                else:
                    # Get general context (smaller components with source code)
                    source_data = self.db_manager.get_component_source_code(
                        session_id, max_size=20000
                    )
                    if source_data['success']:
                        # Include up to 2 small components with full source
                        small_components = [c for c in source_data['components'] 
                                        if c.get('source_strategy') == 'full'][:2]
                        context['components'] = small_components
                        context['source_code_included'] = len(small_components) > 0

            # Get dependencies - especially important for program call questions
            all_dependencies = self.db_manager.get_dependencies(session_id)
            
            # If user is asking about calls or programs, include relevant dependencies
            if any(word in message.lower() for word in ['call', 'calls', 'program', 'programs', 'link', 'xctl', 'invoke', 'execute']):
                if component_mentioned:
                    # Get dependencies for specific component
                    component_deps = [
                        dep for dep in all_dependencies 
                        if dep['source_component'] == component_mentioned or dep['target_component'] == component_mentioned
                    ]
                    context['dependencies'] = component_deps
                    
                    # Extract program calls specifically
                    program_calls = [
                        dep for dep in component_deps
                        if dep['relationship_type'] in ['PROGRAM_CALL', 'CICS_LINK', 'CICS_XCTL', 'CICS_START']
                    ]
                    context['program_calls'] = program_calls
                    
                    logger.info(f"Found {len(program_calls)} program calls for {component_mentioned}")
                else:
                    # Include all dependencies for general program call questions
                    context['dependencies'] = all_dependencies[:10]  # Limit to prevent overflow
                    context['program_calls'] = [
                        dep for dep in all_dependencies
                        if dep['relationship_type'] in ['PROGRAM_CALL', 'CICS_LINK', 'CICS_XCTL', 'CICS_START']
                    ][:10]
            else:
                # For non-program-call questions, include limited dependencies
                context['dependencies'] = all_dependencies[:5]
            
            # Get other context as before
            context['record_layouts'] = self.db_manager.get_record_layouts(session_id)[:5]
            
            # Get field mappings if relevant
            if any(word in message.lower() for word in ['field', 'mapping', 'oracle', 'conversion']):
                try:
                    # Try to get field mappings for any mentioned layouts or files
                    layouts = context['record_layouts']
                    for layout in layouts[:3]:
                        layout_name = layout.get('layout_name', '')
                        if layout_name:
                            mappings = self.db_manager.get_field_mappings(session_id, layout_name)
                            context['field_mappings'].extend(mappings[:5])
                except Exception as e:
                    logger.debug(f"Error loading field mappings: {e}")
            
            # Log context summary
            logger.info(f"Enhanced context for '{message[:50]}...': "
                    f"Components: {len(context['components'])}, "
                    f"Dependencies: {len(context['dependencies'])}, "
                    f"Program calls: {len(context['program_calls'])}, "
                    f"Source included: {context['source_code_included']}")
            
        except Exception as e:
            logger.error(f"Error getting enhanced context: {str(e)}")
        
        return context
    
    def _extract_component_name_from_message(self, message: str) -> Optional[str]:
        """Extract a likely component or program name from a user message.

        This is a heuristic extractor that looks for common patterns such as:
        - "program PROGRAM-NAME"
        - "component COMPONENT-NAME"
        - bare identifiers that look like COBOL program names (ALL-CAPS, dashes, numbers)
        - filenames ending with .cob/.cbl/.cpy/etc.

        Returns the extracted name in the original-like form (uppercased, with dashes) or None.
        """
        if not message:
            return None

        try:
            msg = message.strip()

            # First try file-like names with extensions
            file_match = re.search(r"\b([A-Za-z0-9_\-]+)\.(cob|cbl|cpy|cpyk|cpyb)\b", msg, re.IGNORECASE)
            if file_match:
                name = file_match.group(1)
                return name.upper()

            # Try common keyword patterns
            patterns = [
                r"\bprogram\s+([A-Z][A-Z0-9\-]{2,})\b",
                r"\bcomponent\s+([A-Z][A-Z0-9\-]{2,})\b",
                r"\babout\s+([A-Z][A-Z0-9\-]{2,})\b",
                r"\bshow\s+(?:me\s+)?([A-Z][A-Z0-9\-]{2,})\b",
                r"\bopen\s+([A-Z][A-Z0-9\-]{2,})\b",
                r"\bwhat\s+does\s+([A-Z][A-Z0-9\-]{2,})\s+do\b",
            ]

            for pat in patterns:
                m = re.search(pat, msg, re.IGNORECASE)
                if m:
                    return m.group(1).upper().replace('_', '-')

            # As a last resort, look for an ALL-CAPS token that resembles a program/component name
            tokens = re.findall(r"\b[A-Z][A-Z0-9\-]{2,}\b", msg)
            if tokens:
                # Prefer the first token not in common English words
                for t in tokens:
                    if t.upper() not in self.cobol_keywords:
                        return t.upper()

        except Exception:
            logger.debug("_extract_component_name_from_message failed", exc_info=True)

        return None

    def _build_enhanced_prompt(self, message: str, context: Dict) -> str:
        """Build enhanced prompt with program call context"""
        prompt_parts = [
            "You are a COBOL/mainframe analysis expert. Answer the user's question using the provided context.",
            "",
            f"USER QUESTION: {message}",
            ""
        ]
        
        # Add program call information if available
        if context.get('program_calls'):
            prompt_parts.extend([
                "=== PROGRAM CALLS AND DEPENDENCIES ===",
                ""
            ])
            for call in context['program_calls']:
                prompt_parts.append(
                    f"- {call['relationship_type']}: {call['source_component']} → {call['target_component']}"
                )
            prompt_parts.append("")
        
        # Add source code context
        if context.get('source_code_included') and context.get('components'):
            prompt_parts.extend([
                "=== SOURCE CODE CONTEXT ===",
                ""
            ])
            
            for component in context['components'][:2]:
                prompt_parts.extend([
                    f"COMPONENT: {component['component_name']} ({component['component_type']})",
                    f"LINES: {component.get('total_lines', 0)}",
                    "",
                    "SOURCE CODE:",
                    component.get('source_for_chat', 'No source available'),
                    "",
                    "=" * 50,
                    ""
                ])
        
        prompt_parts.extend([
            "",
            "Please provide a helpful, accurate response based on this context. If you reference program calls or dependencies, include the specific relationship types (CALL, CICS LINK, CICS XCTL)."
        ])
        
        return '\n'.join(prompt_parts)
    
    def _search_for_similar_fields(self, session_id: str, query_term: str) -> List[str]:
        """Search for fields similar to the query term"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Search for fields that contain the query term
                cursor.execute('''
                    SELECT DISTINCT field_name FROM field_analysis_details 
                    WHERE session_id = ? AND (
                        UPPER(field_name) LIKE ? OR
                        UPPER(business_purpose) LIKE ?
                    )
                    ORDER BY field_name 
                    LIMIT 5
                ''', (session_id, f'%{query_term.upper()}%', f'%{query_term.upper()}%'))
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error searching for similar fields: {str(e)}")
            return []

    def _extract_field_names(self, message: str) -> List[str]:
        """Extract COBOL field names more intelligently"""
        field_names = set()
        
        try:
            # More specific field extraction patterns
            specific_patterns = [
                r'\b([A-Z][A-Z0-9\-]{4,})\b',                    # At least 5 chars, COBOL-style
                r'field\s+([A-Za-z][A-Za-z0-9\-_]{3,})',         # "field CUSTOMER-NAME"
                r'about\s+([A-Z][A-Z0-9\-_]{3,})',               # "about ACCOUNT-NO" 
                r'([A-Z][A-Z0-9\-_]{4,})\s+field',               # "CUSTOMER-NAME field"
                r'tell\s+me\s+about\s+([A-Z][A-Z0-9\-_]{3,})',   # "tell me about FIELD-NAME"
                r'what\s+is\s+([A-Z][A-Z0-9\-_]{3,})',           # "what is FIELD-NAME" (only ALL CAPS)
                r'show\s+me\s+([A-Z][A-Z0-9\-_]{3,})',           # "show me FIELD-NAME"
                r'\b([A-Z]{2,}[-_][A-Z0-9\-_]{2,})\b'            # COBOL naming convention
            ]
            
            # English stop words to exclude
            english_words = {
                'WHAT', 'WHERE', 'WHEN', 'HOW', 'WHO', 'WHY', 'WHICH', 'THE', 'THIS', 'THAT',
                'CUSTOMER', 'RECORD', 'STRUCTURE', 'FIELD', 'FIELDS', 'PROGRAM', 'PROGRAMS',
                'LAYOUT', 'LAYOUTS', 'FILE', 'FILES', 'TABLE', 'TABLES', 'SHOW', 'TELL',
                'ABOUT', 'EXPLAIN', 'DESCRIBE', 'LIST', 'FIND', 'SEARCH', 'HELP', 'CAN',
                'WILL', 'WOULD', 'SHOULD', 'COULD', 'AND', 'OR', 'BUT', 'FOR', 'WITH',
                'FROM', 'INTO', 'ONTO', 'OVER', 'UNDER', 'ABOVE', 'BELOW'
            }
            
            for pattern in specific_patterns:
                matches = re.findall(pattern, message, re.IGNORECASE)
                for match in matches:
                    cobol_name = match.upper().replace('_', '-')
                    
                    # More strict filtering
                    if (len(cobol_name) > 3 and 
                        cobol_name not in self.cobol_keywords and
                        cobol_name not in english_words and
                        not cobol_name.isdigit() and
                        '-' in cobol_name or len(cobol_name) > 6):  # Either has dash or is long
                        field_names.add(cobol_name)
            
            result = list(field_names)
            logger.debug(f"Extracted field names from '{message}': {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting field names: {str(e)}")
            return []
    
    def _handle_field_query(self, session_id: str, field_names: List[str], message: str) -> str:
        """Handle queries about specific fields with better error messages"""
        try:
            response_parts = []
            found_fields = []
            not_found_fields = []
            
            for field_name in field_names[:2]:
                field_info = self._get_comprehensive_field_info(session_id, field_name, message)
                if field_info and "was not found" not in field_info:
                    response_parts.append(field_info)
                    found_fields.append(field_name)
                else:
                    not_found_fields.append(field_name)
            
            # If no fields found, search for similar ones
            if not response_parts and not_found_fields:
                similar_suggestions = []
                for field_name in not_found_fields:
                    similar = self._search_for_similar_fields(session_id, field_name)
                    similar_suggestions.extend(similar)
                
                if similar_suggestions:
                    return f"I couldn't find exact matches for {', '.join(not_found_fields)}, but I found these similar fields: {', '.join(similar_suggestions[:5])}. Try asking about one of these specific field names."
                else:
                    available_fields = self._get_available_fields_sample(session_id)
                    return f"I couldn't find fields matching {', '.join(not_found_fields)}.\n\nAvailable fields include: {available_fields}\n\nTry asking about one of these field names, or ask a general question about your programs."
            
            return '\n\n'.join(response_parts)
            
        except Exception as e:
            logger.error(f"Error handling field query: {str(e)}")
            return f"Error analyzing fields: {str(e)}"
    
    def _get_comprehensive_field_info(self, session_id: str, field_name: str, original_message: str) -> str:
        """Get comprehensive field information with LLM enhancement"""
        try:
            logger.info(f"Getting comprehensive info for field: {field_name}")
            
            # Get field context from database
            context = self.db_manager.get_context_for_field(session_id, field_name)
            
            if context and context.get('field_details'):
                # Use LLM to create intelligent response
                return self._create_llm_enhanced_field_response(field_name, context, original_message, session_id)
            else:
                # Perform live analysis
                return self._perform_live_field_analysis(session_id, field_name)
                
        except Exception as e:
            logger.error(f"Error getting field info: {str(e)}")
            return f"Error analyzing {field_name}: {str(e)}"

    def _create_llm_enhanced_field_response(self, field_name: str, context: Dict, 
                                        original_message: str, session_id: str) -> str:
        """Create LLM-enhanced field response"""
        try:
            field_details = context.get('field_details', [])
            field_mappings = context.get('field_mappings', [])
            
            if not field_details:
                return f"No detailed information found for field {field_name}"
            
            primary_field = field_details[0]
            
            # Build comprehensive context for LLM
            field_context = {
                'field_name': field_name,
                'program_name': primary_field.get('program_name', 'Unknown'),
                'usage_type': primary_field.get('usage_type', 'Unknown'),
                'business_purpose': primary_field.get('business_purpose', ''),
                'total_references': primary_field.get('total_program_references', 0),
                'definition_code': primary_field.get('definition_code', ''),
                'operations': {
                    'move_source': primary_field.get('move_source_count', 0),
                    'move_target': primary_field.get('move_target_count', 0),
                    'arithmetic': primary_field.get('arithmetic_count', 0),
                    'conditional': primary_field.get('conditional_count', 0),
                    'cics': primary_field.get('cics_count', 0)
                }
            }
            
            # Parse field references for context
            field_refs = []
            field_refs_json = primary_field.get('field_references_json', '[]')
            try:
                field_refs = json.loads(field_refs_json) if field_refs_json else []
            except:
                pass
            
            # Create LLM prompt for field analysis
            prompt = f"""
    You are a mainframe COBOL expert with prita bank wealth management domain analyzing field usage. Based on the field analysis data provided, create a comprehensive, conversational response about this field.

    User asked: "{original_message}"

    Field Analysis Data:
    - Field Name: {field_name}
    - Program: {field_context['program_name']}
    - Usage Type: {field_context['usage_type']}
    - Total References: {field_context['total_references']}
    - Business Purpose: {field_context['business_purpose']}

    Field Definition:
    {field_context['definition_code']}

    Operations Summary:
    - Data Input Operations: {field_context['operations']['move_target']}
    - Data Output Operations: {field_context['operations']['move_source']}
    - Mathematical Operations: {field_context['operations']['arithmetic']}
    - Conditional Logic: {field_context['operations']['conditional']}
    - CICS Operations: {field_context['operations']['cics']}

    Key Source Code References:
    {chr(10).join([f"Line {ref.get('line_number', 'N/A')}: {ref.get('line_content', '')}" for ref in field_refs[:3]])}

    Please provide a conversational, expert analysis that:
    1. Explains what this field does in business terms
    2. Describes how it's used in the program flow
    3. Highlights any important technical details
    4. Answers the user's specific question
    5. Provides actionable insights for mainframe-to-modern migration

    Keep the response informative but conversational, as if you're explaining to a colleague.
    """

            # Call LLM for enhanced response
            response = self.llm_client.call_llm(prompt, max_tokens=800, temperature=0.3)
            
            # Log LLM call
            self.db_manager.log_llm_call(
                session_id, 'chat_field_analysis', 1, 1,
                response.prompt_tokens, response.response_tokens, response.processing_time_ms,
                response.success, response.error_message
            )
            
            if response.success and response.content:
                # Add technical details footer
                technical_footer = f"\n\n**Technical Details:**\n"
                technical_footer += f"• Program: {field_context['program_name']}\n"
                technical_footer += f"• Definition: `{field_context['definition_code']}`\n"
                technical_footer += f"• Total References: {field_context['total_references']}\n"
                
                if field_mappings:
                    mapping = field_mappings[0]
                    technical_footer += f"• Oracle Mapping: {mapping.get('oracle_data_type', 'Not mapped')}\n"
                
                return response.content + technical_footer
            else:
                # Fallback to formatted database response
                return self._format_database_field_response(field_name, context, original_message)
                
        except Exception as e:
            logger.error(f"Error creating LLM enhanced response: {str(e)}")
            # Fallback to database formatting
            return self._format_database_field_response(field_name, context, original_message)
    
    def _format_database_field_response(self, field_name: str, context: Dict, message: str) -> str:
        """Format comprehensive field response from database context"""
        try:
            field_details = context.get('field_details', [])
            field_mappings = context.get('field_mappings', [])
            
            primary_field = field_details[0]
            response = f"Field Analysis: {field_name}\n"
            response += "=" * (len(field_name) + 16) + "\n"
            
            # Basic information
            response += f"Program: {primary_field.get('program_name', 'Unknown')}\n"
            response += f"Usage Type: {primary_field.get('usage_type', 'Unknown')}\n"
            
            # Business purpose
            business_purpose = primary_field.get('business_purpose', '')
            if business_purpose:
                response += f"Business Purpose: {business_purpose}\n"
            
            # Field definition with source code
            definition_code = primary_field.get('definition_code', '')
            if definition_code:
                response += f"\nField Definition:\n  {definition_code}\n"
            
            # Usage statistics
            total_refs = primary_field.get('total_program_references', 0)
            if total_refs > 0:
                response += f"\nUsage Statistics:\n"
                response += f"  Total References: {total_refs}\n"
                
                # Detailed breakdown
                stats = []
                if primary_field.get('move_target_count', 0) > 0:
                    stats.append(f"Receives data: {primary_field['move_target_count']} operations")
                if primary_field.get('move_source_count', 0) > 0:
                    stats.append(f"Provides data: {primary_field['move_source_count']} operations")
                if primary_field.get('arithmetic_count', 0) > 0:
                    stats.append(f"Calculations: {primary_field['arithmetic_count']} operations")
                if primary_field.get('conditional_count', 0) > 0:
                    stats.append(f"Conditions: {primary_field['conditional_count']} operations")
                if primary_field.get('cics_count', 0) > 0:
                    stats.append(f"CICS operations: {primary_field['cics_count']} operations")
                
                if stats:
                    response += f"  Usage Breakdown: {'; '.join(stats)}\n"
            
            # Source code examples
            field_refs_json = primary_field.get('field_references_json', '[]')
            try:
                references = json.loads(field_refs_json) if field_refs_json else []
                if references:
                    response += f"\nSource Code Examples:\n"
                    
                    # Show definition first
                    def_refs = [ref for ref in references if ref.get('operation_type') == 'DEFINITION']
                    if def_refs:
                        def_ref = def_refs[0]
                        response += f"  Definition (Line {def_ref['line_number']}):\n"
                        response += f"    {def_ref['line_content']}\n"
                    
                    # Show key operations
                    operation_refs = [ref for ref in references if ref.get('operation_type') != 'DEFINITION']
                    operation_refs.sort(key=lambda x: x.get('line_number', 0))
                    
                    for ref in operation_refs[:5]:  # Show up to 5 usage examples
                        response += f"  {ref.get('operation_type', 'Usage')} (Line {ref['line_number']}):\n"
                        response += f"    {ref['line_content']}\n"
                        if ref.get('business_context'):
                            response += f"    -> {ref['business_context']}\n"
                    
                    # Show detailed context for most important operation
                    if operation_refs and any(word in message.lower() for word in ['how', 'where', 'usage', 'context']):
                        important_ref = operation_refs[0]
                        context_block = important_ref.get('context_block', '')
                        if context_block:
                            response += f"\nDetailed Code Context:\n{context_block}\n"
                            
            except Exception as ref_error:
                logger.warning(f"Error parsing field references: {str(ref_error)}")
            
            # Field mappings if available
            if field_mappings:
                mapping = field_mappings[0]
                response += f"\nData Type Mapping:\n"
                response += f"  Mainframe: {mapping.get('mainframe_data_type', 'Unknown')}\n"
                response += f"  Oracle: {mapping.get('oracle_data_type', 'Unknown')}\n"
                if mapping.get('business_logic_description'):
                    response += f"  Logic: {mapping['business_logic_description']}\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting field response: {str(e)}")
            return f"Error formatting response for {field_name}: {str(e)}"
    
    def _perform_live_field_analysis(self, session_id: str, field_name: str) -> str:
        """Perform live field analysis when not in database"""
        try:
            logger.info(f"Performing live analysis for: {field_name}")
            
            components = self.db_manager.get_session_components(session_id)
            
            for component in components:
                if component.get('component_type') != 'PROGRAM':
                    continue
                
                # Get program source code
                analysis_json = component.get('analysis_result_json', '{}')
                analysis_data = json.loads(analysis_json) if analysis_json else {}
                source_content = analysis_data.get('content') or component.get('source_content', '')
                
                if not source_content:
                    continue
                
                # Check if field exists in this program
                if field_name.upper() in source_content.upper():
                    logger.info(f"Found {field_name} in {component['component_name']}")
                    
                    # Analyze field usage
                    field_analysis = self._analyze_field_in_program(
                        field_name, source_content, component['component_name']
                    )
                    
                    return self._format_live_analysis_response(field_name, field_analysis)
            
            return f"Field {field_name} was not found in any analyzed program source code. Please verify the field name and ensure the containing program has been uploaded and analyzed."
            
        except Exception as e:
            logger.error(f"Error in live field analysis: {str(e)}")
            return f"Error performing live analysis for {field_name}: {str(e)}"
    
    def _analyze_field_in_program(self, field_name: str, source_content: str, program_name: str) -> Dict:
        """Complete field analysis in program source code"""
        analysis = {
            'field_name': field_name,
            'program_name': program_name,
            'definition': None,
            'references': [],
            'usage_patterns': {
                'input_operations': [],
                'output_operations': [],
                'arithmetic_operations': [],
                'conditional_operations': [],
                'cics_operations': []
            },
            'business_summary': ''
        }
        
        try:
            lines = source_content.split('\n')
            field_upper = field_name.upper()
            
            for line_idx, line in enumerate(lines, 1):
                line_stripped = line.strip()
                line_upper = line_stripped.upper()
                
                # Skip comments and empty lines
                if not line_stripped or line_stripped.startswith('*'):
                    continue
                
                if field_upper in line_upper:
                    # Determine operation type
                    operation_type, business_context = self._classify_field_operation(line_upper, field_upper)
                    
                    # Get surrounding context
                    context_start = max(0, line_idx - 3)
                    context_end = min(len(lines), line_idx + 2)
                    context_lines = lines[context_start:context_end]
                    
                    reference = {
                        'line_number': line_idx,
                        'line_content': line_stripped,
                        'operation_type': operation_type,
                        'business_context': business_context,
                        'context_lines': context_lines,
                        'context_display': '\n'.join([
                            f"{context_start + i + 1:4d}: {ctx_line}"
                            for i, ctx_line in enumerate(context_lines)
                        ])
                    }
                    
                    # Categorize by operation type
                    if operation_type == 'DEFINITION':
                        analysis['definition'] = reference
                    elif operation_type == 'MOVE_TARGET':
                        analysis['usage_patterns']['input_operations'].append(reference)
                    elif operation_type == 'MOVE_SOURCE':
                        analysis['usage_patterns']['output_operations'].append(reference)
                    elif operation_type == 'ARITHMETIC':
                        analysis['usage_patterns']['arithmetic_operations'].append(reference)
                    elif operation_type == 'CONDITIONAL':
                        analysis['usage_patterns']['conditional_operations'].append(reference)
                    elif operation_type == 'CICS':
                        analysis['usage_patterns']['cics_operations'].append(reference)
                    
                    analysis['references'].append(reference)
            
            # Generate business summary
            analysis['business_summary'] = self._generate_field_business_summary(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing field {field_name}: {str(e)}")
            return analysis
    
    def _classify_field_operation(self, line_upper: str, field_upper: str) -> tuple:
        """Classify the type of operation involving the field"""
        # Field definition
        if ('PIC' in line_upper and 
            re.match(r'^\s*\d{2}\s+' + re.escape(field_upper), line_upper)):
            return 'DEFINITION', 'Data structure definition with type and length specification'
        
        # MOVE operations
        elif 'MOVE' in line_upper:
            # Field receives data (MOVE source TO field)
            if re.search(rf'MOVE\s+.+\s+TO\s+{re.escape(field_upper)}', line_upper):
                source_match = re.search(r'MOVE\s+([A-Z0-9\-\(\)]+)', line_upper)
                source = source_match.group(1) if source_match else 'unknown source'
                return 'MOVE_TARGET', f'Receives data from {source}'
            
            # Field provides data (MOVE field TO target)
            elif re.search(rf'MOVE\s+{re.escape(field_upper)}\s+TO', line_upper):
                target_match = re.search(rf'MOVE\s+{re.escape(field_upper)}\s+TO\s+([A-Z0-9\-\(\)]+)', line_upper)
                target = target_match.group(1) if target_match else 'unknown target'
                return 'MOVE_SOURCE', f'Provides data to {target}'
        
        # Arithmetic operations
        elif any(op in line_upper for op in ['COMPUTE', 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE']):
            return 'ARITHMETIC', 'Used in mathematical calculation or business computation'
        
        # Conditional operations
        elif any(op in line_upper for op in ['IF', 'WHEN', 'EVALUATE']):
            return 'CONDITIONAL', 'Used in business logic decision or program flow control'
        
        # CICS operations
        elif 'CICS' in line_upper:
            return 'CICS', 'Used in CICS transaction processing or screen handling'
        
        # File operations
        elif any(op in line_upper for op in ['READ', 'WRITE', 'REWRITE']):
            return 'FILE_IO', 'Used in file input/output operations'
        
        # General reference
        else:
            return 'REFERENCE', 'Referenced in program logic'
    
    def _generate_field_business_summary(self, analysis: Dict) -> str:
        """Generate business summary from field analysis"""
        patterns = analysis['usage_patterns']
        field_name = analysis['field_name']
        
        summary_parts = []
        
        if patterns['input_operations']:
            summary_parts.append(f"receives data ({len(patterns['input_operations'])} times)")
        
        if patterns['output_operations']:
            summary_parts.append(f"provides data ({len(patterns['output_operations'])} times)")
        
        if patterns['arithmetic_operations']:
            summary_parts.append(f"mathematical calculations ({len(patterns['arithmetic_operations'])} times)")
        
        if patterns['conditional_operations']:
            summary_parts.append(f"business decisions ({len(patterns['conditional_operations'])} times)")
        
        if patterns['cics_operations']:
            summary_parts.append(f"CICS transactions ({len(patterns['cics_operations'])} times)")
        
        if summary_parts:
            return f"{field_name} is actively used for: {', '.join(summary_parts)}"
        elif analysis['definition']:
            return f"{field_name} is defined but not actively used in the main program logic"
        else:
            return f"{field_name} usage pattern could not be determined"
    
    def _format_live_analysis_response(self, field_name: str, analysis: Dict) -> str:
        """Format response from live analysis"""
        response = f"Field Analysis: {field_name} (Live Analysis)\n"
        response += "=" * (len(field_name) + 28) + "\n"
        
        response += f"Program: {analysis['program_name']}\n"
        response += f"Business Summary: {analysis['business_summary']}\n"
        
        # Show definition
        if analysis['definition']:
            def_ref = analysis['definition']
            response += f"\nField Definition:\n"
            response += f"  Line {def_ref['line_number']}: {def_ref['line_content']}\n"
        
        # Show usage patterns
        patterns = analysis['usage_patterns']
        total_operations = sum(len(ops) for ops in patterns.values())
        
        if total_operations > 0:
            response += f"\nUsage Patterns ({total_operations} total operations):\n"
            
            if patterns['input_operations']:
                response += f"  Data Input ({len(patterns['input_operations'])} operations):\n"
                for op in patterns['input_operations'][:2]:
                    response += f"    Line {op['line_number']}: {op['line_content']}\n"
            
            if patterns['output_operations']:
                response += f"  Data Output ({len(patterns['output_operations'])} operations):\n"
                for op in patterns['output_operations'][:2]:
                    response += f"    Line {op['line_number']}: {op['line_content']}\n"
            
            if patterns['arithmetic_operations']:
                response += f"  Calculations ({len(patterns['arithmetic_operations'])} operations):\n"
                for op in patterns['arithmetic_operations'][:2]:
                    response += f"    Line {op['line_number']}: {op['line_content']}\n"
            
            if patterns['conditional_operations']:
                response += f"  Business Logic ({len(patterns['conditional_operations'])} operations):\n"
                for op in patterns['conditional_operations'][:2]:
                    response += f"    Line {op['line_number']}: {op['line_content']}\n"
        
        # Show detailed context for first significant operation
        significant_ops = (patterns['input_operations'] + patterns['output_operations'] + 
                          patterns['arithmetic_operations'])
        if significant_ops:
            important_op = significant_ops[0]
            response += f"\nDetailed Context Example:\n"
            response += important_op['context_display']
        
        return response
    
    def _handle_layout_query(self, session_id: str, message: str) -> str:
        """Handle queries about record layouts"""
        try:
            # Extract layout names from message
            layout_names = re.findall(r'\b([A-Z][A-Z0-9\-]{2,})\b', message.upper())
            
            if layout_names:
                # Get specific layout info
                layouts = self.db_manager.get_record_layouts(session_id)
                matching_layouts = []
                
                for layout_name in layout_names:
                    matches = [l for l in layouts if layout_name in l['layout_name'].upper()]
                    matching_layouts.extend(matches)
                
                if matching_layouts:
                    response = f"Record Layout Analysis:\n\n"
                    for layout in matching_layouts[:2]:
                        response += f"Layout: {layout['layout_name']}\n"
                        response += f"Program: {layout['program_name']}\n"
                        response += f"Level: {layout.get('level_number', '01')}\n"
                        response += f"Fields: {layout.get('fields_count', 0)}\n"
                        if layout.get('business_purpose'):
                            response += f"Purpose: {layout['business_purpose']}\n"
                        response += "\n"
                    
                    return response
            
            # General layout information
            return self._get_general_layout_info(session_id)
            
        except Exception as e:
            logger.error(f"Error handling layout query: {str(e)}")
            return f"Error analyzing layouts: {str(e)}"
    
    def _get_general_layout_info(self, session_id: str) -> str:
        """Get general information about record layouts"""
        try:
            layouts = self.db_manager.get_record_layouts(session_id)
            
            if not layouts:
                return "No record layouts found. Upload COBOL programs with data structures first."
            
            response = f"Record Layouts ({len(layouts)} found):\n\n"
            
            for layout in layouts[:5]:  # Show first 5
                response += f"Layout: {layout['layout_name']}\n"
                response += f"Program: {layout['program_name']}\n"
                response += f"Fields: {layout.get('fields_count', 0)}\n"
                if layout.get('business_purpose'):
                    response += f"Purpose: {layout['business_purpose']}\n"
                response += "\n"
            
            if len(layouts) > 5:
                response += f"... and {len(layouts) - 5} more layouts\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting layout info: {str(e)}")
            return "Error retrieving layout information"
    
    def _handle_program_query(self, session_id: str, message: str) -> str:
        """Handle queries about programs"""
        try:
            components = self.db_manager.get_session_components(session_id)
            programs = [c for c in components if c.get('component_type') == 'PROGRAM']
            
            if not programs:
                return "No programs have been analyzed yet. Please upload COBOL program files first."
            
            response = f"Program Analysis ({len(programs)} programs):\n\n"
            
            for program in programs:
                response += f"Program: {program['component_name']}\n"
                response += f"Lines: {program.get('total_lines', 0)}\n"
                response += f"Fields: {program.get('total_fields', 0)}\n"
                
                if program.get('business_purpose'):
                    response += f"Purpose: {program['business_purpose']}\n"
                
                response += "\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling program query: {str(e)}")
            return f"Error analyzing programs: {str(e)}"
    
    def _handle_summary_query(self, session_id: str) -> str:
        """Handle summary/overview queries"""
        try:
            metrics = self.db_manager.get_session_metrics(session_id)
            components = self.db_manager.get_session_components(session_id)
            
            response = "Project Analysis Summary:\n"
            response += "=" * 25 + "\n"
            
            # Basic metrics
            response += f"Total Components: {metrics.get('total_components', 0)}\n"
            response += f"Total Fields: {metrics.get('total_fields', 0)}\n"
            response += f"Lines of Code: {metrics.get('total_lines', 0)}\n"
            
            # Component breakdown
            if components:
                component_types = {}
                for comp in components:
                    comp_type = comp.get('component_type', 'Unknown')
                    component_types[comp_type] = component_types.get(comp_type, 0) + 1
                
                response += f"\nComponent Breakdown:\n"
                for comp_type, count in component_types.items():
                    response += f"  {comp_type}: {count}\n"
            
            # Token usage
            token_usage = metrics.get('token_usage', {})
            if token_usage:
                total_tokens = token_usage.get('total_prompt_tokens', 0) + token_usage.get('total_response_tokens', 0)
                response += f"\nToken Usage: {total_tokens:,} tokens\n"
                response += f"LLM Calls: {token_usage.get('total_calls', 0)}\n"
            
            response += f"\nAsk me about specific fields, record layouts, or programs!"
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling summary query: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def _handle_general_query(self, session_id: str, message: str) -> str:
        """Handle general queries with LLM context"""
        try:
            # Check for help request
            if any(word in message.lower() for word in ['help', 'what can you', 'how do i']):
                return self._get_help_response()
            
            # Get session context for LLM
            session_context = self._get_session_context(session_id)
            
            # Create LLM prompt with session context
            prompt = f"""
    You are a mainframe COBOL analysis expert in private bank wealth management domain. The user has asked: "{message}"

    Current Analysis Context:
    - Programs Analyzed: {session_context.get('total_programs', 0)}
    - Record Layouts: {session_context.get('total_layouts', 0)}
    - Total Fields: {session_context.get('total_fields', 0)}

    Available Components: {', '.join(session_context.get('component_names', [])[:5])}
    Available Fields: {', '.join(session_context.get('field_names', [])[:10])}

    Please provide a helpful response that:
    1. Addresses their question directly
    2. Uses the available analysis data
    3. Suggests specific fields, programs, or layouts they can ask about
    4. Provides actionable insights

    If you don't have enough context to answer fully, suggest what they should upload or analyze next.
    """
            
            response = self.llm_client.call_llm(prompt, max_tokens=600, temperature=0.3)
            
            # Log LLM call
            self.db_manager.log_llm_call(
                session_id, 'chat_general', 1, 1,
                response.prompt_tokens, response.response_tokens, response.processing_time_ms,
                response.success, response.error_message
            )
            
            if response.success and response.content:
                return response.content
            else:
                # Fallback response
                return self._get_contextual_help_response(session_context)
                
        except Exception as e:
            logger.error(f"Error handling general query: {str(e)}")
            return "I can help analyze your mainframe code. Ask me about specific fields, programs, or record layouts!"

    def _get_session_context(self, session_id: str) -> Dict:
        """Get current session context for LLM"""
        try:
            metrics = self.db_manager.get_session_metrics(session_id)
            components = self.db_manager.get_session_components(session_id)
            
            component_names = [c['component_name'] for c in components[:10]]
            field_names = self._get_available_fields_sample(session_id).split(', ')[:10]
            
            return {
                'total_programs': metrics.get('component_counts', {}).get('PROGRAM', 0),
                'total_layouts': metrics.get('component_counts', {}).get('RECORD_LAYOUT', 0),
                'total_fields': metrics.get('total_fields', 0),
                'component_names': component_names,
                'field_names': field_names
            }
        except Exception as e:
            logger.error(f"Error getting session context: {str(e)}")
            return {}

    def _get_contextual_help_response(self, session_context: Dict) -> str:
        """Generate contextual help based on session state"""
        if session_context.get('total_programs', 0) == 0:
            return ("I don't see any analyzed programs yet. Please upload COBOL files first, "
                "then I can help you understand fields, record layouts, and business logic!")
        
        programs = session_context.get('component_names', [])[:3]
        fields = session_context.get('field_names', [])[:5]
        
        response = f"I can help you analyze your mainframe code! "
        
        if programs:
            response += f"\n\nYour programs: {', '.join(programs)}"
        if fields:
            response += f"\n\nSample fields: {', '.join(fields)}"
        
        response += "\n\nTry asking:\n"
        response += "• 'Tell me about [FIELD-NAME]'\n"
        response += "• 'How is [FIELD-NAME] used?'\n"
        response += "• 'Show me the record layouts'\n"
        response += "• 'What does this program do?'"
        
        return response
    
    def _get_help_response(self) -> str:
        """Provide help information"""
        return ("Mainframe Code Analyzer Help:\n\n"
               "I can analyze your COBOL programs and provide detailed information about:\n\n"
               "1. Field Analysis:\n"
               "   - Field definitions and data types\n"
               "   - How fields are used (input, output, calculations)\n"
               "   - Source code examples showing field usage\n"
               "   - Business purpose and data flow\n\n"
               "2. Program Structure:\n"
               "   - Record layouts and data structures\n"
               "   - Component relationships and dependencies\n"
               "   - CICS transaction processing\n"
               "   - File operations and data flow\n\n"
               "Example questions:\n"
               "   - 'Tell me about CUSTOMER-NAME field'\n"
               "   - 'How is ACCOUNT-BALANCE calculated?'\n"
               "   - 'Show me the EMPLOYEE-RECORD layout'\n"
               "   - 'What programs use TRANSACTION-CODE?'\n\n"
               "Just ask about any field name and I'll show you the actual COBOL code!")
    
    def _get_available_fields_sample(self, session_id: str) -> str:
        """Get sample of available fields"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT field_name FROM field_analysis_details 
                    WHERE session_id = ? 
                    ORDER BY field_name 
                    LIMIT 10
                ''', (session_id,))
                
                fields = [row[0] for row in cursor.fetchall()]
                return ', '.join(fields) if fields else 'No fields analyzed yet'
                
        except Exception as e:
            logger.error(f"Error getting available fields: {str(e)}")
            return 'Error retrieving field list'
