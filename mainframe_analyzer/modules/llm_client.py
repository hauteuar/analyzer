"""
LLM Client Module for VLLM Integration
Handles communication with VLLM server with retry logic and error handling
"""

import requests
import json
import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import backoff
import re

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    success: bool
    content: str
    prompt_tokens: int = 0
    response_tokens: int = 0
    processing_time_ms: int = 0
    error_message: str = ""

class LLMClient:
    def __init__(self, endpoint: str = "http://localhost:8100/generate"):
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.max_retries = 3
        self.rate_limit_delay = 1.0  # 1 second between calls
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
        """
        Call VLLM endpoint with retry logic and error handling
        """
        self._apply_rate_limit()
        
        # Use provided values or defaults
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if temperature is None:
            temperature = self.default_temperature
        
        logger.info(f"🤖 Making LLM call to {self.endpoint}")
        logger.info(f"📊 Request params: max_tokens={max_tokens}, temperature={temperature}")
        logger.info(f"📝 Prompt length: {len(prompt)} characters (~{len(prompt)//4} tokens)")
        
        start_time = time.time()
        
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": ["</analysis>", "END_OF_RESPONSE"],
                "stream": False
            }
            
            logger.debug("📤 Sending request to VLLM server...")
            
            response = self.session.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            if response.status_code == 200:
                logger.info(f"✅ LLM request successful in {processing_time}ms")
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
                prompt_tokens = usage.get('prompt_tokens', len(prompt) // 4)  # Estimate
                completion_tokens = usage.get('completion_tokens', len(content) // 4)  # Estimate
                
                logger.info(f"📈 Token usage: {prompt_tokens} prompt + {completion_tokens} response = {prompt_tokens + completion_tokens} total")
                logger.info(f"📄 Response length: {len(content)} characters")
                
                return LLMResponse(
                    success=True,
                    content=content,
                    prompt_tokens=prompt_tokens,
                    response_tokens=completion_tokens,
                    processing_time_ms=processing_time
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ LLM request failed: {error_msg}")
                
                return LLMResponse(
                    success=False,
                    content="",
                    processing_time_ms=processing_time,
                    error_message=error_msg
                )
                
        except requests.exceptions.Timeout:
            error_msg = f"LLM request timeout after {self.timeout}s"
            logger.error(f"⏰ {error_msg}")
            return LLMResponse(
                success=False,
                content="",
                processing_time_ms=int((time.time() - start_time) * 1000),
                error_message=error_msg
            )
            
        except requests.exceptions.ConnectionError:
            error_msg = f"Cannot connect to LLM server at {self.endpoint}"
            logger.error(f"🔌 {error_msg}")
            return LLMResponse(
                success=False,
                content="",
                processing_time_ms=int((time.time() - start_time) * 1000),
                error_message=error_msg
            )
            
        except Exception as e:
            error_msg = f"Unexpected error calling LLM: {str(e)}"
            logger.error(f"💥 {error_msg}")
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
            # First, try to parse the entire response as JSON
            return json.loads(response_content.strip())
        except json.JSONDecodeError:
            pass
        
        # Clean the response content
        cleaned_content = response_content.strip()
        
        # Look for JSON blocks in code fences
        json_patterns = [
            r'```json\s*(.*?)\s*```',  # ```json ... ```
            r'```\s*([\s\S]*?)\s*```',  # ``` ... ``` (any code block)
            r'<json>(.*?)</json>',      # <json>...</json>
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, cleaned_content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    # Clean the match
                    clean_match = match.strip()
                    if clean_match:
                        return json.loads(clean_match)
                except json.JSONDecodeError:
                    continue
        
        # Look for JSON-like objects using bracket matching
        brace_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested objects
            r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'  # Simple nested arrays
        ]
        
        for pattern in brace_patterns:
            matches = re.findall(pattern, cleaned_content, re.DOTALL)
            for match in matches:
                try:
                    # Try to parse each potential JSON object
                    parsed = json.loads(match.strip())
                    if isinstance(parsed, (dict, list)) and parsed:  # Valid non-empty structure
                        return parsed
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
        
        # Last resort: try to fix common JSON issues
        try:
            # Fix common issues like trailing commas, single quotes, etc.
            fixed_content = cleaned_content
            
            # Replace single quotes with double quotes (be careful with contractions)
            fixed_content = re.sub(r"'([^']*)':", r'"\1":', fixed_content)  # Keys
            fixed_content = re.sub(r":\s*'([^']*)'", r': "\1"', fixed_content)  # Values
            
            # Remove trailing commas
            fixed_content = re.sub(r',(\s*[}\]])', r'\1', fixed_content)
            
            # Try to extract JSON from the fixed content
            for pattern in [r'\{.*\}', r'\[.*\]']:
                matches = re.findall(pattern, fixed_content, re.DOTALL)
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.warning(f"Error in JSON extraction fallback: {str(e)}")
        
        logger.warning(f"Could not extract JSON from LLM response. Content preview: {cleaned_content[:200]}...")
        return cleaned_content  # Return raw content if all else fails
    
    def call_with_structured_output(self, prompt: str, expected_structure: str = "json") -> Tuple[bool, Dict]:
        """
        Call LLM and expect structured output (JSON)
        """
        enhanced_prompt = f"{prompt}\n\nRespond with valid JSON only. No additional text or explanations."
        
        response = self.call_llm(enhanced_prompt)
        
        if not response.success:
            return False, {"error": response.error_message}
        
        parsed_json = self.extract_json_from_response(response.content)
        
        if parsed_json is None:
            return False, {
                "error": "Could not parse JSON from response",
                "raw_response": response.content[:500]  # First 500 chars for debugging
            }
        
        return True, parsed_json
    
    def health_check(self) -> bool:
        """Check if LLM server is healthy"""
        try:
            test_prompt = "Hello, respond with 'OK' if you can process this request."
            response = self.call_llm(test_prompt, max_tokens=10)
            return response.success and "OK" in response.content.upper()
        except Exception as e:
            logger.error(f"LLM health check failed: {str(e)}")
            return False