"""
Multi-Provider AI Model Integration
Supports OpenAI, DeepSeek, Gemini, and Llama models with unified interfaces
"""

import os
import requests
import json
import time
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
import anthropic

load_dotenv()

class MultiProviderLLM:
    """Unified interface for multiple AI providers"""
    
    def __init__(self):
        # Load API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Initialize clients
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_client = None
        
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        if self.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.anthropic_api_key
            )
        
        if self.gemini_api_key:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
    
    def get_provider_from_model_id(self, model_id: str) -> str:
        """Determine provider from model ID"""
        model_id_lower = model_id.lower()
        
        if model_id_lower.startswith(('gpt-', 'text-', 'dall-e')):
            return "openai"
        elif model_id_lower.startswith(('deepseek', 'deepseek-chat')):
            return "deepseek"
        elif model_id_lower.startswith(('gemini', 'gemini-pro')):
            return "gemini"
        elif model_id_lower.startswith(('claude', 'sonnet', 'opus')):
            return "anthropic"
        elif model_id_lower.startswith(('llama', 'mixtral', 'llama3')):
            return "groq"  # Using Groq for Llama models
        else:
            # Default to Groq for unknown models
            return "groq"
    
    def format_messages_for_provider(self, messages: List[Dict[str, str]], provider: str) -> Any:
        """Format messages according to provider requirements"""
        if provider in ["openai", "groq", "deepseek"]:
            return messages
        elif provider == "anthropic":
            # Anthropic uses different message format
            formatted = []
            for msg in messages:
                if msg["role"] == "user":
                    formatted.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    formatted.append({"role": "assistant", "content": msg["content"]})
                elif msg["role"] == "system":
                    # Anthropic doesn't support system messages directly
                    # We'll prepend to the first user message
                    if formatted and formatted[0]["role"] == "user":
                        formatted[0]["content"] = f"{msg['content']}\n\n{formatted[0]['content']}"
            return formatted
        elif provider == "gemini":
            # Gemini uses a different format
            formatted = []
            for msg in messages:
                if msg["role"] == "user":
                    formatted.append({"role": "user", "parts": [{"text": msg["content"]}]})
                elif msg["role"] == "assistant":
                    formatted.append({"role": "model", "parts": [{"text": msg["content"]}]})
            return formatted
        return messages
    
    async def generate_text(
        self, 
        model_id: str, 
        messages: Union[str, List[Dict[str, str]]], 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate text using the appropriate provider"""


        
        # Handle string input (legacy support)
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        provider = self.get_provider_from_model_id(model_id)
        formatted_messages = self.format_messages_for_provider(messages, provider)
        
        # Default parameters
        default_params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.95
        }
        if parameters:
            default_params.update(parameters)
        
        try:
            if provider == "openai":
                return await self.call_openai(model_id, formatted_messages, default_params)
            elif provider == "deepseek":
                return await self.call_deepseek(model_id, formatted_messages, default_params)
            elif provider == "gemini":
                return await self.call_gemini(model_id, formatted_messages, default_params)
                
            elif provider == "anthropic":
                return await self.call_anthropic(model_id, formatted_messages, default_params)
            elif provider == "groq":
                return await self.call_groq(model_id, formatted_messages, default_params)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "provider": provider,
                "model_id": model_id
            }
    
    async def call_openai(self, model_id: str, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Call OpenAI API"""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 1000),
                top_p=params.get("top_p", 0.95)
            )
            
            return {
                "success": True,
                "output": response.choices[0].message.content,
                "provider": "openai",
                "model_id": model_id,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"OpenAI API error: {str(e)}",
                "provider": "openai",
                "model_id": model_id
            }
    
    async def call_deepseek(self, model_id: str, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Call DeepSeek API"""
        if not self.deepseek_api_key:
            raise ValueError("DeepSeek API key not configured")
        
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 1000),
            "top_p": params.get("top_p", 0.95)
        }
        
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
            
            result = response.json()
            return {
                "success": True,
                "output": result["choices"][0]["message"]["content"],
                "provider": "deepseek",
                "model_id": model_id,
                "usage": result.get("usage", {})
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"DeepSeek API error: {str(e)}",
                "provider": "deepseek",
                "model_id": model_id
            }
    
    async def call_gemini(self, model_id: str, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.gemini_client:
            raise ValueError("Gemini API key not configured")
        try:
            def extract_text(msg):
                if "content" in msg:
                    return msg["content"]
                elif "parts" in msg and isinstance(msg["parts"], list):
                    return "".join(part.get("text", "") for part in msg["parts"])
                return ""
            prompt = "\n".join([extract_text(msg) for msg in messages if msg["role"] == "user"])
            response = self.gemini_client.models.generate_content(
                model=model_id,
                contents=prompt
            )
            return {
                "success": True,
                "output": response.text,
                "provider": "gemini",
                "model_id": model_id,
                "usage": {}  # Gemini doesn't provide detailed usage
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Gemini API error: {str(e)}",
                "provider": "gemini",
                "model_id": model_id
            }
    
    async def call_anthropic(self, model_id: str, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Call Anthropic API"""
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not configured")
        
        try:
            # Anthropic uses a different message format
            system_message = None
            formatted_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            response = self.anthropic_client.messages.create(
                model=model_id,
                max_tokens=params.get("max_tokens", 1000),
                temperature=params.get("temperature", 0.7),
                system=system_message,
                messages=formatted_messages
            )
            
            return {
                "success": True,
                "output": response.content[0].text,
                "provider": "anthropic",
                "model_id": model_id,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Anthropic API error: {str(e)}",
                "provider": "anthropic",
                "model_id": model_id
            }
    
    async def call_groq(self, model_id: str, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Call Groq API"""
        if not self.groq_api_key:
            raise ValueError("Groq API key not configured")
        
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 1000),
            "top_p": params.get("top_p", 0.95)
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Groq API error: {response.status_code} - {response.text}")
            
            result = response.json()
            return {
                "success": True,
                "output": result["choices"][0]["message"]["content"],
                "provider": "groq",
                "model_id": model_id,
                "usage": result.get("usage", {})
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Groq API error: {str(e)}",
                "provider": "groq",
                "model_id": model_id
            }
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models by provider"""
        models = {
            "openai": [
                "gpt-4o",
                "gpt-4o-mini", 
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo"
            ],
            "deepseek": [
                "deepseek-chat",
                "deepseek-coder"
            ],
            "gemini": [
                "gemini-2.5-pro",
                "gemini-2.5-flash"
            ],
            "anthropic": [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ],
            "groq": [
                "llama3-70b-8192",
                "llama3-8b-8192",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ]
        }
        return models
    
    def estimate_cost(self, provider: str, model_id: str, tokens_used: int) -> float:
        """Estimate cost based on provider and model"""
        # Rough cost estimates per 1K tokens (in USD)
        cost_estimates = {
            "openai": {
                "gpt-4o": 0.005,
                "gpt-4o-mini": 0.00015,
                "gpt-4-turbo": 0.01,
                "gpt-4": 0.03,
                "gpt-3.5-turbo": 0.0005
            },
            "deepseek": {
                "deepseek-chat": 0.00014,
                "deepseek-coder": 0.00014
            },
            "gemini": {
                "gemini-pro": 0.0005,
                "gemini-pro-vision": 0.0005
            },
            "anthropic": {
                "claude-3-5-sonnet-20241022": 0.003,
                "claude-3-5-haiku-20241022": 0.00025,
                "claude-3-opus-20240229": 0.015,
                "claude-3-sonnet-20240229": 0.003,
                "claude-3-haiku-20240307": 0.00025
            },
            "groq": {
                "llama3-70b-8192": 0.00024,
                "llama3-8b-8192": 0.00005,
                "mixtral-8x7b-32768": 0.00024,
                "gemma2-9b-it": 0.0001
            }
        }
        
        base_cost = cost_estimates.get(provider, {}).get(model_id, 0.001)
        return (tokens_used / 1000) * base_cost

# Global instance
multi_provider_llm = MultiProviderLLM() 