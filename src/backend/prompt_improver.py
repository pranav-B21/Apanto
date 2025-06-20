import os
import requests
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class PromptImprover:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_base_url = "https://api.groq.com/openai/v1"
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
    
    def improve_prompt(self, original_prompt: str) -> Dict[str, Any]:
        """
        Improve a prompt using Groq API with Llama model
        """
        try:
            # Create the system prompt for prompt improvement
            system_prompt = """You are an expert prompt engineer. Your task is to improve user prompts by making them more specific, clear, and effective. 

Consider these aspects when improving prompts:
1. **Clarity**: Make the request unambiguous and easy to understand
2. **Specificity**: Add relevant details, context, and constraints
3. **Structure**: Organize the prompt logically with clear sections if needed
4. **Output Format**: Specify desired output format (JSON, markdown, code, etc.)
5. **Constraints**: Add relevant limitations or requirements
6. **Examples**: Include examples when helpful
7. **Context**: Add relevant background information
8. **Tone**: Ensure appropriate tone for the task
9. **Completeness**: Ensure all necessary information is included
10. **Actionability**: Make sure the prompt leads to actionable results

Return your response as a JSON object with these fields:
- "improved_prompt": The enhanced version of the original prompt
- "improvements_made": List of specific improvements made
- "reasoning": Brief explanation of why these improvements were made
- "confidence": Confidence score (0-100) in the improvement quality

Focus on making the prompt more effective while preserving the user's original intent."""

            # Prepare the request to Groq API
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama3-70b-8192",  # Using Llama model as requested
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please improve this prompt:\n\n{original_prompt}"}
                ],
                "temperature": 0.3,  # Lower temperature for more consistent results
                "max_tokens": 1000
            }
            
            # Make the API call
            response = requests.post(
                f"{self.groq_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Groq API error: {response.status_code} - {response.text}")
            
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            
            # Try to parse the JSON response
            try:
                # Extract JSON from the response (it might be wrapped in markdown)
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    json_str = ai_response[json_start:json_end]
                    parsed_response = json.loads(json_str)
                else:
                    # Fallback: treat the entire response as the improved prompt
                    parsed_response = {
                        "improved_prompt": ai_response,
                        "improvements_made": ["Enhanced clarity and structure"],
                        "reasoning": "AI provided an improved version of the prompt",
                        "confidence": 85
                    }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                parsed_response = {
                    "improved_prompt": ai_response,
                    "improvements_made": ["Enhanced clarity and structure"],
                    "reasoning": "AI provided an improved version of the prompt",
                    "confidence": 85
                }
            
            return {
                "success": True,
                "original_prompt": original_prompt,
                "improved_prompt": parsed_response.get("improved_prompt", ai_response),
                "improvements_made": parsed_response.get("improvements_made", []),
                "reasoning": parsed_response.get("reasoning", ""),
                "confidence": parsed_response.get("confidence", 85),
                "model_used": "llama3-70b-8192",
                "tokens_used": result.get("usage", {}).get("total_tokens", 0)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "original_prompt": original_prompt,
                "improved_prompt": original_prompt,  # Fallback to original
                "improvements_made": [],
                "reasoning": "Failed to improve prompt due to API error",
                "confidence": 0,
                "model_used": "llama3-70b-8192",
                "tokens_used": 0
            }
    
    def get_improvement_suggestions(self, original_prompt: str) -> Dict[str, Any]:
        """
        Get specific suggestions for improving a prompt
        """
        try:
            system_prompt = """You are a prompt engineering expert. Analyze the given prompt and provide specific, actionable suggestions for improvement.

Focus on these areas:
1. **Missing Context**: What additional information would make this prompt clearer?
2. **Output Format**: What format specification would be helpful?
3. **Constraints**: What limitations or requirements should be added?
4. **Examples**: Would examples help clarify the request?
5. **Specificity**: What details are too vague or missing?

Return your response as a JSON object with:
- "suggestions": Array of specific improvement suggestions as strings
- "priority": "high", "medium", or "low" based on how much the prompt needs improvement
- "estimated_impact": Brief description of how these improvements would help"""

            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this prompt and provide improvement suggestions:\n\n{original_prompt}"}
                ],
                "temperature": 0.2,
                "max_tokens": 800
            }
            
            response = requests.post(
                f"{self.groq_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Groq API error: {response.status_code}")
            
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            
            # Try to parse JSON response
            try:
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    json_str = ai_response[json_start:json_end]
                    parsed_response = json.loads(json_str)
                else:
                    parsed_response = {
                        "suggestions": ["Add more specific details", "Specify output format"],
                        "priority": "medium",
                        "estimated_impact": "Would make the prompt clearer and more actionable"
                    }
            except json.JSONDecodeError:
                parsed_response = {
                    "suggestions": ["Add more specific details", "Specify output format"],
                    "priority": "medium",
                    "estimated_impact": "Would make the prompt clearer and more actionable"
                }
            
            return {
                "success": True,
                "suggestions": parsed_response.get("suggestions", []),
                "priority": parsed_response.get("priority", "medium"),
                "estimated_impact": parsed_response.get("estimated_impact", ""),
                "model_used": "llama3-70b-8192"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "suggestions": [],
                "priority": "low",
                "estimated_impact": "Unable to analyze due to API error"
            } 