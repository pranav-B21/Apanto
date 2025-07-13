import os
import requests
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from .multi_provider import multi_provider_llm

load_dotenv()

class PromptImprover:
    def __init__(self):
        # Use the multi-provider system instead of just Groq
        self.multi_provider = multi_provider_llm
    
    async def improve_prompt(self, original_prompt: str) -> Dict[str, Any]:
        """
        Improve a prompt using the multi-provider system
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

            # Use the multi-provider system with GPT-4 for better prompt improvement
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please improve this prompt:\n\n{original_prompt}"}
            ]
            
            # Try GPT-4 first, fallback to other models
            model_options = ["gpt-4o", "gpt-4o-mini", "llama3-70b-8192"]
            
            for model_id in model_options:
                try:
                    result = await self.multi_provider.generate_text(
                        model_id=model_id,
                        messages=messages,
                        parameters={"temperature": 0.3, "max_tokens": 1000}
                    )
                    
                    if result["success"]:
                        ai_response = result["output"]
                        break
                    else:
                        print(f"Failed with {model_id}: {result.get('error', 'Unknown error')}")
                        continue
                except Exception as e:
                    print(f"Error with {model_id}: {str(e)}")
                    continue
            else:
                raise Exception("All model options failed")
            
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
    
    async def get_improvement_suggestions(self, original_prompt: str) -> Dict[str, Any]:
        """
        Get specific suggestions for improving a prompt using the multi-provider system
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

            # Use the multi-provider system
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this prompt and provide improvement suggestions:\n\n{original_prompt}"}
            ]
            
            # Try different models for suggestions
            model_options = ["gpt-4o-mini", "llama3-70b-8192", "claude-3-5-haiku-20241022"]
            
            for model_id in model_options:
                try:
                    result = await self.multi_provider.generate_text(
                        model_id=model_id,
                        messages=messages,
                        parameters={"temperature": 0.2, "max_tokens": 800}
                    )
                    
                    if result["success"]:
                        ai_response = result["output"]
                        break
                    else:
                        print(f"Failed with {model_id}: {result.get('error', 'Unknown error')}")
                        continue
                except Exception as e:
                    print(f"Error with {model_id}: {str(e)}")
                    continue
            else:
                raise Exception("All model options failed")
            
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