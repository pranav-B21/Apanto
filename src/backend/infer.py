import os
import requests
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from .hf_hosting import HuggingFaceInference
from .groq_client import GroqClient

# Load environment variables from .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def run_prompt_on_llm(model_id: str, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        raise Exception(f"Groq API error: {resp.text}")

    return resp.json()["choices"][0]["message"]["content"]

class InferenceService:
    def __init__(self):
        self.hf_inference = HuggingFaceInference()
        self.groq_client = GroqClient()
    
    def run_inference(self, model_id: str, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run inference on a model, routing to appropriate service based on model ID prefix
        """
        # Default parameters
        if parameters is None:
            parameters = {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.95
            }
        
        # Route to appropriate service based on model ID prefix
        if model_id.startswith('hf-'):
            # Remove the hf- prefix to get the actual model ID
            hf_model_id = model_id[3:]
            return self.hf_inference.run_inference(
                model_id=hf_model_id,
                prompt=prompt,
                parameters=parameters
            )
        else:
            # Use Groq for non-HF models
            return self.groq_client.run_inference(
                model_id=model_id,
                prompt=prompt,
                parameters=parameters
            )
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """
        Get the status of a model
        """
        if model_id.startswith('hf-'):
            # For HF models, we need to check the hosted_models table
            # This would be implemented in your database layer
            return self._get_hf_model_status(model_id[3:])
        else:
            # For Groq models, check their status
            return self.groq_client.get_model_status(model_id)
    
    def _get_hf_model_status(self, hosted_model_id: str) -> Dict[str, Any]:
        """
        Get status of a hosted HF model from the database
        """
        # This would be implemented in your database layer
        # For now, return a placeholder
        return {
            "status": "active",
            "model_id": f"hf-{hosted_model_id}"
        }

