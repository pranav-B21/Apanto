import os
import requests
from dotenv import load_dotenv

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

