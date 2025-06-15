import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch

# Import our existing modules
from infer import run_prompt_on_llm
from scorer import load_models, select_best_model
from analyzer import classify_prompt
from database import load_models_from_database, db_manager
from hf_integration import ModelLoader, HuggingFaceHostingService

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Smart Prompt Stream Backend", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Primary Vite dev port
        "http://localhost:5174",  # Vite backup port
        "http://localhost:3000",  # Alternative React port
        "http://127.0.0.1:5173",  # IPv4 localhost
        "http://127.0.0.1:5174",  # IPv4 localhost backup
        "http://localhost:8081",  # Backup port 1
        "http://localhost:8082",  # Backup port 2
        "http://localhost:4173"   # Vite preview port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    prompt: str
    priority: str = "accuracy"  # accuracy, speed, cost
    session_id: Optional[str] = None
    model_id: Optional[str] = None

class ChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    response: str
    model_used: str
    task_type: str
    confidence: float
    response_time: float
    tokens_used: int
    estimated_cost: float
    top_3_models: List[str]

class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    name: str
    model_id: str
    scores: Dict[str, Dict[str, float]]

class EnhancementSuggestion(BaseModel):
    id: str
    suggestion: str
    type: str
    confidence: int

class HostModelRequest(BaseModel):
    model_url: str
    custom_name: Optional[str] = None

# Global variables to cache models
cached_models = None

# --- LLM Memory Store ---
from threading import Lock

class SessionMemory:
    def __init__(self, max_turns: int = 8):
        self.memory: Dict[str, List[Dict[str, str]]] = {}
        self.lock = Lock()
        self.max_turns = max_turns

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self.lock:
            return self.memory.get(session_id, [])

    def append(self, session_id: str, role: str, content: str):
        with self.lock:
            if session_id not in self.memory:
                self.memory[session_id] = []
            self.memory[session_id].append({"role": role, "content": content})
            # Keep only the last N turns
            self.memory[session_id] = self.memory[session_id][-self.max_turns:]

    def clear(self, session_id: str):
        with self.lock:
            if session_id in self.memory:
                del self.memory[session_id]

# Instantiate global memory store
session_memory = SessionMemory(max_turns=8)

# Instantiate global model loader
model_loader = ModelLoader()

# Instantiate global HuggingFaceHostingService
hosting_service = HuggingFaceHostingService(db_manager)

def get_models():
    """Get cached models or load them if not cached"""
    global cached_models
    if cached_models is None:
        cached_models = load_models()
    return cached_models

@app.get("/")
async def root():
    return {"message": "AI Smart Prompt Stream Backend is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is operational"}

@app.get("/models")
async def get_available_models():
    """Get all available models from the database"""
    try:
        models = get_models()
        return {
            "models": [
                {
                    "name": model["name"],
                    "model_id": model["model_id"],
                    "scores": model.get("scores", {})
                }
                for model in models
            ],
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint that processes prompts and returns AI responses"""
    try:
        import time
        start_time = time.time()
        
        # 0. Get session_id (use a default if not provided)
        session_id = request.session_id or "default"
        
        # 1. Classify the prompt type
        task_type = classify_prompt(request.prompt)
        
        # 2. Get available models
        models = get_models()
        if not models:
            raise HTTPException(status_code=500, detail="No models available")
        
        # 3. Select the model: use provided model_id if present, else select best
        if request.model_id:
            selected_model = next((m for m in models if m["model_id"] == request.model_id), None)
            if not selected_model:
                raise HTTPException(status_code=400, detail=f"Model ID {request.model_id} not found")
            # For top_3_models, just return the selected one first
            top_3_models = [selected_model["name"]] + [m["name"] for m in models if m["model_id"] != request.model_id][:2]
        else:
            selected_model, top_3_models = select_best_model(
                models, 
                task_type, 
                request.priority
            )
        
        # 4. Prepare LLM context (last N messages)
        history = session_memory.get_history(session_id)
        context_messages = history[-session_memory.max_turns:] if history else []
        context_messages.append({"role": "user", "content": request.prompt})
        
        # 5. Run the prompt on the selected model (pass context if supported)
        try:
            # If your LLM supports chat history, pass context_messages instead of just prompt
            # For now, join messages for simple LLMs
            if hasattr(selected_model, 'supports_chat') and selected_model.supports_chat:
                ai_response = model_loader.generate_text(
                    selected_model["model_id"],
                    context_messages
                )
                if isinstance(ai_response, dict) and ai_response.get('success'):
                    ai_response = ai_response.get('text', '')
                elif isinstance(ai_response, dict):
                    raise Exception(ai_response.get('error', 'Unknown error'))
            else:
                # Fallback: concatenate context for plain LLMs
                prompt_with_context = "\n".join([m["content"] for m in context_messages])
                ai_response = model_loader.generate_text(
                    selected_model["model_id"],
                    prompt_with_context
                )
                if isinstance(ai_response, dict) and ai_response.get('success'):
                    ai_response = ai_response.get('text', '')
                elif isinstance(ai_response, dict):
                    raise Exception(ai_response.get('error', 'Unknown error'))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM inference failed: {str(e)}")
        
        # 6. Calculate metrics
        end_time = time.time()
        response_time = end_time - start_time
        
        # Estimate tokens (rough approximation)
        tokens_used = len(request.prompt.split()) + len(ai_response.split())
        
        # Estimate cost (rough approximation based on model)
        estimated_cost = tokens_used * 0.0001  # $0.0001 per token as baseline
        
        # Get confidence from model selection (or default)
        confidence = 85.0  # You could enhance this based on model scoring
        
        # 7. Update memory with user and assistant turns
        session_memory.append(session_id, "user", request.prompt)
        session_memory.append(session_id, "assistant", ai_response)
        
        return ChatResponse(
            response=ai_response,
            model_used=selected_model["name"],
            task_type=task_type,
            confidence=confidence,
            response_time=response_time,
            tokens_used=tokens_used,
            estimated_cost=estimated_cost,
            top_3_models=top_3_models
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/enhance-prompt")
async def enhance_prompt(request: dict):
    """Generate enhancement suggestions for a prompt"""
    try:
        prompt = request.get("prompt", "")
        
        if len(prompt) < 10:
            return {"enhancements": []}
        
        # Generate context-aware suggestions
        enhancements = []
        
        # Basic enhancement suggestions based on prompt analysis
        if "code" in prompt.lower() or "programming" in prompt.lower():
            enhancements.extend([
                {
                    "id": "code_1",
                    "suggestion": "Specify the programming language",
                    "type": "context",
                    "confidence": 92
                },
                {
                    "id": "code_2",
                    "suggestion": "Include expected input/output format",
                    "type": "format",
                    "confidence": 85
                },
                {
                    "id": "code_3",
                    "suggestion": "Add error handling requirements",
                    "type": "requirements",
                    "confidence": 78
                }
            ])
        elif "write" in prompt.lower() or "essay" in prompt.lower():
            enhancements.extend([
                {
                    "id": "write_1",
                    "suggestion": "Specify target audience",
                    "type": "context",
                    "confidence": 88
                },
                {
                    "id": "write_2",
                    "suggestion": "Add desired tone and style",
                    "type": "style",
                    "confidence": 82
                },
                {
                    "id": "write_3",
                    "suggestion": "Include word count requirement",
                    "type": "format",
                    "confidence": 75
                }
            ])
        else:
            enhancements.extend([
                {
                    "id": "general_1",
                    "suggestion": "Add output format specification",
                    "type": "format",
                    "confidence": 85
                },
                {
                    "id": "general_2",
                    "suggestion": "Include example of desired result",
                    "type": "example",
                    "confidence": 78
                },
                {
                    "id": "general_3",
                    "suggestion": "Provide more context or background",
                    "type": "context",
                    "confidence": 72
                }
            ])
        
        return {"enhancements": enhancements[:3]}  # Return top 3 suggestions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate enhancements: {str(e)}")

@app.get("/analytics/models")
async def get_model_analytics():
    """Get analytics about model performance and usage"""
    try:
        models = get_models()
        
        # Generate mock analytics for now - you can enhance this with real usage data
        analytics = {
            "total_models": len(models),
            "categories": {},
            "top_performers": []
        }
        
        # Count models by category
        for model in models:
            for category in model.get("scores", {}):
                if category not in analytics["categories"]:
                    analytics["categories"][category] = 0
                analytics["categories"][category] += 1
        
        # Get top performers (mock data)
        analytics["top_performers"] = [
            {"name": model["name"], "score": 95.5} for model in models[:5]
        ]
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@app.post("/host-model")
async def host_model(request: HostModelRequest):
    """Host a new Hugging Face model"""
    try:
        model_url = request.model_url.strip()
        custom_name = request.custom_name

        # Use HuggingFaceHostingService to parse, validate, and register the model
        result = hosting_service.register_model(user_id="default_user", model_url=model_url, custom_name=custom_name)
        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error', 'Failed to register model'))
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in host-model endpoint:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 