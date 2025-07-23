import os
import requests
from dotenv import load_dotenv
from .multi_provider import multi_provider_llm
import traceback
import asyncio
from .templates import get_template, get_expected_format

# Load environment variables from .env
load_dotenv()

# Define rich formatting system message
RICH_FORMAT_SYSTEM_MESSAGE = """You are a helpful AI assistant that provides detailed, well-formatted responses.

ALWAYS format your responses using markdown for better readability. Use these markdown features when appropriate:
- Headings (# ## ###)
- Bold (**text**)
- Italic (*text*)
- Lists (ordered and unordered)
- Code blocks (``` ```)
- Tables
- Blockquotes (>)
- Links

For any content that could be visualized as a diagram, use Mermaid syntax. Common diagram types to use:
- Flowcharts (graph TD)
- Sequence diagrams (sequenceDiagram)
- Class diagrams (classDiagram)
- State diagrams (stateDiagram)
- Entity Relationship diagrams (erDiagram)
- User Journey diagrams (journey)

Example Mermaid diagram:
```mermaid
graph TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
```

DO NOT include images or external content. Focus on text-based formatting and diagrams only."""

async def run_prompt_on_llm(model_id: str, prompt: str) -> str:
    """
    Run a prompt on the specified LLM using the multi-provider system.
    Supports OpenAI, DeepSeek, Gemini, Anthropic, and Groq models.
    """
    try:
        # Add system message for rich formatting
        messages = [
            {"role": "system", "content": RICH_FORMAT_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}
        ]
        
        # Use the multi-provider system
        result = await multi_provider_llm.generate_text(model_id, messages)
        
        if result["success"]:
            return result["output"]
        else:
            raise Exception(f"LLM generation failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        raise Exception(f"Error running prompt on LLM: {str(e)}")

async def run_prompt_with_context(model_id: str, messages: list) -> str:
    """
    Run a prompt with conversation context using the multi-provider system.
    """
    try:
        # Insert system message at the beginning for rich formatting
        messages_with_system = [{"role": "system", "content": RICH_FORMAT_SYSTEM_MESSAGE}] + messages
        
        # Use the multi-provider system with message history
        result = await multi_provider_llm.generate_text(model_id, messages_with_system)
        
        if result["success"]:
            return result["output"]
        else:
            raise Exception(f"LLM generation failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print("\n💥 LLM inference error (run_prompt_with_context):")
        print(str(e))
        traceback.print_exc()
        raise Exception(f"Error running prompt with context on LLM: {str(e)}")

def get_available_providers():
    """Get list of available AI providers"""
    return list(multi_provider_llm.get_available_models().keys())

def get_models_by_provider(provider: str):
    """Get available models for a specific provider"""
    all_models = multi_provider_llm.get_available_models()
    return all_models.get(provider, [])

def estimate_cost(model_id: str, tokens_used: int) -> float:
    """Estimate cost for a specific model and token usage"""
    provider = multi_provider_llm.get_provider_from_model_id(model_id)
    return multi_provider_llm.estimate_cost(provider, model_id, tokens_used)

async def test_model(model):
    """Test if a model can generate a response"""
    model_id = model["model_id"]
    prompt = f"Say hello from {model_id}!"
    try:
        result = await multi_provider_llm.generate_text(
            model_id=model_id,
            messages=prompt,
            parameters={"temperature": 0.2, "max_tokens": 16}
        )
        if not result.get("success"):
            return False
        if not result.get("output"):
            return False
        return True
    except Exception as e:
        return False

async def filter_models_by_inference_test(models):
    """Filter models by running inference tests on each one"""
    tasks = [test_model(m) for m in models]
    results = await asyncio.gather(*tasks)
    return [m for m, passed in zip(models, results) if passed]

def run_inference_tests_sync(models):
    """Run inference tests synchronously (for use in non-async contexts)"""
    if not models:
        return []
    
    try:
        # Try to run in a new event loop
        filtered_models = asyncio.run(filter_models_by_inference_test(models))
        return filtered_models
    except RuntimeError:
        # If already in an event loop (e.g. in FastAPI), try a different approach
        try:
            import nest_asyncio
            nest_asyncio.apply()
            
            # Get the current event loop or create a new one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function
            filtered_models = loop.run_until_complete(filter_models_by_inference_test(models))
            return filtered_models
            
        except Exception as e:
            print(f"❌ Error running inference tests: {e}")
            return models

async def run_inference_tests_async(models):
    """Run inference tests asynchronously (for use in async contexts like FastAPI endpoints)"""
    return await filter_models_by_inference_test(models)
