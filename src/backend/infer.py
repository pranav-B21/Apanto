import os
import requests
from dotenv import load_dotenv
from .multi_provider import multi_provider_llm
import traceback
import asyncio
import sys
from .templates import get_template, get_expected_format

def use_default_loop():
    """Switch to default asyncio event loop policy"""
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    
def use_uvloop():
    """Switch back to uvloop event loop policy if available"""
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass  # uvloop not available, stick with default

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
        # Create messages in the correct format
        messages = [{"role": "user", "content": prompt}]
        
        # Use asyncio.shield to prevent cancellation during testing
        result = await asyncio.shield(
            multi_provider_llm.generate_text(
                model_id=model_id,
                messages=messages,
                parameters={"temperature": 0.2, "max_tokens": 16}
            )
        )
        
        if not result.get("success"):
            print(f"Model {model_id} test failed: {result.get('error', 'Unknown error')}")
            return False
            
        if not result.get("output"):
            print(f"Model {model_id} test failed: No output generated")
            return False
            
        print(f"Model {model_id} test passed")
        return True
        
    except asyncio.TimeoutError:
        print(f"Model {model_id} test failed: Timeout")
        return False
    except Exception as e:
        print(f"Model {model_id} test failed: {str(e)}")
        return False

async def filter_models_by_inference_test(models):
    """Filter models by running inference tests on each one"""
    tasks = [test_model(m) for m in models]
    results = await asyncio.gather(*tasks)
    return [m for m, passed in zip(models, results) if passed]

async def run_inference_tests_async(models):
    """Run inference tests asynchronously"""
    if not models:
        return []
        
    filtered_models = []
    for model in models:
        try:
            result = await test_model(model)
            if result:
                filtered_models.append(model)
        except Exception as e:
            print(f"Failed to test model {model['model_id']}: {str(e)}")
            continue
    return filtered_models
