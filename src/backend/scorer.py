import json
import os
import requests
import random
from dotenv import load_dotenv
from .database import load_models_from_database
from .infer import run_inference_tests_async, test_model
import asyncio
import traceback
# Load environment variables from .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

def get_live_groq_model_ids():
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    try:
        resp = requests.get(GROQ_MODELS_URL, headers=headers)
        model_list = resp.json()["data"]
        return {m["id"] for m in model_list}
    except Exception as e:
        print("Failed to fetch model list from Groq:", e)
        return set()

async def load_models():
    """
    Load models from Supabase database, then filter out any decommissioned models
    by checking against Groq's live model list. Only return models that pass a basic inference test, except local HuggingFace models which are always included.
    """
    try:
        # Load models from database instead of models.json
        all_models = load_models_from_database()
        
        if not all_models:
            print("No models loaded from database!")
            return []

        print(f"\n📋 Loaded {len(all_models)} models from database:")
        for model in all_models:
            print(f"  - {model['name']} ({model['model_id']}) - HF: {model.get('is_huggingface', False)}")

        live_model_ids = get_live_groq_model_ids()
        
        if live_model_ids is None:
            # If Groq API is not available, return all models from database
            print(f"⚠️ Groq API not available, returning all models from database")
            return all_models

        print(f"✅ Groq API available, proceeding with model filtering")

        # Separate local (HuggingFace) and non-local models
        local_models = [m for m in all_models if m.get('is_huggingface')]
        remote_models = [m for m in all_models if not m.get('is_huggingface')]

        print(f"\n🔍 Model Separation:")
        print(f"  📦 Local (HuggingFace) models: {len(local_models)}")
        print(f"  🌐 Remote models: {len(remote_models)}")
        
        if remote_models:
            print(f"  🌐 Remote models to test: {[m['model_id'] for m in remote_models]}")

        # Filter only remote models by inference test
        print(f"\n🧪 Testing {len(remote_models)} remote models...")
        
        # Test models concurrently
        filtered_remote_models = []
        for model in remote_models:
            try:
                result = await test_model(model)
                if result:
                    filtered_remote_models.append(model)
            except Exception as e:
                print(f"Failed to test model {model['model_id']}: {str(e)}")
                continue

        # Always include local models
        filtered_models = filtered_remote_models + local_models
        
        # Calculate working vs non-working models
        working_remote = len(filtered_remote_models)
        failed_remote = len(remote_models) - working_remote
        working_local = len(local_models)
        
        print(f"\nModel Testing Results:")
        print(f"Working remote models: {working_remote}")
        print(f"Failed remote models: {failed_remote}")
        print(f"Local models (always included): {working_local}")
        print(f"Total available models: {len(filtered_models)}")
        
        return filtered_models
        
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        return []

def load_models_sync():
    """Synchronous wrapper for load_models"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in a running loop, create a task
            return asyncio.create_task(load_models())
        else:
            # If loop exists but not running, use it directly
            return loop.run_until_complete(load_models())
    except RuntimeError:
        # No event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(load_models())
        finally:
            loop.close()

def score_model(model: dict, category: str, priority: str) -> float:
    """
    Compute a combined score for `model` given a prompt `category` (e.g. "qa", "code", "creative")
    and a user `priority` ("accuracy", "speed", or "cost").

    - eval_score comes from model["scores"][category][priority]
    - latency_norm = 1 / latency   (higher is better)
    - cost_norm = 1 / cost         (higher is better)

    We weight these three components by (w_eval, w_latency, w_cost) based on `priority`.
    """
    # 1) Category-specific "eval" score: fallback 0.0 if missing
    eval_score = model.get("scores", {}).get(category, {}).get(priority, 0.0)

    # 2) Latency and Cost: fallback to some default if missing
    latency = model.get("latency", 1.0)    # if not provided, assume 1s
    cost    = model.get("cost",    0.01)   # if not provided, assume $0.01/token

    # 3) Choose weights based on user priority
    weights = {
        "accuracy": (0.5, 0.25, 0.25),
        "speed":    (0.2, 0.6,  0.2),
        "cost":     (0.2, 0.2,  0.6)
    }.get(priority, (0.5, 0.25, 0.25))

    w_eval, w_latency, w_cost = weights

    # 4) Normalize each component
    eval_norm    = eval_score / 1.0                    # eval_score is already 0–1 or 0–100?
    latency_norm = (1.0 / latency)   if latency else 0  # smaller latency → bigger norm
    cost_norm    = (1.0 / cost)      if cost else 0     # smaller cost → bigger norm

    combined = (
        w_eval    * eval_norm +
        w_latency * latency_norm +
        w_cost    * cost_norm
    )
    return combined

def select_best_model(models: list[dict], prompt_type: str, priority: str, return_debug: bool=False):
    """
    1. Score every model in `models` dynamically via score_model().
    2. Sort by the combined score, descending.
    3. Extract top-3 names for logging or 'top 3' leaderboard.
    4. With probability ε, pick a random model among the top 3 (exploration);
       otherwise pick the single best (exploitation).
    5. Return (best_model_dict, [top3_names]) or (best_model_dict, [top3_names], debug_scores).

    This ε-greedy step ensures we occasionally "explore" other high-scoring LLMs,
    so that if your metadata is slightly off, you'll still gather signals.
    """
    ε = 0.10  # 10% exploration rate
    scored_list = []

    for m in models:
        try:
            s = score_model(m, prompt_type, priority)
            scored_list.append((m, s))
        except Exception as e:
            print(f"⚠️ Skipping model {m.get('name')} due to scoring error: {e}")

    if not scored_list:
        raise ValueError(f"No models support prompt type '{prompt_type}' with priority '{priority}'")

    # Sort descending by score
    scored_list.sort(key=lambda pair: pair[1], reverse=True)

    # Top 3 candidate names for debug / "leaderboard"
    top_three = [model_dict["name"] for model_dict, _ in scored_list[:3]]

    #  ε-greedy: sometimes choose a random model among top 3
    if len(scored_list) > 1 and random.random() < ε:
        # pick any of the top 3 (if available), else fallback to best
        top_k = scored_list[: min(3, len(scored_list))]
        chosen = random.choice(top_k)[0]
    else:
        chosen = scored_list[0][0]

    if return_debug:
        debug_scores = {m["name"]: s for m, s in scored_list}
        return chosen, top_three, debug_scores
    else:
        return chosen, top_three
