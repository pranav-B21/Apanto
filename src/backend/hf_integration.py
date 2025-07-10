"""
Hugging Face Model Hosting Service
Handles registration, validation, and management of user-submitted HF models
"""

import os
import re
import json
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import psutil
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from fastapi import HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
import torch
from templates import get_template, get_expected_format
import asyncio

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Optional, for private models
HF_INFERENCE_API_BASE = "https://api-inference.huggingface.co/models/"

# Global variable to store the progress callback
progress_callback = None

def set_progress_callback(callback):
    """Set the callback function for progress updates"""
    global progress_callback
    progress_callback = callback

async def send_progress_update(progress_data: Dict[str, Any]):
    """Send progress update if callback is set"""
    global progress_callback
    if progress_callback:
        try:
            await progress_callback(progress_data)
        except Exception as e:
            print(f"Error sending progress update: {e}")

class LocalModelLoader:
    """Handles loading and managing local Hugging Face models"""
    
    def __init__(self):
        self.loaded_models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a model locally and cache it"""
        print(f"📦 Starting to load model: {model_id}")
        print(f"🔍 Checking if model is already loaded...")
        
        if model_id in self.loaded_models:
            print(f"✅ Model {model_id} already loaded, returning cached version")
            return self.loaded_models[model_id]
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"💾 Initial memory usage: {initial_memory:.1f} MB")
        
        # Send initial progress update
        asyncio.create_task(send_progress_update({
            "stage": "starting",
            "message": f"Starting to load model: {model_id}",
            "progress": 0,
            "memory_usage": initial_memory
        }))
        
        print(f"🔄 Model {model_id} not in cache, downloading from Hugging Face...")
        print(f"💻 Using device: {self.device}")
        print(f"🕐 Download started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Load tokenizer with progress tracking
            print(f"🔤 Downloading tokenizer for {model_id}...")
            print(f"📁 Tokenizer files being downloaded...")
            
            # Send tokenizer progress update
            asyncio.create_task(send_progress_update({
                "stage": "tokenizer",
                "message": f"Downloading tokenizer for {model_id}...",
                "progress": 10,
                "memory_usage": initial_memory
            }))
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            print(f"✅ Tokenizer downloaded successfully")
            print(f"📊 Tokenizer vocabulary size: {tokenizer.vocab_size}")
            
            # Check memory after tokenizer
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"💾 Memory after tokenizer: {current_memory:.1f} MB (+{current_memory - initial_memory:.1f} MB)")
            
            # Send tokenizer completion update
            asyncio.create_task(send_progress_update({
                "stage": "tokenizer_complete",
                "message": f"Tokenizer downloaded successfully (vocab size: {tokenizer.vocab_size})",
                "progress": 20,
                "memory_usage": current_memory
            }))
            
            # Try to determine model type from config
            print(f"⚙️ Loading model configuration...")
            print(f"📋 Fetching model config from Hugging Face...")
            
            # Send config progress update
            asyncio.create_task(send_progress_update({
                "stage": "config",
                "message": "Loading model configuration...",
                "progress": 25,
                "memory_usage": current_memory
            }))
            
            try:
                config = AutoConfig.from_pretrained(model_id)
                model_type = config.model_type if hasattr(config, 'model_type') else None
                print(f"📋 Model type detected: {model_type}")
                print(f"📊 Model config parameters: {config}")
                
                # Estimate model size before downloading
                param_count = None
                if hasattr(config, 'num_parameters'):
                    param_count = config.num_parameters
                    print(f"📊 Estimated parameters: {param_count:,}")
                    if param_count > 1_000_000_000:
                        print(f"⚠️ Large model detected ({param_count:,} parameters) - this may take a while...")
                        print(f"💾 Estimated memory usage: ~{param_count * 4 / 1_000_000_000:.1f} GB (FP32) or ~{param_count * 2 / 1_000_000_000:.1f} GB (FP16)")
                
                # Send config completion update
                param_display = f"{param_count:,}" if param_count else "Unknown"
                asyncio.create_task(send_progress_update({
                    "stage": "config_complete",
                    "message": f"Model configuration loaded (type: {model_type}, params: {param_display})",
                    "progress": 30,
                    "memory_usage": current_memory,
                    "model_info": {
                        "type": model_type,
                        "parameters": param_count
                    }
                }))
                
                # Load appropriate model based on type with progress tracking
                print(f"🤖 Downloading model weights...")
                print(f"📥 This may take several minutes for large models...")
                print(f"💾 Downloading to cache directory...")
                
                # Check memory before model download
                pre_model_memory = process.memory_info().rss / 1024 / 1024  # MB
                print(f"💾 Memory before model download: {pre_model_memory:.1f} MB")
                
                # Send model download start update
                asyncio.create_task(send_progress_update({
                    "stage": "model_download",
                    "message": "Downloading model weights (this may take several minutes)...",
                    "progress": 35,
                    "memory_usage": pre_model_memory
                }))
                
                if model_type in ['bert', 'roberta', 'distilbert']:
                    print(f"📥 Loading BERT-style model: {model_id}")
                    model = AutoModel.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    )
                else:
                    # Default to causal LM for other types
                    print(f"📥 Loading causal language model: {model_id}")
                    print(f"🔄 Downloading model files (this may take a while)...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    )
                    
            except ValueError as e:
                # Handle models without proper config
                if "Unrecognized model" in str(e) or "model_type" in str(e):
                    print(f"❌ Model {model_id} is not supported. This appears to be a PEFT adapter model or has an unsupported configuration.")
                    print(f"💡 Please use a standard Hugging Face model (not an adapter/fine-tuned model)")
                    
                    # Send error update
                    asyncio.create_task(send_progress_update({
                        "stage": "error",
                        "message": f"Model {model_id} is not supported. Please use a standard Hugging Face model.",
                        "progress": 0,
                        "error": "PEFT adapter models are not supported. Please use a standard model."
                    }))
                    
                    raise Exception(f"Model {model_id} is not supported. This appears to be a PEFT adapter model. Please use a standard Hugging Face model.")
                else:
                    raise e
            
            # Check memory after model download
            post_model_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"✅ Model weights downloaded and loaded successfully")
            print(f"📍 Model device: {model.device}")
            print(f"💾 Memory after model download: {post_model_memory:.1f} MB (+{post_model_memory - pre_model_memory:.1f} MB)")
            print(f"📊 Model loaded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Send model download completion update
            asyncio.create_task(send_progress_update({
                "stage": "model_complete",
                "message": "Model weights downloaded and loaded successfully",
                "progress": 90,
                "memory_usage": post_model_memory
            }))
            
            # Print model information
            if hasattr(model, 'config'):
                print(f"📋 Model config loaded successfully")
                if hasattr(model.config, 'num_parameters'):
                    print(f"📊 Total parameters: {model.config.num_parameters:,}")
            
            # Cache the loaded model
            self.loaded_models[model_id] = {
                'model': model,
                'tokenizer': tokenizer,
                'device': self.device,
                'model_type': model_type
            }
            
            # Final memory check
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"💾 Final memory usage: {final_memory:.1f} MB (total increase: {final_memory - initial_memory:.1f} MB)")
            print(f"💾 Model {model_id} cached for future use")
            print(f"🎉 Model loading completed successfully!")
            
            # Send final completion update
            asyncio.create_task(send_progress_update({
                "stage": "complete",
                "message": f"Model {model_id} loaded successfully!",
                "progress": 100,
                "memory_usage": final_memory,
                "total_time": (datetime.now() - datetime.now()).total_seconds()  # Placeholder
            }))
            
            return self.loaded_models[model_id]
            
        except Exception as e:
            print(f"❌ Failed to load model {model_id}: {str(e)}")
            print(f"🔍 Error details: {type(e).__name__}")
            import traceback
            print(f"📄 Full traceback:")
            traceback.print_exc()
            
            # Send error update
            asyncio.create_task(send_progress_update({
                "stage": "error",
                "message": f"Failed to load model: {str(e)}",
                "progress": 0,
                "error": str(e)
            }))
            
            raise Exception(f"Failed to load model {model_id}: {str(e)}")
    
    def generate_text(self, model_id: str, prompt_with_context: str, task_type: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate text or embeddings using a locally loaded model"""
        print(f"🎯 Starting text generation for model: {model_id}")
        print(f"📝 Task type: {task_type}")
        print(f"📄 Input prompt length: {len(prompt_with_context)} characters")
        
        if model_id not in self.loaded_models:
            print(f"⚠️ Model {model_id} not loaded, loading now...")
            self.load_model(model_id)
        
        model_data = self.loaded_models[model_id]
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        model_type = model_data.get('model_type')

        # Get the appropriate template for the task type
        template = get_template(task_type)
        expected_format = get_expected_format(task_type)
        
        # Format the prompt with the context
        prompt = template.format(prompt_with_context=prompt_with_context)
        print(f'📋 Task type: {task_type}')
        print(f'📝 Formatted prompt length: {len(prompt)} characters')
        
        try:
            # Tokenize input
            print(f"🔤 Tokenizing input...")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            print(f"✅ Input tokenized, tensor shape: {inputs['input_ids'].shape}")
            
            # Handle different model types
            if model_type in ['bert', 'roberta', 'distilbert']:
                # For BERT-like models, return embeddings
                print(f"🧠 Generating embeddings with {model_type} model...")
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                    print(f"✅ Embeddings generated, shape: {embeddings.shape}")
                    return {
                        'success': True,
                        'embeddings': embeddings.cpu().numpy().tolist(),
                        'latency': None
                    }
            else:
                # For causal models, generate text
                print(f"🤖 Generating text with causal model...")
                default_params = {
                    "max_new_tokens": 1000,  # Increased for more detailed responses
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True,
                    "repetition_penalty": 1.2  # Added to reduce repetition
                }
                
                if parameters:
                    default_params.update(parameters)
                
                print(f"⚙️ Generation parameters: {default_params}")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=default_params["max_new_tokens"],
                        temperature=default_params["temperature"],
                        top_p=default_params["top_p"],
                        do_sample=default_params["do_sample"],
                        repetition_penalty=default_params["repetition_penalty"]
                    )
                
                print(f"✅ Text generation completed, output shape: {outputs.shape}")
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"📄 Generated text length: {len(generated_text)} characters")
                
                # Remove the prompt from the generated text
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                    print(f"✂️ Removed prompt from output, final length: {len(generated_text)} characters")
                
                return {
                    'success': True,
                    'output': generated_text,
                    'format': expected_format,
                    'latency': None
                }
            
        except Exception as e:
            print(f"❌ Generation error: {str(e)}")
            return {
                'success': False,
                'error': f'Generation error: {str(e)}'
            }

class HuggingFaceHostingService:
    """Manages Hugging Face model hosting and registration"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.hf_token = HF_API_TOKEN
        self.local_loader = LocalModelLoader()
        
    def parse_hf_url(self, url: str) -> Optional[Dict[str, str]]:
        """
        Parse Hugging Face model URL to extract org/user and model name
        Examples:
        - https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
        - huggingface.co/google/flan-t5-base
        - meta-llama/Llama-2-7b-chat-hf
        """
        # Remove protocol if present
        url = url.strip()
        url = re.sub(r'^https?://', '', url)
        
        # Remove huggingface.co if present
        url = re.sub(r'^(www\.)?huggingface\.co/', '', url)
        
        # Extract org/model pattern
        pattern = r'^([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)$'
        match = re.match(pattern, url)
        
        if match:
            return {
                'organization': match.group(1),
                'model_name': match.group(2),
                'model_id': f"{match.group(1)}/{match.group(2)}"
            }
        return None
    
    def validate_model_exists(self, model_id: str) -> Dict[str, Any]:
        """
        Validate that a model exists on Hugging Face and get its metadata
        """
        print(f"🔍 Validating model: {model_id}")
        
        headers = {}
        if self.hf_token:
            headers['Authorization'] = f'Bearer {self.hf_token}'
            print(f"🔑 Using HF token for authentication")
        else:
            print(f"⚠️ No HF token provided, using public access")
        
        # Check model info endpoint
        info_url = f"https://huggingface.co/api/models/{model_id}"
        print(f"🌐 Checking URL: {info_url}")
        
        try:
            print(f"📡 Making HTTP request to Hugging Face API...")
            response = requests.get(info_url, headers=headers, timeout=10)
            print(f"📊 Response status code: {response.status_code}")
            print(f"📄 Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                model_info = response.json()
                print(f"✅ Model found! Model info: {model_info}")
                
                # Extract relevant metadata
                return {
                    'exists': True,
                    'model_id': model_id,
                    'pipeline_tag': model_info.get('pipeline_tag', 'text-generation'),
                    'tags': model_info.get('tags', []),
                    'library_name': model_info.get('library_name', 'transformers'),
                    'language': model_info.get('language', ['en']),
                    'license': model_info.get('license', 'unknown'),
                    'downloads': model_info.get('downloads', 0),
                    'likes': model_info.get('likes', 0),
                    'created_at': model_info.get('created_at'),
                    'model_size': self._estimate_model_size(model_info)
                }
            elif response.status_code == 404:
                print(f"❌ Model not found (404): {model_id}")
                print(f"📄 Response content: {response.text}")
                return {'exists': False, 'error': 'Model not found'}
            else:
                print(f"❌ Unexpected status code: {response.status_code}")
                print(f"📄 Response content: {response.text}")
                return {'exists': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            print(f"💥 Exception during validation: {str(e)}")
            return {'exists': False, 'error': str(e)}
    
    def _estimate_model_size(self, model_info: Dict) -> str:
        """Estimate model size category based on metadata"""
        # This is a simplified estimation - you might want to enhance this
        safetensors = model_info.get('safetensors', {})
        if safetensors:
            total_params = safetensors.get('total', 0)
            if total_params > 100_000_000_000:  # 100B+
                return 'XXL'
            elif total_params > 10_000_000_000:  # 10B+
                return 'XL'
            elif total_params > 1_000_000_000:   # 1B+
                return 'L'
            elif total_params > 100_000_000:     # 100M+
                return 'M'
            else:
                return 'S'
        return 'Unknown'
    
    def test_model_inference(self, model_id: str, test_prompt: str = "Hello") -> Dict[str, Any]:
        """
        Test if the model can be loaded locally
        """
        print(f"🧪 Starting inference test for model: {model_id}")
        print(f"📝 Test prompt: '{test_prompt}'")
        
        try:
            # Try loading the model locally
            print(f"📦 Loading model for inference test...")
            self.local_loader.load_model(model_id)
            print(f"✅ Model loaded successfully for testing")
            
            # Test generation
            print(f"🎯 Running inference test...")
            result = self.local_loader.generate_text(model_id, test_prompt, None)
            print(f"📊 Inference test result: {result}")
            
            if result['success']:
                print(f"✅ Inference test passed! Response: {result.get('output', 'N/A')}")
                return {
                    'success': True,
                    'response': result['text'],
                    'latency': None
                }
            else:
                print(f"❌ Inference test failed: {result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'error': result['error']
                }
                
        except Exception as e:
            print(f'💥 Exception in test_model_inference: {e}')
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Model loading error: {str(e)}'
            }
    
    def register_model(self, user_id: str, model_url: str, custom_name: str = None) -> Dict[str, Any]:
        """
        Register a new Hugging Face model for a user
        """
        print(f"🔧 Starting model registration for user: {user_id}")
        print(f"🔗 Model URL: {model_url}")
        print(f"🏷️ Custom name: {custom_name}")
        
        # Parse the URL
        print("🔍 Parsing Hugging Face URL...")
        parsed = self.parse_hf_url(model_url)
        print('✅ Parsed model URL:', parsed)
        if not parsed:
            print("❌ Failed to parse model URL")
            return {
                'success': False,
                'error': 'Invalid Hugging Face model URL format'
            }
        
        model_id = parsed['model_id']
        print(f"🆔 Extracted model ID: {model_id}")
        
        # Validate model exists
        print("🔍 Validating model exists on Hugging Face...")
        validation = self.validate_model_exists(model_id)
        print('✅ Model validation:', validation)
        if not validation.get('exists'):
            print(f"❌ Model validation failed: {validation.get('error', 'Unknown error')}")
            return {
                'success': False,
                'error': f"Model validation failed: {validation.get('error', 'Unknown error')}"
            }
        
        print(f"✅ Model exists! Size: {validation.get('model_size', 'Unknown')}, Pipeline: {validation.get('pipeline_tag', 'Unknown')}")
        
        # Test inference
        print("🧪 Testing model inference (this will download the model)...")
        inference_test = self.test_model_inference(model_id)
        print('✅ Inference test result:', inference_test)
        if not inference_test.get('success'):
            if inference_test.get('error', '').startswith('Inference failed: HTTP 404'):
                print("❌ Model not available for hosted inference")
                return {
                    'success': False,
                    'error': "This model is not available for hosted inference via the Hugging Face API. Please choose a different model."
                }
            print(f"❌ Model inference test failed: {inference_test.get('error', 'Unknown error')}")
            return {
                'success': False,
                'error': f"Model inference test failed: {inference_test.get('error', 'Unknown error')}",
                'details': inference_test.get('details', None)
            }
        
        print("✅ Model inference test passed!")
        
        # Generate unique ID for this hosted model instance
        hosted_model_id = self._generate_hosted_model_id(user_id, model_id)
        print(f"🆔 Generated hosted model ID: {hosted_model_id}")
        
        # Prepare model metadata
        model_data = {
            'hosted_model_id': hosted_model_id,
            'user_id': user_id,
            'hf_model_id': model_id,
            'custom_name': custom_name or f"Custom {parsed['model_name']}",
            'organization': parsed['organization'],
            'model_name': parsed['model_name'],
            'pipeline_tag': validation.get('pipeline_tag', 'text-generation'),
            'model_size': validation.get('model_size', 'Unknown'),
            'license': validation.get('license', 'unknown'),
            'status': 'loading' if inference_test.get('status') == 'loading' else 'active',
            'registered_at': datetime.utcnow().isoformat(),
            'metadata': {
                'tags': validation.get('tags', []),
                'language': validation.get('language', ['en']),
                'downloads': validation.get('downloads', 0),
                'likes': validation.get('likes', 0),
                'test_latency': inference_test.get('latency', None)
            }
        }
        print('📊 Prepared model_data for DB:', model_data)
        
        # Store in database
        try:
            print("💾 Storing model in database...")
            self._store_hosted_model(model_data)
            print("✅ Model stored in hosted_models table")
            
            # Also add to the models table with estimated scores
            print("📈 Adding model to main models table with scores...")
            self._add_to_models_table(model_data)
            print("✅ Model added to main models table")
            
            print("🎉 Model registration completed successfully!")
            return {
                'success': True,
                'hosted_model_id': hosted_model_id,
                'model_id': model_id,
                'custom_name': model_data['custom_name'],
                'status': model_data['status'],
                'message': 'Model successfully registered' if model_data['status'] == 'active' else 'Model registered and is loading'
            }
            
        except Exception as e:
            print('💥 Exception in /api/host-model:', e)
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    def _generate_hosted_model_id(self, user_id: str, model_id: str) -> str:
        """Generate a unique ID for a hosted model instance"""
        timestamp = datetime.utcnow().isoformat()
        hash_input = f"{user_id}:{model_id}:{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _store_hosted_model(self, model_data: Dict[str, Any]):
        """Store hosted model in database"""
        print(f"💾 Connecting to database to store hosted model...")
        conn = self.db_manager.get_connection()
        if not conn:
            print('❌ DB connection failed in _store_hosted_model')
            raise Exception("Database connection failed")
        
        try:
            cursor = conn.cursor()
            
            print('📊 Inserting into hosted_models table...')
            print(f'📋 Model data: {model_data}')
            
            # Create hosted_models table if it doesn't exist
            print("🔨 Creating hosted_models table if it doesn't exist...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hosted_models (
                    hosted_model_id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    hf_model_id VARCHAR(255) NOT NULL,
                    custom_name VARCHAR(255),
                    organization VARCHAR(255),
                    model_name VARCHAR(255),
                    pipeline_tag VARCHAR(100),
                    model_size VARCHAR(10),
                    license VARCHAR(100),
                    status VARCHAR(50),
                    registered_at TIMESTAMP,
                    metadata JSONB,
                    UNIQUE(user_id, hf_model_id)
                );
            """)
            print("✅ Table structure verified")
            
            # Insert the model
            print("📝 Executing INSERT statement...")
            cursor.execute("""
                INSERT INTO hosted_models (
                    hosted_model_id, user_id, hf_model_id, custom_name,
                    organization, model_name, pipeline_tag, model_size,
                    license, status, registered_at, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, hf_model_id) 
                DO UPDATE SET 
                    custom_name = EXCLUDED.custom_name,
                    status = EXCLUDED.status,
                    metadata = EXCLUDED.metadata;
            """, (
                model_data['hosted_model_id'],
                model_data['user_id'],
                model_data['hf_model_id'],
                model_data['custom_name'],
                model_data['organization'],
                model_data['model_name'],
                model_data['pipeline_tag'],
                model_data['model_size'],
                model_data['license'],
                model_data['status'],
                model_data['registered_at'],
                json.dumps(model_data['metadata'])
            ))
            
            print("💾 Committing transaction...")
            conn.commit()
            print("✅ Model successfully stored in hosted_models table")
            
        except Exception as e:
            print('💥 Exception in _store_hosted_model:', e)
            import traceback
            traceback.print_exc()
            raise
        finally:
            cursor.close()
            print("🔒 Database cursor closed")
    
    def _add_to_models_table(self, model_data: Dict[str, Any]):
        """Add the hosted model to the main models table with estimated scores"""
        print(f"💾 Connecting to database to add model to main models table...")
        conn = self.db_manager.get_connection()
        if not conn:
            print('❌ DB connection failed in _add_to_models_table')
            raise Exception("Database connection failed")
        
        try:
            cursor = conn.cursor()
            
            print('📊 Adding to main models table...')
            print(f'📋 Model data: {model_data}')
            
            # First, ensure the required columns exist
            print("🔨 Ensuring required columns exist in models table...")
            cursor.execute("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'models' AND column_name = 'is_huggingface') THEN
                        ALTER TABLE models ADD COLUMN is_huggingface BOOLEAN DEFAULT FALSE;
                    END IF;
                    
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'models' AND column_name = 'huggingface_url') THEN
                        ALTER TABLE models ADD COLUMN huggingface_url TEXT;
                    END IF;
                END $$;
            """)
            print("✅ Table structure verified")
            
            # Insert into models table
            model_id = f"hf-{model_data['hosted_model_id']}"
            huggingface_url = f"https://huggingface.co/{model_data['hf_model_id']}"
            
            print(f"📝 Inserting into models table with ID: {model_id}")
            cursor.execute("""
                INSERT INTO models (name, model_id, is_huggingface, huggingface_url)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (model_id) DO NOTHING;
            """, (
                model_data['custom_name'],
                model_id,  # Prefix to distinguish hosted models
                True,
                huggingface_url
            ))
            print("✅ Model inserted into main models table")
            
            # Add estimated scores based on model size and metadata
            # This is a simplified scoring - you might want to enhance this
            size_scores = {
                'XXL': {'accuracy': 0.95, 'speed': 0.3, 'cost': 0.9},
                'XL': {'accuracy': 0.9, 'speed': 0.4, 'cost': 0.8},
                'L': {'accuracy': 0.85, 'speed': 0.5, 'cost': 0.7},
                'M': {'accuracy': 0.75, 'speed': 0.7, 'cost': 0.5},
                'S': {'accuracy': 0.65, 'speed': 0.9, 'cost': 0.2},
                'Unknown': {'accuracy': 0.7, 'speed': 0.6, 'cost': 0.5}
            }
            
            base_scores = size_scores.get(model_data['model_size'], size_scores['Unknown'])
            print(f"📊 Estimated scores for {model_data['model_size']} model: {base_scores}")
            
            # Determine categories based on pipeline_tag
            categories = ['qa', 'reasoning']
            if model_data['pipeline_tag'] == 'text-generation':
                categories.extend(['coding', 'creative'])
            elif model_data['pipeline_tag'] == 'text2text-generation':
                categories.extend(['summarization', 'translation'])
            
            print(f"📋 Adding scores for categories: {categories}")
            
            # Insert scores for each category
            for category in categories:
                print(f"📈 Adding scores for category: {category}")
                cursor.execute("""
                    INSERT INTO model_scores (model_id, category, accuracy, speed, cost)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (model_id, category) DO UPDATE SET
                        accuracy = EXCLUDED.accuracy,
                        speed = EXCLUDED.speed,
                        cost = EXCLUDED.cost;
                """, (
                    model_id,
                    category,
                    base_scores['accuracy'],
                    base_scores['speed'],
                    base_scores['cost']
                ))
            
            print("💾 Committing transaction...")
            conn.commit()
            print("✅ Model scores successfully added to model_scores table")
            
        except Exception as e:
            print('💥 Exception in _add_to_models_table:', e)
            import traceback
            traceback.print_exc()
            raise
        finally:
            cursor.close()
            print("🔒 Database cursor closed")
    
    def get_user_hosted_models(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all models hosted by a specific user"""
        conn = self.db_manager.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM hosted_models
                WHERE user_id = %s
                ORDER BY registered_at DESC;
            """, (user_id,))
            
            return cursor.fetchall()
            
        finally:
            cursor.close()
    
    def update_model_status(self, hosted_model_id: str, status: str):
        """Update the status of a hosted model"""
        conn = self.db_manager.get_connection()
        if not conn:
            raise Exception("Database connection failed")
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE hosted_models
                SET status = %s
                WHERE hosted_model_id = %s;
            """, (status, hosted_model_id))
            
            conn.commit()
            
        finally:
            cursor.close()

