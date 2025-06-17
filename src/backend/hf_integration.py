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
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from fastapi import HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
import torch
from templates import get_template, get_expected_format

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Optional, for private models
HF_INFERENCE_API_BASE = "https://api-inference.huggingface.co/models/"

class LocalModelLoader:
    """Handles loading and managing local Hugging Face models"""
    
    def __init__(self):
        self.loaded_models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a model locally and cache it"""
        print("first model id" + model_id)
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Try to determine model type from config
            config = AutoConfig.from_pretrained(model_id)
            model_type = config.model_type if hasattr(config, 'model_type') else None
            
            # Load appropriate model based on type
            if model_type in ['bert', 'roberta', 'distilbert']:
                model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            else:
                # Default to causal LM for other types
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            
            # Cache the loaded model
            self.loaded_models[model_id] = {
                'model': model,
                'tokenizer': tokenizer,
                'device': self.device,
                'model_type': model_type
            }
            
            return self.loaded_models[model_id]
            
        except Exception as e:
            raise Exception(f"Failed to load model {model_id}: {str(e)}")
    
    def generate_text(self, model_id: str, prompt_with_context: str, task_type: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate text or embeddings using a locally loaded model"""
        if model_id not in self.loaded_models:
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
        print('task type:' + task_type)
        print('prompt going into model: '+ prompt)
        
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Handle different model types
            if model_type in ['bert', 'roberta', 'distilbert']:
                # For BERT-like models, return embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                    return {
                        'success': True,
                        'embeddings': embeddings.cpu().numpy().tolist(),
                        'latency': None
                    }
            else:
                # For causal models, generate text
                default_params = {
                    "max_new_tokens": 1000,  # Increased for more detailed responses
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True,
                    "repetition_penalty": 1.2  # Added to reduce repetition
                }
                
                if parameters:
                    default_params.update(parameters)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=default_params["max_new_tokens"],
                        temperature=default_params["temperature"],
                        top_p=default_params["top_p"],
                        do_sample=default_params["do_sample"],
                        repetition_penalty=default_params["repetition_penalty"]
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the prompt from the generated text
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                return {
                    'success': True,
                    'output': generated_text,
                    'format': expected_format,
                    'latency': None
                }
            
        except Exception as e:
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
        headers = {}
        if self.hf_token:
            headers['Authorization'] = f'Bearer {self.hf_token}'
        
        # Check model info endpoint
        info_url = f"https://huggingface.co/api/models/{model_id}"
        
        try:
            response = requests.get(info_url, headers=headers, timeout=10)
            if response.status_code == 200:
                model_info = response.json()
                
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
                return {'exists': False, 'error': 'Model not found'}
            else:
                return {'exists': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
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
        try:
            # Try loading the model locally
            self.local_loader.load_model(model_id)
            
            # Test generation
            result = self.local_loader.generate_text(model_id, test_prompt, None)
            
            if result['success']:
                return {
                    'success': True,
                    'response': result['text'],
                    'latency': None  # TODO: Add timing
                }
            else:
                return {
                    'success': False,
                    'error': result['error']
                }
                
        except Exception as e:
            print('Exception in test_model_inference:', e)
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
        # Parse the URL
        parsed = self.parse_hf_url(model_url)
        print('Parsed model URL:', parsed)
        if not parsed:
            return {
                'success': False,
                'error': 'Invalid Hugging Face model URL format'
            }
        
        model_id = parsed['model_id']
        
        # Validate model exists
        validation = self.validate_model_exists(model_id)
        print('Model validation:', validation)
        if not validation.get('exists'):
            return {
                'success': False,
                'error': f"Model validation failed: {validation.get('error', 'Unknown error')}"
            }
        
        
        # Test inference
        inference_test = self.test_model_inference(model_id)
        print('Inference test result:', inference_test)
        if not inference_test.get('success'):
            if inference_test.get('error', '').startswith('Inference failed: HTTP 404'):
                return {
                    'success': False,
                    'error': "This model is not available for hosted inference via the Hugging Face API. Please choose a different model."
                }
            return {
                'success': False,
                'error': f"Model inference test failed: {inference_test.get('error', 'Unknown error')}",
                'details': inference_test.get('details', None)
            }
        
        
        # Generate unique ID for this hosted model instance
        hosted_model_id = self._generate_hosted_model_id(user_id, model_id)
        
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
        print('Prepared model_data for DB:', model_data)
        
        # Store in database
        try:
            self._store_hosted_model(model_data)
            
            # Also add to the models table with estimated scores
            self._add_to_models_table(model_data)
            
            return {
                'success': True,
                'hosted_model_id': hosted_model_id,
                'model_id': model_id,
                'custom_name': model_data['custom_name'],
                'status': model_data['status'],
                'message': 'Model successfully registered' if model_data['status'] == 'active' else 'Model registered and is loading'
            }
            
        except Exception as e:
            print('Exception in /api/host-model:', e)
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
        conn = self.db_manager.get_connection()
        if not conn:
            print('DB connection failed in _store_hosted_model')
            raise Exception("Database connection failed")
        
        try:
            cursor = conn.cursor()
            
            print('Inserting into hosted_models:', model_data)
            # Create hosted_models table if it doesn't exist
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
            
            # Insert the model
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
            
            conn.commit()
            
        except Exception as e:
            print('Exception in _store_hosted_model:', e)
            import traceback
            traceback.print_exc()
            raise
        finally:
            cursor.close()
    
    def _add_to_models_table(self, model_data: Dict[str, Any]):
        """Add the hosted model to the main models table with estimated scores"""
        conn = self.db_manager.get_connection()
        if not conn:
            print('DB connection failed in _add_to_models_table')
            raise Exception("Database connection failed")
        
        try:
            cursor = conn.cursor()
            
            print('Inserting into models table:', model_data)
            
            # First, ensure the required columns exist
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
            
            # Insert into models table
            cursor.execute("""
                INSERT INTO models (name, model_id, is_huggingface, huggingface_url)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (model_id) DO NOTHING;
            """, (
                model_data['custom_name'],
                f"hf-{model_data['hosted_model_id']}",  # Prefix to distinguish hosted models
                True,
                f"https://huggingface.co/{model_data['hf_model_id']}"
            ))
            
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
            
            # Determine categories based on pipeline_tag
            categories = ['qa', 'reasoning']
            if model_data['pipeline_tag'] == 'text-generation':
                categories.extend(['coding', 'creative'])
            elif model_data['pipeline_tag'] == 'text2text-generation':
                categories.extend(['summarization', 'translation'])
            
            # Insert scores for each category
            for category in categories:
                cursor.execute("""
                    INSERT INTO model_scores (model_id, category, accuracy, speed, cost)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (model_id, category) DO UPDATE SET
                        accuracy = EXCLUDED.accuracy,
                        speed = EXCLUDED.speed,
                        cost = EXCLUDED.cost;
                """, (
                    f"hf-{model_data['hosted_model_id']}",
                    category,
                    base_scores['accuracy'],
                    base_scores['speed'],
                    base_scores['cost']
                ))
            
            conn.commit()
            
        except Exception as e:
            print('Exception in _add_to_models_table:', e)
            import traceback
            traceback.print_exc()
            raise
        finally:
            cursor.close()
    
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

