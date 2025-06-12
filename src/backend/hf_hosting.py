"""
Hugging Face Model Hosting Service
Handles registration, validation, and management of user-submitted HF models
"""

import os
import re
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
from dotenv import load_dotenv

import json
from psycopg2.extras import RealDictCursor

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Optional, for private models
HF_INFERENCE_API_BASE = "https://api-inference.huggingface.co/models/"


class HuggingFaceHostingService:
    """Manages Hugging Face model hosting and registration"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.hf_token = HF_API_TOKEN
        
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
        Test if the model can be used for inference via HF Inference API
        """
        headers = {
            'Content-Type': 'application/json'
        }
        if self.hf_token:
            headers['Authorization'] = f'Bearer {self.hf_token}'
        
        payload = {
            "inputs": test_prompt,
            "parameters": {
                "max_new_tokens": 50,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        
        inference_url = f"{HF_INFERENCE_API_BASE}{model_id}"
        
        try:
            response = requests.post(
                inference_url, 
                json=payload, 
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'response': response.json(),
                    'latency': response.elapsed.total_seconds()
                }
            elif response.status_code == 503:
                # Model is loading
                return {
                    'success': False,
                    'error': 'Model is loading, please try again in a few minutes',
                    'status': 'loading'
                }
            else:
                return {
                    'success': False,
                    'error': f'Inference failed: HTTP {response.status_code}',
                    'details': response.text
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Inference error: {str(e)}'
            }
    
    def register_model(self, user_id: str, model_url: str, custom_name: str = None) -> Dict[str, Any]:
        """
        Register a new Hugging Face model for a user
        """
        # Parse the URL
        parsed = self.parse_hf_url(model_url)
        if not parsed:
            return {
                'success': False,
                'error': 'Invalid Hugging Face model URL format'
            }
        
        model_id = parsed['model_id']
        
        # Validate model exists
        validation = self.validate_model_exists(model_id)
        if not validation.get('exists'):
            return {
                'success': False,
                'error': f"Model validation failed: {validation.get('error', 'Unknown error')}"
            }
        
        # Test inference
        inference_test = self.test_model_inference(model_id)
        if not inference_test.get('success') and inference_test.get('status') != 'loading':
            return {
                'success': False,
                'error': f"Model inference test failed: {inference_test.get('error', 'Unknown error')}"
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
            return {
                'success': False,
                'error': f'Failed to register model: {str(e)}'
            }
    
    def _generate_hosted_model_id(self, user_id: str, model_id: str) -> str:
        """Generate a unique ID for a hosted model instance"""
        timestamp = datetime.utcnow().isoformat()
        hash_input = f"{user_id}:{model_id}:{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _store_hosted_model(self, model_data: Dict[str, Any]):
        """Store hosted model in database"""
        conn = self.db_manager.get_connection()
        if not conn:
            raise Exception("Database connection failed")
        
        try:
            cursor = conn.cursor()
            
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
            
        finally:
            cursor.close()
    
    def _add_to_models_table(self, model_data: Dict[str, Any]):
        """Add the hosted model to the main models table with estimated scores"""
        conn = self.db_manager.get_connection()
        if not conn:
            raise Exception("Database connection failed")
        
        try:
            cursor = conn.cursor()
            
            # Insert into models table
            cursor.execute("""
                INSERT INTO models (name, model_id)
                VALUES (%s, %s)
                ON CONFLICT (model_id) DO NOTHING;
            """, (
                model_data['custom_name'],
                f"hf-{model_data['hosted_model_id']}"  # Prefix to distinguish hosted models
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
            categories = ['qa', 'reasoning']  # Default categories
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


# HF Inference wrapper for hosted models
class HuggingFaceInference:
    """Handles inference for Hugging Face hosted models"""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or HF_API_TOKEN
        self.base_url = HF_INFERENCE_API_BASE
    
    def run_inference(self, model_id: str, prompt: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run inference on a Hugging Face model
        """
        headers = {
            'Content-Type': 'application/json'
        }
        if self.hf_token:
            headers['Authorization'] = f'Bearer {self.hf_token}'
        
        # Default parameters
        default_params = {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "return_full_text": False
        }
        
        if parameters:
            default_params.update(parameters)
        
        payload = {
            "inputs": prompt,
            "parameters": default_params
        }
        
        inference_url = f"{self.base_url}{model_id}"
        
        try:
            response = requests.post(
                inference_url,
                json=payload,
                headers=headers,
                timeout=60  # Longer timeout for larger models
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    # Standard text generation response
                    return {
                        'success': True,
                        'text': result[0].get('generated_text', ''),
                        'latency': response.elapsed.total_seconds()
                    }
                elif isinstance(result, dict):
                    # Some models return dict directly
                    return {
                        'success': True,
                        'text': result.get('generated_text', str(result)),
                        'latency': response.elapsed.total_seconds()
                    }
                else:
                    return {
                        'success': True,
                        'text': str(result),
                        'latency': response.elapsed.total_seconds()
                    }
                    
            elif response.status_code == 503:
                # Model is loading
                estimated_time = response.headers.get('X-Estimated-Time', '60')
                return {
                    'success': False,
                    'error': f'Model is loading. Estimated time: {estimated_time}s',
                    'status': 'loading',
                    'estimated_time': estimated_time
                }
            else:
                return {
                    'success': False,
                    'error': f'Inference failed: HTTP {response.status_code}',
                    'details': response.text
                }
                
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timed out. The model might be too large or busy.'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Inference error: {str(e)}'
            }
