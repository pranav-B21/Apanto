"""
Database utility module for connecting to Supabase and querying models data.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from typing import List, Dict, Any
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DatabaseManager:
    """Manages database connections and model data queries."""
    
    def __init__(self):
        # Load database credentials from environment variables
        self.db_host = os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com")
        self.db_port = os.getenv("DB_PORT", "6543")
        self.db_name = os.getenv("DB_NAME", "postgres")
        self.db_user = os.getenv("DB_USER", "postgres.jqvayaoaqjkytejrypxs")
        self.db_password = os.getenv("DB_PASSWORD")
        
        if not self.db_password:
            raise ValueError("DB_PASSWORD environment variable is required but not set")
        
        # Construct connection string from environment variables
        self.connection_string = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        self._connection = None
    
    def get_connection(self):
        """Get database connection, creating one if it doesn't exist."""
        try:
            if self._connection is None or self._connection.closed:
                self._connection = psycopg2.connect(self.connection_string)
            return self._connection
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            return None
    
    def close_connection(self):
        """Close the database connection."""
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None
    
    def load_models_from_db(self) -> List[Dict[str, Any]]:
        """
        Load all models and their scores from the database.
        Returns data in the same format as the original models.json.
        """
        conn = self.get_connection()
        if not conn:
            print("⚠️ Could not connect to database, falling back to empty list")
            return []
        
        cursor = None
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Query to get all models with their scores
            query = """
                SELECT 
                    m.name,
                    m.model_id,
                    m.is_huggingface,
                    m.huggingface_url,
                    ms.category,
                    ms.accuracy,
                    ms.speed,
                    ms.cost
                FROM models m
                JOIN model_scores ms ON m.model_id = ms.model_id
                ORDER BY m.name, ms.category;
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Transform the flat result into the nested structure expected by the app
            models = {}
            
            for row in rows:
                model_id = row['model_id']
                
                # Initialize model structure if not exists
                if model_id not in models:
                    models[model_id] = {
                        'name': row['name'],
                        'model_id': row['model_id'],
                        'is_huggingface': row.get('is_huggingface', False),
                        'huggingface_url': row.get('huggingface_url'),
                        'scores': {}
                    }
                
                # Add scores for this category
                models[model_id]['scores'][row['category']] = {
                    'accuracy': float(row['accuracy']),
                    'speed': float(row['speed']),
                    'cost': float(row['cost'])
                }
            
            # Convert to list format
            models_list = list(models.values())
            print(f"✅ Loaded {len(models_list)} models from database")
            
            return models_list
            
        except Exception as e:
            print(f"❌ Error loading models from database: {e}")
            if conn:
                conn.rollback()  # Rollback the transaction on error
            return []
        finally:
            if cursor:
                cursor.close()
    
    def get_model_by_id(self, model_id: str) -> Dict[str, Any]:
        """Get a specific model by its model_id."""
        models = self.load_models_from_db()
        for model in models:
            if model['model_id'] == model_id:
                return model
        return None
    
    def get_models_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all models that have scores for a specific category."""
        models = self.load_models_from_db()
        return [model for model in models if category in model.get('scores', {})]

    def load_all_models_raw(self) -> list[dict]:
        """
        Load all models from the models table, regardless of whether they have scores.
        Returns a list of dicts with all columns from the models table.
        """
        conn = self.get_connection()
        if not conn:
            print("⚠️ Could not connect to database, falling back to empty list")
            return []
        cursor = None
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = "SELECT * FROM models ORDER BY name;"
            cursor.execute(query)
            rows = cursor.fetchall()
            print(f"✅ Loaded {len(rows)} models from models table (raw)")
            return rows
        except Exception as e:
            print(f"❌ Error loading all models from models table: {e}")
            if conn:
                conn.rollback()  # Rollback the transaction on error
            return []
        finally:
            if cursor:
                cursor.close()


# Global database manager instance
db_manager = DatabaseManager()


def load_models_from_database() -> List[Dict[str, Any]]:
    """
    Convenience function to load models from database.
    This replaces the old models.json loading functionality.
    """
    return db_manager.load_models_from_db()


def cleanup_database_connection():
    """Cleanup function to close database connections."""
    db_manager.close_connection() 