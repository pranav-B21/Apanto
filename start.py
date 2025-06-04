#!/usr/bin/env python3
"""
Start script for the AI Smart Prompt Stream backend
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Change to the src/backend directory
    backend_dir = Path(__file__).parent / "src" / "backend"
    
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        sys.exit(1)
    
    # Check if .env file exists
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("⚠️  .env file not found. Please create one based on env.example")
        print("   Copy env.example to .env and fill in your API keys")
        sys.exit(1)
    
    print("🚀 Starting AI Smart Prompt Stream backend...")
    print(f"📁 Backend directory: {backend_dir}")
    
    # Change to backend directory and start the server
    os.chdir(backend_dir)
    
    try:
        # Start the FastAPI server
        subprocess.run([
            "python3", "-m", "uvicorn", 
            "main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Backend server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start backend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 