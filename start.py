#!/usr/bin/env python3
"""
Deployment script for Render.com
This script starts the FastAPI backend server
"""

import uvicorn
import os

if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8000))
    
    # Start the server
    uvicorn.run(
        "src.backend.main:app",
        host="0.0.0.0",  # Bind to all interfaces for deployment
        port=port,
        reload=False  # Disable reload in production
    ) 