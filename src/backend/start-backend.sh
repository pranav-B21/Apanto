#!/bin/bash

# AI Smart Prompt Stream - Backend Startup Script
# This script ensures the backend starts with the correct Python environment

set -e  # Exit on any error

echo "🚀 Starting AI Smart Prompt Stream Backend..."

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from the src/backend directory."
    exit 1
fi

# Define Python path
PYTHON_PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "⚠️  Warning: $PYTHON_PATH not found, falling back to system python3"
    PYTHON_PATH="python3"
fi

# Test Python and required packages
echo "🔍 Checking Python environment..."
$PYTHON_PATH -c "
import sys
print(f'✅ Python: {sys.version}')

try:
    import fastapi, uvicorn, pydantic
    print('✅ All required packages found')
except ImportError as e:
    print(f'❌ Missing package: {e}')
    print('Run: pip3 install fastapi uvicorn pydantic python-dotenv requests')
    sys.exit(1)
" || {
    echo "❌ Python environment check failed"
    exit 1
}

# Start the server
echo "🌐 Starting backend server on http://127.0.0.1:8000"
exec $PYTHON_PATH -m uvicorn main:app --reload --host 127.0.0.1 --port 8000 