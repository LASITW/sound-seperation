#!/bin/bash
set -e

# Install API dependencies if needed
pip install -r requirements_api.txt -q

# Create runtime directories
mkdir -p uploads outputs

# Open browser once server is ready
(sleep 1.5 && open http://localhost:8000) &

# Start the server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
