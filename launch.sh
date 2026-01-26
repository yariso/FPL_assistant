#!/bin/bash
cd "$(dirname "$0")"
echo "Starting FPL Assistant..."
source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || echo "No venv found"
streamlit run src/fpl_assistant/app.py
