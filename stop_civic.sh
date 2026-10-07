#!/usr/bin/env bash
# Stop the two Streamlit apps and n8n. Ollama and PostgreSQL keep running (other projects use them).
# Usage:  bash stop_civic.sh        or   bash stop_civic.sh --all   (also frees the Ollama model from memory)
pkill -f "streamlit run citizen_app.py" 2>/dev/null && echo "Citizen app stopped"
pkill -f "streamlit run department_app.py" 2>/dev/null && echo "Department app stopped"
docker stop n8n >/dev/null 2>&1 && echo "n8n stopped"
if [ "$1" = "--all" ]; then
  docker exec ollama ollama stop civic-qwen >/dev/null 2>&1 && echo "civic-qwen unloaded from memory"
fi
echo "Done."
