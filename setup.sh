#!/usr/bin/env bash
# One-time setup for the Civic Complaint Resolution Crew (Ubuntu / WSL2).
# Needs: Docker, PostgreSQL 16, Ollama (native or the "ollama" Docker container), Python 3.12.
# Run from the project folder:  bash setup.sh
set -e
cd "$(dirname "$0")"

echo "== 1/5 Python environment"
if [ -z "$VIRTUAL_ENV" ]; then
  python3 -m venv .venv
  source .venv/bin/activate
fi
pip install -q -r requirements.txt

echo "== 2/5 PostgreSQL database 'civic'"
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='civic_user'" | grep -q 1 || \
  sudo -u postgres psql -c "CREATE USER civic_user WITH PASSWORD 'civic123';"
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='civic'" | grep -q 1 || \
  sudo -u postgres psql -c "CREATE DATABASE civic OWNER civic_user;"

echo "== 3/5 Ollama models (qwen2.5 7B, bge-m3, and civic-qwen with an 8K context)"
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx ollama; then
  OLLAMA="docker exec -i ollama ollama"
  MF_CMD='docker exec -i ollama sh -c'
else
  OLLAMA="ollama"
  MF_CMD='sh -c'
fi
$OLLAMA pull qwen2.5:latest
$OLLAMA pull bge-m3:latest
$MF_CMD 'printf "FROM qwen2.5:latest\nPARAMETER num_ctx 8192\nPARAMETER temperature 0.1\n" > /tmp/Mf && ollama create civic-qwen -f /tmp/Mf'

echo "== 4/5 Synthetic data, charter and vector index"
python 01_setup_and_data.py

echo "== 5/5 Notifications table (filled by the n8n workflow)"
psql "postgresql://civic_user:civic123@localhost:5432/civic" -c \
  "CREATE TABLE IF NOT EXISTS notifications (id SERIAL PRIMARY KEY, channel TEXT, complaint_id TEXT,
   recipient TEXT, language TEXT, priority TEXT, message TEXT, payload TEXT, sent_at TIMESTAMP DEFAULT now());"

echo
echo "Setup done. Next:"
echo "  1. Start n8n (docker) on port 5678, import n8n_civic_workflow.json, add the Postgres credential, Publish."
echo "  2. python 02_crew.py --test-llm"
echo "  3. streamlit run citizen_app.py --server.port 8501   and   streamlit run department_app.py --server.port 8502"
