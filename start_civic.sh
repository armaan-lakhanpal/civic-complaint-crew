#!/usr/bin/env bash
# Start everything for the Civic Complaint Resolution Crew, in the background.
# Usage (from the project folder):  bash start_civic.sh
cd "$(dirname "$0")"
source ~/crewai_env/bin/activate
mkdir -p logs

echo "1/4 PostgreSQL"
pg_isready -q -h localhost || sudo service postgresql start >/dev/null

echo "2/4 Ollama"
docker start ollama >/dev/null 2>&1

echo "3/4 n8n"
if ! docker ps --format '{{.Names}}' | grep -qx n8n; then
  docker run -d --rm --name n8n --network host \
    -e NODE_OPTIONS="--max-old-space-size=4096" \
    -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n >/dev/null
fi

echo "4/4 Streamlit apps"
pkill -f "streamlit run citizen_app.py" 2>/dev/null
pkill -f "streamlit run department_app.py" 2>/dev/null
sleep 1
nohup streamlit run citizen_app.py --server.port 8501 --server.headless true > logs/citizen.log 2>&1 &
nohup streamlit run department_app.py --server.port 8502 --server.headless true > logs/department.log 2>&1 &

echo
echo "Waiting for everything to answer (up to 60 s)..."
check() {  # name url
  for _ in $(seq 1 30); do
    curl -s -o /dev/null -m 2 "$2" && { echo "  OK    $1  $2"; return; }
    sleep 2
  done
  echo "  DOWN  $1  $2"
}
check "Ollama        " http://localhost:11434/api/version
check "n8n           " http://localhost:5678
check "Citizen app   " http://localhost:8501
check "Department app" http://localhost:8502
echo
echo "Logs: logs/citizen.log, logs/department.log   Stop: bash stop_civic.sh"
