#!/usr/bin/env bash
# Health check for the Civic Complaint Resolution Crew. Changes nothing (the n8n test row is removed).
# Usage (from the project folder):  bash check_civic.sh
cd "$(dirname "$0")"
source ~/crewai_env/bin/activate 2>/dev/null
DB="postgresql://civic_user:civic123@localhost:5432/civic"
ok()   { echo "  OK    $*"; }
bad()  { echo "  FAIL  $*"; }
warn() { echo "  WARN  $*"; }

echo "== Services"
pg_isready -q -h localhost && ok "PostgreSQL" || bad "PostgreSQL down: bash start_civic.sh"
curl -s -m 3 -o /dev/null http://localhost:11434/api/version && ok "Ollama" || bad "Ollama down: docker start ollama"
docker exec ollama ollama list 2>/dev/null | grep -q "^civic-qwen" && ok "civic-qwen model present" || bad "civic-qwen missing (see setup.sh step 3)"
docker exec ollama ollama list 2>/dev/null | grep -q "^bge-m3" && ok "bge-m3 model present" || bad "bge-m3 missing: docker exec ollama ollama pull bge-m3"
curl -s -m 3 -o /dev/null http://localhost:5678 && ok "n8n" || bad "n8n down: bash start_civic.sh"
code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" -X POST http://localhost:5678/webhook/civic-ack \
  -H "Content-Type: application/json" -d '{"type":"acknowledgement","complaint_id":"HEALTHCHECK","message":"test"}')
if [ "$code" = "200" ]; then
  sleep 2; psql -q "$DB" -c "DELETE FROM notifications WHERE complaint_id = 'HEALTHCHECK';" 2>/dev/null
  ok "n8n webhook published and saving"
else
  bad "n8n webhook answered HTTP $code (404 = workflow not published)"
fi
curl -s -m 3 -o /dev/null http://localhost:8501 && ok "Citizen app :8501" || bad "Citizen app down: bash start_civic.sh"
curl -s -m 3 -o /dev/null http://localhost:8502 && ok "Department app :8502" || bad "Department app down: bash start_civic.sh"

echo "== Project files"
for f in 01_setup_and_data.py 02_crew.py citizen_app.py department_app.py config.json n8n_civic_workflow.json \
         data/citizen_charter.md README.md setup.sh start_civic.sh stop_civic.sh requirements.txt .gitignore .env.example; do
  [ -f "$f" ] && ok "$f" || bad "$f missing"
done
grep -q "load_dotenv" 02_crew.py && ok "02_crew.py reads .env" || bad "02_crew.py is an old version (no .env loading)"
grep -q "time.tzset" 02_crew.py && ok "02_crew.py uses India time" || bad "02_crew.py is an old version (UTC times)"
ls *Zone.Identifier data/*Zone.Identifier >/dev/null 2>&1 && warn "Windows tag files present: rm -f *Zone.Identifier data/*Zone.Identifier"

echo "== API key"
python - <<'EOF'
from dotenv import dotenv_values
k = (dotenv_values('.env').get('GEMINI_API_KEY') or '') if __import__('os').path.exists('.env') else ''
print(f"  OK    .env has GEMINI_API_KEY ({len(k)} characters)" if k else "  WARN  no key in .env (API mode will not work; Local still does)")
EOF

echo "== Database"
psql -tA "$DB" -c "SELECT '  OK    ' || count(*) || ' complaints (' || count(*) FILTER (WHERE status='new') || ' new, ' || count(*) FILTER (WHERE status IN ('open','in_progress')) || ' open)' FROM complaints;" 2>/dev/null || bad "cannot query complaints"
psql -tA "$DB" -c "SELECT '  OK    ' || count(*) || ' agent actions, ' || (SELECT count(*) FROM notifications) || ' notifications' FROM agent_actions;" 2>/dev/null
tz=$(psql -tA "$DB" -c "SHOW timezone;" 2>/dev/null)
[ "$tz" = "Asia/Kolkata" ] && ok "database timezone $tz" || warn "database timezone is '$tz' (restart the apps once with the new 02_crew.py)"

echo "== Git"
if git ls-files --error-unmatch .env >/dev/null 2>&1; then bad ".env IS TRACKED BY GIT: git rm --cached .env && git commit -m 'untrack .env'"; else ok ".env not in Git"; fi
changes=$(git status --porcelain | wc -l)
[ "$changes" = "0" ] && ok "everything committed" || warn "$changes uncommitted change(s): git status"
git fetch -q origin 2>/dev/null && [ "$(git rev-parse HEAD)" = "$(git rev-parse @{u} 2>/dev/null)" ] && ok "in sync with GitHub" || warn "not in sync with GitHub: git push"

echo "== Memory"
free -h | awk '/Mem:/ {print "  INFO  WSL memory used " $3 " of " $2}'

echo "== Recent app errors"
for l in logs/citizen.log logs/department.log; do
  [ -f "$l" ] && { n=$(grep -ciE "error|traceback" "$l"); [ "$n" = "0" ] && ok "$l clean" || warn "$l has $n error line(s): tail -30 $l"; }
done
echo "Done."
