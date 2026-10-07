# Civic Complaint Resolution Crew

A multi-agent system that handles citizen complaints for a city municipality (demo city: Gurugram).
Citizens report problems in English, Hindi or Hinglish. Five AI agents understand each complaint,
detect duplicates, route it to the right department with the right priority, reply in the citizen's
language, and escalate anything that misses its deadline. Every decision cites a rule from the city's
citizen charter and is logged for audit.

Built for the Agentic AI for Business Automation course (FORE School of Management, PGDM BDA 2025 to 27).
All data is synthetic and the charter is fictional.

## The problem

* Complaints arrive as messy free text in mixed languages and get sent to the wrong department.
* The same pothole or garbage heap is reported many times, so crews are dispatched twice.
* Service deadlines (SLAs) are missed with no escalation, and nobody can explain why a decision was made.

## Architecture

```
 Citizen app (Streamlit)          Department app (Streamlit)
          |                                   ^
          v                                   |
   PostgreSQL  <--------------------->  CrewAI Flow (5 agents)  ----> Ollama (local) or Gemini API
   complaints, actions,                       |          |              with automatic fallback
   notifications                              |          v
                                              |     ChromaDB + bge-m3
                                              |     (charter RAG, duplicate search)
                                              v
                                     n8n webhook --> notifications (citizen replies, officer alerts)
```

**Design principle: the LLM decides, rules guard, n8n delivers.**
Agents handle language understanding and judgement. Exact rules (duplicate window, safety priorities,
deadline maths, officer assignment) are enforced in Python and every override is logged.

## The five agents

| # | Agent | Job | Tools |
|---|---|---|---|
| 1 | Intake | Reads the complaint in any language, extracts issue, locality, landmark, urgency | LLM |
| 2 | Duplicate checker | Finds an open complaint about the same issue (charter CC-5.1) | ChromaDB vector search, SQL rule check |
| 3 | Router | Picks department and priority, must cite charter clauses | Charter RAG (ChromaDB) |
| 4 | Communication | Writes the acknowledgement in the citizen's language (CC-7) | LLM, n8n webhook |
| 5 | SLA monitor | Escalates breached complaints to level 2 or level 3 officers (CC-6) | SQL, LLM for the covering note, n8n |

Flow A (per complaint) is a CrewAI Flow with a branch: Intake, then Duplicate check, then either
Merge or Router, then Communication. Flow B is the SLA monitor, run on demand or on a schedule.

## Guardrails

* **Duplicate rule check:** the database confirms CC-5.1 (same department, landmark, open, under 72 hours) and overrides the agent if needed.
* **Safety floor:** live wires, open manholes, sewage entering homes and fallen trees blocking traffic are always critical (CC-3.1).
* **No over-escalation:** high or critical priority without any urgency signal is downgraded to normal.
* **Reply checks:** the complaint ID must appear; the agent may not promise a time earlier than the target (CC-7.3).
* **Exact escalation lists:** officer-wise lists are built in Python, the agent only writes the cover note.
* **Model fallback:** if the API model fails (for example 503 overloaded), each agent retries on backup models and finally on the local model.

## Tech stack

CrewAI 1.15 (Agents, Flows, tools) · Ollama (qwen2.5 7B as `civic-qwen`, bge-m3 embeddings) · Google Gemini API (optional) ·
ChromaDB (embedded) · PostgreSQL 16 · n8n (Docker) · Streamlit · Python 3.12 on Ubuntu 24.04 (WSL2)

## Project structure

| File | Purpose |
|---|---|
| `01_setup_and_data.py` | Generates 500 synthetic complaints (English, Hindi, Hinglish), departments, SLA table, answer key; writes the citizen charter; builds the ChromaDB index |
| `02_crew.py` | Tools, the five agents, both flows, guardrails, evaluation, reset, command line |
| `citizen_app.py` | Citizen chat: submit a complaint, watch the agents live, track by ID |
| `department_app.py` | Staff dashboard: KPIs, deadlines, status updates, audit trail, notifications, SLA monitor, evaluation |
| `n8n_civic_workflow.json` | n8n workflow: Webhook, Format notification, Save to notifications |
| `data/citizen_charter.md` | The 27-clause rulebook every decision cites |
| `config.json` | Service addresses and model choice |
| `setup.sh` | One-time setup |

## Setup

Prerequisites: Ubuntu or WSL2, Docker, PostgreSQL 16, Ollama, Python 3.12.

```bash
git clone <this repo> civic_crew && cd civic_crew
bash setup.sh
```

Then start n8n and import the workflow:

```bash
docker run -d --rm --name n8n --network host -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n
```

Open http://localhost:5678, import `n8n_civic_workflow.json`, set the Postgres credential on
*Save to notifications* (host `localhost`, database `civic`, user `civic_user`, password `civic123`), then **Publish**.

Optional, for the faster API mode: copy `.env.example` to `.env` and put your key in it.
`.env` is git-ignored, so the key stays on your machine and is never pushed.

```bash
cp .env.example .env
code .env        # set GEMINI_API_KEY=your-key
```

## Running

```bash
python 02_crew.py --test-llm                 # check the model(s)
python 02_crew.py --limit 3                  # process 3 new complaints on the local model
python 02_crew.py --llm api --limit 20       # all 20 via API (falls back to local if needed)
python 02_crew.py --sla                      # SLA monitor
python 02_crew.py --evaluate                 # accuracy against the answer key
python 02_crew.py --reset                    # undo all agent decisions for a fresh demo

streamlit run citizen_app.py --server.port 8501
streamlit run department_app.py --server.port 8502
```

## Results

Scored against the answer key (`complaint_labels`) on the complaints that completed the full flow.
The API run was stopped early to stay within the Gemini free-tier quota (15 requests per minute),
so the sample is small and should be read as an indication, not a benchmark.

**API run (Gemini Flash Lite, 5 complaints: 3 English, 2 Hindi, 1 planted duplicate)**

| Metric | Result |
|---|---|
| Department accuracy | 5 / 5 (100%) |
| Priority accuracy (routed complaints) | 4 / 4 (100%) |
| Duplicates caught | 1 / 1, no false alarms |
| Time per complaint | about 10 to 15 seconds |

**Local run (qwen2.5 7B on CPU, earlier test of 4 complaints)**

| Metric | Result |
|---|---|
| Department accuracy | 4 / 4 |
| Duplicates | the agent missed the planted duplicate; the rule guardrail caught it |
| Priority | live-wire complaint first rated high; fixed by the CC-3.1 safety floor |
| Time per complaint | about 1.5 to 4 minutes |

The comparison shows the point of the guardrails: the larger API model got everything right on its own,
while the small local model needed the rule checks to reach the same final decisions.

## Limitations and next steps

* Data and charter are synthetic; templated complaints are easier than real ones.
* The small local model often misses duplicates and over-rates priority; the rule guardrails catch these, which is why they exist.
* CPU-only local inference takes about 2 to 3 minutes per complaint.
* Next: real delivery channels in n8n (WhatsApp, email), photo and GPS intake, a scheduled SLA run, testing on real municipal complaint data.
