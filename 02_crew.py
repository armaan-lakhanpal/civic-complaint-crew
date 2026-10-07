#!/usr/bin/env python3
"""02_crew.py: Civic complaint resolution crew (AL_AABA project)

Flow A (each new complaint, CrewAI Flow with a branch):
    1 Intake agent  ->  2 Duplicate checker  --duplicate-->  merge into parent  --+
                                             --new issue-->  3 Router agent  ------+-->  4 Communication agent  ->  n8n
Flow B (on demand / scheduled):
    5 SLA monitor agent  ->  escalations to level 2 / level 3 officers  ->  n8n

Usage (inside ~/aaba_crewai_p1/civic_crew with crewai_env active):
    python 02_crew.py --test-llm              check the model answers
    python 02_crew.py --limit 3               process the 3 oldest new complaints (local model)
    python 02_crew.py --ids GGN-0485 GGN-0488 process specific complaints
    python 02_crew.py --llm api --limit 20    same, using the API model in config.json (key in .env)
    python 02_crew.py --sla                   run the SLA monitor
    python 02_crew.py --evaluate              compare decisions with the answer key
    python 02_crew.py --reset                 undo all crew decisions so you can demo again
Add --verbose to see each agent's reasoning and tool calls.
"""
import os
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

import argparse
import json
import warnings
import re
import time
import datetime as dt
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
import ollama
import pandas as pd
import requests
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai.flow.flow import Flow, start, listen, router, or_

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =====================================================================
# Shared setup
# =====================================================================
PROJECT_DIR = Path(__file__).resolve().parent

# API keys live in a local .env file next to this script (git-ignored, never pushed).
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env", override=False)
except ImportError:
    pass
CONFIG = json.loads((PROJECT_DIR / "config.json").read_text())

# WSL runs on UTC; deadlines must be shown in the city's local time (Python and every DB session).
TIMEZONE = CONFIG.get("timezone", "Asia/Kolkata")
os.environ["TZ"] = TIMEZONE
time.tzset()

LOCAL_MODEL = CONFIG.get("local_model", "civic-qwen")
API_MODEL = CONFIG.get("api_model", "gpt-4o-mini")
VERBOSE = False

engine = create_engine(CONFIG["pg_url"], connect_args={"options": f"-c timezone={TIMEZONE}"})
oc = ollama.Client(host=CONFIG["ollama_url"])
chroma = chromadb.PersistentClient(path=CONFIG["chroma_dir"])
charter_col = chroma.get_collection("citizen_charter")
complaint_col = chroma.get_collection("open_complaints")


def q(sql, **params):
    """Run a SELECT and return a DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def run_sql(sql, **params):
    with engine.begin() as conn:
        conn.execute(text(sql), params)


try:  # sessions opened by n8n get the same timezone (stamps on notifications)
    with engine.begin() as _conn:
        _db = _conn.execute(text("SELECT current_database()")).scalar()
        _conn.execute(text(f"ALTER DATABASE \"{_db}\" SET timezone TO '{TIMEZONE}'"))
except Exception:
    pass

DEPTS = q("SELECT * FROM departments").set_index("dept_code").to_dict("index")
DEPT_CODES = list(DEPTS)
DEPT_HINTS = {
    "PWD_ROADS": "roads, potholes, footpaths",
    "WATER": "water supply, pressure, dirty tap water, pipeline leaks",
    "SEWER": "sewer overflow, blocked drains, waterlogging, open manholes",
    "SANITATION": "garbage collection, garbage heaps or burning",
    "ELECTRICAL": "streetlights, poles, live wires",
    "HORTICULTURE": "trees, parks, pruning",
}
LOCALITIES = [x for x in q("SELECT DISTINCT locality FROM complaints").locality.tolist() if x]
LANDMARKS = [x for x in q("SELECT DISTINCT landmark FROM complaints").landmark.tolist() if x]
PRIORITIES = ["critical", "high", "normal"]

HINGLISH_WORDS = {"mein", "ke", "paas", "nahi", "hai", "hain", "se", "din", "aa", "raha", "rahe", "kooda",
                  "paani", "sadak", "gaddhe", "naali", "koi", "ho", "gayi", "gaya", "bahut", "wali",
                  "saamne", "chahiye", "band", "pada", "tooti", "hui", "uthane"}
# Charter CC-3.1 safety conditions: these can never be routed below critical
CRITICAL_PATTERNS = [r"live wire", r"exposed .{0,20}wire", r"open manhole", r"manhole without",
                     r"blocking traffic", r"घरों में घुस"]


def detect_language(complaint_text):
    """Rule-based language label (more reliable than a small LLM for this)."""
    if re.search(r"[\u0900-\u097F]", complaint_text):
        return "hi"
    words = re.findall(r"[a-z]+", complaint_text.lower())
    return "hinglish" if sum(w in HINGLISH_WORDS for w in words) >= 2 else "en"


def target_hours(dept, priority):
    """Charter CC-3.1 to CC-3.3 and CC-4.1."""
    if priority == "critical":
        return 6
    base = DEPTS[dept]["sla_hours"]
    return base / 2 if priority == "high" else base


def embed(texts):
    return oc.embed(model=CONFIG["embed_model"], input=texts)["embeddings"]


def parse_json(raw):
    """Pull the first JSON object out of an LLM answer. Returns {} if none."""
    raw = re.sub(r"```(json)?", "", str(raw))
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def match_known(value, complaint_text, known):
    """Map the agent's locality or landmark onto a known name (guards against typos)."""
    v = str(value or "").strip().lower()
    for k in sorted(known, key=len, reverse=True):  # what the citizen actually wrote wins
        if k.lower() in (complaint_text or "").lower():
            return k
    for k in known:
        if k.lower() == v:
            return k
    for k in known:
        if v and (v in k.lower() or k.lower() in v):
            return k
    return str(value or "")


def log_action(cid, agent, action, detail, rule=None):
    run_sql("INSERT INTO agent_actions (complaint_id, agent, action, detail, rule_cited) "
            "VALUES (:c, :a, :ac, CAST(:d AS JSONB), :r)",
            c=cid, a=agent, ac=action, d=json.dumps(detail, ensure_ascii=False, default=str), r=rule)


N8N_STATUS = {"last": ""}


def send_to_n8n(payload):
    try:
        r = requests.post(CONFIG["n8n_webhook"], json=payload, timeout=10)
        N8N_STATUS["last"] = {404: "webhook not found (import and publish the workflow)"}.get(
            r.status_code, f"HTTP {r.status_code}")
        return r.status_code < 300
    except requests.RequestException:
        N8N_STATUS["last"] = "n8n not running"
        return False


def make_llm(mode):
    if mode == "api":
        # api_model in config.json decides the provider, e.g. "gpt-4o-mini" or "gemini/gemini-flash-latest"
        key_var = "GEMINI_API_KEY" if API_MODEL.startswith("gemini/") else "OPENAI_API_KEY"
        if not os.getenv(key_var) and not (key_var == "GEMINI_API_KEY" and os.getenv("GOOGLE_API_KEY")):
            raise SystemExit(f"API key missing: add {key_var}=your-key to the .env file in the project "
                             "folder, then restart the app")
        return LLM(model=API_MODEL, temperature=0.1)
    return LLM(model=f"ollama/{LOCAL_MODEL}", base_url=CONFIG["ollama_url"], temperature=0.1)


# Backup models tried in order when the main model fails (e.g. API overloaded, no internet).
# Filled by setup_agents(): for "api" mode = config "api_fallbacks" models, then the local model.
FALLBACK_LLMS = []
FALLBACK_LOG = []
RATE_LIMIT_RETRIES = 3


def rate_limit_wait(error):
    """If the error is an API rate limit (HTTP 429), return how many seconds to wait, else None."""
    msg = str(error)
    if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg and "rate limit" not in msg.lower():
        return None
    if "per day" in msg.lower() or "PerDay" in msg:
        return None  # daily quota: waiting won't help, use the backup models
    m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"retryDelay'?:\s*'?(\d+)s", msg)
    return min(float(m.group(1)) + 2, 65) if m else 30


def setup_agents(mode):
    """Create the five agents on the chosen model and prepare the backup chain."""
    llm = make_llm(mode)
    AGENTS.clear()
    AGENTS.update(build_agents(llm))
    FALLBACK_LLMS.clear()
    if mode == "api":
        for m in CONFIG.get("api_fallbacks", []):
            FALLBACK_LLMS.append(LLM(model=m, temperature=0.1))
        FALLBACK_LLMS.append(make_llm("local"))
    return llm


def run_agent(agent, description, expected_output):
    """Run one agent on one task, falling back to backup models if the main one fails.
    Returns (raw text, seconds taken)."""
    t0 = time.time()
    primary, last_error = agent.llm, None
    try:
        for llm in [primary] + FALLBACK_LLMS:
            agent.llm = llm
            for attempt in range(RATE_LIMIT_RETRIES + 1):
                try:
                    task = Task(description=description, expected_output=expected_output, agent=agent)
                    out = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=VERBOSE).kickoff()
                    if llm is not primary:
                        FALLBACK_LOG.append(f"{agent.role}: used backup model {llm.model}")
                    return out.raw, round(time.time() - t0, 1)
                except Exception as e:
                    last_error = e
                    wait = rate_limit_wait(e)
                    if wait and attempt < RATE_LIMIT_RETRIES:
                        # Quota per minute exceeded: waiting is faster than dropping to the slow local model
                        print(f"[rate limit] {llm.model}: waiting {wait:.0f} s, then retrying {agent.role}")
                        time.sleep(wait)
                        continue
                    print(f"[backup] {agent.role} failed on {llm.model}: {str(e)[:100]}")
                    break
        raise last_error
    finally:
        agent.llm = primary  # next call tries the main model again


# =====================================================================
# Tools (what agents can look up on their own)
# =====================================================================
@tool("search_open_complaints")
def search_open_complaints(summary: str, locality: str) -> str:
    """Find OPEN complaints similar to a new complaint in the same locality.
    Inputs: summary (short English description of the issue) and locality (e.g. 'Sector 56').
    Returns one line per candidate: ID, department, landmark, age in hours, similarity and text."""
    loc = match_known(locality, "", LOCALITIES)
    try:
        res = complaint_col.query(query_embeddings=embed([summary]), n_results=5, where={"locality": loc})
    except Exception:
        return "No open complaints found in this locality."
    now = time.time()
    lines = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        age_h = (now - meta["created_ts"]) / 3600
        lines.append(f"{cid} | dept={meta['dept_code']} | landmark={meta['landmark']} | "
                     f"age_hours={age_h:.0f} | similarity={1 - dist:.2f} | text={doc}")
    return "\n".join(lines) or "No open complaints found in this locality."


@tool("search_citizen_charter")
def search_citizen_charter(query: str) -> str:
    """Search the city's citizen charter (the rulebook for departments, priorities, SLAs).
    Input: a short English description of the issue. Returns the most relevant clauses with IDs like CC-2.1."""
    res = charter_col.query(query_embeddings=embed([query]), n_results=5)
    return "\n".join(res["documents"][0])


# =====================================================================
# The five agents
# =====================================================================
def build_agents(llm):
    common = dict(llm=llm, allow_delegation=False, verbose=VERBOSE)
    return {
        "intake": Agent(
            role="Complaint Intake Officer",
            goal="Turn each raw citizen complaint (English, Hindi or Hinglish) into clean structured data",
            backstory="You work at the Gurugram municipal call centre and read Hindi, Hinglish and English fluently.",
            max_iter=2, **common),
        "duplicate": Agent(
            role="Duplicate Complaint Checker",
            goal="Decide whether a new complaint repeats an issue that is already open, following charter rule CC-5.1",
            backstory="You stop the city from sending two crews to the same pothole. You always check the open complaints first.",
            tools=[search_open_complaints], max_iter=3, **common),
        "router": Agent(
            role="Complaint Routing Officer",
            goal="Assign the correct department and priority and justify both with citizen charter clause IDs",
            backstory="You know the charter well, but you always search it before deciding and you never decide without citing a clause.",
            tools=[search_citizen_charter], max_iter=3, **common),
        "comms": Agent(
            role="Citizen Communication Officer",
            goal="Write short, polite, accurate acknowledgements in the citizen's own language",
            backstory="You follow charter section 7: give the complaint ID, department and target time, never over-promise.",
            max_iter=1, **common),
        "sla": Agent(
            role="SLA Monitor",
            goal="Tell senior officers clearly which complaints breached their resolution target and what they must do",
            backstory="You prepare the hourly escalation note for Executive Engineers and Joint Commissioners.",
            max_iter=1, **common),
    }


AGENTS = {}

# =====================================================================
# Flow A: one complaint
# =====================================================================
class ComplaintState(BaseModel):
    complaint_id: str = ""
    text: str = ""
    created_at: str = ""
    intake: dict = {}
    duplicate_of: Optional[str] = None
    dept_code: Optional[str] = None
    priority: Optional[str] = None
    rules_cited: list = []
    due_at: Optional[str] = None
    message: str = ""
    sent_to_n8n: bool = False
    notes: list = []
    timings: dict = {}


class ComplaintFlow(Flow[ComplaintState]):

    @start()
    def intake_step(self):
        s = self.state
        row = q("SELECT text, created_at FROM complaints WHERE complaint_id = :c", c=s.complaint_id).iloc[0]
        s.text, s.created_at = row.text, str(row.created_at)
        dept_list = "\n".join(f"- {k}: {v}" for k, v in DEPT_HINTS.items())
        raw, secs = run_agent(AGENTS["intake"], f"""
Read this citizen complaint and extract structured data.

Complaint ID: {s.complaint_id}
Complaint text: {s.text}

Known localities: {", ".join(LOCALITIES)}
Department codes:
{dept_list}

Return ONLY a JSON object with these keys:
language: "en", "hi" (Devanagari Hindi) or "hinglish" (Hindi in Roman letters)
summary_en: one short English sentence describing the problem
issue_type: two to four words
likely_dept: one department code from the list
locality: the locality name
landmark: the landmark mentioned
urgency_signals: list of words that show danger or a full service failure (empty list if none)
""", "A single JSON object with the seven keys.")
        d = parse_json(raw)
        d["language_agent"] = d.get("language")
        d["language"] = detect_language(s.text)
        d["locality"] = match_known(d.get("locality"), s.text, LOCALITIES)
        d["landmark"] = match_known(d.get("landmark"), s.text, LANDMARKS)
        if d.get("likely_dept") not in DEPT_CODES:
            d["likely_dept"] = None
        d.setdefault("summary_en", s.text)
        s.intake, s.timings["intake"] = d, secs
        log_action(s.complaint_id, "intake", "extracted", d)

    @router(intake_step)
    def duplicate_step(self):
        s, d = self.state, self.state.intake
        raw, secs = run_agent(AGENTS["duplicate"], f"""
A new complaint has arrived. Decide whether it is a duplicate of an open complaint.

New complaint {s.complaint_id}
Summary: {d['summary_en']}
Likely department: {d['likely_dept']}
Locality: {d['locality']}
Landmark: {d['landmark']}
Original text: {s.text}

Step 1: call the search_open_complaints tool with the summary and the locality.
Step 2: apply charter rule CC-5.1. It is a duplicate ONLY if a candidate has the same department,
the same landmark, and age_hours of 72 or less. If several qualify, pick the oldest (CC-5.2).

Return ONLY a JSON object with keys:
is_duplicate: true or false
parent_id: the complaint ID it duplicates, or null
reason: one sentence
""", "A single JSON object with is_duplicate, parent_id and reason.")
        s.timings["duplicate"] = secs
        ans = parse_json(raw)

        # Guardrail: the database confirms rule CC-5.1 exactly
        since = pd.Timestamp(s.created_at) - pd.Timedelta(hours=72)
        rule_hits = [] if not d["landmark"] else q("""
            SELECT complaint_id FROM complaints
            WHERE status IN ('open', 'in_progress') AND locality = :loc AND landmark = :lm
              AND dept_code = :dept AND created_at >= :since AND complaint_id <> :c
            ORDER BY created_at""",
            loc=d["locality"], lm=d["landmark"], dept=d["likely_dept"] or "", since=since.to_pydatetime(),
            c=s.complaint_id).complaint_id.tolist()
        agent_parent = ans.get("parent_id") if ans.get("is_duplicate") in (True, "true", "True") else None

        if agent_parent and agent_parent in rule_hits:
            s.duplicate_of, verdict = agent_parent, "agent decision confirmed by rule check"
        elif agent_parent and not rule_hits:
            verdict = (f"agent said duplicate of {agent_parent}, overridden: rule CC-5.1 not met"
                       + ("" if d["landmark"] else " (no landmark given)"))
        elif rule_hits:
            s.duplicate_of = rule_hits[0]
            verdict = f"agent missed it, rule check found {rule_hits[0]}" if not agent_parent else \
                      f"agent picked {agent_parent}, corrected to oldest match {rule_hits[0]} (CC-5.2)"
        else:
            verdict = "agent decision confirmed: new issue"
        s.notes.append(f"Duplicate check: {verdict}")
        log_action(s.complaint_id, "duplicate_checker", "duplicate" if s.duplicate_of else "new_issue",
                   {"agent_answer": ans, "rule_matches": rule_hits, "verdict": verdict}, "CC-5.1")
        return "duplicate" if s.duplicate_of else "new_issue"

    @listen("duplicate")
    def merge_step(self):
        s = self.state
        p = q("SELECT dept_code, priority, status, created_at FROM complaints WHERE complaint_id = :p",
              p=s.duplicate_of).iloc[0]
        run_sql("""UPDATE complaints SET status = 'merged', duplicate_of = :p, dept_code = :d, priority = :pr
                   WHERE complaint_id = :c""", p=s.duplicate_of, d=p.dept_code, pr=p.priority, c=s.complaint_id)
        s.dept_code, s.priority, s.rules_cited = p.dept_code, p.priority, ["CC-5.1", "CC-5.2"]

        # CC-5.3: five or more duplicates in 24 hours raise the parent's priority
        n_dups = q("""SELECT count(*) AS n FROM complaints WHERE duplicate_of = :p AND created_at >= :since""",
                   p=s.duplicate_of, since=(pd.Timestamp.now() - pd.Timedelta(hours=24)).to_pydatetime()).n[0]
        if n_dups >= 5 and p.priority != "critical":
            new_p = PRIORITIES[PRIORITIES.index(p.priority) - 1]
            run_sql("UPDATE complaints SET priority = :pr WHERE complaint_id = :p", pr=new_p, p=s.duplicate_of)
            s.priority = new_p
            s.rules_cited.append("CC-5.3")
            log_action(s.duplicate_of, "duplicate_checker", "priority_raised", {"from": p.priority, "to": new_p}, "CC-5.3")
        s.due_at = str(pd.Timestamp(p.created_at) + pd.Timedelta(hours=target_hours(s.dept_code, s.priority)))
        log_action(s.complaint_id, "duplicate_checker", "merged", {"parent": s.duplicate_of}, "CC-5.2")

    @listen("new_issue")
    def route_step(self):
        s, d = self.state, self.state.intake
        dept_list = "\n".join(f"- {k}: {v}" for k, v in DEPT_HINTS.items())
        raw, secs = run_agent(AGENTS["router"], f"""
Route this complaint to a department and set its priority.

Complaint {s.complaint_id}
Summary: {d['summary_en']}
Original text: {s.text}
Intake officer's first guess: {d['likely_dept']}
Urgency signals: {d.get('urgency_signals', [])}

Department codes:
{dept_list}

Step 1: call search_citizen_charter with the summary.
Step 2: decide the department. Boundary rules in charter section 2 override the general lists in section 1.
Step 3: decide the priority. Start from "normal" (CC-3.3) and move up ONLY if the text clearly reports
one of these conditions:
- critical (CC-3.1): live or exposed wires, an open manhole, a fallen tree blocking traffic, sewage entering homes
- high (CC-3.2): no water at all, sewer overflowing onto a road, an accident or injury caused by the issue,
  a whole stretch fully dark, garbage being burned, a tree about to fall
Everything else is normal, for example dirty or smelly tap water, low pressure, a blocked drain,
waterlogging, broken roads, uncollected garbage, one broken streetlight, overgrown grass.

Return ONLY a JSON object with keys:
dept_code: one department code
priority: "critical", "high" or "normal"
rules_cited: list of clause IDs you relied on, for example ["CC-2.1", "CC-3.2"]
reason: one sentence
""", "A single JSON object with dept_code, priority, rules_cited and reason.")
        s.timings["router"] = secs
        ans = parse_json(raw)
        fallback = []
        dept = ans.get("dept_code")
        if dept not in DEPT_CODES:
            dept = d["likely_dept"] or "SANITATION"
            fallback.append("dept")
        prio = str(ans.get("priority", "")).lower()
        if prio not in PRIORITIES:
            prio = "normal"
            fallback.append("priority")
        if prio in ("critical", "high") and not d.get("urgency_signals"):
            fallback.append(f"priority {prio} downgraded to normal: intake found no urgency signals")
            prio = "normal"
        if prio != "critical" and any(re.search(p, s.text, re.I) for p in CRITICAL_PATTERNS):
            fallback.append(f"priority {prio} raised to critical: safety condition in CC-3.1")
            prio = "critical"
        rules = [r for r in ans.get("rules_cited", []) if re.fullmatch(r"CC-\d+\.\d+", str(r))]
        if not rules:
            fallback.append("rules")
        target = target_hours(dept, prio)
        s.dept_code, s.priority, s.rules_cited = dept, prio, rules
        s.due_at = str(pd.Timestamp(s.created_at) + pd.Timedelta(hours=target))

        run_sql("UPDATE complaints SET status = 'open', dept_code = :d, priority = :p WHERE complaint_id = :c",
                d=dept, p=prio, c=s.complaint_id)
        complaint_col.upsert(ids=[s.complaint_id], documents=[s.text], embeddings=embed([s.text]),
                          metadatas=[{"locality": d["locality"], "landmark": d["landmark"], "dept_code": dept,
                                      "status": "open", "created_ts": int(pd.Timestamp(s.created_at).timestamp())}])
        log_action(s.complaint_id, "router", "routed",
                   {"dept": dept, "priority": prio, "target_hours": target, "reason": ans.get("reason"),
                    "fallback_used": fallback}, ", ".join(rules) or None)
        if fallback:
            s.notes.append(f"Router guardrail: {'; '.join(fallback)}")
        if prio == "critical":  # CC-6.4
            log_action(s.complaint_id, "router", "copied_to_level2", {"officer": DEPTS[dept]["l2_officer"]}, "CC-6.4")
            s.notes.append(f"Critical: copied to {DEPTS[dept]['l2_officer']} (CC-6.4)")

    @listen(or_(merge_step, route_step))
    def communicate_step(self):
        s = self.state
        lang_name = {"en": "English", "hi": "Hindi in Devanagari script", "hinglish": "Hinglish (Hindi in Roman letters)"}[s.intake["language"]]
        due = pd.Timestamp(s.due_at).strftime("%d %b %Y, %I:%M %p")
        if s.duplicate_of:
            parent_status = q("SELECT status FROM complaints WHERE complaint_id = :p", p=s.duplicate_of).status[0]
            facts = (f"This issue was already reported as complaint {s.duplicate_of} (status: {parent_status}). "
                     f"The citizen's complaint {s.complaint_id} has been linked to it. Department: "
                     f"{DEPTS[s.dept_code]['dept_name']}. Target resolution by {due}.")
        else:
            facts = (f"Complaint ID {s.complaint_id}. Department: {DEPTS[s.dept_code]['dept_name']}. "
                     f"Priority: {s.priority}. Target resolution by {due}.")
        raw, secs = run_agent(AGENTS["comms"], f"""
Write an acknowledgement message to the citizen.

Citizen's original complaint: {s.text}
Facts to include: {facts}
Write in: {lang_name}

Rules: include the complaint ID, the department and the target time (CC-7.1); reply in the citizen's
language (CC-7.2); state the target time exactly as "by {due}" and never say "before", "earlier" or
"soon", and never blame another department (CC-7.3); no phone numbers or personal data (CC-7.4).
Maximum three sentences.
Return only the message text.
""", "The message text only.")
        s.timings["comms"] = secs
        s.message = str(raw).strip().strip('"')
        quote_id = s.duplicate_of or s.complaint_id
        if s.complaint_id not in s.message and quote_id not in s.message:  # CC-7.1 check
            s.message = f"[{s.complaint_id}] {s.message}"
        s.sent_to_n8n = send_to_n8n({
            "type": "acknowledgement", "complaint_id": s.complaint_id, "language": s.intake["language"],
            "department": s.dept_code, "priority": s.priority, "duplicate_of": s.duplicate_of,
            "due_at": s.due_at, "message": s.message})
        log_action(s.complaint_id, "communication", "acknowledged",
                   {"message": s.message, "sent_to_n8n": s.sent_to_n8n}, "CC-7.1")


def process_complaint(cid, quiet=False):
    """Run Flow A on one complaint. Returns the final state (used by the citizen app)."""
    t0 = time.time()
    flow = ComplaintFlow(suppress_flow_events=not VERBOSE)
    flow.kickoff(inputs={"complaint_id": cid})
    s = flow.state
    if quiet:
        return s
    d = s.intake
    print("\n" + "=" * 90)
    print(f"{cid}   ({time.time() - t0:.0f} s; per agent: {s.timings})")
    print(f"Text      : {s.text}")
    print(f"Intake    : {d.get('language')} | {d.get('issue_type')} | likely {d.get('likely_dept')} | "
          f"{d.get('locality')} / {d.get('landmark')}")
    if s.duplicate_of:
        print(f"Duplicate : YES, merged into {s.duplicate_of}")
    else:
        print(f"Route     : {s.dept_code}, {s.priority}, rules {s.rules_cited}, due {s.due_at[:16]}")
    for n in s.notes:
        print(f"Note      : {n}")
    print(f"Reply     : {s.message}")
    print(f"n8n       : {'sent' if s.sent_to_n8n else N8N_STATUS['last'] + ', message saved in agent_actions'}")
    return s


# =====================================================================
# Flow B: SLA monitor
# =====================================================================
def run_sla_monitor(quiet=False):
    """Escalate breached complaints. Python decides who gets what (exact); the agent only writes the cover note.
    Returns a dict (used by the department dashboard)."""
    df = q("""SELECT c.complaint_id, c.dept_code, c.priority, c.created_at, c.escalation_level, c.locality,
                     c.text, d.dept_name, d.l1_officer, d.l2_officer, d.l3_officer
              FROM complaints c JOIN departments d USING (dept_code)
              WHERE c.status IN ('open', 'in_progress')""")
    now = pd.Timestamp.now()
    df["target_h"] = [target_hours(a, b) for a, b in zip(df.dept_code, df.priority)]
    df["age_h"] = (now - pd.to_datetime(df.created_at)).dt.total_seconds() / 3600
    df["overdue_h"] = (df.age_h - df.target_h).round(0)
    df["level"] = np.select([df.age_h > 2 * df.target_h, df.age_h > df.target_h], [3, 2], 0)
    esc = df[df.level > df.escalation_level].sort_values(["level", "overdue_h"], ascending=False).copy()
    if esc.empty:
        if not quiet:
            print("SLA monitor: no new breaches.")
        return {"count": 0, "escalations": esc, "note": "No new breaches.", "sent": False}

    esc["officer"] = np.where(esc.level == 3, esc.l3_officer, esc.l2_officer)
    esc["rule"] = np.where(esc.level == 3, "CC-6.3", "CC-6.2")
    for r in esc.itertuples():
        run_sql("UPDATE complaints SET escalation_level = :l WHERE complaint_id = :c", l=int(r.level), c=r.complaint_id)
        log_action(r.complaint_id, "sla_monitor", f"escalated_L{r.level}",
                   {"officer": r.officer, "overdue_hours": r.overdue_h, "target_hours": r.target_h}, r.rule)

    # Exact officer-wise list built in Python (no LLM, so nothing can be misgrouped or dropped)
    blocks = []
    for officer, g in esc.groupby("officer", sort=False):
        lines = [f"{officer} ({len(g)} complaints)"]
        for r in g.itertuples():
            lines.append(f"  - {r.complaint_id} | {r.dept_name} | {r.priority} | {r.locality} | "
                         f"overdue {r.overdue_h:.0f} h | level {r.level} ({r.rule})")
        blocks.append("\n".join(lines))
    listing = "\n\n".join(blocks)
    n2, n3 = int((esc.level == 2).sum()), int((esc.level == 3).sum())
    by_dept = esc.groupby("dept_name").size().sort_values(ascending=False)
    summary = (f"{len(esc)} complaints breached their targets: {n2} escalated to level 2 (CC-6.2), "
               f"{n3} to level 3 (CC-6.3). By department: "
               + ", ".join(f"{k} {v}" for k, v in by_dept.items())
               + f". Most overdue: {esc.iloc[0].complaint_id} ({esc.iloc[0].overdue_h:.0f} h).")

    secs = 0
    if AGENTS:
        raw, secs = run_agent(AGENTS["sla"], f"""
Write a short covering note (3 to 4 sentences) for senior municipal officers about today's SLA breaches.
Facts: {summary}
Say which department needs the most attention, ask each officer to assign a crew today and update the
status in the system, and mention that the full officer-wise list is attached below.
Do NOT list complaint IDs yourself. Return only the note.
""", "A 3 to 4 sentence covering note.")
        cover = str(raw).strip()
    else:
        cover = summary
    note = f"{cover}\n\n{listing}"
    sent = send_to_n8n({"type": "escalation", "note": note, "count": len(esc),
                        "items": esc[["complaint_id", "dept_code", "level", "overdue_h"]].to_dict("records")})
    if not quiet:
        print(f"\nSLA monitor: {len(esc)} complaints escalated ({n2} to level 2, {n3} to level 3) in {secs} s")
        print("\nEscalation note:\n" + note)
        print(f"\nn8n: {'sent' if sent else N8N_STATUS['last'] + ', actions saved in agent_actions'}")
    return {"count": len(esc), "escalations": esc, "note": note, "sent": sent}


# =====================================================================
# Evaluation and reset
# =====================================================================
def evaluation_data():
    """Compare crew decisions with the answer key. Returns (metrics dict, detail DataFrame)."""
    df = q("""SELECT c.complaint_id, c.status, c.dept_code, c.priority, c.duplicate_of,
                     l.true_dept, l.true_priority, l.true_duplicate_of, l.language
              FROM complaints c JOIN complaint_labels l USING (complaint_id)
              WHERE c.complaint_id IN (SELECT DISTINCT complaint_id FROM agent_actions WHERE agent = 'communication')
              ORDER BY c.complaint_id""")  # only complaints that finished the whole flow (reply sent)
    if df.empty:
        return {}, df
    routed = df[df.status != "merged"]
    pred_dup, true_dup = df.duplicate_of.notna(), df.true_duplicate_of.notna()
    correct_dup = pred_dup & true_dup & (df.duplicate_of == df.true_duplicate_of)
    df["dept_ok"] = df.dept_code == df.true_dept
    df["priority_ok"] = df.priority == df.true_priority
    metrics = {
        "processed": len(df),
        "dept_accuracy": float(df.dept_ok.mean()),
        "priority_accuracy": float((routed.priority == routed.true_priority).mean()) if len(routed) else None,
        "dups_planted": int(true_dup.sum()),
        "dups_caught": int(correct_dup.sum()),
        "false_alarms": int((pred_dup & ~true_dup).sum()),
    }
    return metrics, df


def evaluate():
    m, df = evaluation_data()
    if not m:
        print("Nothing processed yet.")
        return
    print(f"Complaints processed      : {m['processed']}")
    print(f"Department accuracy       : {m['dept_accuracy']:.0%}")
    if m["priority_accuracy"] is not None:
        print(f"Priority accuracy (routed): {m['priority_accuracy']:.0%}")
    print(f"Duplicates planted        : {m['dups_planted']}   caught correctly: {m['dups_caught']}   "
          f"false alarms: {m['false_alarms']}")
    print("\n" + df[["complaint_id", "language", "dept_code", "true_dept", "priority", "true_priority",
                     "duplicate_of", "true_duplicate_of", "dept_ok"]].to_string(index=False))


def reset():
    chat_ids = q("SELECT complaint_id FROM complaints WHERE channel = 'citizen_chat'").complaint_id.tolist()
    # Department app status changes on synthetic complaints: restore the status each had before the first change
    dept = q("""SELECT DISTINCT ON (a.complaint_id) a.complaint_id, a.detail, c.text, c.locality, c.landmark,
                       c.dept_code, c.created_at
                FROM agent_actions a JOIN complaints c USING (complaint_id)
                WHERE a.agent = 'department' AND c.channel <> 'citizen_chat'
                ORDER BY a.complaint_id, a.action_id""")
    for r in dept.itertuples():
        before = (r.detail or {}).get("from", "open")
        run_sql("UPDATE complaints SET status = :s, resolved_at = NULL WHERE complaint_id = :c",
                s=before, c=r.complaint_id)
        complaint_col.upsert(ids=[r.complaint_id], documents=[r.text], embeddings=embed([r.text]),
                             metadatas=[{"locality": r.locality, "landmark": r.landmark, "dept_code": r.dept_code,
                                         "status": before, "created_ts": int(pd.Timestamp(r.created_at).timestamp())}])
    ids = q("SELECT DISTINCT complaint_id FROM agent_actions WHERE agent = 'intake'").complaint_id.tolist()
    run_sql("""UPDATE complaints SET status = 'new', dept_code = NULL, priority = NULL, duplicate_of = NULL
               WHERE complaint_id IN (SELECT DISTINCT complaint_id FROM agent_actions WHERE agent = 'intake')""")
    run_sql("DELETE FROM complaints WHERE channel = 'citizen_chat'")
    run_sql("UPDATE complaints SET escalation_level = 0")
    run_sql("DELETE FROM agent_actions")
    run_sql("DELETE FROM notifications") if q("SELECT to_regclass('notifications') AS t").t[0] else None
    stale = [i for i in set(ids) | set(chat_ids)]
    if stale:
        complaint_col.delete(ids=stale)
    print(f"Reset done: {len(ids)} complaints set back to 'new', {len(chat_ids)} chat complaints removed, "
          f"{len(dept)} department status changes undone, escalations, action log and notifications cleared.")
    print("(Priority raised by CC-5.3 on a parent is not reverted. Re-run 01 for a fully clean dataset.)")


# =====================================================================
# Main
# =====================================================================
def main():
    global VERBOSE
    ap = argparse.ArgumentParser(description="Civic complaint resolution crew")
    ap.add_argument("--llm", choices=["local", "api"], default="local")
    ap.add_argument("--limit", type=int, default=3, help="how many new complaints to process")
    ap.add_argument("--ids", nargs="*", help="specific complaint IDs to process")
    ap.add_argument("--sla", action="store_true", help="run the SLA monitor")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--test-llm", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    VERBOSE = args.verbose

    if args.reset:
        return reset()
    if args.evaluate:
        return evaluate()

    llm = setup_agents(args.llm)
    print(f"Model: {API_MODEL if args.llm == 'api' else 'ollama/' + LOCAL_MODEL}"
          + (f" (backups: {', '.join(l.model for l in FALLBACK_LLMS)})" if FALLBACK_LLMS else ""))
    if args.test_llm:
        for candidate in [llm] + FALLBACK_LLMS:
            t0 = time.time()
            try:
                print(f"{candidate.model}: {candidate.call('Reply with the single word OK.')} "
                      f"({time.time() - t0:.1f} s)")
            except Exception as e:
                print(f"{candidate.model}: FAILED ({str(e)[:90]})")
        return

    if args.sla:
        return run_sla_monitor()

    ids = args.ids or q("SELECT complaint_id FROM complaints WHERE status = 'new' ORDER BY created_at LIMIT :n",
                        n=args.limit).complaint_id.tolist()
    if not ids:
        print("No new complaints left. Run with --reset to demo again.")
        return
    print(f"Processing {len(ids)} complaint(s): {', '.join(ids)}")
    for cid in ids:
        process_complaint(cid)


if __name__ == "__main__":
    main()
