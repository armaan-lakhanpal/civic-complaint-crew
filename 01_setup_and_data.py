#!/usr/bin/env python3
"""01_setup_and_data.py
Civic complaint resolution crew (AL_AABA project)
1. Checks Ollama (Docker, port 11434) and PostgreSQL (WSL, port 5432)
2. Generates synthetic Gurugram complaints (English, Hinglish, Hindi), departments, SLA data
3. Loads everything into PostgreSQL
4. Writes the citizen charter and indexes it (plus open complaints) in ChromaDB with bge-m3

Run:  source ~/crewai_env/bin/activate && python 01_setup_and_data.py
Safe to re-run: tables and collections are rebuilt.
"""


# ======================================================================
# 01: Setup and data
# ======================================================================


# ======================================================================
# 0. Config
# ======================================================================
import os, json, random, re
import datetime as dt
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import create_engine, text

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "ollama_url": "http://localhost:11434",
    "llm_model": "qwen2.5:latest",
    "embed_model": "bge-m3:latest",
    "pg_url": "postgresql+psycopg2://civic_user:civic123@localhost:5432/civic",
    "chroma_dir": str(PROJECT_DIR / "chroma_store"),
    "n8n_webhook": "http://localhost:5678/webhook/civic-ack",
    "local_model": "civic-qwen",
    "api_model": "gemini/gemini-flash-latest",
    "api_fallbacks": ["gemini/gemini-flash-lite-latest"],
    "timezone": "Asia/Kolkata",
}
_cfg_path = PROJECT_DIR / "config.json"
if _cfg_path.exists():  # keep settings already chosen (models, webhook); always refresh the local path
    _saved = json.loads(_cfg_path.read_text())
    _saved.pop("chroma_dir", None)
    CONFIG.update(_saved)
_cfg_path.write_text(json.dumps(CONFIG, indent=2))

# WSL runs on UTC; generate and stamp everything in the city's local time
import time as _time
os.environ["TZ"] = CONFIG.get("timezone", "Asia/Kolkata")
_time.tzset()
RANDOM_SEED = 42
print("Project folder:", PROJECT_DIR)


# ======================================================================
# 1. Connectivity check
# ======================================================================
tags = requests.get(f"{CONFIG['ollama_url']}/api/tags", timeout=10).json()
models = [m["name"] for m in tags["models"]]
for m in (CONFIG["llm_model"], CONFIG["embed_model"]):
    print(("OK       " if m in models else "MISSING  ") + m)

engine = create_engine(CONFIG["pg_url"], connect_args={"options": f"-c timezone={os.environ['TZ']}"})
with engine.connect() as conn:
    print(conn.execute(text("select version()")).scalar())


# ======================================================================
# 2. Departments and SLA targets
# ======================================================================
DEPTS = {
    "PWD_ROADS":    {"name": "Roads and Footpaths",    "sla_hours": 72,  "l1": "JE Ravi Malik",   "l2": "EE Sunita Rao",   "l3": "JC Arvind Mehta"},
    "WATER":        {"name": "Water Supply",           "sla_hours": 24,  "l1": "JE Pooja Yadav",  "l2": "EE Karan Bhatia", "l3": "JC Arvind Mehta"},
    "SEWER":        {"name": "Sewerage and Drainage",  "sla_hours": 48,  "l1": "JE Imran Khan",   "l2": "EE Neha Sethi",   "l3": "JC Arvind Mehta"},
    "SANITATION":   {"name": "Garbage and Sanitation", "sla_hours": 24,  "l1": "JE Deepak Saini", "l2": "EE Ritu Sharma",  "l3": "JC Meera Joshi"},
    "ELECTRICAL":   {"name": "Streetlights",           "sla_hours": 48,  "l1": "JE Vikas Dahiya", "l2": "EE Anil Kapoor",  "l3": "JC Meera Joshi"},
    "HORTICULTURE": {"name": "Parks and Trees",        "sla_hours": 120, "l1": "JE Sanjay Rawat", "l2": "EE Kavita Arora", "l3": "JC Meera Joshi"},
}
DEPT_WEIGHTS = {"SANITATION": 0.24, "WATER": 0.20, "PWD_ROADS": 0.18, "SEWER": 0.16, "ELECTRICAL": 0.12, "HORTICULTURE": 0.10}

def target_hours(dept, priority):
    """Resolution target as per charter clauses CC-3.1 to CC-3.3."""
    if priority == "critical":
        return 6
    base = DEPTS[dept]["sla_hours"]
    return base / 2 if priority == "high" else base

departments_df = pd.DataFrame([
    {"dept_code": k, "dept_name": v["name"], "sla_hours": v["sla_hours"],
     "l1_officer": v["l1"], "l2_officer": v["l2"], "l3_officer": v["l3"],
     "email": f"{k.lower()}@civic-demo.example.org"}
    for k, v in DEPTS.items()])
print(departments_df.to_string(index=False))


# ======================================================================
# 3. Complaint templates
# ======================================================================
LOCALITIES = ["Sector 14", "Sector 29", "DLF Phase 3", "Sohna Road", "Sector 56",
              "Palam Vihar", "Sector 45", "Sushant Lok 1", "Sector 10A", "Old Gurgaon"]
LANDMARKS = ["Metro Station", "Main Market", "School Gate", "Shiv Mandir",
             "Community Centre", "Gate No. 2", "Petrol Pump", "Bus Stop"]

TEMPLATES = [
    # Roads
    ("PWD_ROADS", "en", "high", "Huge pothole near {lm} in {loc}. Two bikes skidded this week, someone will get hurt."),
    ("PWD_ROADS", "en", "normal", "The road near {lm}, {loc} is badly broken and full of small potholes."),
    ("PWD_ROADS", "hinglish", "normal", "{loc} mein {lm} ke paas sadak poori tooti hui hai, gaddhe hi gaddhe hain"),
    ("PWD_ROADS", "hi", "high", "{loc} में {lm} के पास सड़क पर गहरा गड्ढा है, रात में दिखता नहीं, कल एक स्कूटी गिर गई"),
    ("PWD_ROADS", "en", "normal", "Footpath tiles are broken outside {lm}, {loc}. Senior citizens find it hard to walk."),
    # Water
    ("WATER", "en", "high", "No water supply in {loc} for {days} days. The entire block near {lm} is affected."),
    ("WATER", "hinglish", "high", "{loc} mein {days} din se paani nahi aa raha, {lm} wali gali mein sab pareshaan hain"),
    ("WATER", "hi", "normal", "{loc} में नल से गंदा और बदबूदार पानी आ रहा है, {lm} के पास"),
    ("WATER", "en", "normal", "Very low water pressure in {loc} since last week, tanks near {lm} do not fill up."),
    ("WATER", "hinglish", "normal", "{lm} ke paas pipeline leak ho rahi hai {loc} mein, paani waste ho raha hai"),
    # Sewer and drainage
    ("SEWER", "en", "high", "Sewer is overflowing on the main road near {lm}, {loc}. Terrible smell and dirty water everywhere."),
    ("SEWER", "hinglish", "normal", "{lm} ke saamne naali jam hai {loc} mein, gutter ka paani road par aa raha hai"),
    ("SEWER", "hi", "critical", "{loc} में {lm} के पास सीवर का गंदा पानी घरों में घुस रहा है"),
    ("SEWER", "en", "normal", "The road near {lm} in {loc} gets waterlogged every time it rains, the drain seems blocked."),
    ("SEWER", "en", "critical", "Open manhole without a cover on the road near {lm}, {loc}. Very dangerous at night."),
    # Sanitation
    ("SANITATION", "en", "normal", "Garbage has not been collected for {days} days in {loc}, stray dogs are spreading it near {lm}."),
    ("SANITATION", "hinglish", "normal", "{loc} mein {lm} ke paas kooda {days} din se pada hai, koi uthane nahi aaya"),
    ("SANITATION", "hi", "normal", "{lm} के पास कूड़े का बड़ा ढेर लगा है, {loc} में मच्छर बहुत हो गए हैं"),
    ("SANITATION", "en", "high", "Someone is burning garbage near {lm} in {loc} every evening, the smoke is unbearable."),
    ("SANITATION", "en", "normal", "People are dumping garbage inside the park near {lm}, {loc}."),
    # Streetlights
    ("ELECTRICAL", "en", "high", "Streetlights on the whole stretch near {lm}, {loc} are off. Women feel unsafe walking at night."),
    ("ELECTRICAL", "hinglish", "normal", "{loc} mein {lm} ke paas street light {days} din se band hai"),
    ("ELECTRICAL", "hi", "normal", "{loc} में {lm} के पास स्ट्रीट लाइट खराब है, रात को अंधेरा रहता है"),
    ("ELECTRICAL", "en", "critical", "A streetlight pole near {lm}, {loc} has exposed live wires hanging low. Kids play there."),
    # Parks and trees
    ("HORTICULTURE", "en", "critical", "A big tree branch fell on the road near {lm}, {loc} and is blocking traffic."),
    ("HORTICULTURE", "hinglish", "normal", "{loc} ke park mein ghaas bahut badi ho gayi hai {lm} ke paas, safai chahiye"),
    ("HORTICULTURE", "hi", "high", "{lm} के पास एक पेड़ झुक गया है और कभी भी गिर सकता है, {loc}"),
    ("HORTICULTURE", "en", "normal", "Tree branches near {lm} in {loc} are touching the wires and need pruning."),
]
print(len(TEMPLATES), "templates")


# ======================================================================
# 4. Generate complaints
# ======================================================================
random.seed(RANDOM_SEED)
NOW = dt.datetime.now().replace(second=0, microsecond=0)
N_HIST, N_NEW = 480, 20
CHANNELS = ["web", "whatsapp", "call_centre", "mobile_app"]

def pick_template(dept=None, exclude=None):
    if dept is None:
        dept = random.choices(list(DEPT_WEIGHTS), weights=list(DEPT_WEIGHTS.values()))[0]
    pool = [t for t in TEMPLATES if t[0] == dept and t[3] != exclude] or [t for t in TEMPLATES if t[0] == dept]
    return random.choice(pool)

complaints, labels = [], []

def add(cid, created, tpl_row, loc, lm, status, dup_of=None, resolved_at=None, triaged=True):
    dept, lang, prio, tpl = tpl_row
    complaints.append({
        "complaint_id": cid, "created_at": created, "locality": loc, "landmark": lm,
        "text": tpl.format(loc=loc, lm=lm, days=random.randint(2, 6)),
        "channel": random.choice(CHANNELS), "status": status,
        "dept_code": dept if triaged else None, "priority": prio if triaged else None,
        "duplicate_of": dup_of if triaged else None, "resolved_at": resolved_at,
        "escalation_level": 0, "_template": tpl})
    labels.append({"complaint_id": cid, "true_dept": dept, "true_priority": prio,
                   "true_duplicate_of": dup_of, "language": lang})

# Historical complaints, oldest first
ages = sorted((random.uniform(1, 240) for _ in range(N_HIST)), reverse=True)
for i, age in enumerate(ages, start=1):
    cid, created = f"GGN-{i:04d}", NOW - dt.timedelta(hours=age)
    parents = [c for c in complaints if c["status"] in ("open", "in_progress")
               and dt.timedelta(0) < created - c["created_at"] <= dt.timedelta(hours=72)]
    if parents and random.random() < 0.10:
        p = random.choice(parents)
        add(cid, created, pick_template(p["dept_code"], exclude=p["_template"]),
            p["locality"], p["landmark"], "merged", dup_of=p["complaint_id"])
        continue
    row = pick_template()
    tgt = target_hours(row[0], row[2])
    if random.random() < min(0.92, age / (tgt * 1.3)):
        took = min(age * 0.95, random.uniform(0.3, 1.4) * tgt)
        status, resolved_at = "resolved", created + dt.timedelta(hours=took)
    else:
        status, resolved_at = random.choice(["open", "in_progress"]), None
    add(cid, created, row, random.choice(LOCALITIES), random.choice(LANDMARKS), status, resolved_at=resolved_at)

# New incoming complaints
open_recent = [c for c in complaints if c["status"] in ("open", "in_progress")
               and NOW - c["created_at"] <= dt.timedelta(hours=72)]
for j in range(N_NEW):
    cid = f"GGN-{N_HIST + j + 1:04d}"
    created = NOW - dt.timedelta(minutes=random.randint(5, 120))
    if open_recent and j % 4 == 0:
        p = random.choice(open_recent)
        add(cid, created, pick_template(p["dept_code"], exclude=p["_template"]),
            p["locality"], p["landmark"], "new", dup_of=p["complaint_id"], triaged=False)
        continue
    row = pick_template()
    for _ in range(30):  # make sure a "fresh" complaint does not accidentally match an open one
        loc, lm = random.choice(LOCALITIES), random.choice(LANDMARKS)
        if not any(c["dept_code"] == row[0] and c["locality"] == loc and c["landmark"] == lm for c in open_recent):
            break
    add(cid, created, row, loc, lm, "new", triaged=False)

complaints_df = pd.DataFrame(complaints).drop(columns=["_template"]).sort_values("created_at").reset_index(drop=True)
labels_df = pd.DataFrame(labels)
complaints_df.to_csv(DATA_DIR / "complaints.csv", index=False)
labels_df.to_csv(DATA_DIR / "complaint_labels.csv", index=False)

print(complaints_df["status"].value_counts().to_string(), "\n")
print(labels_df["language"].value_counts().to_string())
print(complaints_df.sample(8, random_state=1)[["complaint_id", "locality", "text", "status", "dept_code", "priority"]].to_string(index=False))


# ======================================================================
# 5. Load into PostgreSQL
# ======================================================================
with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS agent_actions, complaint_labels, complaints, departments CASCADE"))

departments_df.to_sql("departments", engine, index=False)
complaints_df.to_sql("complaints", engine, index=False)
labels_df.to_sql("complaint_labels", engine, index=False)

with engine.begin() as conn:
    conn.execute(text("ALTER TABLE departments ADD PRIMARY KEY (dept_code)"))
    conn.execute(text("ALTER TABLE complaints ADD PRIMARY KEY (complaint_id)"))
    conn.execute(text("ALTER TABLE complaint_labels ADD PRIMARY KEY (complaint_id)"))
    conn.execute(text("""
        CREATE TABLE agent_actions (
            action_id    SERIAL PRIMARY KEY,
            complaint_id TEXT,
            agent        TEXT,
            action       TEXT,
            detail       JSONB,
            rule_cited   TEXT,
            created_at   TIMESTAMP DEFAULT now()
        )"""))

print(pd.read_sql("""
    SELECT status, count(*) AS n FROM complaints GROUP BY status ORDER BY n DESC
""", engine).to_string(index=False))
# Preview: how many open complaints are already past their target time (the SLA monitor will act on these)
open_df = complaints_df[complaints_df.status.isin(["open", "in_progress"])].copy()
open_df["target_h"] = [target_hours(d, p) for d, p in zip(open_df.dept_code, open_df.priority)]
open_df["age_h"] = (NOW - open_df.created_at).dt.total_seconds() / 3600
open_df["breach_ratio"] = open_df.age_h / open_df.target_h
open_df["would_escalate_to"] = pd.cut(open_df.breach_ratio, [0, 1, 2, float("inf")], labels=["none", "L2", "L3"])
print("\nOpen complaints by escalation need:")
print(open_df.groupby(["dept_code", "would_escalate_to"], observed=False).size().unstack(fill_value=0))


# ======================================================================
# 6. Citizen charter (the rulebook)
# ======================================================================
CHARTER = """# Gurugram Municipal Services: Citizen Charter (fictional, for demo use)

## 1. Department responsibilities
CC-1.1 Roads (PWD_ROADS): potholes, broken roads, damaged footpaths, speed breakers, missing covers of road (non sewer) manholes.
CC-1.2 Water Supply (WATER): no water, low pressure, dirty or smelly tap water, pipeline leaks, illegal connections.
CC-1.3 Sewerage and Drainage (SEWER): sewer overflow, blocked drains, waterlogging after rain, open sewer manholes.
CC-1.4 Sanitation (SANITATION): garbage not collected, garbage heaps, garbage burning, dead animals, public toilet cleanliness.
CC-1.5 Streetlights (ELECTRICAL): streetlights not working, damaged poles, exposed or live wires on public poles.
CC-1.6 Parks and Trees (HORTICULTURE): fallen or dangerous trees, overgrown parks, tree pruning.

## 2. Boundary rules
CC-2.1 Waterlogging on a road after rain goes to SEWER, not WATER or PWD_ROADS.
CC-2.2 Dirty water from household taps goes to WATER even if a sewer leak may be the cause; WATER informs SEWER.
CC-2.3 A fallen tree or branch blocking a road goes to HORTICULTURE as lead, with priority critical.
CC-2.4 An open or missing manhole cover on a road goes to SEWER unless it is clearly not a sewer line.
CC-2.5 Garbage dumped in a park goes to SANITATION, not HORTICULTURE.
CC-2.6 Tree branches touching electric wires go to HORTICULTURE for pruning.

## 3. Priority levels
CC-3.1 Critical: immediate risk to life or safety (live wires, open manhole, fallen tree blocking traffic, sewage entering homes). Target 6 hours for any department.
CC-3.2 High: health or safety risk or service fully down (no water, sewer overflow on a road, accidents due to a pothole, a fully dark stretch, garbage burning, a tree about to fall). Target is half the department SLA.
CC-3.3 Normal: all other complaints. Target is the department SLA.

## 4. Department SLA for normal priority
CC-4.1 WATER 24 hours, SANITATION 24 hours, SEWER 48 hours, ELECTRICAL 48 hours, PWD_ROADS 72 hours, HORTICULTURE 120 hours.

## 5. Duplicate complaints
CC-5.1 A complaint is a duplicate if it reports the same issue type at the same locality and landmark as an open complaint created in the last 72 hours.
CC-5.2 Duplicates are merged into the earliest open complaint; the citizen receives the parent complaint ID and its current status.
CC-5.3 Five or more duplicates on one parent within 24 hours raise the parent priority by one level.

## 6. Escalation
CC-6.1 The Level 1 Junior Engineer owns a complaint from assignment.
CC-6.2 If the target time is breached, escalate to the Level 2 Executive Engineer.
CC-6.3 If the complaint is open for more than twice its target time, escalate to the Level 3 Joint Commissioner.
CC-6.4 Critical complaints are copied to Level 2 at the time of assignment.

## 7. Communication
CC-7.1 Acknowledge every complaint within 15 minutes with the complaint ID, department and target resolution time.
CC-7.2 Reply in the citizen's language (English, Hindi or Hinglish).
CC-7.3 Never promise a time earlier than the target and never blame another department.
CC-7.4 Do not include phone numbers or personal data in replies.
"""
(DATA_DIR / "citizen_charter.md").write_text(CHARTER, encoding="utf-8")

clauses, section = [], ""
for line in CHARTER.splitlines():
    if line.startswith("## "):
        section = line[3:].strip()
    m = re.match(r"^(CC-\d+\.\d+)\s+(.*)", line.strip())
    if m:
        clauses.append({"id": m.group(1), "section": section,
                        "text": f"{m.group(1)} ({section}): {m.group(2)}"})
print(len(clauses), "clauses parsed")


# ======================================================================
# 7. Index in ChromaDB (embedded mode, no server needed)
# ======================================================================
import chromadb
import ollama

oc = ollama.Client(host=CONFIG["ollama_url"])

def embed(texts, batch=16):
    vectors = []
    for i in range(0, len(texts), batch):
        vectors += oc.embed(model=CONFIG["embed_model"], input=texts[i:i + batch])["embeddings"]
    return vectors

chroma = chromadb.PersistentClient(path=CONFIG["chroma_dir"])

def fresh_collection(name):
    try:
        chroma.delete_collection(name)
    except Exception:
        pass
    try:
        return chroma.create_collection(name, configuration={"hnsw": {"space": "cosine"}})
    except Exception:
        return chroma.create_collection(name, metadata={"hnsw:space": "cosine"})

charter_col = fresh_collection("citizen_charter")
charter_col.add(ids=[c["id"] for c in clauses],
                documents=[c["text"] for c in clauses],
                metadatas=[{"section": c["section"]} for c in clauses],
                embeddings=embed([c["text"] for c in clauses]))
print("Charter clauses indexed:", charter_col.count())

idx = complaints_df[complaints_df.status.isin(["open", "in_progress"])]
complaint_col = fresh_collection("open_complaints")
complaint_col.add(
    ids=idx.complaint_id.tolist(),
    documents=idx.text.tolist(),
    metadatas=[{"locality": r.locality, "landmark": r.landmark, "dept_code": r.dept_code,
                "status": r.status, "created_ts": int(r.created_at.timestamp())} for r in idx.itertuples()],
    embeddings=embed(idx.text.tolist()))
print("Open complaints indexed:", complaint_col.count())


# ======================================================================
# 8. Sanity checks
# ======================================================================
# Does the charter search work across languages?
for q in ["बारिश के बाद सड़क पर पानी भर गया है", "tree fell on the road and cars cannot pass", "kooda roz shaam ko jalaya jata hai"]:
    res = charter_col.query(query_embeddings=embed([q]), n_results=2)
    print(f"{q}\n   -> {res['ids'][0]}")
# Can vector search find the parent of each planted duplicate?
new_df = complaints_df[complaints_df.status == "new"].merge(labels_df, on="complaint_id")
for r in new_df[new_df.true_duplicate_of.notna()].itertuples():
    res = complaint_col.query(query_embeddings=embed([r.text]), n_results=3,
                              where={"locality": r.locality})
    hit = "HIT " if r.true_duplicate_of in res["ids"][0] else "MISS"
    print(f"{hit} {r.complaint_id} true parent {r.true_duplicate_of}, top matches {res['ids'][0]}")
    print("     ", r.text)


# ======================================================================
# Done
# ======================================================================
