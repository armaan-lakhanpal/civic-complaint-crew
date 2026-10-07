"""citizen_app.py: Citizen complaint chat (AL_AABA civic crew)

Run:  streamlit run citizen_app.py --server.port 8501
A citizen picks their area, types a complaint in English, Hindi or Hinglish, and watches the
agents handle it live. They can also track any complaint by its ID.
"""
import datetime as dt
import importlib.util
import threading
import time
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="Gurugram Civic Helpdesk", page_icon="🏙️", layout="centered")


@st.cache_resource(show_spinner="Starting the agents...")
def load_crew():
    spec = importlib.util.spec_from_file_location("civic_crew", PROJECT_DIR / "02_crew.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


crew = load_crew()


def use_model(mode):
    if st.session_state.get("model_mode") != mode:
        try:
            crew.setup_agents(mode)
        except SystemExit as e:
            st.error(str(e)); st.stop()
        st.session_state.model_mode = mode


def next_complaint_id():
    n = crew.q("SELECT max(substring(complaint_id from 5)::int) AS n FROM complaints").n[0]
    return f"GGN-{int(n or 0) + 1:04d}"


def save_complaint(text, locality, landmark):
    cid = next_complaint_id()
    crew.run_sql("""INSERT INTO complaints (complaint_id, created_at, locality, landmark, text, channel,
                                            status, escalation_level)
                    VALUES (:c, :t, :loc, :lm, :txt, 'citizen_chat', 'new', 0)""",
                 c=cid, t=dt.datetime.now().replace(microsecond=0), loc=locality, lm=landmark, txt=text)
    return cid


STEP_TEXT = {
    "extracted": lambda d: f"🧾 **Intake agent** understood it as *{d.get('issue_type', '?')}* "
                           f"(language: {d.get('language')}, likely {d.get('likely_dept')})",
    "new_issue": lambda d: f"🔍 **Duplicate checker**: new issue ({d.get('verdict')})",
    "duplicate": lambda d: f"🔍 **Duplicate checker**: already reported ({d.get('verdict')})",
    "merged": lambda d: f"🔗 Linked to existing complaint **{d.get('parent')}** (rule CC-5.2)",
    "routed": lambda d: f"🧭 **Router agent**: {d.get('dept')}, priority **{d.get('priority')}**, "
                        f"target {d.get('target_hours'):.0f} h",
    "copied_to_level2": lambda d: f"🚨 Critical: copied to {d.get('officer')} (CC-6.4)",
    "acknowledged": lambda d: "✉️ **Communication agent** wrote your reply and sent it via n8n"
                              if d.get("sent_to_n8n") else "✉️ **Communication agent** wrote your reply",
}


def describe(row):
    fn = STEP_TEXT.get(row.action)
    text = fn(row.detail or {}) if fn else f"{row.agent}: {row.action}"
    rule = row.rule_cited if isinstance(row.rule_cited, str) else ""
    return text + (f"  `{rule}`" if rule else "")


def run_with_live_progress(cid):
    """Run the crew in a background thread and show each agent's step as soon as it is logged."""
    result = {}

    def worker():
        try:
            result["state"] = crew.process_complaint(cid, quiet=True)
        except Exception as e:  # shown to the user below
            result["error"] = e

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    seen, t0 = set(), time.time()
    with st.status("Our agents are working on your complaint...", expanded=True) as status:
        while True:
            rows = crew.q("""SELECT action_id, agent, action, rule_cited, detail FROM agent_actions
                             WHERE complaint_id = :c ORDER BY action_id""", c=cid)
            for r in rows.itertuples():
                if r.action_id not in seen:
                    seen.add(r.action_id)
                    st.write(describe(r))
            if not th.is_alive():
                break
            status.update(label=f"Our agents are working on your complaint... ({time.time() - t0:.0f} s)")
            time.sleep(2)
        if "error" in result:
            status.update(label="Something went wrong", state="error")
        else:
            status.update(label=f"Done in {time.time() - t0:.0f} s", state="complete", expanded=False)
    return result


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Your details")
    locality = st.selectbox("Your area", sorted(crew.LOCALITIES))
    landmark = st.selectbox("Nearest landmark", sorted(crew.LANDMARKS) + ["Not sure"])
    st.divider()
    mode = st.radio("AI model", ["local", "api"],
                    format_func=lambda m: "Local (Ollama, free)" if m == "local" else f"API: {crew.API_MODEL} (fast)")
    st.caption("Local model: about 2 to 3 minutes per complaint on CPU.")
    st.divider()
    st.subheader("Track a complaint")
    track_id = st.text_input("Complaint ID", placeholder="GGN-0501").strip().upper()
    if track_id:
        t = crew.q("""SELECT c.status, c.priority, c.duplicate_of, c.created_at, c.dept_code, d.dept_name
                      FROM complaints c LEFT JOIN departments d USING (dept_code)
                      WHERE c.complaint_id = :c""", c=track_id)
        if t.empty:
            st.warning("No complaint with that ID.")
        else:
            r = t.iloc[0]
            st.write(f"**Status:** {r.status}")
            if r.dept_name:
                due = r.created_at + dt.timedelta(hours=crew.target_hours(r.dept_code, r.priority))
                st.write(f"**Department:** {r.dept_name}")
                st.write(f"**Target:** {due:%d %b, %I:%M %p}")
            if r.duplicate_of:
                st.write(f"**Linked to:** {r.duplicate_of}")

# ---------------------------------------------------------------- chat
st.title("🏙️ Gurugram Civic Helpdesk")
st.caption("Report a civic problem in English, हिंदी or Hinglish. Demo system with fictional data.")

if "chat" not in st.session_state:
    st.session_state.chat = [{"role": "assistant",
                              "content": "Namaste! Choose your area on the left, then tell me the problem "
                                         "(pothole, water, garbage, streetlight, sewer, trees...)."}]
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Describe the problem...")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    text = prompt
    if locality.lower() not in prompt.lower():
        text += f" ({locality}" + (f", near {landmark})" if landmark != "Not sure" else ")")
    lm = landmark if landmark != "Not sure" else ""

    use_model(mode)
    cid = save_complaint(text, locality, lm)
    with st.chat_message("assistant"):
        st.markdown(f"Complaint registered as **{cid}**.")
        result = run_with_live_progress(cid)
        if "error" in result:
            reply = f"Sorry, something went wrong while processing {cid}: `{result['error']}`"
        else:
            s = result["state"]
            reply = s.message
            facts = (f"Linked to **{s.duplicate_of}**" if s.duplicate_of
                     else f"{crew.DEPTS[s.dept_code]['dept_name']} · priority **{s.priority}**")
            reply += f"\n\n<small>{facts} · target {s.due_at[:16]}</small>"
        st.markdown(reply, unsafe_allow_html=True)
    st.session_state.chat.append({"role": "assistant", "content": f"Complaint **{cid}**: {reply}"})