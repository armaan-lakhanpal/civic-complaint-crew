"""department_app.py: Department dashboard (AL_AABA civic crew)

Run:  streamlit run department_app.py --server.port 8502
For municipal staff: live board of open complaints and deadlines, status updates,
the agents' audit trail, n8n notifications, the SLA monitor and accuracy evaluation.
"""
import datetime as dt
import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="Civic Ops Dashboard", page_icon="🏛️", layout="wide")


@st.cache_resource(show_spinner="Connecting to the crew...")
def load_crew():
    spec = importlib.util.spec_from_file_location("civic_crew", PROJECT_DIR / "02_crew.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


crew = load_crew()
DEPT_NAMES = {k: v["dept_name"] for k, v in crew.DEPTS.items()}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🏛️ Department view")
    dept_choice = st.selectbox("Department", ["ALL"] + list(DEPT_NAMES),
                               format_func=lambda k: "All departments" if k == "ALL" else DEPT_NAMES[k])
    if st.button("🔄 Refresh", width="stretch"):
        st.rerun()
    st.caption(f"Last refreshed {dt.datetime.now():%H:%M:%S}")


def load_complaints():
    df = crew.q("""SELECT complaint_id, created_at, locality, landmark, text, channel, status, dept_code,
                          priority, duplicate_of, escalation_level, resolved_at
                   FROM complaints""")
    df["created_at"] = pd.to_datetime(df.created_at)
    now = pd.Timestamp.now()
    has = df.dept_code.notna() & df.priority.notna()
    df.loc[has, "target_h"] = [crew.target_hours(d, p) for d, p in zip(df.dept_code[has], df.priority[has])]
    df["due_at"] = df.created_at + pd.to_timedelta(df.target_h, unit="h")
    df["hours_left"] = ((df.due_at - now).dt.total_seconds() / 3600).round(1)
    df["department"] = df.dept_code.map(DEPT_NAMES)
    return df


df = load_complaints()
view = df if dept_choice == "ALL" else df[df.dept_code == dept_choice]
open_ = view[view.status.isin(["open", "in_progress"])]
ai_ids = set(crew.q("SELECT DISTINCT complaint_id FROM agent_actions WHERE agent = 'intake'").complaint_id)

st.title("Civic Operations Dashboard")
st.caption("Gurugram Municipal Services (fictional demo data) · decisions made by the CrewAI agents, delivered by n8n")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Open", len(open_))
k2.metric("Overdue", int((open_.hours_left < 0).sum()))
k3.metric("Critical open", int((open_.priority == "critical").sum()))
k4.metric("Waiting for AI", int((view.status == "new").sum()))
k5.metric("Handled by AI", len(ai_ids & set(view.complaint_id)))
k6.metric("Duplicates merged", int((view.status == "merged").sum()))

tab_board, tab_charts, tab_audit, tab_notif, tab_sla, tab_eval = st.tabs(
    ["📋 Open complaints", "📊 Charts", "🧾 Audit trail", "📨 Notifications", "⏰ SLA monitor", "✅ Evaluation"])

# ---------------------------------------------------------------- board
with tab_board:
    board = open_.sort_values("hours_left").copy()
    board["deadline"] = board.hours_left.apply(
        lambda h: f"🔴 {abs(h):.0f} h overdue" if h < 0 else (f"🟠 {h:.0f} h left" if h < 6 else f"🟢 {h:.0f} h left"))
    board["AI"] = board.complaint_id.isin(ai_ids).map({True: "🤖", False: ""})
    board["escalated"] = board.escalation_level.map({0: "", 2: "L2", 3: "L3"}).fillna("")
    st.dataframe(board[["AI", "complaint_id", "department", "priority", "status", "deadline", "escalated",
                        "locality", "landmark", "text"]],
                 hide_index=True, width="stretch", height=420)

    st.subheader("Update a complaint")
    c1, c2, c3 = st.columns([2, 2, 1])
    pick = c1.selectbox("Complaint", board.complaint_id.tolist() or ["(none open)"])
    new_status = c2.selectbox("New status", ["in_progress", "resolved"])
    if c3.button("Update", width="stretch") and pick.startswith("GGN"):
        resolved_at = dt.datetime.now().replace(microsecond=0) if new_status == "resolved" else None
        crew.run_sql("UPDATE complaints SET status = :s, resolved_at = :r WHERE complaint_id = :c",
                     s=new_status, r=resolved_at, c=pick)
        if new_status == "resolved":
            crew.complaint_col.delete(ids=[pick])  # no longer a candidate for duplicate matching
        row = df[df.complaint_id == pick].iloc[0]
        msg = (f"Update on complaint {pick}: the {row.department} department has marked it "
               f"{'resolved' if new_status == 'resolved' else 'in progress'}.")
        crew.log_action(pick, "department", f"status_{new_status}",
                        {"by": dept_choice, "from": str(df.loc[df.complaint_id == pick, "status"].iloc[0])})
        sent = crew.send_to_n8n({"type": "status_update", "complaint_id": pick, "language": "en",
                                 "priority": row.priority, "message": msg})
        st.session_state.flash = f"{pick} marked {new_status}. Citizen notified via n8n: {'yes' if sent else 'no'}"
        st.rerun()
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))

# ---------------------------------------------------------------- charts
with tab_charts:
    c1, c2 = st.columns(2)
    c1.markdown("**Open complaints by department**")
    c1.bar_chart(open_.groupby("department").size().rename("open"))
    c2.markdown("**Open complaints by priority**")
    c2.bar_chart(open_.groupby("priority").size().reindex(["critical", "high", "normal"]).fillna(0).rename("open"))
    c3, c4 = st.columns(2)
    c3.markdown("**Overdue vs on time (open)**")
    c3.bar_chart(open_.assign(state=open_.hours_left.lt(0).map({True: "overdue", False: "on time"}))
                 .groupby(["department", "state"]).size().unstack(fill_value=0))
    c4.markdown("**Complaints received per day**")
    c4.line_chart(view.groupby(view.created_at.dt.date).size().rename("complaints"))

# ---------------------------------------------------------------- audit trail
with tab_audit:
    st.markdown("Every agent decision, the charter rule it cited, and any guardrail override.")
    handled = sorted(ai_ids & set(view.complaint_id), reverse=True)
    if not handled:
        st.info("No complaints processed by the agents yet.")
    else:
        cid = st.selectbox("Complaint", handled)
        c = df[df.complaint_id == cid].iloc[0]
        st.markdown(f"> {c.text}")
        st.caption(f"{c.locality} · {c.landmark} · via {c.channel} · {c.created_at:%d %b %H:%M}")
        acts = crew.q("""SELECT created_at, agent, action, rule_cited, detail FROM agent_actions
                         WHERE complaint_id = :c ORDER BY action_id""", c=cid)
        for a in acts.itertuples():
            d = a.detail or {}
            override = (d.get("verdict") and "overrid" in d.get("verdict", "")) or \
                       (d.get("verdict") and "missed" in d.get("verdict", "")) or d.get("fallback_used")
            icon = "🛡️" if override else "✅"
            with st.expander(f"{icon} {a.created_at:%H:%M:%S} · {a.agent} · {a.action}"
                             + (f" · `{a.rule_cited}`" if isinstance(a.rule_cited, str) else "")):
                if override:
                    st.warning("Guardrail stepped in: "
                               + str(d.get("verdict") or "; ".join(d.get("fallback_used", []))))
                st.json(d)

    st.markdown("**Guardrail overrides across all complaints**")
    g = crew.q("""SELECT complaint_id, agent, detail FROM agent_actions
                  WHERE (agent = 'duplicate_checker' AND (detail->>'verdict' ILIKE '%overrid%'
                                                          OR detail->>'verdict' ILIKE '%missed%'
                                                          OR detail->>'verdict' ILIKE '%corrected%'))
                     OR (agent = 'router' AND jsonb_array_length(COALESCE(detail->'fallback_used', '[]')) > 0)""")
    if g.empty:
        st.caption("None so far.")
    else:
        g["what happened"] = g.detail.apply(lambda d: d.get("verdict") or "; ".join(d.get("fallback_used", [])))
        st.dataframe(g[["complaint_id", "agent", "what happened"]], hide_index=True, width="stretch")

# ---------------------------------------------------------------- notifications
with tab_notif:
    st.markdown("Messages delivered through the **n8n webhook** (stored by the n8n workflow).")
    try:
        n = crew.q("SELECT sent_at, channel, complaint_id, recipient, language, message FROM notifications "
                   "ORDER BY sent_at DESC LIMIT 200")
    except Exception:
        n = pd.DataFrame()
    if n.empty:
        st.info("No notifications yet. Check that n8n is running and the workflow is published.")
    else:
        kind = st.radio("Show", ["all", "citizen_reply", "officer_alert"], horizontal=True)
        if kind != "all":
            n = n[n.channel == kind]
        for r in n.itertuples():
            icon = "🚨" if r.channel == "officer_alert" else "💬"
            with st.expander(f"{icon} {r.sent_at:%d %b %H:%M} · {r.complaint_id[:60]} · {r.recipient}"):
                st.text(r.message)

# ---------------------------------------------------------------- SLA monitor
with tab_sla:
    st.markdown("Finds open complaints past their target time and escalates them: "
                "level 2 when the target is breached (CC-6.2), level 3 when open for more than twice the "
                "target (CC-6.3). The officer lists are computed exactly; the agent writes the covering note.")
    mode = st.radio("Model for the covering note", ["local", "api"], horizontal=True)
    if st.button("Run SLA monitor now", type="primary"):
        try:
            crew.setup_agents(mode)
        except SystemExit as e:
            st.error(str(e)); st.stop()
        with st.spinner("SLA monitor agent is writing the escalation note..."):
            res = crew.run_sla_monitor(quiet=True)
        st.session_state.sla_result = res
    res = st.session_state.get("sla_result")
    if res:
        if res["count"] == 0:
            st.success("No new breaches.")
        else:
            st.success(f"{res['count']} complaints escalated. Sent via n8n: {'yes' if res['sent'] else 'no'}")
            st.text(res["note"])

# ---------------------------------------------------------------- evaluation
with tab_eval:
    st.markdown("Agent decisions compared with the answer key (`complaint_labels`), "
                "for the synthetic complaints only.")
    m, ev = crew.evaluation_data()
    if not m:
        st.info("Nothing processed yet.")
    else:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Processed", m["processed"])
        e2.metric("Department accuracy", f"{m['dept_accuracy']:.0%}")
        e3.metric("Priority accuracy", "n/a" if m["priority_accuracy"] is None else f"{m['priority_accuracy']:.0%}")
        e4.metric("Duplicates caught", f"{m['dups_caught']} / {m['dups_planted']}",
                  delta=f"{m['false_alarms']} false alarms", delta_color="inverse")
        st.dataframe(ev[["complaint_id", "language", "dept_code", "true_dept", "priority", "true_priority",
                         "duplicate_of", "true_duplicate_of", "dept_ok", "priority_ok"]],
                     hide_index=True, width="stretch")
