"""SENTINEL AI - Autonomous Inventory Intelligence & Decision Centre.

Interactive Streamlit + HTML/CSS + Plotly dashboard for the RDMU lab exercise.
Launch from the project root with:

    streamlit run dashboard/app.py

Every probability, cost and recommendation shown here is computed by the
shared engine in `src/` (the same modules the notebook imports). This file
contains layout and interaction wiring only.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import pandas as pd
import streamlit as st

import components as ui
from src import visualization as viz
from src.config import (ACTION_KEYS, ACTION_SHORT, ACTIONS, CHANNEL_KEYS, CHANNELS, COST_UNIT,
                        DEMAND_MEAN_UNITS, DEMAND_SD_UNITS, STATES, default_config, inventory_band_label)
from src.decision_policy import (BeliefMyopicPolicy, FixedPolicy, InfographicThresholdPolicy, OraclePolicy,
                                 PeriodicReorderPolicy, RecordTrustingPolicy, policy_region_grid)
from src.inventory_simulator import evaluate_policies
from src.metrics import (BELIEF_COLUMNS, calibration_table, expected_calibration_error, gap_closed,
                         summarise_replications)
from src.pomdp_engine import SentinelEngine
from src.scenarios import DASHBOARD_PRESETS, SCENARIOS

st.set_page_config(page_title="SENTINEL AI Inventory Dashboard", page_icon="🤖", layout="wide",
                   initial_sidebar_state="collapsed")

AGENT_NAME = "SENTINEL belief agent"
BASELINE_NAME = "Record-trusting heuristic"
ORACLE_NAME = "Full-information myopic oracle"
EXEC_OPTIONS = ["recommended"] + list(ACTION_KEYS)


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    if "engine" not in ss:
        ss.engine = SentinelEngine(default_config())
    ss.setdefault("auto_running", False)
    ss.setdefault("flash", None)
    ss.setdefault("cfg_version", 0)
    ss.setdefault("batch", None)
    ss.setdefault("batch_request", False)
    ss.setdefault("alert_channel", CHANNEL_KEYS[0])
    ss.setdefault("dedupe_guard", True)
    ss.setdefault("observe_env", True)
    ss.setdefault("exec_choice", "recommended")
    ss.setdefault("auto_interval", 2.0)
    ss.setdefault("alert_note", "")
    ss.setdefault("scenario_key", DASHBOARD_PRESETS[0])
    ss.setdefault("reveal_truth", False)
    ss.setdefault("batch_reps", 60)


def engine() -> SentinelEngine:
    return st.session_state.engine


def flash(kind: str, text: str) -> None:
    st.session_state.flash = (kind, text)


# ----------------------------------------------------------------------------
# Callbacks (every control is wired to the shared engine)
# ----------------------------------------------------------------------------
def cb_inject() -> None:
    ss = st.session_state
    channel = ss.alert_channel
    try:
        result = engine().inject_alert(channel, int(ss[f"alert_obs_{channel}"]),
                                       reliability=float(ss[f"alert_rel_{channel}"]) / 100.0,
                                       origin="manual", deduplicate=bool(ss.dedupe_guard), note=ss.alert_note)
    except ValueError as error:
        flash("error", f"Alert rejected: {error}")
        return
    label = f"{CHANNELS[channel]['label']}: {CHANNELS[channel]['observations'][int(ss[f'alert_obs_{channel}'])]}"
    if result["status"] == "duplicate ignored":
        flash("warning", f"Duplicate ignored — {result['note']} Advance the simulation for a new independent "
                         f"reading, or switch the duplicate guard off to see double counting.")
    elif result["status"].startswith("rejected"):
        flash("error", result["note"])
    else:
        change = (f"Recommendation changed to {ACTIONS[result['new_action']]}." if result["policy_changed"]
                  else f"Recommendation unchanged ({ACTIONS[result['new_action']]}).")
        extra = f" {result['note']}" if result["note"] else ""
        kind = "warning" if "guard off" in result["status"] else "success"
        flash(kind, f"Alert injected — {label}. P(Critical) {100 * result['belief_before'][2]:.1f}% → "
                    f"{100 * result['belief_after'][2]:.1f}%. {change}{extra}")


def _execute_step() -> dict:
    choice = st.session_state.exec_choice
    action = None if choice == "recommended" else choice
    return engine().run_next_step(action=action, observe=bool(st.session_state.observe_env))


def cb_step() -> None:
    try:
        out = _execute_step()
    except ValueError as error:
        flash("error", f"Step failed: {error}")
        return
    followed = "the recommendation" if out["executed"] == out["recommended"] else "a manual override"
    seen = "; ".join(out["observations"].values()) if out["observations"] else "no environment observations requested"
    flash("success", f"Period {engine().step}: executed {ACTIONS[out['executed']]} ({followed}). Evidence: {seen}. "
                     f"New recommendation: {ACTIONS[out['new_recommendation']]}.")


def cb_start() -> None:
    if engine().step >= int(engine().config.horizon):
        flash("warning", "Simulation duration already reached. Reset the simulation or raise the duration "
                         "in the What-If Lab.")
        return
    st.session_state.auto_running = True
    flash("info", f"Auto-simulation started: one period every {st.session_state.auto_interval:.0f} s "
                  f"until period {engine().config.horizon}.")


def cb_pause() -> None:
    was_running = st.session_state.auto_running
    st.session_state.auto_running = False
    flash("info", "Auto-simulation paused." if was_running else "Auto-simulation is not running.")


def cb_reset() -> None:
    st.session_state.auto_running = False
    engine().reset()
    flash("info", "Simulation reset: prior belief, logs and hidden simulator restored to the initial configuration.")


def cb_load_scenario() -> None:
    st.session_state.auto_running = False
    scenario = SCENARIOS[st.session_state.scenario_key]
    engine().play_scenario(scenario)
    flash("success", f"Preset loaded — {scenario.name}: {len(engine().event_log)} scripted alerts over "
                     f"{engine().step + 1} period(s). Recommendation: {engine().decision.action_name}.")


def cb_restore_defaults() -> None:
    st.session_state.auto_running = False
    engine().apply_config(default_config())
    st.session_state.cfg_version += 1
    st.session_state.batch = None
    flash("info", "Default model configuration restored and all outputs recomputed.")


def cb_request_batch() -> None:
    st.session_state.batch_request = True


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def chart(fig, key: str, alt: str) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config=viz.CHART_CONFIG, key=key)
    st.caption(alt)


def show_flash() -> None:
    message = st.session_state.flash
    if not message:
        return
    kind, text = message
    {"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}[kind](text)
    st.session_state.flash = None


def batch_factories() -> dict:
    return {
        AGENT_NAME: lambda m: BeliefMyopicPolicy(m.transition, m.cost),
        BASELINE_NAME: lambda m: RecordTrustingPolicy(),
        "Infographic threshold rule": lambda m: InfographicThresholdPolicy(),
        "Fixed: always reorder": lambda m: FixedPolicy("reorder"),
        "Fixed: reorder every 3 periods": lambda m: PeriodicReorderPolicy(3),
        ORACLE_NAME: lambda m: OraclePolicy(m.transition, m.cost),
    }


def debug_table(truth: pd.DataFrame) -> pd.DataFrame:
    """Readable version of the hidden ground-truth log (evaluation view only)."""
    def action_name(value):
        return "—" if pd.isna(value) else ACTIONS[int(value)]
    table = pd.DataFrame({
        "Period": truth["step"],
        "Date": pd.to_datetime(truth["date"]).dt.strftime("%Y-%m-%d"),
        "Hidden state": truth["hidden_state_name"],
        "Action executed into period": truth["action_into_period"].map(action_name),
        "Inventory snapshot (units, proxy)": truth["inventory_units"].round(0).astype(int),
        "Demand (units)": truth["demand_units"].round(0).astype(int),
        "Units short": truth["units_short"].round(0).astype(int),
        "Fill rate": (100 * truth["fill_rate"]).map("{:.1f}%".format),
        "Stock-out": truth["stockout"].map({True: "yes", False: "no"}),
    })
    if "b_healthy" in truth:
        table["Agent belief at decision (H / L / C)"] = [
            "—" if pd.isna(h) else f"{100 * h:.1f}% / {100 * l:.1f}% / {100 * c:.1f}%"
            for h, l, c in zip(truth["b_healthy"], truth["b_low"], truth["b_critical"])]
        table[f"Realised cost ({COST_UNIT})"] = truth["realised_cost"].map(
            lambda v: "—" if pd.isna(v) else f"{v:.0f}")
    return table.iloc[::-1]


def run_batch(reps: int) -> dict:
    cfg = engine().config
    results, trajectories = evaluate_policies(cfg, batch_factories(), n_replications=reps,
                                              horizon=int(cfg.horizon), seed=int(cfg.seed), keep_trajectories=True)
    return {"results": results, "summary": summarise_replications(results), "agent": trajectories[AGENT_NAME],
            "reps": reps, "horizon": int(cfg.horizon), "seed": int(cfg.seed)}


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
init_state()
st.session_state["_in_full_run"] = True
eng = engine()
st.html(f"<style>{ui.load_css()}</style>")

kpis = eng.kpis()
decision = eng.decision
explanation = eng.explanation()
timeline = eng.timeline_frame()

# ---- SECTION 1: command-centre header ---------------------------------------
st.html(ui.header_html(kpis["period"], kpis["timestamp"].strftime("%Y-%m-%d %H:%M"),
                       st.session_state.auto_running, int(eng.config.horizon)))

# ---- SECTION 2: executive KPI overview --------------------------------------
st.html(ui.kpi_cards_html(kpis, COST_UNIT, ACTIONS[decision.runner_up], decision.margin))

# ---- SECTION 3: real-time belief-state monitor ------------------------------
with st.container(key="row_belief"):
    col_a, col_b = st.columns([5, 7], gap="medium")
    with col_a:
        with st.container(key="panel_belief"):
            st.html(ui.section_title("03", "Real-Time Belief-State Monitor",
                                     "The agent never sees the true inventory state. This is its current "
                                     "probability distribution over the three hidden conditions."))
            st.html(ui.belief_monitor_html(eng.belief))
            st.html(ui.note(f"Belief concentration <b>{100 * kpis['concentration']:.0f} / 100</b> · expected "
                            f"stock-out risk <b>{100 * kpis['stockout_risk']:.1f}%</b> · expected service level "
                            f"<b>{100 * kpis['expected_service_level']:.1f}%</b> (belief-weighted model values)."))
    with col_b:
        with st.container(key="panel_evolution"):
            st.html(ui.section_title("03b", "Belief-State Evolution",
                                     "Each point is one event: an injected alert, a prediction after an executed "
                                     "action, or a joint correction. Hover for the event, timestamp and exact values."))
            chart(viz.fig_belief_evolution(timeline, height=430), "chart_evolution",
                  "Stacked area: the three beliefs always sum to 100%. A widening coral band means the evidence "
                  "is shifting probability towards Critical inventory.")

# ---- SECTION 4 + 5: alert control room and AI decision centre ---------------
with st.container(key="row_control"):
    col_a, col_b = st.columns([6, 6], gap="medium")
    with col_a:
        with st.container(key="panel_control"):
            st.html(ui.section_title("04", "Alert & Observation Control Room",
                                     "Inject evidence by hand, or let the simulated environment emit it. An alert "
                                     "only corrects the belief; nothing is executed until you run the next step."))
            show_flash()
            st.radio("Information channel", CHANNEL_KEYS, key="alert_channel", horizontal=True,
                     format_func=lambda k: f"{CHANNELS[k]['icon']} {CHANNELS[k]['label']}")
            channel = st.session_state.alert_channel
            meta = CHANNELS[channel]
            c1, c2 = st.columns([7, 5])
            with c1:
                st.selectbox("Observation · alert severity grade", list(range(len(meta["observations"]))),
                             key=f"alert_obs_{channel}",
                             format_func=lambda i, meta=meta: f"{meta['observations'][i]} · {meta['severity'][i]}",
                             help="Each severity level is a distinct observation symbol with its own likelihood "
                                  "column, so severity enters the update through Bayes' rule, not a multiplier.")
            with c2:
                st.slider("Observation reliability (%)", 0, 100,
                          int(round(100 * eng.observation_model.reliability[channel])), 5,
                          key=f"alert_rel_{channel}",
                          help="Probability that this report is a genuine reading rather than state-independent "
                               "noise. 0% makes the alert uninformative.")
            c1, c2 = st.columns([7, 5])
            with c1:
                st.text_input("Optional operator note (stored in the event log)", key="alert_note",
                              max_chars=200, placeholder="e.g. phone call from supplier account manager")
            with c2:
                st.toggle("Duplicate-event prevention", key="dedupe_guard",
                          help="On: one reading per channel per period (repeats ignored, revisions replace). "
                               "Off: repeats are counted as independent evidence and inflate confidence.")
            st.button("INJECT ALERT", key="btn_inject", type="primary", on_click=cb_inject, width="stretch",
                      help="Apply this observation as evidence in the current period.")

            st.html(ui.label("Simulation event progression"))
            c1, c2 = st.columns([8, 4])
            with c1:
                st.selectbox("Action to execute on the next step", EXEC_OPTIONS, key="exec_choice",
                             format_func=lambda k: "NOVA's current recommendation" if k == "recommended"
                             else f"Manual override: {ACTIONS[ACTION_KEYS.index(k)]}")
            with c2:
                st.number_input("Auto-run interval (s)", 1.0, 10.0, step=1.0, key="auto_interval")
            st.toggle("Simulated environment emits one observation per channel after each step", key="observe_env",
                      help="After the executed action, the hidden simulator emits one reading per channel, "
                           "combined in a single joint correction.")
            b1, b2 = st.columns(2)
            b1.button("RUN NEXT STEP", key="btn_step", on_click=cb_step, width="stretch",
                      help="Execute the selected action and advance one period.")
            b2.button("RESET SIMULATION", key="btn_reset", on_click=cb_reset, width="stretch")
            b3, b4 = st.columns(2)
            b3.button("START AUTO-SIMULATION", key="btn_start", on_click=cb_start, width="stretch",
                      disabled=st.session_state.auto_running)
            b4.button("PAUSE SIMULATION", key="btn_pause", on_click=cb_pause, width="stretch",
                      disabled=not st.session_state.auto_running)

            st.html(ui.label(f"Evidence counted in period {eng.step} (ranked by belief shift)"))
            st.html(ui.evidence_table_html(eng.evidence_summary()))

            st.html(ui.label("Event log (newest first)"))
            log = eng.event_log_frame()
            if log.empty:
                st.html(ui.note("No observations yet. Inject an alert or run a step to populate the log."))
            else:
                st.dataframe(log.iloc[::-1], hide_index=True, height=250, width="stretch", key="event_log_table")
    with col_b:
        with st.container(key="panel_decision"):
            st.html(ui.section_title("05", "AI Decision Intelligence Centre",
                                     "The recommendation is the arg-min of the expected one-step cost over all "
                                     "four actions under the current belief."))
            executed_note = ("RECOMMENDED — NOT EXECUTED. Use “Run next step” to execute an action."
                             if not st.session_state.auto_running else
                             "RECOMMENDED — auto-simulation executes it at the next tick.")
            st.html(ui.recommendation_html(
                explanation["headline"], explanation["paragraph"], explanation["nova"],
                explanation["main_evidence"], eng.belief, decision.expected_cost, decision.margin,
                ACTIONS[decision.runner_up], COST_UNIT, kpis["timestamp"].strftime("%Y-%m-%d %H:%M"), executed_note))
            st.html(ui.label("Comparison with alternative actions"))
            st.html(ui.alternatives_table_html(ACTIONS, decision.q_values, decision.action,
                                               decision.predicted_beliefs[:, 2], COST_UNIT))

# ---- cost comparison and operational centre ----------------------------------
with st.container(key="row_cost"):
    col_a, col_b = st.columns([6, 6], gap="medium")
    with col_a:
        with st.container(key="panel_cost"):
            st.html(ui.section_title("05b", "Expected Cost by Operational Action",
                                     "Q(b, a) = direct cost + expected current-state cost + expected "
                                     "post-transition cost. The outlined bar is the engine's arg-min."))
            chart(viz.fig_action_costs(decision, height=430), "chart_costs",
                  "The winning bar is chosen by the Python engine, then outlined and labelled; it is not picked by "
                  "a visual threshold. Grey = holding/shortage exposure that remains after the action.")
    with col_b:
        with st.container(key="panel_ops"):
            st.html(ui.section_title("07", "Operational Action & Performance Centre",
                                     "Recommended actions are advice; executed actions change the hidden state. "
                                     "The two are tracked separately."))
            history = eng.action_history()
            tab_time, tab_trend, tab_log = st.tabs(["Action timeline", "Recovery & risk trends", "Action log"])
            with tab_time:
                chart(viz.fig_policy_timeline(timeline, history, height=380), "chart_policy_timeline",
                      "Cyan line: the standing recommendation after every event. Stars: actions actually executed "
                      "(× marks a manual override that differed from the recommendation).")
            with tab_trend:
                chart(viz.fig_recovery_trend(timeline, height=330), "chart_recovery",
                      "Belief-based recovery: P(Healthy) should rise and P(Critical) fall after a replenishment "
                      "action (triangles) if the action worked.")
                chart(viz.fig_risk_trend(timeline, height=330), "chart_risk",
                      "Predicted outcome measures: belief-weighted stock-out risk and fill rate implied by the model's "
                      "per-state snapshot and demand assumptions (not a physical stock projection).")
            with tab_log:
                n_exec = len(history)
                orders = int((history["executed"] != 0).sum()) if n_exec else 0
                overrides = int((history["executed"] != history["recommended"]).sum()) if n_exec else 0
                spent = float(history["expected_cost"].sum()) if n_exec else 0.0
                st.html(ui.simple_table_html(
                    ["Operational measure", "Value"],
                    [["Periods executed", f"{n_exec}"],
                     ["Replenishment activity (non-monitoring actions executed)", f"{orders}"],
                     ["Manual overrides of the recommendation", f"{overrides}"],
                     ["Policy switches (recommendation changes)", f"{kpis['policy_changes']}"],
                     [f"Estimated operational cost so far (sum of expected costs, {COST_UNIT})", f"{spent:.1f}"]]))
                actions_frame = eng.action_log_frame()
                if actions_frame.empty:
                    st.html(ui.note("No action has been executed yet. Recommendations alone never change the "
                                    "simulated inventory."))
                else:
                    st.dataframe(actions_frame.iloc[::-1], hide_index=True, height=260, width="stretch",
                                 key="action_log_table")

# ---- SECTION 6: POMDP model explorer -----------------------------------------
with st.container(key="panel_model"):
    st.html(ui.section_title("06", "POMDP Model Explorer",
                             "POMDP = (S, A, T, O, Z, R, γ). Inspect every component the agent reasons with. "
                             "All values are synthetic simulation assumptions."))
    tab_t, tab_z, tab_b, tab_p, tab_a = st.tabs([
        "A · State transition matrix", "B · Observation likelihoods", "C · Bayesian update explorer",
        "Policy map", "D · Model assumptions"])
    with tab_t:
        c1, c2 = st.columns([7, 5], gap="medium")
        with c1:
            action_key = st.selectbox("Action", ACTION_KEYS, key="explore_action",
                                      format_func=lambda k: ACTIONS[ACTION_KEYS.index(k)])
            chart(viz.fig_transition_heatmap(eng.transition_model.matrix(action_key),
                                             ACTIONS[ACTION_KEYS.index(action_key)], height=380),
                  "chart_transition",
                  "Rows are the current state, columns the next state; each row sums to 100%.")
        with c2:
            m = eng.transition_model.matrix(action_key)
            st.html(ui.note(
                f"<b>Reading the matrix.</b> If inventory is Critical and this action is executed, it stays "
                f"Critical with probability <b>{100 * m[2, 2]:.1f}%</b> and recovers to Healthy with "
                f"<b>{100 * m[2, 0]:.1f}%</b>. From Healthy it slips to Low or Critical with "
                f"<b>{100 * (m[0, 1] + m[0, 2]):.1f}%</b>."))
            st.html(ui.note(
                f"<b>Environment modifiers in force.</b> Demand volatility κ = {eng.config.demand_volatility:.2f} "
                f"(extra one-level depletion per period); supplier delivery risk ρ = {eng.config.supplier_risk:.2f} "
                f"(probability that a reorder or expedite fails to land). The matrix shown already includes both."))
            st.html(ui.note("The transition model is applied to the belief <b>only when an action is executed</b> "
                            "(Run next step). Injected alerts never trigger it."))
    with tab_z:
        c1, c2 = st.columns([7, 5], gap="medium")
        with c1:
            view_channel = st.radio("Observation model", CHANNEL_KEYS, key="explore_channel", horizontal=True,
                                    format_func=lambda k: CHANNELS[k]["label"])
            view_rel = st.slider("Reliability shown (%)", 0, 100,
                                 int(round(100 * eng.observation_model.reliability[view_channel])), 5,
                                 key=f"explore_rel_{view_channel}")
            chart(viz.fig_observation_heatmap(
                eng.observation_model.effective_matrix(view_channel, view_rel / 100.0),
                eng.observation_model.labels(view_channel), CHANNELS[view_channel]["label"], view_rel / 100.0,
                height=380), "chart_observation",
                "Each row is P(observation | hidden state) and sums to 100%. An observation is evidence about the "
                "state, not the state itself.")
        with c2:
            st.latex(r"Z_r(o \mid s) = r\,Z(o \mid s) + (1-r)\,\tfrac{1}{|O|}")
            st.html(ui.note(
                "<b>Reliability model.</b> With probability r the channel reports a genuine reading; otherwise it "
                "reports noise that does not depend on the inventory state. At r = 0% every row becomes identical, "
                "so the observation cannot move the belief. Move the slider to watch the rows flatten."))
            st.html(ui.note(
                "<b>Likelihood is not belief.</b> These numbers answer “how probable is this alert if the state "
                "were s?”. The belief answers the reverse question, and Bayes' rule converts one into the other."))
            st.html(ui.note(
                "<b>Independence assumption.</b> Channels are treated as conditionally independent given the "
                "hidden state. Duplicate or correlated alerts violate this and inflate confidence; the duplicate "
                "guard in the control room protects against the simplest case."))
    with tab_b:
        update = eng.bayes_breakdown()
        chart(viz.fig_bayes_explorer(update.prior, update.predicted, update.joint_likelihood, update.posterior,
                                     title=f"Bayesian update explorer — period {eng.step}", height=380),
              "chart_bayes",
              "Left to right: belief at the end of the previous period, belief after the executed action's "
              "transition, combined likelihood of this period's evidence, and the normalised posterior.")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.html(ui.label("Stage 1 · prediction (only when an action is executed)"))
            st.latex(r"b^{-}(s') = \sum_{s} T(s' \mid s, a)\, b(s)")
            last = "none yet (period 0: the prediction equals the prior)" if eng.last_executed_action is None \
                else ACTIONS[eng.last_executed_action]
            st.html(ui.note(f"Prediction asks “where could inventory have moved?”. Action executed into this "
                            f"period: <b>{last}</b>."))
        with c2:
            st.html(ui.label("Stage 2 · correction (all evidence of the period, jointly)"))
            st.latex(r"b'(s') = \eta \prod_{k} Z_k(o_k \mid s')\; b^{-}(s'), \qquad "
                     r"\eta^{-1} = \sum_{s'} \prod_{k} Z_k(o_k \mid s')\, b^{-}(s')")
            st.html(ui.note(f"Correction asks “which states explain what we just saw?”. Evidence items counted: "
                            f"<b>{len(eng.evidence)}</b>; probability of this evidence under the prediction "
                            f"η⁻¹ = <b>{update.evidence_probability:.4g}</b>."))
        st.html(ui.simple_table_html(
            ["Quantity", "Healthy", "Low", "Critical"],
            [["Prior b(s)"] + [f"{100 * v:.2f}%" for v in update.prior],
             ["Predicted b⁻(s′)"] + [f"{100 * v:.2f}%" for v in update.predicted],
             ["Joint likelihood ∏ Z(o|s′)"] + [f"{v:.6f}" for v in update.joint_likelihood],
             ["Posterior b′(s′)"] + [f"{100 * v:.2f}%" for v in update.posterior]]))
    with tab_p:
        c1, c2 = st.columns([7, 5], gap="medium")
        with c1:
            grid = policy_region_grid(eng.transition_model, eng.cost_model, 34)
            chart(viz.fig_policy_map(grid, eng.belief, height=470), "chart_policy_map",
                  "Every point is a possible belief; its marker shows which action has the lowest expected cost "
                  "there. The white × is the current belief.")
        with c2:
            share = grid["action_name"].value_counts(normalize=True)
            st.html(ui.simple_table_html(
                ["Action", "Share of belief space where it is optimal"],
                [[a, f"{100 * share.get(a, 0.0):.1f}%"] for a in ACTIONS]))
            st.html(ui.note(
                "<b>Why regions, not thresholds.</b> Q(b, a) is linear in the belief, so each action wins on a "
                "convex region. The boundaries are produced by the cost model; no cut-off such as "
                "“P(Critical) &gt; 20%” is hard-coded. Change a cost in the What-If Lab and the regions move."))
            st.html(ui.note(
                "<b>Same alert, different action.</b> An alert moves the × by an amount that depends on where it "
                "starts. The same alert can therefore cross a boundary from one prior and stay inside a region "
                "from another."))
    with tab_a:
        cfg = eng.config
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.html(ui.label(f"Cost model C(s, a, s′) = direct + current-state + post-transition ({COST_UNIT})"))
            st.html(ui.simple_table_html(
                ["Action", "Direct cost", "Extra cost if currently Healthy"],
                [[ACTIONS[i], f"{cfg.action_cost[i]:.1f}", f"{cfg.current_state_cost[0, i]:.1f}"] for i in range(4)]))
            st.html(ui.simple_table_html(
                ["Resulting state s′", "Post-transition cost", "Meaning"],
                [["Healthy", f"{cfg.post_state_cost[0]:.1f}", "holding cost"],
                 ["Low", f"{cfg.post_state_cost[1]:.1f}", "service degradation"],
                 ["Critical", f"{cfg.post_state_cost[2]:.1f}", "stock-out penalty"]], numeric_from=1))
            st.html(ui.note(f"Reward is the negative of cost: R(s, a, s′) = −C(s, a, s′). Discount factor "
                            f"γ = {cfg.discount:.2f} completes the tuple; the myopic policy looks one period ahead."))
        with c2:
            st.html(ui.label("Observation channels"))
            st.html(ui.simple_table_html(
                ["Channel", "Symbols", "Assumed reliability"],
                [[CHANNELS[k]["label"], f"{len(CHANNELS[k]['observations'])}",
                  f"{100 * eng.observation_model.reliability[k]:.0f}%"] for k in CHANNEL_KEYS]))
            st.html(ui.label("Synthetic unit proxies (simulator only; never shown to the agent)"))
            st.html(ui.simple_table_html(
                ["State", "Snapshot band (units)", "Model P(stock-out)", "Model fill rate"],
                [[STATES[i], inventory_band_label(i),
                  f"{100 * eng.stockout_probability[i]:.1f}%", f"{100 * eng.fill_rate[i]:.1f}%"] for i in range(3)]))
            st.html(ui.note(
                "<b>These are state-conditioned snapshots, not a stock ledger.</b> Each period the simulator draws "
                "an inventory snapshot uniformly inside the hidden state's band and an independent demand; it does "
                "not roll stock forward from one period to the next. Stock-out and fill rate are computed per period "
                "from that snapshot and demand. Only the three-state hidden condition evolves over time."))
            st.html(ui.note(
                "Demand per period: " + "; ".join(
                    f"{CHANNELS['demand']['observations'][i].lower()} ≈ {DEMAND_MEAN_UNITS[i]:.0f} ± "
                    f"{DEMAND_SD_UNITS[i]:.0f} units" for i in range(4)) + "."))
        st.html(ui.note("To change any of these, use the model-configuration form in the What-If Lab below. "
                        "Applying it rebuilds the model and recomputes the belief, costs and recommendation."))

# ---- SECTIONS 8 + 9: what-if lab, simulation evaluation, debug view -----------
with st.container(key="panel_lab"):
    st.html(ui.section_title("08 · 09", "Simulation What-If Lab & Evaluation",
                             "Test scenarios, change assumptions and measure the policy against baselines. "
                             "All performance figures are synthetic simulation output, not real-world results."))
    if st.session_state.batch_request:
        with st.spinner("Running Monte Carlo evaluation with the shared simulator…"):
            st.session_state.batch = run_batch(int(st.session_state.batch_reps))
        st.session_state.batch_request = False
    batch = st.session_state.batch
    batch_label = ("No batch run yet." if batch is None else
                   f"Last run: {batch['reps']} replications × {batch['horizon']} periods.")
    st.html(ui.scope_legend_html(eng.step, eng.alerts_processed, batch_label))
    tab_lab, tab_eval, tab_debug = st.tabs(["Simulation & what-if lab", "Simulation evaluation",
                                            "Evaluation / debug view (hidden state)"])

    with tab_lab:
        st.html(ui.label("Preset scenarios (scripted and deterministic; loading one resets the live session)"))
        c1, c2, c3 = st.columns([4, 6, 2], gap="medium", vertical_alignment="bottom")
        with c1:
            st.selectbox("Scenario", DASHBOARD_PRESETS, key="scenario_key",
                         format_func=lambda k: f"{DASHBOARD_PRESETS.index(k) + 1}. {SCENARIOS[k].name}")
        with c2:
            st.html(ui.note(SCENARIOS[st.session_state.scenario_key].description))
        with c3:
            st.button("LOAD SCENARIO", key="btn_scenario", on_click=cb_load_scenario, width="stretch",
                      help="Resets the session and replays the scripted alerts through the engine.")
        st.html(ui.label("Model configuration (applies to the engine when you press Apply)"))
        with st.container(key="lab_config"):
            v = st.session_state.cfg_version
            cfg = eng.config
            with st.form(key=f"config_form_{v}", border=True):
                st.caption("Initial belief (%) — must sum to 100. Takes effect at period 0 or after a reset.")
                f1, f2, f3, f4 = st.columns(4)
                b0 = [f1.number_input("Healthy (%)", 0.0, 100.0, float(round(100 * cfg.initial_belief[0], 4)), 1.0, key=f"cfg_b0_{v}"),
                      f2.number_input("Low (%)", 0.0, 100.0, float(round(100 * cfg.initial_belief[1], 4)), 1.0, key=f"cfg_b1_{v}"),
                      f3.number_input("Critical (%)", 0.0, 100.0, float(round(100 * cfg.initial_belief[2], 4)), 1.0, key=f"cfg_b2_{v}")]
                duration = f4.number_input("Simulation duration (periods)", 5, 365, int(cfg.horizon), 5, key=f"cfg_h_{v}")
                st.caption("Observation reliability per channel (%)")
                r_cols = st.columns(4)
                rel = {k: r_cols[i].slider(CHANNELS[k]["label"], 0, 100, int(round(100 * cfg.reliability[k])), 5,
                                           key=f"cfg_rel_{k}_{v}") for i, k in enumerate(CHANNEL_KEYS)}
                e1, e2, e3 = st.columns(3)
                kappa = e1.slider("Demand volatility κ", 0.0, 0.5, float(cfg.demand_volatility), 0.01, key=f"cfg_kappa_{v}",
                                  help="Extra probability per period that inventory slips one level.")
                rho = e2.slider("Supplier delivery risk ρ", 0.0, 1.0, float(cfg.supplier_risk), 0.05, key=f"cfg_rho_{v}",
                                help="Probability that a reorder or expedite fails to land.")
                seed = e3.number_input("Random seed (applies on reset)", 0, 10_000_000, int(cfg.seed), 1, key=f"cfg_seed_{v}")
                st.caption(f"Operational action costs ({COST_UNIT})")
                a_cols = st.columns(4)
                action_cost = [a_cols[i].number_input(ACTION_SHORT[i], 0.0, 1000.0, float(cfg.action_cost[i]), 1.0,
                                                      key=f"cfg_ac_{i}_{v}") for i in range(4)]
                st.caption(f"Post-transition state costs ({COST_UNIT}): holding, service degradation, stock-out penalty")
                s_cols = st.columns(3)
                post_cost = [s_cols[i].number_input(f"Ending {STATES[i]}", 0.0, 5000.0, float(cfg.post_state_cost[i]), 1.0,
                                                    key=f"cfg_pc_{i}_{v}") for i in range(3)]
                st.caption("Transition probabilities T(s′ | s, a) — each row must sum to 1.00 (validated, not auto-fixed)")
                edited = {}
                t_cols = st.columns(2)
                for i, key in enumerate(ACTION_KEYS):
                    frame = pd.DataFrame(cfg.transitions[key], index=[f"from {s}" for s in STATES],
                                         columns=[f"to {s}" for s in STATES])
                    with t_cols[i % 2]:
                        st.caption(ACTIONS[i])
                        edited[key] = st.data_editor(
                            frame, key=f"cfg_T_{key}_{v}", width="stretch",
                            column_config={c: st.column_config.NumberColumn(c, min_value=0.0, max_value=1.0,
                                                                            step=0.005, format="%.3f")
                                           for c in frame.columns})
                s1, s2 = st.columns(2)
                apply_clicked = s1.form_submit_button("APPLY CONFIGURATION", type="primary", width="stretch")
                s2.form_submit_button("RESTORE DEFAULTS", on_click=cb_restore_defaults, width="stretch")
            if apply_clicked:
                new_cfg = eng.config.copy()
                new_cfg.initial_belief = np.array(b0, dtype=float) / 100.0
                new_cfg.horizon = int(duration)
                new_cfg.reliability = {k: rel[k] / 100.0 for k in CHANNEL_KEYS}
                new_cfg.demand_volatility, new_cfg.supplier_risk, new_cfg.seed = float(kappa), float(rho), int(seed)
                new_cfg.action_cost = np.array(action_cost, dtype=float)
                new_cfg.post_state_cost = np.array(post_cost, dtype=float)
                try:
                    new_cfg.transitions = {k: edited[k].to_numpy(dtype=float) for k in ACTION_KEYS}
                    problems = new_cfg.validate()
                except (ValueError, TypeError) as error:
                    problems = [f"Transition tables must contain numbers only ({error})."]
                if problems:
                    st.error("Configuration not applied. Please fix the following:\n\n- " + "\n- ".join(problems))
                else:
                    eng.seed = int(seed)
                    eng.apply_config(new_cfg)
                    st.session_state.batch = None
                    flash("success", "Model configuration applied. Belief, expected costs, recommendation, heatmaps "
                                     "and policy map were recomputed with the new parameters.")
                    st.rerun()

    with tab_eval:
        c1, c2 = st.columns([3, 9], gap="medium")
        with c1:
            st.slider("Monte Carlo replications", 20, 300, key="batch_reps", step=20)
            st.button("RUN BATCH EVALUATION", key="btn_batch", type="primary", on_click=cb_request_batch,
                      width="stretch")
            st.html(ui.note(f"Each policy is simulated for {int(eng.config.horizon)} periods on the same random "
                            f"seeds (common random numbers) with the current model configuration."))
        with c2:
            if batch is None:
                st.info("No historical simulation results yet. Press “Run batch evaluation” to compare the belief "
                        "agent with fixed policies, a record-trusting heuristic and a full-information oracle.")
            else:
                summary = batch["summary"].set_index("policy")
                agent, base, oracle = summary.loc[AGENT_NAME], summary.loc[BASELINE_NAME], summary.loc[ORACLE_NAME]
                efficiency = gap_closed(base["total_cost"], agent["total_cost"], oracle["total_cost"])
                eff_text = "n/a" if not np.isfinite(efficiency) else f"{100 * efficiency:.0f}"
                st.html(ui.metric_cards_html([
                    ("Total simulated operating cost", f'{agent["total_cost"]:,.0f}<span class="unit">{COST_UNIT}</span>',
                     f'Belief agent, mean of {batch["reps"]} runs (±{agent["total_cost_ci"]:,.0f}, 95% CI)', "var(--blue)", "Σ"),
                    ("Stock-out rate", f'{100 * agent["stockout_rate"]:.1f}<span class="unit">%</span>',
                     "Periods where demand exceeded the state-conditioned inventory snapshot", "var(--critical)", "■"),
                    ("Service level (fill rate)", f'{100 * agent["service_level"]:.1f}<span class="unit">%</span>',
                     "Units served ÷ units demanded (snapshot-based proxy)", "var(--healthy)", "●"),
                    ("Belief-estimation accuracy", f'{100 * agent["accuracy"]:.1f}<span class="unit">%</span>',
                     "Most probable state equals the hidden state", "var(--cyan)", "◎"),
                    ("Brier score (0–2)", f'{agent["brier"]:.3f}', "Multiclass, original Brier form; lower is better",
                     "var(--violet)", "∑"),
                    ("Log loss", f'{agent["log_loss"]:.3f}', "−ln b(true state); lower is better", "var(--violet)", "ℓ"),
                    ("Policy efficiency", f'{eff_text}<span class="unit">%</span>',
                     "Cost gap closed between the record-trusting heuristic and the oracle", "var(--cyan)", "⇄"),
                    ("Baseline cost", f'{base["total_cost"]:,.0f}<span class="unit">{COST_UNIT}</span>',
                     f'Record-trusting heuristic; oracle {oracle["total_cost"]:,.0f} {COST_UNIT}', "var(--low)", "◆"),
                ]))
        if batch is not None:
            # Same nine result columns as before, split into two logically grouped tables placed under
            # their related charts, so every heading and value is readable without hidden scrolling.
            summary_rows = batch["summary"].reset_index(drop=True)
            agent_row = int(summary_rows.index[summary_rows["policy"] == AGENT_NAME][0])
            runs = f"{batch['reps']} runs × {batch['horizon']} periods"
            with st.container(key="row_eval"):
                g1, g2 = st.columns(2, gap="medium")
                with g1:
                    chart(viz.fig_policy_comparison(batch["summary"], "total_cost",
                                                    f"Total simulated cost by policy ({batch['reps']} runs × {batch['horizon']} periods)",
                                                    highlight=AGENT_NAME, height=430), "chart_batch_cost",
                          "Bars are means across replications; whiskers are 95% confidence intervals. The oracle sees "
                          "the hidden state and is not implementable; it only bounds what visibility is worth.")
                    agent_traj = batch["agent"]
                    per_period = agent_traj.groupby("step").agg(units_served=("units_served", "sum"),
                                                                demand_units=("demand_units", "sum")).reset_index()
                    per_period["fill_rate"] = per_period["units_served"] / per_period["demand_units"]
                    chart(viz.fig_service_trend(per_period, "Service-level trend — belief agent, pooled over runs",
                                                height=360), "chart_batch_service",
                          "Fill rate by simulation period, pooled across replications, with a 5-period rolling mean. "
                          "Each period compares demand with that period's state-conditioned inventory snapshot.")
                    st.html(ui.results_table_html(
                        ["Policy", f"Mean total cost ({COST_UNIT})", f"± 95% CI ({COST_UNIT})", "Stock‑out rate",
                         "Service level (fill rate)", "Intervention rate"],
                        [[r.policy, f"{r.total_cost:,.0f}", f"{r.total_cost_ci:,.0f}", f"{100 * r.stockout_rate:.1f}%",
                          f"{100 * r.service_level:.1f}%", f"{100 * r.intervention_rate:.1f}%"]
                         for r in summary_rows.itertuples()],
                        ["", "Mean total simulated cost per episode, across replications",
                         "Half-width of the 95% Student-t confidence interval of the mean cost",
                         "Share of periods in which demand exceeded the state-conditioned inventory snapshot",
                         "Units served divided by units demanded (snapshot-based proxy)",
                         "Share of periods in which an action other than Continue Monitoring was executed"],
                        highlight_row=agent_row, caption=f"Operational performance · {runs}"))
                    st.caption("Highlighted row: the SENTINEL belief agent. Hover a heading (ⓘ) for its definition.")
                with g2:
                    table = calibration_table(batch["agent"][BELIEF_COLUMNS].to_numpy(),
                                              batch["agent"]["hidden_state"].to_numpy(), n_bins=10)
                    chart(viz.fig_calibration(table, "Belief calibration — top-label reliability diagram",
                                              expected_calibration_error(table), height=430), "chart_batch_calibration",
                          "Points on the diagonal mean stated confidence matches observed accuracy. Here the simulator "
                          "uses the agent's own model, so good calibration verifies the implementation rather than "
                          "proving real-world accuracy.")
                    st.html(ui.results_table_html(
                        ["Policy", "Brier score (0–2)", "Log loss", "Belief accuracy"],
                        [[r.policy, f"{r.brier:.3f}", f"{r.log_loss:.3f}", f"{100 * r.accuracy:.1f}%"]
                         for r in summary_rows.itertuples()],
                        ["", "Multiclass Brier score, original 0–2 form; lower is better",
                         "Mean of −ln b(true state); lower is better",
                         "Share of periods in which the most probable state equals the hidden state"],
                        highlight_row=agent_row, caption=f"Belief quality · {runs}"))
                    st.caption("Highlighted row: the SENTINEL belief agent. Belief metrics are computed by running the "
                               "same Bayes filter alongside every policy, so they differ only because the policies "
                               "steer the system into different states.")

    with tab_debug:
        st.warning("Evaluation only. The hidden simulated state shown here is never passed to the belief update, "
                   "the expected-cost calculation or the recommendation. It exists so you can check the agent "
                   "against the simulator's ground truth.")
        st.toggle("Reveal hidden simulated ground truth", key="reveal_truth")
        if st.session_state.reveal_truth:
            truth = eng.evaluation_frame()
            live = eng.evaluation_metrics()
            if live is None:
                st.html(ui.note(f"Hidden state right now: <b>{STATES[eng.env.evaluation_state()]}</b>. Execute at "
                                f"least one step to accumulate evaluation metrics."))
            else:
                st.html(ui.metric_cards_html([
                    ("Hidden state now", STATES[eng.env.evaluation_state()], "Simulator ground truth (evaluation only)",
                     "var(--violet)", "◎"),
                    ("Session Brier score (0–2)", f'{live["brier"]:.3f}', f'Over {live["periods"]} completed periods',
                     "var(--violet)", "∑"),
                    ("Session belief accuracy", f'{100 * live["accuracy"]:.0f}<span class="unit">%</span>',
                     "Most probable state equals hidden state", "var(--cyan)", "●"),
                    ("Realised cost so far", f'{live["realised_cost"]:,.0f}<span class="unit">{COST_UNIT}</span>',
                     "Sum of C(s, a, s′) on the hidden path", "var(--blue)", "Σ"),
                ]))
                st.caption("Hand-injected alerts are user-supplied what-if evidence; they need not agree with the "
                           "hidden state, so session metrics can look worse than the batch evaluation.")
                d1, d2 = st.columns(2, gap="medium")
                with d1:
                    chart(viz.fig_state_timeline(truth, height=360), "chart_debug_states",
                          "White line: hidden state. Markers: the agent's most probable state when it decided "
                          "(× where it differed).")
                with d2:
                    chart(viz.fig_inventory_units(truth, height=360), "chart_debug_units",
                          "Synthetic inventory snapshots: drawn independently inside the hidden state's band each "
                          "period (no stock roll-forward). They exist only to compute stock-outs and fill rate.")
                st.dataframe(debug_table(truth), hide_index=True, height=240, width="stretch", key="debug_table")
        else:
            st.html(ui.note("Ground truth is hidden. Switch the toggle on to compare the agent's beliefs with the "
                            "simulator's actual state."))

st.html(ui.FOOTER)


# ----------------------------------------------------------------------------
# Auto-simulation ticker (supported Streamlit mechanism: a timed fragment)
# ----------------------------------------------------------------------------
def auto_tick() -> bool:
    """Advance one period for the auto-simulation. Returns True if a step was executed."""
    ss = st.session_state
    if not ss.get("auto_running"):
        return False
    if engine().step >= int(engine().config.horizon):
        ss.auto_running = False
        flash("info", f"Auto-simulation finished: simulation duration of {engine().config.horizon} periods reached.")
        return False
    out = _execute_step()
    flash("info", f"Auto-simulation · period {engine().step}: executed {ACTIONS[out['executed']]}; "
                  f"recommendation now {ACTIONS[out['new_recommendation']]}.")
    return True


@st.fragment(run_every=(float(st.session_state.auto_interval) if st.session_state.auto_running else None))
def auto_ticker() -> None:
    # During a normal full-script run this body does nothing; only the timer-driven
    # fragment rerun (which happens after the full run has finished) advances the simulation.
    if st.session_state.get("_in_full_run", True) or not st.session_state.get("auto_running"):
        return
    auto_tick()
    st.rerun()


auto_ticker()
st.session_state["_in_full_run"] = False
