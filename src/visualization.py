"""Plotly figure builders shared by the notebook and the dashboard.

Colour semantics are fixed across the whole project:
    green = Healthy / favourable      amber = Low / warning      coral = Critical
    cyan / blue = AI reasoning, probabilities, system information
    violet = model analytics and selected highlights
Status is never conveyed by colour alone: every series is also named in a
legend, axis category or text label. Probabilities are always shown in percent.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import ACTION_SHORT, ACTIONS, COST_UNIT, N_ACTIONS, STATES

PALETTE = {
    "bg": "#07111F", "bg2": "#0B1728", "panel": "#10243A", "blue": "#247BFF", "cyan": "#00D9FF",
    "violet": "#875BFF", "healthy": "#2DD6A0", "low": "#FFB547", "critical": "#FF596B",
    "text": "#F2F7FF", "muted": "#A6B9D0", "border": "#233B56", "slate": "#7F93AD",
}
STATE_COLORS = [PALETTE["healthy"], PALETTE["low"], PALETTE["critical"]]
ACTION_COLORS = [PALETTE["slate"], PALETTE["blue"], PALETTE["violet"], PALETTE["cyan"]]
STATE_SYMBOLS = ["circle", "diamond", "square"]
ACTION_SYMBOLS = ["circle", "diamond", "star", "square"]
FONT_BODY = "Inter, Manrope, 'Segoe UI', Arial, sans-serif"
FONT_HEAD = "'Space Grotesk', Inter, 'Segoe UI', Arial, sans-serif"
FONT_MONO = "'IBM Plex Mono', Consolas, 'Courier New', monospace"
SEQUENTIAL = [[0.0, "#0B1728"], [0.35, "#17407F"], [0.7, "#247BFF"], [1.0, "#00D9FF"]]


def _label_heatmap(fig: go.Figure, z: np.ndarray, x: Sequence, y: Sequence, row: Optional[int] = None,
                   col: Optional[int] = None, size: int = 13) -> None:
    """Write each cell's value with a text colour chosen for contrast against the cell colour."""
    for i, y_label in enumerate(y):
        for j, x_label in enumerate(x):
            dark_text = z[i][j] >= 55          # bright cyan / blue cells get dark text
            kwargs = dict(x=x_label, y=y_label, text=f"{z[i][j]:.1f}%", showarrow=False,
                          font=dict(family=FONT_MONO, size=size, color="#04101E" if dark_text else PALETTE["text"]))
            if row is not None:
                fig.add_annotation(row=row, col=col, **kwargs)
            else:
                fig.add_annotation(**kwargs)


def _integer_periods(fig: go.Figure, periods) -> None:
    """Periods are whole numbers: never show 0.5-style ticks."""
    n = max(1, int(pd.Series(periods).nunique()))
    fig.update_xaxes(tickformat="d", dtick=max(1, int(np.ceil(n / 12))))


def _split_title(title: str, limit: int = 46):
    """Long titles become a title line plus a smaller subtitle line, so they fit narrow screens."""
    if len(title) <= limit or "<br>" in title:
        return title, "<br>" in title
    for sep in (" — ", ": ", " (", " vs. ", ", "):
        if sep in title:
            head, tail = title.split(sep, 1)
            if head.count("(") != head.count(")"):      # never break inside a formula such as T(s′ | s, a)
                continue
            if sep == " (":
                tail = "(" + tail
            elif sep == " vs. ":
                tail = "vs. " + tail
            if 8 <= len(head) <= limit + 10:
                return (f"{head}<br><span style='font-size:12px;color:{PALETTE['muted']}'>{tail}</span>", True)
    return title, False


def _style(fig: go.Figure, title: str, height: int = 380, legend: bool = True,
           x_title: Optional[str] = None, y_title: Optional[str] = None) -> go.Figure:
    """Apply the shared dark theme. Legends sit below the plot so they never cover data or titles."""
    title, wrapped = _split_title(title)
    fig.update_layout(
        # fixed pixel offset from the top (not a fraction of the height), so two-line titles are never
        # clipped on short charts
        title=dict(text=title, x=0.01, xanchor="left", yref="container", y=1.0, yanchor="top", pad=dict(t=14),
                   font=dict(family=FONT_HEAD, size=16, color=PALETTE["text"])),
        height=height, paper_bgcolor=PALETTE["bg2"], plot_bgcolor=PALETTE["bg2"],
        font=dict(family=FONT_BODY, size=12, color=PALETTE["text"]),
        margin=dict(l=64, r=28, t=80 if wrapped else 58, b=88 if legend else 56),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="left", x=0.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=PALETTE["muted"])),
        hoverlabel=dict(bgcolor=PALETTE["panel"], bordercolor=PALETTE["border"],
                        font=dict(family=FONT_MONO, size=12, color=PALETTE["text"])),
    )
    axis = dict(gridcolor="rgba(35,59,86,0.65)", zerolinecolor=PALETTE["border"],
                linecolor=PALETTE["border"], tickfont=dict(color=PALETTE["muted"], size=11),
                title_font=dict(color=PALETTE["muted"], size=12), automargin=True)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    return fig


# ----------------------------------------------------------------------------
# Belief charts
# ----------------------------------------------------------------------------
def fig_belief_distribution(belief: Sequence[float], title: str = "Current belief distribution",
                            height: int = 340) -> go.Figure:
    b = 100 * np.asarray(belief, dtype=float)
    fig = go.Figure(go.Bar(
        x=list(STATES), y=b, marker=dict(color=STATE_COLORS, line=dict(color=PALETTE["border"], width=1)),
        text=[f"{v:.1f}%" for v in b], textposition="outside", textfont=dict(family=FONT_MONO, size=13),
        hovertemplate="%{x} inventory<br>Belief: %{y:.2f}%<extra></extra>", cliponaxis=False))
    _style(fig, title, height, legend=False, x_title="Hidden inventory state", y_title="Belief probability (%)")
    fig.update_yaxes(range=[0, 112], ticksuffix="%")
    return fig


def fig_belief_evolution(timeline: pd.DataFrame, title: str = "Belief-state evolution",
                         height: int = 400) -> go.Figure:
    """Stacked area of the three beliefs over simulation events.

    With a single event (the initial belief only) a time axis has nothing to
    show, so the chart draws the initial belief as one labelled stacked bar
    with a note that the evolution begins at the first alert or executed step.
    """
    if len(timeline) == 1:
        return _initial_belief_panel(timeline.iloc[0], title, height)
    fig = go.Figure()
    custom = np.stack([timeline["label"].astype(str), timeline["timestamp"].astype(str).str.slice(0, 16),
                       timeline["period"].astype(str), timeline["recommended_name"].astype(str)], axis=-1)
    for i, col in enumerate(["b_healthy", "b_low", "b_critical"]):
        fig.add_trace(go.Scatter(
            x=timeline["event_index"], y=100 * timeline[col], name=f"{STATES[i]} inventory",
            mode="lines+markers", stackgroup="belief", line=dict(color=STATE_COLORS[i], width=1.6),
            marker=dict(symbol=STATE_SYMBOLS[i], size=5, color=STATE_COLORS[i]),
            customdata=custom,
            hovertemplate=(f"<b>{STATES[i]}: %{{y:.1f}}%</b><br>Event %{{x}} · period %{{customdata[2]}}"
                           "<br>%{customdata[1]}<br>%{customdata[0]}"
                           "<br>Recommendation: %{customdata[3]}<extra></extra>")))
    _style(fig, title, height, x_title="Simulation event index (alerts, predictions, corrections)",
           y_title="Belief probability (%)")
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    fig.update_xaxes(dtick=max(1, int(np.ceil(len(timeline) / 12))))
    fig.update_layout(hovermode="x unified", legend=dict(traceorder="normal"))
    return fig


def _initial_belief_panel(row: pd.Series, title: str, height: int) -> go.Figure:
    """Single-event view of the belief-evolution chart: one stacked bar for the initial belief."""
    values = [100 * float(row["b_healthy"]), 100 * float(row["b_low"]), 100 * float(row["b_critical"])]
    category = ["Initial belief b₀ (event 0)"]
    fig = go.Figure()
    for i, value in enumerate(values):
        fig.add_trace(go.Bar(
            x=category, y=[value], name=f"{STATES[i]} inventory", marker=dict(color=STATE_COLORS[i]),
            text=[f"{STATES[i]} {value:.1f}%"], textposition="inside", insidetextanchor="middle",
            textfont=dict(family=FONT_MONO, size=12, color="#04101E"), width=0.42,
            hovertemplate=(f"<b>{STATES[i]}: %{{y:.1f}}%</b><br>{str(row['timestamp'])[:16]}"
                           f"<br>{row['label']}<extra></extra>")))
    _style(fig, title, height, x_title="Simulation event index (alerts, predictions, corrections)",
           y_title="Belief probability (%)")
    fig.update_layout(barmode="stack", legend=dict(traceorder="normal"))
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    fig.add_annotation(xref="paper", yref="paper", x=0.5, y=1.02, yanchor="bottom", showarrow=False,
                       text="Only the initial belief exists so far. The evolution is drawn from the first alert or executed step.",
                       font=dict(size=11, color=PALETTE["muted"]))
    fig.update_layout(margin=dict(t=84))
    return fig


def fig_before_after(stages: Dict[str, Sequence[float]], title: str = "Belief before and after the evidence",
                     height: int = 360) -> go.Figure:
    """Grouped bars: one group per inventory state, one bar per stage (e.g. prior / posterior)."""
    shades = [PALETTE["slate"], PALETTE["blue"], PALETTE["cyan"], PALETTE["violet"]]
    fig = go.Figure()
    for k, (name, belief) in enumerate(stages.items()):
        b = 100 * np.asarray(belief, dtype=float)
        fig.add_trace(go.Bar(
            x=list(STATES), y=b, name=name, marker=dict(color=shades[k % len(shades)]),
            text=[f"{v:.1f}%" for v in b], textposition="outside", textfont=dict(family=FONT_MONO, size=11),
            hovertemplate=f"{name}<br>%{{x}}: %{{y:.2f}}%<extra></extra>", cliponaxis=False))
    _style(fig, title, height, x_title="Hidden inventory state", y_title="Belief probability (%)")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(range=[0, 115], ticksuffix="%")
    return fig


def fig_bayes_explorer(prior, predicted, likelihood, posterior, title: str = "Bayesian update explorer",
                       height: int = 360) -> go.Figure:
    """Four panels: prior -> predicted -> joint likelihood -> posterior."""
    panels = [("1 · Prior b(s)", 100 * np.asarray(prior), "%", True),
              ("2 · Predicted b⁻(s′)", 100 * np.asarray(predicted), "%", True),
              ("3 · Joint likelihood ∏ Z(o|s′)", np.asarray(likelihood, dtype=float), "", False),
              ("4 · Posterior b′(s′)", 100 * np.asarray(posterior), "%", True)]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[p[0] for p in panels], horizontal_spacing=0.07)
    for c, (_, values, suffix, is_pct) in enumerate(panels, start=1):
        text = [f"{v:.1f}%" for v in values] if is_pct else [f"{v:.4f}" if v >= 1e-4 else f"{v:.1e}" for v in values]
        fig.add_trace(go.Bar(x=list(STATES), y=values, marker=dict(color=STATE_COLORS), text=text,
                             textposition="outside", textfont=dict(family=FONT_MONO, size=10),
                             hovertemplate="%{x}: %{text}<extra></extra>", cliponaxis=False, showlegend=False),
                      row=1, col=c)
        top = 115 if is_pct else max(float(np.max(values)) * 1.25, 1e-12)
        fig.update_yaxes(range=[0, top], ticksuffix=suffix, row=1, col=c)
    _style(fig, title, height, legend=False)
    fig.update_yaxes(title_text="Probability (%)", row=1, col=1)
    fig.update_yaxes(title_text="Likelihood (unitless)", row=1, col=3)
    fig.update_xaxes(title_text="Inventory state", tickfont=dict(size=10))
    fig.update_annotations(font=dict(size=12, color=PALETTE["muted"], family=FONT_BODY))
    fig.update_layout(margin=dict(t=100, b=56))
    return fig


# ----------------------------------------------------------------------------
# Model heatmaps
# ----------------------------------------------------------------------------
def fig_transition_heatmap(matrix: np.ndarray, action_name: str, height: int = 360) -> go.Figure:
    z = 100 * np.asarray(matrix, dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{s}" for s in STATES], y=[f"{s}" for s in STATES], zmin=0, zmax=100,
        colorscale=SEQUENTIAL,
        colorbar=dict(title=dict(text="Transition probability (%)", side="right", font=dict(size=11)),
                      ticksuffix="%", thickness=12, len=0.9),
        hovertemplate="From %{y} to %{x}<br>P = %{z:.2f}%<extra></extra>", xgap=3, ygap=3))
    _label_heatmap(fig, z, list(STATES), list(STATES))
    _style(fig, f"Transition matrix T(s′ | s, a) — {action_name}", height, legend=False,
           x_title="Next inventory state s′", y_title="Current inventory state s")
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_transition_grid(tensor: np.ndarray, height: int = 640) -> go.Figure:
    """All four action-specific transition matrices in one figure (notebook)."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(ACTIONS), horizontal_spacing=0.16,
                        vertical_spacing=0.2)
    for a in range(N_ACTIONS):
        r, c = divmod(a, 2)
        fig.add_trace(go.Heatmap(
            z=100 * tensor[a], x=list(STATES), y=list(STATES), zmin=0, zmax=100, colorscale=SEQUENTIAL,
            showscale=(a == 0),
            colorbar=dict(title=dict(text="Transition probability (%)", side="right", font=dict(size=11)),
                          ticksuffix="%", thickness=12),
            hovertemplate="From %{y} to %{x}<br>P = %{z:.2f}%<extra></extra>", xgap=3, ygap=3),
            row=r + 1, col=c + 1)
        _label_heatmap(fig, 100 * tensor[a], list(STATES), list(STATES), row=r + 1, col=c + 1, size=12)
        fig.update_yaxes(autorange="reversed", title_text="Current state s", row=r + 1, col=c + 1)
        fig.update_xaxes(title_text="Next state s′", row=r + 1, col=c + 1)
    _style(fig, "Action-specific transition matrices T(s′ | s, a)", height, legend=False)
    for annotation in fig.layout.annotations[:N_ACTIONS]:          # the four subplot titles
        annotation.font = dict(size=12, color=PALETTE["muted"])
    fig.update_layout(margin=dict(t=100, r=110))
    return fig


def fig_observation_heatmap(matrix: np.ndarray, labels: Sequence[str], channel_label: str,
                            reliability: Optional[float] = None, height: int = 360) -> go.Figure:
    z = 100 * np.asarray(matrix, dtype=float)
    wrapped = [lab.replace(" / ", "<br>").replace(" inventory ", "<br>inventory ") if len(lab) > 18 else lab
               for lab in labels]
    wrapped = [w if "<br>" in w or len(w) <= 18 else "<br>".join(_wrap_words(w, 16)) for w in wrapped]
    suffix = "" if reliability is None else f" · reliability {100 * reliability:.0f}%"
    fig = go.Figure(go.Heatmap(
        z=z, x=wrapped, y=list(STATES), zmin=0, zmax=100, colorscale=SEQUENTIAL,
        colorbar=dict(title=dict(text="Likelihood P(o | s) (%)", side="right", font=dict(size=11)),
                      ticksuffix="%", thickness=12, len=0.9),
        customdata=np.tile(np.array(list(labels), dtype=object), (len(STATES), 1)),
        hovertemplate="State %{y}<br>Observation: %{customdata}<br>P(o | s) = %{z:.2f}%<extra></extra>",
        xgap=3, ygap=3))
    _label_heatmap(fig, z, wrapped, list(STATES), size=12)
    _style(fig, f"Observation likelihood Z(o | s) — {channel_label}{suffix}", height, legend=False,
           x_title="Observation o", y_title="Hidden inventory state s")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=0, tickfont=dict(size=10))
    fig.update_layout(margin=dict(b=84))
    return fig


def _wrap_words(text: str, width: int):
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


# ----------------------------------------------------------------------------
# Decision charts
# ----------------------------------------------------------------------------
def fig_action_costs(decision, title: str = "Expected cost by operational action", height: int = 400) -> go.Figure:
    """Stacked expected-cost components per action; the engine's argmin is labelled in text."""
    names = [f"{ACTION_SHORT[i]}" for i in range(N_ACTIONS)]
    comps = [("Direct action cost", decision.direct, PALETTE["blue"]),
             ("Current-state cost (surplus / disruption)", decision.current_state, PALETTE["violet"]),
             ("Post-transition cost (holding / shortage)", decision.post_transition, PALETTE["slate"])]
    fig = go.Figure()
    for label, values, color in comps:
        fig.add_trace(go.Bar(
            x=names, y=values, name=label, marker=dict(color=color, line=dict(color=PALETTE["bg2"], width=1)),
            hovertemplate=f"%{{x}}<br>{label}: %{{y:.2f}} {COST_UNIT}<extra></extra>"))
    q = decision.q_values
    top = float(q.max())
    for i in range(N_ACTIONS):
        best = i == decision.action
        fig.add_annotation(x=names[i], y=q[i], yshift=10 if not best else 22, showarrow=False,
                           text=f"<b>{q[i]:.1f}</b> {COST_UNIT}",
                           font=dict(family=FONT_MONO, size=12,
                                     color=PALETTE["healthy"] if best else PALETTE["text"]))
        if best:
            fig.add_annotation(x=names[i], y=q[i], yshift=8, showarrow=False, text="▼ LOWEST EXPECTED COST",
                               font=dict(family=FONT_MONO, size=10, color=PALETTE["healthy"]))
    _style(fig, title, height, x_title="Operational action", y_title=f"Expected total cost ({COST_UNIT})")
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_yaxes(range=[0, top * 1.28])
    # outline the engine's recommended action (shape, not just colour)
    fig.add_shape(type="rect", xref="x", yref="y", x0=decision.action - 0.36, x1=decision.action + 0.36,
                  y0=0, y1=q[decision.action], line=dict(color=PALETTE["healthy"], width=2.5),
                  fillcolor="rgba(0,0,0,0)")
    return fig


def fig_policy_timeline(timeline: pd.DataFrame, action_history: Optional[pd.DataFrame] = None,
                        title: str = "Recommended and executed actions over time", height: int = 380,
                        line_name: str = "Recommended by NOVA (not executed)") -> go.Figure:
    """Step line of the standing recommendation, with executed actions marked separately."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline["event_index"], y=[ACTION_SHORT[i] for i in timeline["recommended"]],
        mode="lines+markers", name=line_name, line=dict(color=PALETTE["cyan"], width=2, shape="hv"),
        marker=dict(size=7, color=PALETTE["cyan"], symbol="circle"),
        customdata=np.stack([timeline["label"].astype(str), timeline["period"].astype(str)], axis=-1),
        hovertemplate="Event %{x} · period %{customdata[1]}<br>Action: %{y}<br>%{customdata[0]}<extra></extra>"))
    if action_history is not None and not action_history.empty:
        predict_rows = timeline[timeline["kind"] == "predict"].reset_index(drop=True)
        n = min(len(predict_rows), len(action_history))
        if n:
            executed = action_history["executed"].iloc[:n].to_numpy()
            followed = executed == action_history["recommended"].iloc[:n].to_numpy()
            fig.add_trace(go.Scatter(
                x=predict_rows["event_index"].iloc[:n], y=[ACTION_SHORT[i] for i in executed], mode="markers",
                name="Executed action (transition applied)",
                marker=dict(size=14, symbol=["star" if f else "x" for f in followed], color=PALETTE["violet"],
                            line=dict(color=PALETTE["text"], width=1)),
                customdata=np.where(followed, "followed recommendation", "manual override"),
                hovertemplate="Event %{x}<br>Executed: %{y}<br>%{customdata}<extra></extra>"))
    _style(fig, title, height, x_title="Simulation event index", y_title="Operational action")
    fig.update_yaxes(categoryorder="array", categoryarray=list(ACTION_SHORT), range=[-0.5, N_ACTIONS - 0.5])
    fig.update_xaxes(dtick=max(1, int(np.ceil(len(timeline) / 12))))
    return fig


def fig_risk_trend(timeline: pd.DataFrame, title: str = "Model-expected stock-out risk and service level",
                   height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline["event_index"], y=100 * timeline["expected_service_level"], mode="lines+markers",
        name="Expected service level (fill rate)", line=dict(color=PALETTE["healthy"], width=2),
        marker=dict(symbol="circle", size=5), hovertemplate="Event %{x}<br>Expected fill rate: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=timeline["event_index"], y=100 * timeline["stockout_risk"], mode="lines+markers",
        name="Expected stock-out risk", line=dict(color=PALETTE["critical"], width=2, dash="dash"),
        marker=dict(symbol="square", size=5), hovertemplate="Event %{x}<br>Stock-out risk: %{y:.1f}%<extra></extra>"))
    _style(fig, title, height, x_title="Simulation event index", y_title="Belief-weighted expectation (%)")
    fig.update_yaxes(range=[0, 104], ticksuffix="%")
    fig.update_xaxes(dtick=max(1, int(np.ceil(len(timeline) / 12))))
    return fig


def fig_recovery_trend(timeline: pd.DataFrame,
                       title: str = "Belief-based inventory recovery: P(Healthy) vs P(Critical)",
                       height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timeline["event_index"], y=100 * timeline["b_healthy"], mode="lines+markers",
                             name="P(Healthy)", line=dict(color=PALETTE["healthy"], width=2),
                             marker=dict(symbol="circle", size=5),
                             hovertemplate="Event %{x}<br>P(Healthy): %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=timeline["event_index"], y=100 * timeline["b_critical"], mode="lines+markers",
                             name="P(Critical)", line=dict(color=PALETTE["critical"], width=2, dash="dash"),
                             marker=dict(symbol="square", size=5),
                             hovertemplate="Event %{x}<br>P(Critical): %{y:.1f}%<extra></extra>"))
    predicts = timeline[(timeline["kind"] == "predict") & timeline["executed_into_period"].notna()]
    orders = predicts[predicts["executed_into_period"] != 0]
    if not orders.empty:
        fig.add_trace(go.Scatter(
            x=orders["event_index"], y=[3] * len(orders), mode="markers", name="Replenishment action executed",
            marker=dict(symbol="triangle-up", size=11, color=PALETTE["violet"], line=dict(color=PALETTE["text"], width=1)),
            customdata=[ACTIONS[int(i)] for i in orders["executed_into_period"]],
            hovertemplate="Event %{x}<br>Executed: %{customdata}<extra></extra>"))
    _style(fig, title, height, x_title="Simulation event index", y_title="Belief probability (%)")
    fig.update_yaxes(range=[0, 104], ticksuffix="%")
    fig.update_xaxes(dtick=max(1, int(np.ceil(len(timeline) / 12))))
    return fig


def fig_expected_cost_trend(timeline: pd.DataFrame, title: str = "Expected cost of the standing recommendation",
                            height: int = 340) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=timeline["event_index"], y=timeline["expected_cost"], mode="lines+markers", name="Q(b, a*)",
        line=dict(color=PALETTE["cyan"], width=2), marker=dict(size=6),
        customdata=timeline["recommended_name"],
        hovertemplate=f"Event %{{x}}<br>%{{customdata}}<br>Expected cost: %{{y:.2f}} {COST_UNIT}<extra></extra>"))
    _style(fig, title, height, legend=False, x_title="Simulation event index",
           y_title=f"Expected next-period cost ({COST_UNIT})")
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(dtick=max(1, int(np.ceil(len(timeline) / 12))))
    return fig


def fig_policy_map(grid: pd.DataFrame, current_belief: Optional[Sequence[float]] = None,
                   title: str = "Policy map: which action wins where on the belief simplex",
                   height: int = 460) -> go.Figure:
    """Ternary plot of the engine's argmin action over the whole belief simplex."""
    fig = go.Figure()
    for a in range(N_ACTIONS):
        part = grid[grid["action"] == a]
        if part.empty:
            continue
        fig.add_trace(go.Scatterternary(
            a=100 * part["Critical"], b=100 * part["Healthy"], c=100 * part["Low"], mode="markers",
            name=ACTIONS[a], marker=dict(symbol=ACTION_SYMBOLS[a], size=6, color=ACTION_COLORS[a], opacity=0.9),
            customdata=part["expected_cost"],
            hovertemplate=(f"{ACTIONS[a]}<br>Healthy %{{b:.0f}}% · Low %{{c:.0f}}% · Critical %{{a:.0f}}%"
                           f"<br>Expected cost %{{customdata:.1f}} {COST_UNIT}<extra></extra>")))
    if current_belief is not None:
        cb = 100 * np.asarray(current_belief, dtype=float)
        fig.add_trace(go.Scatterternary(
            a=[cb[2]], b=[cb[0]], c=[cb[1]], mode="markers", name="Current belief",
            marker=dict(symbol="x", size=16, color=PALETTE["text"], line=dict(color=PALETTE["bg"], width=2)),
            hovertemplate="Current belief<br>Healthy %{b:.1f}% · Low %{c:.1f}% · Critical %{a:.1f}%<extra></extra>"))
    _style(fig, title, height)
    axis = dict(gridcolor=PALETTE["border"], linecolor=PALETTE["border"], ticksuffix="%", min=0,
                tickvals=[20, 40, 60, 80], tickfont=dict(size=10, color=PALETTE["muted"]))
    fig.update_layout(
        ternary=dict(sum=100, bgcolor=PALETTE["bg2"],
                     aaxis=dict(title=dict(text="P(Critical)", font=dict(color=PALETTE["critical"], size=12)), **axis),
                     baxis=dict(title=dict(text="P(Healthy)", font=dict(color=PALETTE["healthy"], size=12)), **axis),
                     caxis=dict(title=dict(text="P(Low)", font=dict(color=PALETTE["low"], size=12)), **axis)),
        margin=dict(l=50, r=50, t=90, b=96),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0.0))
    return fig


# ----------------------------------------------------------------------------
# Evaluation charts (need hidden ground truth: simulation only)
# ----------------------------------------------------------------------------
def fig_state_timeline(frame: pd.DataFrame, title: str = "Hidden inventory state vs. the agent's most probable state",
                       height: int = 380, step_col: str = "step") -> go.Figure:
    """Hidden state per period (evaluation only) with the belief's arg-max overlaid."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame[step_col], y=[STATES[int(i)] for i in frame["hidden_state"]], mode="lines",
        name="Hidden true state (evaluation only)", line=dict(color=PALETTE["text"], width=2, shape="hv"),
        hovertemplate="Period %{x}<br>True state: %{y}<extra></extra>"))
    if {"b_healthy", "b_low", "b_critical"}.issubset(frame.columns):
        valid = frame.dropna(subset=["b_healthy"])
        guess = valid[["b_healthy", "b_low", "b_critical"]].to_numpy().argmax(axis=1)
        hit = guess == valid["hidden_state"].to_numpy()
        names = np.array([STATES[i] for i in guess], dtype=object)
        steps = valid[step_col].to_numpy()
        for mask, label, symbol, color in ((hit, "Agent's most probable state: matches truth", "circle", PALETTE["cyan"]),
                                           (~hit, "Agent's most probable state: differs from truth", "x", PALETTE["critical"])):
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=steps[mask], y=names[mask], mode="markers", name=label,
                    marker=dict(size=8, symbol=symbol, color=color, line=dict(color=PALETTE["bg"], width=1)),
                    hovertemplate="Period %{x}<br>Most probable: %{y}<extra>" + label.split(": ")[1] + "</extra>"))
    _style(fig, title, height, x_title="Simulation period", y_title="Inventory state")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(STATES)), range=[-0.5, 2.5])
    _integer_periods(fig, frame[step_col])
    return fig


def fig_inventory_units(frame: pd.DataFrame,
                        title: str = "Synthetic inventory snapshots and demand (evaluation only)",
                        height: int = 380, step_col: str = "step") -> go.Figure:
    fig = go.Figure()
    for lo, hi, color, name in ((600, 1000, PALETTE["healthy"], "Healthy"), (250, 600, PALETTE["low"], "Low"),
                                (0, 250, PALETTE["critical"], "Critical")):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.07, line_width=0)
        # band labels sit in the right margin, outside the plot area, so they never cover data
        fig.add_annotation(xref="paper", x=1.01, xanchor="left", y=(lo + hi) / 2, yref="y", showarrow=False,
                           text=f"{name}<br>band", align="left", font=dict(size=10, color=color))
    fig.add_trace(go.Scatter(x=frame[step_col], y=frame["inventory_units"], mode="lines+markers",
                             name="Inventory snapshot (state-conditioned proxy)", line=dict(color=PALETTE["cyan"], width=2),
                             marker=dict(size=4),
                             hovertemplate="Period %{x}<br>Snapshot: %{y:.0f} units (drawn within the state band)<extra></extra>"))
    fig.add_trace(go.Scatter(x=frame[step_col], y=frame["demand_units"], mode="lines", name="Demand (synthetic)",
                             line=dict(color=PALETTE["violet"], width=1.6, dash="dot"),
                             hovertemplate="Period %{x}<br>Demand: %{y:.0f} units<extra></extra>"))
    short = frame[frame["stockout"]]
    if not short.empty:
        fig.add_trace(go.Scatter(x=short[step_col], y=short["inventory_units"], mode="markers",
                                 name="Stock-out (demand > snapshot)",
                                 marker=dict(symbol="x", size=10, color=PALETTE["critical"]),
                                 hovertemplate="Period %{x}<br>Stock-out<extra></extra>"))
    _style(fig, title, height, x_title="Simulation period", y_title="Units (synthetic proxy)")
    fig.update_yaxes(range=[0, 1050])
    fig.update_layout(margin=dict(r=78))
    _integer_periods(fig, frame[step_col])
    return fig


def fig_policy_comparison(summary: pd.DataFrame, metric: str = "total_cost",
                          title: str = "Total simulated cost by policy", y_title: Optional[str] = None,
                          highlight: Optional[str] = None, percent: bool = False, height: int = 420) -> go.Figure:
    """Bar chart with 95% confidence-interval error bars across replications."""
    scale = 100.0 if percent else 1.0
    values = scale * summary[metric].to_numpy()
    errors = scale * summary[f"{metric}_ci"].to_numpy() if f"{metric}_ci" in summary else None
    names = summary["policy"].tolist()
    colors = [PALETTE["cyan"] if (highlight and n == highlight) else
              (PALETTE["violet"] if "oracle" in n.lower() else PALETTE["slate"]) for n in names]
    wrapped = ["<br>".join(_wrap_words(n, 16)) for n in names]
    fmt = "%{y:.1f}%" if percent else "%{y:,.0f}"
    fig = go.Figure(go.Bar(
        x=wrapped, y=values, marker=dict(color=colors, line=dict(color=PALETTE["border"], width=1)),
        error_y=dict(type="data", array=errors, color=PALETTE["text"], thickness=1.4, width=5) if errors is not None else None,
        text=[f"{v:.1f}%" if percent else f"{v:,.0f}" for v in values], textposition="inside",
        insidetextanchor="start", textangle=0, textfont=dict(family=FONT_MONO, size=12, color=PALETTE["bg"]),
        customdata=np.array(names, dtype=object),
        hovertemplate=f"%{{customdata}}<br>{fmt} (mean; bar = 95% CI)<extra></extra>"))
    _style(fig, title, height, legend=False, x_title="Policy type",
           y_title=y_title or (f"Mean total simulated cost ({COST_UNIT})" if metric == "total_cost" else metric))
    # tick angle left on "auto": labels stay horizontal when they fit and are angled on narrow screens
    fig.update_xaxes(tickfont=dict(size=10))
    if percent:
        fig.update_yaxes(ticksuffix="%")
    # bar values stay horizontal; on a narrow screen a value that cannot fit its bar is hidden (hover keeps it)
    fig.update_layout(margin=dict(b=96), uniformtext=dict(minsize=10, mode="hide"))
    return fig


def fig_calibration(table: pd.DataFrame, title: str = "Belief calibration (reliability diagram)",
                    ece: Optional[float] = None, height: int = 420, series_name: str = "Belief agent") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect calibration",
                             line=dict(color=PALETTE["muted"], dash="dash", width=1.5), hoverinfo="skip"))
    if not table.empty:
        size = 8 + 22 * np.sqrt(table["count"] / table["count"].max())
        fig.add_trace(go.Scatter(
            x=100 * table["mean_predicted"], y=100 * table["observed_frequency"], mode="lines+markers",
            name=series_name + (f" (ECE {100 * ece:.1f} pts)" if ece is not None else ""),
            line=dict(color=PALETTE["cyan"], width=2), marker=dict(size=size, color=PALETTE["cyan"], opacity=0.85,
                                                                    line=dict(color=PALETTE["bg"], width=1)),
            customdata=table["count"],
            hovertemplate="Predicted %{x:.1f}%<br>Observed %{y:.1f}%<br>n = %{customdata}<extra></extra>"))
    _style(fig, title, height, x_title="Predicted probability (%)", y_title="Observed frequency (%)")
    fig.update_xaxes(range=[0, 102], ticksuffix="%")
    fig.update_yaxes(range=[0, 102], ticksuffix="%")
    return fig


def fig_calibration_multi(tables: Dict[str, pd.DataFrame], title: str, height: int = 440) -> go.Figure:
    """Overlay several reliability curves (e.g. well-specified vs. mismatched)."""
    shades = [PALETTE["cyan"], PALETTE["low"], PALETTE["critical"], PALETTE["violet"], PALETTE["healthy"]]
    symbols = ["circle", "diamond", "square", "star", "triangle-up"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect calibration",
                             line=dict(color=PALETTE["muted"], dash="dash", width=1.5), hoverinfo="skip"))
    for k, (name, table) in enumerate(tables.items()):
        fig.add_trace(go.Scatter(
            x=100 * table["mean_predicted"], y=100 * table["observed_frequency"], mode="lines+markers", name=name,
            line=dict(color=shades[k % 5], width=2), marker=dict(size=8, symbol=symbols[k % 5]),
            customdata=table["count"],
            hovertemplate=f"{name}<br>Predicted %{{x:.1f}}%<br>Observed %{{y:.1f}}%<br>n = %{{customdata}}<extra></extra>"))
    _style(fig, title, height, x_title="Predicted probability (%)", y_title="Observed frequency (%)")
    fig.update_xaxes(range=[0, 102], ticksuffix="%")
    fig.update_yaxes(range=[0, 102], ticksuffix="%")
    return fig


def fig_service_trend(frame: pd.DataFrame, title: str = "Service-level trend", height: int = 360,
                      x_col: str = "step", window: int = 5) -> go.Figure:
    """Realised fill rate per period with a rolling mean (simulation evaluation)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame[x_col], y=100 * frame["fill_rate"], mode="markers",
                             name="Fill rate per period (demand vs state-conditioned snapshot)",
                             marker=dict(size=5, color=PALETTE["slate"]),
                             hovertemplate="Period %{x}<br>Fill rate %{y:.1f}%<extra></extra>"))
    rolling = (frame["units_served"].rolling(window, min_periods=1).sum()
               / frame["demand_units"].rolling(window, min_periods=1).sum())
    fig.add_trace(go.Scatter(x=frame[x_col], y=100 * rolling, mode="lines", name=f"{window}-period rolling service level",
                             line=dict(color=PALETTE["healthy"], width=2.4),
                             hovertemplate="Period %{x}<br>Rolling service level %{y:.1f}%<extra></extra>"))
    _style(fig, title, height, x_title="Simulation period", y_title="Service level - fill rate (%)")
    fig.update_yaxes(range=[0, 104], ticksuffix="%")
    _integer_periods(fig, frame[x_col])
    return fig


def fig_lines_with_bands(frame: pd.DataFrame, x: str, series: str, y: str, ci: str, title: str,
                         x_title: str, y_title: str, height: int = 400, percent_x: bool = False) -> go.Figure:
    """Mean lines with shaded 95% CI bands, one line per value of `series` (sensitivity analysis)."""
    shades = [PALETTE["cyan"], PALETTE["low"], PALETTE["violet"], PALETTE["healthy"], PALETTE["critical"], PALETTE["slate"]]
    symbols = ["circle", "diamond", "star", "square", "triangle-up", "cross"]
    fig = go.Figure()
    for k, (name, part) in enumerate(frame.groupby(series, sort=False)):
        part = part.sort_values(x)
        xs = (100 * part[x]) if percent_x else part[x]
        color = shades[k % len(shades)]
        fig.add_trace(go.Scatter(x=xs, y=part[y], mode="lines+markers", name=str(name),
                                 line=dict(color=color, width=2), marker=dict(size=7, symbol=symbols[k % len(symbols)]),
                                 error_y=dict(type="data", array=part[ci], color=color, thickness=1, width=3),
                                 hovertemplate=f"{name}<br>{x_title}: %{{x}}<br>{y_title}: %{{y:,.1f}}<extra></extra>"))
    _style(fig, title, height, x_title=x_title, y_title=y_title)
    if percent_x:
        fig.update_xaxes(ticksuffix="%")
    return fig


# ----------------------------------------------------------------------------
# Architecture diagram (notebook, Section 2)
# ----------------------------------------------------------------------------
def fig_architecture(height: int = 430) -> go.Figure:
    """The POMDP feedback loop of the assignment infographic, drawn as a labelled diagram."""
    boxes = [
        (0.02, "1 · REAL-WORLD<br>ENVIRONMENT", "Hidden inventory state<br>Healthy / Low / Critical", PALETTE["slate"]),
        (0.185, "2 · PARTIAL<br>OBSERVATION", "Sensor · Supplier<br>Demand · System alerts", PALETTE["blue"]),
        (0.35, "3 · BELIEF STATE", "Bayes filter<br>predict → correct", PALETTE["cyan"]),
        (0.515, "4 · DECISION<br>POLICY", "Expected cost Q(b, a)<br>for all four actions", PALETTE["violet"]),
        (0.68, "5 · OPERATIONAL<br>ACTION", "Monitor · Reorder<br>Expedite · Reallocate", PALETTE["low"]),
        (0.845, "6 · COST /<br>REWARD", "Ordering, holding,<br>stock-out costs", PALETTE["healthy"]),
    ]
    width = 0.135
    fig = go.Figure()
    for x0, head, body, color in boxes:
        fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x0 + width, y0=0.42, y1=0.86,
                      line=dict(color=color, width=2), fillcolor=PALETTE["panel"], layer="below")
        fig.add_annotation(xref="x", yref="y", x=x0 + width / 2, y=0.74, text=f"<b>{head}</b>",
                           showarrow=False, font=dict(size=11, color=color, family=FONT_HEAD), align="center")
        fig.add_annotation(xref="x", yref="y", x=x0 + width / 2, y=0.53, text=body, showarrow=False,
                           font=dict(size=10, color=PALETTE["muted"]), align="center")
    for (x0, *_), (x1, *_) in zip(boxes[:-1], boxes[1:]):
        fig.add_annotation(xref="x", yref="y", x=x1 - 0.003, y=0.64, ax=x0 + width + 0.003, ay=0.64,
                           axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.2,
                           arrowwidth=1.6, arrowcolor=PALETTE["muted"], text="")
    # feedback loop: action changes the hidden state; new observations update the belief again
    fig.add_shape(type="path", xref="x", yref="y",
                  path="M 0.7475,0.42 L 0.7475,0.2 L 0.0875,0.2 L 0.0875,0.40",
                  line=dict(color=PALETTE["cyan"], width=1.6, dash="dot"))
    fig.add_annotation(xref="x", yref="y", x=0.0875, y=0.42, ax=0.0875, ay=0.3, axref="x",
                       ayref="y", showarrow=True, arrowhead=2, arrowcolor=PALETTE["cyan"], arrowwidth=1.6, text="")
    fig.add_annotation(xref="x", yref="y", x=0.4175, y=0.13, showarrow=False,
                       text="Executed action changes the hidden state  →  new observations arrive  →  belief updated again",
                       font=dict(size=11, color=PALETTE["cyan"]))
    fig.add_annotation(xref="x", yref="y", x=0.0875, y=0.93, showarrow=False,
                       text="hidden from the agent", font=dict(size=10, color=PALETTE["critical"]))
    fig.add_annotation(xref="x", yref="y", x=0.5, y=0.93, showarrow=False,
                       text="SENTINEL agent: sees observations only, never the true state",
                       font=dict(size=10, color=PALETTE["muted"]))
    _style(fig, "System architecture: the POMDP belief-state feedback loop", height, legend=False)
    fig.update_xaxes(visible=False, range=[0, 1], fixedrange=True)
    fig.update_yaxes(visible=False, range=[0, 1], fixedrange=True)
    fig.update_layout(margin=dict(l=20, r=20, t=84, b=20))
    return fig


# The mode bar is hidden so that it can never sit on top of a chart title; hover tooltips,
# drag-to-zoom and double-click-to-reset remain available.
CHART_CONFIG = {"displaylogo": False, "responsive": True, "displayModeBar": False}
