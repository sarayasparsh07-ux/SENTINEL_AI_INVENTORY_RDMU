"""HTML building blocks for the SENTINEL AI dashboard (presentation only).

Nothing in this file computes probabilities, costs or recommendations: every
number is passed in from the shared engine in `src/`. The HTML is static
markup styled by assets/styles.css; no JavaScript is used.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

STATE_VARS = ("var(--healthy)", "var(--low)", "var(--critical)")
STATE_GLYPHS = ("●", "◆", "■")          # shape differs per state, so status never relies on colour alone
STATE_NAMES = ("Healthy", "Low", "Critical")
STATE_MEANING = (
    "adequate stock; no immediate replenishment concern",
    "below the preferred operating level; replenishment may be needed",
    "dangerously low; stock-out and service disruption risk",
)
SEVERITY_BADGE = {"Nominal": "ok", "Advisory": "info", "Warning": "warn", "Critical": "crit", "Fault": "ai"}


@lru_cache(maxsize=1)
def load_css() -> str:
    return (ASSETS / "styles.css").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def nova_data_uri() -> str:
    svg = (ASSETS / "robot_identity.svg").read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(svg).decode("ascii")


def section_title(tag: str, title: str, subtitle: str = "") -> str:
    sub = f'<p class="sub">{escape(subtitle)}</p>' if subtitle else ""
    return (f'<div class="sx-title"><span class="tag">{escape(tag)}</span>'
            f'<h3>{escape(title)}</h3>{sub}</div>')


def note(text_html: str) -> str:
    """Small explanatory paragraph. `text_html` may contain <b> tags written by the app itself."""
    return f'<p class="sx-note">{text_html}</p>'


def label(text: str) -> str:
    return f'<div class="sx-label">{escape(text)}</div>'


# ------------------------------------------------------------------ header
def header_html(period: int, timestamp: str, auto_running: bool, horizon: int) -> str:
    if auto_running:
        live = '<span class="sx-dot on"></span>Auto-simulation running'
    else:
        live = '<span class="sx-dot idle"></span>Paused · manual stepping'
    return f"""
<div class="sx-header" role="banner">
  <div class="sx-brand">
    <img class="sx-nova-img" src="{nova_data_uri()}" alt="NOVA robot assistant identity">
    <div class="sx-brand-text">
      <p class="sx-logo">SENTINEL <em>AI</em></p>
      <p class="sx-subtitle">Autonomous Inventory Intelligence &amp; Decision Centre</p>
      <span class="sx-chip">RDMU | POMDP BELIEF-STATE ENGINE</span>
    </div>
  </div>
  <div class="sx-status" aria-label="System status">
    <div class="sx-pill"><span class="k">System status</span>
      <span class="v"><span class="sx-dot info"></span>Engine online</span></div>
    <div class="sx-pill"><span class="k">Simulation clock</span>
      <span class="v">{escape(timestamp)} · period {period} of {horizon}</span></div>
    <div class="sx-pill"><span class="k">Model status</span>
      <span class="v">Bayes filter + myopic expected-cost policy</span></div>
    <div class="sx-pill"><span class="k">Simulation mode</span><span class="v">{live}</span></div>
    <div class="sx-pill" style="grid-column: 1 / -1;"><span class="k">Data source</span>
      <span class="v">Synthetic simulation only — no external live-data connection</span></div>
  </div>
</div>"""


# ------------------------------------------------------------------ KPI cards
def _kpi(label_text: str, value_html: str, detail: str, accent: str, glyph: str, small: bool = False) -> str:
    cls = "v small" if small else "v"
    return (f'<div class="sx-kpi" style="--accent: {accent};"><div class="k"><span class="glyph">{glyph}</span>'
            f'{escape(label_text)}</div><div class="{cls}">{value_html}</div><div class="d">{escape(detail)}</div></div>')


def kpi_cards_html(k: Dict, cost_unit: str, runner_up: str, margin: float) -> str:
    conc = 100 * k["concentration"]
    conc_word = "sharply peaked" if conc >= 60 else "moderately peaked" if conc >= 25 else "diffuse (uncertain)"
    cards = [
        _kpi("P(Healthy inventory)", f'{100 * k["p_healthy"]:.1f}<span class="unit">%</span>',
             "Belief probability of adequate stock", "var(--healthy)", STATE_GLYPHS[0]),
        _kpi("P(Low inventory)", f'{100 * k["p_low"]:.1f}<span class="unit">%</span>',
             "Belief probability of below-target stock", "var(--low)", STATE_GLYPHS[1]),
        _kpi("P(Critical inventory)", f'{100 * k["p_critical"]:.1f}<span class="unit">%</span>',
             "Belief probability of stock-out danger", "var(--critical)", STATE_GLYPHS[2]),
        _kpi("Recommended action", escape(k["recommended_action"]),
             "Computed by the decision engine; not yet executed", "var(--cyan)", "➤", small=True),
        _kpi("Expected operational cost", f'{k["expected_cost"]:.1f}<span class="unit">{escape(cost_unit)}</span>',
             f"Next period, if the recommendation is executed. {margin:.1f} {cost_unit} below {runner_up}.",
             "var(--blue)", "Σ"),
        _kpi("Alerts processed", f'{k["alerts_processed"]:d}', "Observations counted as evidence (integer)",
             "var(--cyan)", "⚑"),
        _kpi("Policy changes", f'{k["policy_changes"]:d}', "Times the recommendation has switched (integer)",
             "var(--violet)", "⇄"),
        _kpi("Belief concentration", f'{conc:.0f}<span class="unit">/ 100</span>',
             f"1 − normalised entropy; {conc_word}. An index, not a probability of being right.",
             "var(--violet)", "◎"),
    ]
    return '<div class="sx-kpis" role="list" aria-label="Key performance indicators">' + "".join(cards) + "</div>"


# ------------------------------------------------------------------ belief monitor
def _interpret(p: float, state: int) -> str:
    if p >= 0.5:
        level = "Dominant hypothesis"
    elif p >= 0.2:
        level = "Material possibility"
    elif p >= 0.05:
        level = "Minor possibility"
    else:
        level = "Unlikely"
    return f"{level}: {STATE_MEANING[state]}."


def belief_monitor_html(belief: Sequence[float]) -> str:
    rows: List[str] = []
    for i, p in enumerate(belief):
        pct = 100 * float(p)
        rows.append(
            f'<div class="sx-belief-row" style="--accent: {STATE_VARS[i]};">'
            f'<div class="name"><span class="glyph">{STATE_GLYPHS[i]}</span>{STATE_NAMES[i]}</div>'
            f'<div class="sx-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" '
            f'aria-valuenow="{pct:.1f}" aria-label="{STATE_NAMES[i]} inventory belief probability">'
            f'<div class="sx-fill" style="width: {pct:.2f}%;"></div></div>'
            f'<div class="pct">{pct:.1f}%</div></div>'
            f'<p class="sx-interp">{escape(_interpret(float(p), i))}</p>')
    segments = "".join(
        f'<div class="seg" style="width: {100 * float(p):.2f}%; background: {STATE_VARS[i]};" '
        f'title="{STATE_NAMES[i]}: {100 * float(p):.1f}%">'
        f'{(STATE_GLYPHS[i] + " " + format(100 * float(p), ".0f") + "%") if p >= 0.09 else ""}</div>'
        for i, p in enumerate(belief))
    legend = "".join(
        f'<span><span class="sw" style="background: {STATE_VARS[i]};"></span>{STATE_GLYPHS[i]} {STATE_NAMES[i]} '
        f'{100 * float(p):.1f}%</span>' for i, p in enumerate(belief))
    return ("".join(rows) + label("Stacked belief composition (sums to 100%)")
            + f'<div class="sx-stack" aria-label="Stacked belief distribution">{segments}</div>'
            + f'<div class="sx-legend">{legend}</div>')


# ------------------------------------------------------------------ recommendation
def recommendation_html(headline: str, paragraph: str, nova_line: str, main_evidence: str, belief: Sequence[float],
                        expected_cost: float, margin: float, runner_up: str, cost_unit: str, timestamp: str,
                        executed_note: str) -> str:
    belief_text = " · ".join(f"{STATE_GLYPHS[i]} {STATE_NAMES[i]} {100 * float(p):.1f}%" for i, p in enumerate(belief))
    return f"""
<div class="sx-rec">
  <div class="eyebrow">Decision engine output</div>
  <div class="headline">{escape(headline)}</div>
  <span class="flag">{escape(executed_note)}</span>
  <div class="sx-nova">
    <img src="{nova_data_uri()}" alt="NOVA">
    <div class="sx-bubble"><span class="who">NOVA · NEURAL OPERATIONS &amp; VISION ASSISTANT</span>{escape(nova_line)}</div>
  </div>
  <p class="why">{escape(paragraph)}</p>
  <div class="sx-facts">
    <div class="sx-fact"><div class="k">Expected total cost (next period)</div>
      <div class="v">{expected_cost:.2f} {escape(cost_unit)}</div></div>
    <div class="sx-fact"><div class="k">Margin over runner-up</div>
      <div class="v">{margin:.2f} {escape(cost_unit)} vs {escape(runner_up)}</div></div>
    <div class="sx-fact"><div class="k">Current belief (H / L / C)</div><div class="v">{escape(belief_text)}</div></div>
    <div class="sx-fact"><div class="k">Main evidence this period</div><div class="v">{escape(main_evidence)}</div></div>
    <div class="sx-fact" style="grid-column: 1 / -1;"><div class="k">Decision timestamp (simulated clock)</div>
      <div class="v">{escape(timestamp)}</div></div>
  </div>
</div>"""


def alternatives_table_html(actions: Sequence[str], q_values: Sequence[float], best: int,
                            predicted_critical: Sequence[float], cost_unit: str) -> str:
    rows = []
    order = sorted(range(len(actions)), key=lambda i: q_values[i])
    for i in order:
        gap = q_values[i] - q_values[best]
        tag = ('<span class="sx-badge ok">✓ recommended</span>' if i == best
               else f'<span class="sx-badge">+{gap:.1f} {escape(cost_unit)}</span>')
        rows.append(
            f'<tr class="{"best" if i == best else ""}"><td>{escape(actions[i])}</td>'
            f'<td class="num">{q_values[i]:.2f}</td>'
            f'<td class="num">{100 * predicted_critical[i]:.1f}%</td><td>{tag}</td></tr>')
    return ('<div class="sx-table-wrap"><table class="sx-table"><thead><tr><th>Action</th>'
            f'<th>Expected total cost ({escape(cost_unit)})</th>'
            '<th>Predicted P(Critical) next period</th><th>Versus best</th></tr></thead><tbody>' + "".join(rows)
            + "</tbody></table></div>")


def evidence_table_html(evidence: Sequence[dict]) -> str:
    if not evidence:
        return note("No observations have been counted in the current period yet. "
                    "The belief shown is the model's prediction.")
    rows = []
    for e in sorted(evidence, key=lambda x: x["impact"], reverse=True):
        badge = SEVERITY_BADGE.get(e["severity"], "info")
        like = " / ".join(f"{100 * v:.1f}" for v in e["likelihood"])
        rows.append(
            f'<tr><td>{escape(e["channel_label"])}</td><td>{escape(e["observation"])}<br>'
            f'<span class="sx-badge {badge}">{escape(e["severity"])}</span> '
            f'<span class="sx-badge">{100 * e["reliability"]:.0f}% reliable · {escape(e["origin"])}</span></td>'
            f'<td class="num">{like}</td><td class="num">{100 * e["impact"]:.1f} pts</td></tr>')
    return ('<div class="sx-table-wrap"><table class="sx-table"><thead><tr><th>Channel</th>'
            '<th>Observation · severity · reliability</th><th>P(o | H / L / C) in %</th><th>Belief shift</th>'
            '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def simple_table_html(headers: Sequence[str], rows: Sequence[Sequence[str]], numeric_from: int = 1) -> str:
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f'<td class="{"num" if j >= numeric_from else ""}">{escape(str(c))}</td>'
                         for j, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="sx-table-wrap"><table class="sx-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def results_table_html(headers: Sequence[str], rows: Sequence[Sequence[str]], tooltips: Sequence[str] = (),
                       highlight_row: int = -1, caption: str = "") -> str:
    """Compact results table: wrapping headings with hover definitions, numeric columns right-aligned.

    The first column is a label column; the remaining columns are numeric. On narrow screens the
    table scrolls horizontally inside its own frame (sx-table-wrap), never the page.
    """
    tips = list(tooltips) + [""] * (len(headers) - len(tooltips))
    head = "".join(
        f'<th scope="col" class="{"num" if j else "label"}"'
        + (f' title="{escape(tips[j])}"' if tips[j] else "")
        + f'>{escape(str(h))}{" ⓘ" if tips[j] else ""}</th>'
        for j, h in enumerate(headers))
    body = "".join(
        f'<tr class="{"best" if i == highlight_row else ""}">'
        + "".join(f'<td class="{"num" if j else "label"}">{escape(str(c))}</td>' for j, c in enumerate(r))
        + "</tr>" for i, r in enumerate(rows))
    # the caption sits OUTSIDE the scrolling frame, so it stays fully visible on narrow screens
    cap = f'<div class="sx-results-cap">{escape(caption)}</div>' if caption else ""
    return (f'{cap}<div class="sx-table-wrap sx-results"><table class="sx-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def scope_legend_html(period: int, alerts: int, batch_label: str) -> str:
    return f"""
<div class="sx-scope">
  <div class="box"><div class="k">① Model configuration</div>
    <div class="v">What the agent assumes: prior, transition, observation and cost parameters. Edited in the form below.</div></div>
  <div class="box"><div class="k">② Current operational simulation</div>
    <div class="v">The live session above: period {period}, {alerts} alerts processed. Changed by alerts, steps and presets.</div></div>
  <div class="box"><div class="k">③ Historical simulation results</div>
    <div class="v">Batch Monte Carlo runs in the “Simulation evaluation” tab. {escape(batch_label)}</div></div>
</div>"""


def metric_cards_html(items: Sequence[tuple]) -> str:
    """items: (label, value_html, detail, accent css var, glyph)."""
    return '<div class="sx-kpis">' + "".join(_kpi(a, b, c, d, e) for a, b, c, d, e in items) + "</div>"


FOOTER = ('<div class="sx-footer">SENTINEL AI · RDMU Session III, Lab Exercise 3 · exact discrete Bayes filter with a '
          'myopic one-step expected-cost policy (not an optimal infinite-horizon POMDP solver) · '
          'all probabilities, costs and results are synthetic simulation assumptions.</div>')
