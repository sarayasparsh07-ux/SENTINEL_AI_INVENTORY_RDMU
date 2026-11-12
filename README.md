# SENTINEL: An Intelligent POMDP-Based Inventory Management and Decision Support System Under Uncertainty

*Bayesian Belief-State Estimation, Adaptive Decision Policies and Interactive Supply Chain Simulation.*

**Course:** Reasoning and Decision Making Under Uncertainty (RDMU) · **Assignment:** Session III, Lab Activity – Exercise 3

> **One-line summary.** SENTINEL implements an exact discrete Bayes filter (Kaelbling et al., 1998) and a transparent **myopic one-step expected-cost policy** for a three-state hidden inventory problem. It is **not** an optimal infinite-horizon POMDP solver and does not value information gathering. **All transition, observation and cost parameters, and all performance results, are synthetic.**

---

## 1. Project overview

In real supply chains, the true inventory condition is often unknown. Decision-makers see it only through sensors, supplier updates, demand signals and system alerts, each of which is incomplete and sometimes wrong. SENTINEL is an AI agent that:

1. observes the inventory environment only through imperfect alerts;
2. keeps a probability distribution (a *belief state*) over three hidden conditions: Healthy, Low and Critical;
3. updates that distribution with Bayes' rule whenever new alerts arrive;
4. evaluates four operational actions by expected cost;
5. recommends the cheapest action under the current belief and explains why;
6. separates what it **recommends** from what is **executed**, and only lets executed actions change the simulated inventory;
7. repeats the cycle as new evidence arrives.

The project has two deliverables built on **one shared Python engine** (`src/`):

| Deliverable | Entry point | Purpose |
|---|---|---|
| Academic Jupyter notebook | `notebooks/SENTINEL_POMDP_Inventory_Management.ipynb` | Mathematics, worked examples, scenarios A–H, evaluation, discussion |
| SENTINEL AI dashboard | `dashboard/app.py` (or `SENTINEL_AI_Inventory_Dashboard.py`) | Interactive command centre where alerts update the belief and the recommendation live |

## 2. Assignment objective

> Design and implement an interactive web-based dashboard using Python and HTML that demonstrates how an AI agent updates its Belief State and changes its decision policy when new alerts are received. The dashboard should be inspired by the provided MDP Belief State (POMDP) infographic and simulate inventory management in a supply chain environment.

The implementation follows the infographic's loop: real-world environment → partial observation → belief state → decision policy → operational action → reward/cost → new observation → belief updated again.

## 3. POMDP introduction

A **Markov Decision Process** assumes the agent sees the state. A **Partially Observable MDP** removes that assumption: the agent receives observations that are correlated with the hidden state and must act on a belief, a probability distribution over states. The belief is a sufficient statistic for the history of actions and observations when it is computed with the correct model (Åström, 1965; Kaelbling et al., 1998). Exact POMDP planning is intractable in general (Papadimitriou & Tsitsiklis, 1987), which is why SENTINEL uses a deliberately simple, explainable one-step policy.

## 4. Mathematical model

**POMDP = (S, A, T, O, Z, R, γ)**

| Component | SENTINEL definition |
|---|---|
| **S** | {Healthy, Low, Critical} (hidden). For evaluation only, each state has a unit band, Healthy [600, 1,000], Low [250, 600), Critical [0, 250) (non-overlapping and gap-free over [0, 1,000]), from which a synthetic inventory snapshot is drawn; see "Inventory units are state-conditioned snapshots" below |
| **A** | Continue Monitoring · Standard Reorder · Expedite Supplier Delivery · Adjust Production / Reallocate Stock |
| **T(s′ \| s, a)** | One 3×3 row-stochastic matrix per action. Optional modifiers: supplier risk ρ (`T′ = (1−ρ)T_a + ρT_monitor` for reorder/expedite) and demand volatility κ (extra one-level depletion) |
| **O** | Four channels: sensor (4 symbols), supplier (5), demand (4), system alerts (5) |
| **Z(o \| s)** | One likelihood matrix per channel, reliability-adjusted: `Z_r(o|s) = r·Z(o|s) + (1−r)/|O|` |
| **R** | `R(s, a, s′) = −C(s, a, s′)`, with `C = c_action(a) + c_current(s, a) + c_post(s′)` |
| **γ** | 0.95 (used only by the optional QMDP comparison; the primary policy looks one period ahead) |
| **b₀** | [0.60, 0.30, 0.10], the illustrative values printed on the lecture infographic (configurable) |

**Timing convention (one period).** (1) evaluate Q(b, a) and *recommend*; (2) *execute* an action; (3) hidden state transitions; (4) cost incurred; (5) each channel emits an observation; (6) **one** prediction with the executed action, then **one** joint correction with all of the period's observations. Alerts injected by hand between steps trigger only the correction, never a transition.

**Belief update.**

```
prediction:  b⁻(s′) = Σ_s T(s′ | s, a) b(s)
correction:  b′(s′) = η · Π_k Z_k(o_k | s′) · b⁻(s′),    η⁻¹ = Σ_s′ Π_k Z_k(o_k | s′) b⁻(s′)
```

The product over channels is the conditional-independence assumption. The correction runs in log-space with log-sum-exp normalisation; likelihoods are floored at 10⁻⁹; evidence with zero probability under the model is rejected and flagged.

**Inventory units are state-conditioned snapshots, not a stock ledger.** The only quantity with dynamics is the three-state hidden condition, which moves according to T. To report stock-outs and fill rate in units, the simulator draws, in every period, a synthetic inventory snapshot uniformly inside the current state's band, plus an independent demand. Snapshots are **not rolled forward**: there is no balance equation of the form *next stock = stock − demand + deliveries*. Stock-out frequency, fill rate and service level are computed per period from the snapshot and the demand, so they are state-driven proxies rather than outputs of a physically continuous inventory simulation. The "inventory recovery" charts are belief-based (P(Healthy) versus P(Critical)), and the evaluation's recovery measure counts periods in the hidden *state*, not units. None of these quantities is ever shown to the agent.

**Decision policy.**

```
Q(b, a) = c_action(a) + Σ_s b(s) c_current(s, a) + Σ_s′ b⁻_a(s′) c_post(s′)
        = Σ_s b(s) Σ_s′ T(s′ | s, a) C(s, a, s′)          (identical; unit-tested)
a*      = argmin_a Q(b, a)                                 R(b, a) = −Q(b, a)
```

Q is linear in b, so each action is optimal on a convex region of the belief simplex. The dashboard and notebook both draw this *policy map*; no threshold is hard-coded.

**Synthetic cost assumptions (simulation cost units, SCU).**

| | Monitor | Reorder | Expedite | Reallocate |
|---|---:|---:|---:|---:|
| Direct cost | 0 | 15 | 40 | 22 |
| Extra cost if currently Healthy (surplus / wasted premium / disruption) | 0 | 6 | 10 | 5 |

Post-transition cost: Healthy 4 (holding), Low 20 (service degradation), Critical 150 (stock-out penalty).

## 5. System architecture

```
                 ┌────────────────────────── src/ (shared engine) ───────────────────────────┐
                 │ config.py            all synthetic parameters + validation                │
                 │ transition_model.py  T(s'|s,a), modifiers ρ, κ                            │
                 │ observation_model.py Z_k(o|s), reliability mixture                        │
                 │ cost_model.py        C(s,a,s'), expected cost                             │
                 │ belief_updater.py    predict / log-space joint correct / diagnostics      │
                 │ decision_policy.py   evaluate_actions, explanations, baselines, oracle    │
                 │ inventory_simulator.py hidden-state environment, episodes, Monte Carlo    │
                 │ metrics.py           Brier, log loss, calibration, CIs                    │
                 │ pomdp_engine.py      SentinelEngine: interactive session (agent + env)    │
                 │ scenarios.py         scripted scenarios A–H and presets                   │
                 │ visualization.py     Plotly figures used by both front ends               │
                 └───────────────┬───────────────────────────────────────┬───────────────────┘
                                 │ import                                │ import
            notebooks/SENTINEL_..._Management.ipynb             dashboard/app.py (+ components.py,
                                                                 assets/styles.css, robot_identity.svg)
```

The simulator owns the hidden state. The only thing passed from the environment to the agent is the list of observations; a unit test confirms that identical evidence produces identical beliefs and recommendations whatever the hidden truth.

## 6. Folder structure

```
SENTINEL_POMDP_PROJECT/
├── notebooks/SENTINEL_POMDP_Inventory_Management.ipynb   executed academic notebook
├── dashboard/
│   ├── app.py                       Streamlit entry point
│   └── components.py                HTML building blocks (presentation only)
├── src/                             shared POMDP engine (see Section 5)
├── tests/                           124 automated tests
├── assets/styles.css                command-centre theme (pure CSS)
├── assets/robot_identity.svg        NOVA robot identity (original SVG)
├── data/sample_events.csv           120-period synthetic event log (generated by the notebook)
├── outputs/charts/                  35 PNG copies of the notebook charts
├── outputs/screenshots/             9 screenshots of the running dashboard
├── outputs/cross_check_notebook.json notebook numbers used by the dashboard cross-check test
├── .streamlit/config.toml           Streamlit theme
├── SENTINEL_AI_Inventory_Dashboard.py  convenience launcher
├── requirements.txt
└── README.md
```

## 7. Technologies and dependencies

Python 3.13 · NumPy · pandas · SciPy · Plotly · Streamlit 1.67.0 · HTML5/CSS3 · Jupyter · pytest.

- **Python version.** **Python 3.12 or newer is required**, because the pinned NumPy 2.5.3 and SciPy 1.18.1 both declare `Requires-Python >= 3.12`. **Python 3.13 is the tested environment.**
- **Version pinning.** Every runtime and test dependency in `requirements.txt` is pinned to an exact version (`==`) that was installed and tested together. The one exception is **JupyterLab**, which has a minimum-version requirement (`jupyterlab>=4.2`) because it is only needed to open the notebook interactively; the clean test installation resolved it to 4.6.6. Kaleido is optional (static PNG copies of the notebook charts).

## 8. Installation

```bash
cd SENTINEL_POMDP_PROJECT
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

All commands below are run from the `SENTINEL_POMDP_PROJECT` folder.

## 9. Running the notebook

```bash
jupyter lab notebooks/SENTINEL_POMDP_Inventory_Management.ipynb
```

The notebook is delivered already executed. To re-run it from top to bottom without opening it:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/SENTINEL_POMDP_Inventory_Management.ipynb
```

A full run takes about 3–4 minutes (most of it is the 300-replication evaluation in Section 9). It regenerates `data/sample_events.csv`, `outputs/cross_check_notebook.json` and, if Chrome/Chromium and Kaleido are available, the PNGs in `outputs/charts/`. Without Chrome the notebook still runs and the charts stay interactive. Charts render natively in JupyterLab and VS Code.

## 10. Running the dashboard

```bash
streamlit run dashboard/app.py
```

or `python SENTINEL_AI_Inventory_Dashboard.py`. The dashboard opens at `http://localhost:8501`. It is a local simulation: it makes no external data connection, and the header says so.

## 11. Dashboard functionality

| Section | What it shows / does |
|---|---|
| **1 · Command-centre header** | SENTINEL AI identity with the NOVA robot, system and model status, simulated clock, auto-simulation indicator, data-source disclaimer |
| **2 · KPI overview** | P(Healthy), P(Low), P(Critical), recommended action, expected cost (SCU), alerts processed, policy changes, belief concentration (1 − normalised entropy, explicitly not a probability of being right) |
| **3 · Belief-state monitor** | Three labelled probability bars with interpretations, stacked composition bar, belief-evolution stacked-area chart with event, timestamp and exact values on hover |
| **4 · Alert control room** | Four channels and every observation/severity symbol; reliability slider; optional operator note; duplicate-event guard; INJECT ALERT, RUN NEXT STEP, START AUTO-SIMULATION, PAUSE SIMULATION, RESET SIMULATION; manual override of the executed action; evidence table; event log (timestamp, source, observation, reliability, belief change, previous and new recommendation) |
| **5 · AI decision centre** | NOVA recommendation card with a dynamically generated explanation, expected cost, margin over the runner-up, main evidence, timestamp; ranked comparison of all four actions; stacked expected-cost chart (the engine's arg-min is outlined and labelled) |
| **6 · POMDP model explorer** | Transition heatmap per action; observation-likelihood heatmap per channel with a live reliability slider; Bayesian update explorer (prior → predicted → likelihood → posterior with equations and table); policy map; model assumptions |
| **7 · Operations centre** | Recommended versus executed action timeline, belief-based recovery trend, model-expected stock-out risk and service level, action log (recommended, executed, predicted outcome, observed subsequent evidence, posterior) |
| **8 · What-if lab** | Nine scripted presets (the seven required plus two extras); validated configuration form for initial belief, reliabilities, demand volatility, supplier risk, action and state costs, all four transition matrices, duration and seed |
| **9 · Simulation evaluation** | Monte Carlo comparison with fixed policies, a record-trusting heuristic, the infographic threshold rule and a full-information oracle: total cost, stock-out rate, service level, belief accuracy, Brier score, log loss, policy efficiency, calibration diagram, service-level trend, and two results tables for all policies (operational performance; belief quality) with hover definitions for every heading |
| **Debug view** | Hidden ground truth (true state, synthetic inventory snapshots and demand), revealed only on request in an isolated tab with a warning; never used by the agent |

Auto-simulation uses Streamlit's supported timed-fragment mechanism (`st.fragment(run_every=…)`); no injected JavaScript is relied on.

## 12. Simulation scenarios

| Notebook | Dashboard preset | Scenario |
|---|---|---|
| A | 1 | Normal inventory operations |
| C | 2 | Unexpected supplier delay (then cancellation) |
| D | 3 | Demand surge |
| E | 4 | Critical inventory warning |
| F | 5 | Conflicting sensor evidence |
| — | 6 | Multiple simultaneous alerts |
| H | 7 | Successful replenishment and recovery |
| B | 8 | Low reading from an unreliable sensor (30% reliable) |
| G | 9 | Multiple consecutive alerts with the operator overriding to "monitor" |

Scenarios are scripted and deterministic, so the notebook and the dashboard produce identical numbers.

## 13. Mathematical assumptions

| Category | Statement |
|---|---|
| **Published findings** | Belief is a sufficient statistic (Åström, 1965; Kaelbling et al., 1998). Finite-horizon POMDPs are PSPACE-complete (Papadimitriou & Tsitsiklis, 1987). 65% of ~370,000 records in 37 stores of one retailer were inaccurate (DeHoratius & Raman, 2008); this describes one retailer only. Independence assumptions push probabilities towards 0 and 1 (Niculescu-Mizil & Caruana, 2005). |
| **Derivations** | Two-stage filter; joint = sequential correction under conditional independence; r = 0 reliability leaves the belief unchanged; decomposed Q equals the Bellman double sum; Q is linear in b. |
| **Synthetic assumptions** | Every number in `src/config.py`: b₀ (from the infographic), all T and Z matrices, reliabilities, costs, inventory bands, demand distributions. |
| **Implementation decisions** | Inventory units are state-conditioned snapshots drawn each period with no stock roll-forward; one reading per channel per period (duplicate guard); likelihood floor 10⁻⁹; log-space normalisation; ties broken towards the less interventionist action; recommendation is never auto-executed except by the auto-simulation or "Run next step". |
| **Experimental results** | Everything in notebook Section 9 and the dashboard's evaluation tab: synthetic simulation output only. |

## 14. Testing and validation

```bash
python -m pytest -q          # 124 tests, about 1 minute
```

| File | Tests | What is checked |
|---|---:|---|
| `test_belief_updates.py` | 18 | Hand-checked worked example; posteriors valid for every observation; **joint = sequential correction in all orders**; one prediction per period; r = 0 is uninformative; reliability monotonicity; impossible and partial-zero evidence; log-space underflow (400 likelihoods); input validation; **Brier reference values 0.26 / 0.86 / 1.26**; log loss |
| `test_transition_model.py` | 24 | Row sums and bounds; modifiers preserve stochasticity (16 combinations); supplier-risk and volatility effects; invalid matrices rejected; readable config validation; observation matrices valid and directional |
| `test_decision_policy.py` | 16 | All actions evaluated; **decomposed Q = Bellman double sum = linear form**; hand-calculated Q at b₀; arg-min selected; every action optimal somewhere; alerts change belief and recommendation; **alerts never apply a transition; only the executed action drives the prediction**; **hidden state never influences beliefs or decisions**; explanations change with the numbers |
| `test_simulation.py` | 32 | Seed reproducibility; valid beliefs over 200 periods; common random numbers; **interactive engine = batch simulator**; duplicate guard and double counting; reset; apply-config recomputation; all 9 scenarios deterministic; belief agent beats naive policies; **model-mismatch degrades Brier and log loss**; metrics helpers; **inventory bands partition [0, 1,000] without gaps or overlaps, and every logged snapshot lies in its state's half-open band** |
| `test_dashboard.py` | 22 | Real `dashboard/app.py` driven headlessly (`streamlit.testing.v1.AppTest`): launch, every button, reliability slider, operator note, duplicate toggle, step, manual override, start/pause/duration limit, reset, **all 9 presets match the engine**, invalid config rejected with a message, valid config recomputes and restores, initial belief editable, batch evaluation, debug view gated |
| `test_integration.py` | 12 | **Notebook ↔ dashboard cross-check** on three scenarios: prior, predicted, likelihood, posterior, expected cost per action and recommendation agree to 10⁻¹⁰ between the notebook's low-level maths, the engine, the dashboard and the numbers saved by the executed notebook |

**Additional checks performed for this submission** (not part of `pytest`):

- A clean Python 3.13.18 virtual environment was created and `pip install -r requirements.txt` completed without errors. The test suite, the notebook and the dashboard were all run from that environment: 124 passed, 0 failed, 0 skipped.
- The notebook was executed top to bottom on that environment's kernel: 44 of 44 code cells ran in order, with no errors.
- The dashboard was driven in Google Chrome with Playwright. Inject alert, duplicate guard, run next step, start auto-simulation (advanced 3 periods in 9 s at a 2 s interval), pause (no further periods over 6 s), load scenario, invalid configuration (rejected, session unchanged), valid configuration, restore defaults, batch evaluation, debug view and reset all worked.
- Layout was inspected at 1440, 1280, 1024, 768 and 420 px wide. No element overflows the page horizontally at any width; on narrow screens wide tables scroll inside their own frame and the tab bar scrolls.

## 15. Limitations

- **Synthetic data only.** All probabilities and costs are assumed; results describe the simulator, not a real supply chain.
- **Myopic policy.** One-period look-ahead. In the synthetic evaluation it is a few percent *dearer* than QMDP and the infographic's threshold rule at the default costs, and too passive when stock-outs are cheap. It never acts in order to learn.
- **Conditional independence.** Cross-channel correlation causes double counting; the duplicate guard handles exact repeats only. The correlated-alert mismatch experiment shows a large calibration error.
- **Model mismatch.** Over-estimated reliabilities make the agent overconfident; mis-specified dynamics raise cost.
- **Inventory units are proxies.** Units are drawn independently inside the hidden state's band each period, with no stock roll-forward. Stock-out frequency, fill rate and service level are per-period, state-driven proxies, not results of a physically continuous inventory simulation; order quantities and lead times are not modelled in units.
- **Three-state abstraction, one item, one location, stationary parameters.**
- **Calibration in the well-specified simulator verifies the code, not the model.**

## 16. Future improvements

Estimate T and Z from logged data (EM / Baum–Welch) and monitor drift; add a costed physical-count action (Kök & Shang, 2007) to study the value of information; finite-horizon lookahead or point-based solvers such as SARSOP (Kurniawati et al., 2008) over the 2-D belief simplex; predictive demand models; correlation-aware evidence fusion; hierarchical and multi-agent extensions to supply-chain networks.

## References

Åström, K. J. (1965). Optimal control of Markov processes with incomplete state information. *Journal of Mathematical Analysis and Applications, 10*(1), 174–205. https://doi.org/10.1016/0022-247X(65)90154-X

Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review, 78*(1), 1–3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

DeHoratius, N., Mersereau, A. J., & Schrage, L. (2008). Retail inventory management when records are inaccurate. *Manufacturing & Service Operations Management, 10*(2), 257–277. https://doi.org/10.1287/msom.1070.0203

DeHoratius, N., & Raman, A. (2008). Inventory record inaccuracy: An empirical analysis. *Management Science, 54*(4), 627–641. https://doi.org/10.1287/mnsc.1070.0789

Domingos, P., & Pazzani, M. (1997). On the optimality of the simple Bayesian classifier under zero-one loss. *Machine Learning, 29*(2–3), 103–130. https://doi.org/10.1023/A:1007413511361

Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association, 102*(477), 359–378. https://doi.org/10.1198/016214506000001437

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning* (Proceedings of Machine Learning Research, Vol. 70, pp. 1321–1330). https://proceedings.mlr.press/v70/guo17a.html

Huber, P. J. (1964). Robust estimation of a location parameter. *The Annals of Mathematical Statistics, 35*(1), 73–101. https://doi.org/10.1214/aoms/1177703732

Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and acting in partially observable stochastic domains. *Artificial Intelligence, 101*(1–2), 99–134. https://doi.org/10.1016/S0004-3702(98)00023-X

Kochenderfer, M. J., Wheeler, T. A., & Wray, K. H. (2022). *Algorithms for decision making*. MIT Press. https://algorithmsbook.com

Kök, A. G., & Shang, K. H. (2007). Inspection and replenishment policies for systems with inventory record inaccuracy. *Manufacturing & Service Operations Management, 9*(2), 185–205. https://doi.org/10.1287/msom.1060.0136

Kurniawati, H., Hsu, D., & Lee, W. S. (2008). SARSOP: Efficient point-based POMDP planning by approximating optimally reachable belief spaces. In *Robotics: Science and Systems IV*. https://doi.org/10.15607/RSS.2008.IV.009

Littman, M. L., Cassandra, A. R., & Kaelbling, L. P. (1995). Learning policies for partially observable environments: Scaling up. In A. Prieditis & S. Russell (Eds.), *Proceedings of the Twelfth International Conference on Machine Learning* (pp. 362–370). Morgan Kaufmann. https://doi.org/10.1016/B978-1-55860-377-6.50052-9

Mersereau, A. J. (2013). Information-sensitive replenishment when inventory records are inaccurate. *Production and Operations Management, 22*(4), 792–810. https://doi.org/10.1111/j.1937-5956.2011.01305.x

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). Association for Computing Machinery. https://doi.org/10.1145/1102351.1102430

Papadimitriou, C. H., & Tsitsiklis, J. N. (1987). The complexity of Markov decision processes. *Mathematics of Operations Research, 12*(3), 441–450. https://doi.org/10.1287/moor.12.3.441

Ross, S., Pineau, J., Paquet, S., & Chaib-draa, B. (2008). Online planning algorithms for POMDPs. *Journal of Artificial Intelligence Research, 32*, 663–704. https://doi.org/10.1613/jair.2567

Silver, E. A., Pyke, D. F., & Thomas, D. J. (2017). *Inventory and production management in supply chains* (4th ed.). CRC Press.

Smallwood, R. D., & Sondik, E. J. (1973). The optimal control of partially observable Markov processes over a finite horizon. *Operations Research, 21*(5), 1071–1088. https://doi.org/10.1287/opre.21.5.1071

Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic robotics*. MIT Press.
