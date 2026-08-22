# Fraud Spike Investigator

**Razorpay Buildathon — Track 02: AI Risk Manager.** A defense-only, merchant-level risk system: it detects sudden abnormal fraud spikes in a merchant's transaction stream, correlates the entities behind them (customers / devices / IPs / payment instruments), and produces an explained, bounded, human-reviewable decision — with honest, temporally-evaluated metrics including false-positive cost in ₹.

**→ Judges: [SUBMISSION.md](SUBMISSION.md) is the one-page summary mapped to the judging criteria.**

Per-order scorers flag individual bad orders. Nobody tells the merchant *"you are under attack right now, here's why, who's behind it, what it costs, and what to do."* This system closes that loop — and it sits **above** per-order scoring, complementary to tools like Thirdwatch/Shield, not competing with them.

## Quickstart

```bash
pip install -r requirements.txt
python -m src.models.select_model    # compare model families on validation → pick winner
python -m src.policy.threshold_sweep # cost-optimize policy cutoffs on validation
python -m src.models.train           # simulate → features → train → metrics → fusion → spike replay
python -m src.models.ablation        # feature-group ablation table
python -m src.agent.eval             # 10-case agent eval (needs ANTHROPIC_API_KEY for live model)
python run_demo.py                   # dashboard + live replay -> http://127.0.0.1:8000
python -m pytest tests/ -q           # safety invariants (46 tests)
```

## Model selection

The transaction scorer is a **GBDT, selected empirically** — not a fixed library choice. Before any tuning, four classifiers were trained with **identical class weighting and default hyperparameters** (no per-model tuning) and compared on the temporal **validation slice only** (days 21–23); the test slice is never touched for this decision. Selection rule, fixed *before* looking at results: highest validation PR-AUC wins; if the top GBDTs land within 0.02 PR-AUC of each other, break the tie by speed + maintainability and say so explicitly.

| Model | Family | Val PR-AUC | Train time | Inference time |
|---|---|---|---|---|
| LightGBM | GBDT | **0.9308** | 1.64s | 0.0085s |
| CatBoost | GBDT | 0.9265 | 7.94s | 0.0047s |
| **XGBoost** ← selected | GBDT | 0.9249 | **0.27s** | 0.0064s |
| LogisticRegression | linear | 0.9124 | 0.13s | 0.0008s |

**Winner: XGBoost — and the tie-break is what chose it, not the raw metric.** All three GBDTs land inside the 0.02 margin (0.9249–0.9308), so the pre-declared rule fired: pick by speed + maintainability. XGBoost trains ~6x faster than LightGBM and ~29x faster than CatBoost for a PR-AUC difference of 0.0059 — well inside noise for a 6,166-row validation slice. Choosing LightGBM on that gap would be optimizing a rounding error.

This is worth stating plainly because **an earlier version of this table looked completely different** (XGBoost 0.669 / LightGBM 0.656 / CatBoost 0.648 / LogReg 0.511) and XGBoost led on raw PR-AUC. The cause was a methodology bug, not model behavior: the validation slice (days 21–23) originally contained *no attack at all*, only ~0.6% ambient fraud — so validation PR-AUC was measuring "can you rank ambient fraud," not "can you catch an attack," which is what the model is selected to do. Moving one historical device-farm attack into day 22 fixed the slice. The winner survived the change, but the *reason* flipped from "leads on PR-AUC" to "wins the tie-break" — which is exactly why the rule was fixed in advance.

Honest caveat: validation now contains one attack type (device farm), so validation PR-AUC is still a narrow measure. It is used only to rank model families, never to report performance — that comes from the test slice.

Reproduce: `python -m src.models.select_model` → writes `artifacts_out/model_selection.csv` and `artifacts_out/model_selection_decision.json` (the persisted winner — `train.py` and `ablation.py` read this file and never hardcode a library).

## Results (temporal held-out test set, synthetic data — honestly labeled as such)

| Metric | Value |
|---|---|
| Model | XGBoost (empirically selected — see Model selection above) |
| PR-AUC | 0.934 |
| Precision / Recall @ cost-optimal threshold | 0.994 / 0.886 |
| Precision@100 / @500 | 1.00 / 1.00 (stable under adversarial tie-breaking) |
| Fraud ₹ prevented (test slice) | ₹10.52L |
| Legitimate ₹ wrongly blocked | ₹5.9K |
| Attack merchants detected at merchant level | 5 / 5 (card-testing, device farm, IP cluster, ATO, fraud ring) |
| False merchant-level alarms | 0 — incl. a 6x legitimate flash-sale spike, correctly NOT flagged |

### Net protected value — the economics of the *decisions*, not the model

Every test transaction is routed through the actual policy engine (allow / step-up / review / restrict), and each action is costed with documented assumptions (₹50 per human review; 7% of legitimate customers abandon at step-up; 90% of fraud fails step-up):

| | |
|---|---|
| Fraud exposure prevented | **₹10.93L** |
| Legitimate revenue impacted | ₹4,814 |
| Human review cost (617 cases × ₹50) | ₹30,850 |
| **Net protected value** | **₹10.57L** |
| ₹ protected per ₹ of cost imposed | ~31x |

(Figures use the validation-optimized step-up cutoff — see Policy thresholds below. Under the old hand-set 85/60 policy: ₹10.20L net, ~33x. The ratio drops slightly while net value rises, because a lower step-up cut catches more fraud *and* touches more legitimate customers — the ₹ number is the one being optimized, not the ratio.)

This reframes the result from "our model has good metrics" to "our system makes economically sensible risk decisions."

### Risk fusion (P1b) — and the honest result that it changes nothing here

The economics loop originally used `risk_score = p_fraud * 100, confidence = 0.85`. That shortcut discarded every signal except the ML probability — the spike z-score, entity-graph structure, and rule hits were all computed and then thrown away — and because confidence was a hardcoded constant, the policy engine's low-confidence escalation branch was unreachable dead code.

`src/policy/fusion.py` replaces it with an explicit, linear, bounded fusion: the calibrated ML probability sets a **floor**, and corroborating context (spike 0.50 / graph 0.30 / rules 0.20, scaled by a 0.6 lift factor) escalates into the headroom above it. Confidence is computed from signal **agreement**, so a high ML score with nothing corroborating comes out *less* confident and routes to a human rather than an automatic restrict.

**The honest headline: fusion changes 0 of 13,987 decisions on this dataset.** Net protected value is identical to the p\*100 shortcut (₹10.20L), and the action mix is unchanged. Context lift reaches 20 risk points at maximum but never crosses a policy threshold, because a model with PR-AUC 0.934 leaves very few transactions in the ambiguous mid-band where corroboration could tip a decision. Low-confidence escalations: 0.

Fusion is kept anyway, and on architectural grounds only — stated plainly rather than dressed up as a metric win:
- **Auditability.** Every decision now carries a per-signal breakdown (`artifacts_out/fusion_scores.csv`) — the "why this score" the P2 investigator and P3 dashboard need.
- **A reachable fail-safe.** The low-confidence branch is now live rather than dead code (pytest-enforced), and ML-unavailable reaches `decide()` as `risk_score=None` so the documented fail-safe fires instead of a context-only score that would understate risk.
- **Headroom for a weaker model.** The value of corroboration scales with model uncertainty; a system running a less separable model, or facing a novel attack the model scores poorly, is where this layer earns its place.

Two bugs found and fixed while building it, both caught by measurement rather than review: (1) the first implementation was a plain weighted average, which capped a `p=1.0` transaction at risk 60 — silently overruling the calibrated model it was supposed to defer to; (2) `COMPONENT_SATURATE=15` handed a full graph-risk bonus to **26% of legitimate transactions**, because ordinary customers already sit in entity components of ~10 via shared ISP IP pools. The graph signal now measures *excess* over the ordinary population, with both constants derived from the train slice only (legit p99 = 25, fraud p90 = 120); legit mean risk dropped 9.02 → 0.67.

### Time-to-detect: how fast do we know a merchant is under attack?

Ground-truth attack start times come from the simulator. Three systems compared — a static volume threshold (2x mean hourly volume), a naive flag-counter (alarm on the 10th model-flagged transaction), and our streaming spike detector (≥8 high-risk among the merchant's last 30 transactions within a bounded span):

| Scenario | Static volume | Flag counter | **Streaming spike** |
|---|---|---|---|
| Card-testing wave | not detected | 62m47s | **55m22s** |
| Device farm | 3m59s | 21m54s | **19m18s** |
| IP cluster | not detected | **26m15s** | 31m03s |
| Account takeover | not detected | 53m38s | **53m30s** |
| Fraud ring | not detected | 100m49s | **69m37s** |
| **False alarms** | **5 merchants (incl. flash sale)** | 0* | **0** |

*The flag-counter shows zero false alarms only because the test window is 6 days — it has no time or rate context, so ambient fraud drip-accumulates flags and it must eventually cry wolf on ordinary merchants; the streaming detector's rate + span guards make that structurally impossible. The static volume threshold, meanwhile, flags the flash sale (it cannot tell a sale from an attack) while missing 4 of 5 real attacks.

Streaming spike beats the naive flag-counter on 4 of 5 scenarios (largest margin: fraud ring, 70m vs 101m) and **loses on IP cluster, 31m vs 26m** — reported, not smoothed over. The static volume threshold is fast on the device farm only because a device farm happens to be a volume event; it misses the four attacks that aren't, and flags the flash sale.

## Ablation study

How much does each feature group — and the merchant-level spike/policy layer on top of the classifier — actually contribute? Same temporal split, same calibration and cost-optimal-threshold procedure as above, evaluated on the test slice, using the empirically-selected model (XGBoost):

| Stage | Features | PR-AUC | Precision | Recall | Fraud ₹ prevented | Legit ₹ wrongly blocked |
|---|---|---|---|---|---|---|
| 1. basics | 6 | 0.661 | 0.988 | 0.572 | ₹5.34L | ₹25.3K |
| 2. + velocity | 12 | 0.631 | 0.920 | 0.577 | ₹5.61L | ₹28.6K |
| 3. + entity/graph (full 22) | 22 | **0.934** | 0.994 | 0.886 | ₹10.52L | ₹5.9K |
| 4. full system (+ spike/policy layer) | 22 | 0.934 | — | — | — | — |

Stage 4 isn't a bigger feature set — stage 3 already uses all 22 features. It replays stage 3's calibrated scores through the merchant `StreamingSpikeDetector` + policy engine and reports **5/5 attack merchants caught, 0 false alarms, flash sale correctly not flagged** — i.e. what the spike/policy layer adds on top of a strong per-transaction classifier (merchant-level, actionable detection), which raw PR-AUC alone doesn't capture.

**Entity/graph features are where the system actually comes from:** +0.30 PR-AUC, recall 0.58 → 0.89, and legitimate ₹ wrongly blocked cut ~5x. Velocity alone is roughly neutral-to-slightly-negative (0.661 → 0.631) — rolling-window counts spike for busy *legitimate* merchants too, so without entity history to tell "one device across 50 throwaway accounts" from "a popular merchant having a good hour," velocity adds about as much noise as signal. It only pays off once entity context is present to condition it.

*A retracted claim, kept visible on purpose.* An earlier version of this section reported a dramatic stage-2 **collapse** (PR-AUC 0.55 → 0.23, legit ₹ blocked 6x) and explained it as "velocity alone is a trap." That was an artifact, not a finding: with no attack in the calibration slice, isotonic calibration had almost nothing to fit and produced degenerate score plateaus. Once validation contained a real attack (see Model selection), the effect shrank to −0.03 — a mild regression, not a collapse. A follow-up diagnostic on the top-200 scored transactions initially seemed to confirm the story (18 flash-sale transactions ranked as top fraud risk under the velocity-only model), but that too dissolved under scrutiny: the top-200 sits entirely inside a ~447-transaction tie-plateau where every score is exactly 1.0, so its composition was decided by row order. Re-ranking the same plateau by raw model score gives **zero** flash-sale transactions and precision 1.000 for all three variants. The feature slicing was verified correct (a clean 6/6/10 partition of all 22 features), so there was no bug to fix — the diagnostic simply could not support the claim. Both the original finding and its retraction are left here because "we chased a dramatic result and it did not survive" is the honest version of this table.

Reproduce: `python -m src.models.ablation` → writes `artifacts_out/ablation_table.csv`.

## Policy thresholds — cost-optimized on validation

The 85/60 restrict/step-up cutoffs were hand-set against the old `p*100` scale and never re-derived. `src/policy/threshold_sweep.py` sweeps both on the **validation slice only** (days 21–23) under the same net-protected-value framework and ₹ assumptions used for the reported test economics. Adoption rule fixed in advance: a cutoff moves only if it beats the incumbent by >2% validation NPV.

| | Restrict cut | Step-up cut | Validation NPV |
|---|---|---|---|
| Old (hand-set) | 85 | 60 | ₹1,16,216 |
| **Adopted** | **85** (unchanged) | **25** | **₹1,26,453** |

**Only step-up moved.** Lowering it 60 → 25 is worth **+8.28%** validation NPV on its own — it catches fraud the old cut let through. The restrict cut stayed at 85 because moving it alone is worth just +0.53%, and its NPV surface is *exactly flat across a 9-wide tie (40–80)*: no validation transaction scores in that band, so the sweep genuinely cannot identify a better value there. Picking 80 out of that tie — which a naive "best pair wins" reading would have done — is choosing an arbitrary point in a dead zone.

Transfer to the untouched test slice, run once after adoption: net protected value **₹10.20L → ₹10.57L (+3.7%)**. Smaller than the +8.28% validation lift, which is the expected direction for a validation-optimized parameter.

Honest caveats: validation holds 125 fraud transactions and 90 of them are one device-farm attack, so this is decided by essentially one attack type — which is why the adopt margin exists. Isotonic calibration is fit on that same slice, making its probabilities mildly in-sample-optimistic; this is a *policy* selection, not a performance estimate. The rule as originally written ("adopt the best pair") turned out under-specified for a degenerate surface, so it was refined to a per-parameter margin check — that refinement was made after seeing the surface, and is recorded in the module rather than folded in quietly.

## P2 — LLM investigator (LangGraph + Claude Haiku)

When the streaming spike detector fires, a LangGraph agent investigates *why* using **six read-only tools** — `get_merchant_baseline`, `get_flagged_transactions`, `get_entity_network`, `get_velocity_summary`, `calculate_exposure`, `write_investigation_report` — and files a structured report: `{cause, evidence[], exposure_inr, recommended_action, confidence}`.

The design is mostly a list of things the agent **cannot** do:

- **It cannot decide.** `recommended_action` passes through `policy.validate_recommendation()`. An out-of-allowlist action degrades to REVIEW and can never escalate. Pytest-enforced: a report recommending `ban_merchant_permanently` comes back as REVIEW.
- **It cannot do the money.** `calculate_exposure` performs all ₹ arithmetic in Python. LLM arithmetic is precisely the kind of silent error a risk team cannot audit, so the model reads a number rather than computing one.
- **It cannot block anyone by failing.** No credentials, API error, timeout, malformed output, or tool-budget exhaustion all produce a deterministic templated report — built from the same read-only tools, so it states measured facts rather than guessing — routed to human REVIEW with `cause: "unclear"` and `confidence: 0.0`.
- **It cannot run away.** The tool loop is capped at 8 rounds.
- **It cannot see the answer.** No tool exposes `is_fraud` or `scenario`; the agent reasons from signals only, as it would in production.

Every tool call is audit-logged as `(tool, inputs hash, output hash, ts)` — hashes, not payloads, so the log carries no raw transaction data but still proves after the fact what was called and what came back. The degraded path is audited too: an audit log that goes quiet exactly when something breaks is worse than none.

### Agent eval — live Claude Haiku 4.5

Ten fixed cases with ground truth (5 attack types, the legitimate flash sale, 4 low-signal merchants where escalating is the *correct* answer) → `artifacts_out/agent_eval.csv`, with full per-case transcripts (every tool call, its real arguments and outputs, and the final JSON) in `artifacts_out/agent_transcripts/`.

**All 10 cases ran the live model** — no case silently fell back, each made all 6 tool calls.

| Metric | Live result |
|---|---|
| Correct cause | **9 / 10** |
| Evidence valid (cited figures + used `calculate_exposure`) | **10 / 10** |
| Correct action | **8 / 10** |
| Escalates when unsure | **9 / 10** |
| **Policy violations** | **0 / 10** |
| `validate_recommendation()` downgrades needed | 0 |

**The check that matters most passed cleanly: 4/4 low-signal merchants refused to invent an attack.** All four returned `legitimate_traffic`, none manufactured a cause from ambient noise. The flash sale was correctly called `legitimate_traffic` → `allow` at 0.92 confidence, citing entity diversity ("instruments per customer = 1.0").

**Money arithmetic stayed in Python: 10/10.** Every `exposure_inr` matches `calculate_exposure`'s output to the paisa — verified against the transcripts, not assumed.

**Evidence traceability: 63 of 64 claims trace to tool output.** The one that doesn't is `ip_cluster`'s *"concentrated in ~90-minute burst"* — no tool reported a 90-minute burst (`flagged_span_hours` was 118.06); the model inferred it from 15 sampled transactions. Two other claims flagged by the automated checker turned out to be legitimate min/max derivations over data the tool did return (`fraud_ring` ₹142–₹7,639, `ip_cluster` ₹96–₹3,947).

**Where it fails, honestly:**
- `card_testing` → labelled `fraud_ring`. The *evidence* was right — it explicitly cited "stolen card testing signature" and "amounts small and varied, consistent with card testing" — but it picked the neighbouring label, because this simulator's card-testing wave genuinely does share attacker devices. Action was still correct (`restrict`).
- `quiet_merchant_b` → `step_up` where `allow` was expected. It did **not** invent a cause; it applied mild friction to a merchant with 2 flagged transactions out of 1,028. Over-cautious, not unsafe.

### The most important result: an agent failure that changed nothing

Running the demo, the agent labelled **m2 — a real account-takeover with ₹5.5L exposure — as `legitimate_traffic` / `allow` at 0.95 confidence.** Its evidence was factually correct ("29 customers, 29 devices, 29 IPs, zero shared entities") and led to exactly the wrong conclusion.

Root cause is a **tool gap, not a model failure**: account takeover is defined by *established customers appearing on new devices, in new geographies, spending atypical amounts*. Those four signals exist in the feature builder and drive the ML scorer — but **no agent tool exposes any of them**, so the agent structurally cannot see ATO. It reasons from entity-sharing, and ATO does not share entities.

The system restricted 68 transactions and sent 8 to human review on m2 anyway, because **the agent was never in the decision path.** A confidently wrong LLM recommendation on a ₹5.5L attack changed exactly zero decisions. That is the entire argument for this architecture, demonstrated rather than asserted.

A second, related finding: `get_merchant_baseline` splits at the 75th percentile of the merchant's window, so an attack that ends mid-window reads as *"baseline 8.99% → recent 0.35%"* — an improvement. The model then rationalises it ("flagged rate jumped from baseline 16.65% to recent 0.38% AFTER accounting for the farm burst"), describing a decrease as a jump. This contributed to both the ATO miss and `ip_cluster`'s downgrade to `step_up`.

**Run-to-run variance is real and unmitigated.** m2 came back as `account_takeover` in the eval and `legitimate_traffic` in the demo — same tools, same data, different sampling. Neither run is cherry-picked here; both are reported.

Neither issue is fixed in this submission: adding ATO-signal tools and reframing the baseline window are design changes, deliberately not made under submission pressure. They are the top two items on the list.

What is also verified without credentials: 15 agent tests drive the full LangGraph loop through a scripted client double — tool dispatch, the policy gate, audit logging, unknown-tool recovery, tool-budget cutoff, read-only enforcement, and the ground-truth-leak check all run deterministically with no network.

Reproduce: `python -m src.agent.eval`.

## P3 — dashboard & demo

```bash
python run_demo.py          # trains, serves, replays → http://127.0.0.1:8000
```

One command, one process, no Docker and **no Node build step**. React is vendored locally as UMD + [htm](https://github.com/developit/htm) tagged templates (144KB in `src/serve/static/vendor/`), so there is no bundler, no `npm install`, and no CDN request at demo time — venue wifi cannot break the demo.

Useful flags: `--speed 400` (transactions/sec; the full 13,987-txn slice takes ~60s at 250), `--no-agent` (skip investigations), `--no-browser`, `--port`.

**What the dashboard shows** — all of it read from the live pipeline, none of it pre-rendered:
- **Merchant grid** — risk gauge /100, flagged-rate delta (baseline → current, with the spike z), ₹ exposure, txns-at-risk, and the investigated cause. Spiking merchants sort to the top and the first one auto-selects, so you are never hunting during a demo.
- **Entity network** — a ~30-line hand-rolled force layout (no graph library). Entities are sized by how many accounts share them, and entities touching only one account are dropped: a device farm renders as one huge hub, and legitimate traffic renders as **nothing at all**.
- **Investigation panel** — the agent's JSON report (cause, evidence, exposure, recommended action, confidence), showing both the raw recommendation *and* what it became after the policy gate, plus a collapsible audit-log viewer.
- **Review queue** — analyst approve / override, with overrides visually marked.
- **Event feed** — the pipeline narrating itself: spike detected → investigating → verdict.

### The 3-minute demo arc

| ~time | What happens | What to point at |
|---|---|---|
| 0:00 | Normal traffic, 12 merchants, all green | Baseline flagged rates ~1% |
| 0:30 | Attack merchants light up red one by one | Flagged rate jumps to 73–93%, z 4.7–5.8 |
| 1:00 | Investigation fires **on the spike**, not a timer | Cause + evidence + audit log |
| 1:30 | Review queue fills; override a case | 617 cases — the system holds, it never acts alone |
| 2:30 | **Finale:** m11 runs a 6× volume flash sale | Peak flagged rate **3%**, z=1.1, **0 restricts, NOT flagged** |

The finale is the whole thesis in one screen: 2,284 transactions of legitimate volume spike, zero restricts, and an **empty** entity graph — because the detector fires on the fraud-score rate, not on volume.

### API

`GET /api/merchants` · `GET /api/merchants/{id}/risk` · `GET /api/merchants/{id}/entity-graph` · `POST /api/merchants/{id}/investigate` · `GET /api/review-queue` · `POST /api/review-queue/{id}/decision` · `GET /api/audit-log` · `POST /api/transactions` (score one transaction through the live fusion→policy path) · `GET /api/status`.

Two properties the serving layer is tested for: `POST /api/transactions` is **side-effect free** (a judge poking the API mid-demo cannot corrupt the numbers on screen), and the frozen allowlist **binds analysts too** — the override endpoint rejects an invented action with 400, exactly as `validate_recommendation()` does for the LLM. An analyst console that accepts arbitrary action strings is the same hole from the other side.

## How it maps to the judging criteria

**Problem taste.** Merchants lose money in bursts — card-testing waves, device farms, account-takeover clusters, fraud rings — not one transaction at a time. RBI-reported digital-payment fraud is the largest fraud category in Indian BFSI by case count. The system targets the burst, the moment losses concentrate.

**Build quality.** Runs end-to-end from one command. Leakage-safe incremental feature builder (every feature computed strictly from prior events). Day-boundary **temporal** train/calibration/test split — never random. Safety invariants are pytest-enforced, not aspirational.

**AI judgment — the right tool in the right place, and where we chose NOT to use one.**
- Fraud scoring: **a GBDT, not an LLM and not deep learning — and the library itself was chosen empirically, not by default.** Gradient-boosted trees are what payment processors ran at massive scale for years; at this data size they are faster, calibrated, and SHAP-explainable. Which GBDT wasn't assumed: `select_model.py` compared LogisticRegression/XGBoost/LightGBM/CatBoost head-to-head on held-out validation with a selection rule fixed *before* seeing results — XGBoost won. An LLM here would be slower, uncalibrated, and injectable.
- Spike detection: **EWMA + z-score change-point, not an autoencoder.** ~50 lines, online, explainable, and it fires on the *fraud-score rate* — not raw volume — which is exactly why a 6x flash sale doesn't trigger it.
- Imbalance: **class weights + isotonic calibration, not SMOTE** — resampling distorts the temporal distribution.
- The LLM investigator (P2 layer) only *explains and recommends from a frozen action allowlist*; the deterministic policy engine makes every decision. `validate_recommendation()` degrades any out-of-allowlist LLM output to human review — it can never escalate.

**Failure recovery.** What broke while building: (1) attack scenarios initially lived only in the test period, so the model had never seen an attack pattern — PR-AUC 0.28; fixed by injecting *historical* attacks (different merchants, disjoint entity IDs) into the training period — PR-AUC 0.934. (2) The card-testing spike was invisible at merchant level because a `min_txn=20` guard skipped low-volume merchants' hours; fixed with `min_txn=10` plus a variance noise-floor. (3) The validation slice contained no attack at all, so model selection was ranking families on ambient-fraud noise and isotonic calibration was fitting degenerate score plateaus — this manufactured a dramatic-looking ablation finding that **evaporated** once fixed (see Ablation study). (4) Risk fusion shipped with two measurement-caught bugs: a weighted average that silently overruled the calibrated model, and a graph threshold that gave 26% of legitimate transactions a risk bonus. At runtime: if the ML scorer is unavailable, decisions fall back to rules and route to human review — an LLM failure cannot block anyone, because the LLM was never in the decision path.

The pattern worth noting across (3) and (4): every one of these was caught by *measuring the thing we had just claimed*, not by reading the code again. The retraction in the ablation section is the clearest example — the finding was dramatic, quotable, and wrong.

## Architecture

```
transaction ──> leakage-safe features ──> GBDT, selected empirically (calibrated) ──> p_fraud
                     │                                                │
                     └──> entity graph (union-find components)        ▼
                                              merchant EWMA/z-score spike detector
                                                                      │
                                    risk fusion: ML floor + spike/graph/rule lift
                                        ──> risk_score 0-100 + confidence 0-1
                                                                      │
                                              deterministic policy engine (allowlist)
                                               allow / step_up / review / restrict
                                                                      │
                                    on spike: LangGraph + Claude Haiku investigator
                                    6 read-only tools · ₹ math in Python · audit log
                                    explains and recommends FROM the allowlist
                                                                      │
                                    any failure ──> deterministic report ──> human review
```

## Repo layout

```
src/sim/        transaction simulator — 6 labeled scenarios incl. legitimate flash sale
src/features/   incremental leakage-safe feature builder (~22 features)
src/models/     empirical model selection, temporal-split training, isotonic
                calibration, cost-optimal threshold, feature-group ablation
src/spike/      merchant-level EWMA + z-score change-point detector
src/policy/     deterministic policy engine + frozen allowlist; risk fusion
                (ML floor + spike/graph/rule lift -> risk_score + confidence);
                validation threshold sweep
src/agent/      LangGraph investigator (Claude Haiku), 6 read-only tools,
                audit log, 10-case ground-truth eval
src/serve/      FastAPI API + replay driver + React SPA (vendored, no build)
run_demo.py     one-command demo: train -> serve -> replay
tests/          safety invariants (fail-safe, LLM cannot escalate, flash-sale
                no-fire, fusion floor/bounds, agent gate/audit/read-only,
                serving side-effect-freedom, analyst allowlist) - 46 tests
```

## Honest limitations

- Data is synthetic (simulator parameters like "0.7%→5% fraud" are design choices, not Razorpay statistics). The simulator explicitly wires shared entities across accounts — fraud rings only exist in a graph if you construct them.
- The account-takeover scenario is diffuse by nature; it is caught primarily at transaction level (p≈0.97 per txn) and only weakly at merchant-spike level.
- No production hardening: single process, SQLite-class persistence, no authn.
