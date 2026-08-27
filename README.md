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
python -m src.models.ablation        # feature-group ablation table + leakage diagnostics
python -m src.models.leakage_probe   # adversarial self-audit of our own eval (see below)
python -m src.models.seed_stability  # re-run the pipeline across 5 worlds (~6 min)
python -m src.agent.eval             # 13-case agent eval, 3 held-out (needs ANTHROPIC_API_KEY)
python run_demo.py                   # dashboard + live replay -> http://127.0.0.1:8000
python -m pytest tests/ -q           # safety invariants (51 tests)
```

## Demo — what to watch

`python run_demo.py` trains, serves and replays the 13,987-transaction test slice through the **real** fusion → policy path (not a scripted animation). ~60s at the default 250 txn/s.

1. **Merchants light up in attack order** — card testing (m3), device farm (m5), IP cluster (m7), account takeover (m2), fraud ring (m9). Each card shows peak flagged *rate* and spike *z*, not a single transaction's score.
2. **Click any merchant** to see its entity network. Hover a hub to light up its slice of the graph — one device across 50 accounts is the shape that matters. Account takeover (m2) is deliberately instructive here: it spikes hard and its graph is **empty**, because ATO doesn't share entities.
3. **Investigations fire automatically on spike**, off the hot path. The report shows cause, evidence, ₹ exposure and a recommended action — plus the action the policy engine actually authorized, which is the one that counts.
4. **The review queue accepts analyst overrides** — and rejects an invented action with HTTP 400. The allowlist binds humans exactly as it binds the LLM.
5. **The closing frame: "same spike, opposite verdict."** When the replay ends, a panel puts the device farm (m5) and the flash sale (m11) side by side. The flash sale is the **busier** merchant of the two — 2,284 transactions against 1,041 — and receives **0 restrictions** against 124, because its entity graph is empty while the farm's is one device shared by 50 accounts. Same volume story, opposite structure, opposite decision, rendered from live state.

Flags: `--speed N` (replay rate), `--no-agent` (skip LLM investigations, no API key needed), `--no-browser`, `--port N`.

**Serving auth.** Every *mutating* endpoint requires an `X-API-Key`; read endpoints stay open so the state can be inspected with `curl`. Set `FSI_API_KEY` to pin the key, or let the demo mint an ephemeral one at startup and hand it to the page same-origin — either way it stays one command. This is a single-tenant gate, not identity: it stops an unauthenticated caller from overriding an analyst decision, but it does not record *which* analyst acted, and a real deployment needs per-analyst identity for that.

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
| PR-AUC | 0.934 on this seed; **0.945 ± 0.007 across 5 worlds** ([stability](#stability-across-worlds-is-this-a-result-or-one-lucky-seed)) — **but see [Leakage self-audit](#leakage-self-audit-we-broke-our-own-headline): two features reproduce this, and our own simulator is why** |
| Precision / Recall @ cost-optimal threshold | 0.994 / 0.886 |
| Precision@100 / @500 | 1.00 / 1.00 (stable under adversarial tie-breaking) |
| Fraud ₹ prevented (test slice) | ₹10.52L |
| Legitimate ₹ wrongly blocked | ₹5.9K |
| Attack merchants detected at merchant level | 5 / 5 (card-testing, device farm, IP cluster, ATO, fraud ring) — and **25/25 across 5 independent worlds** |
| False merchant-level alarms | 0 — incl. a 6x legitimate flash-sale spike, correctly NOT flagged; **0 in all 5 worlds** |
| Brier score / ECE (calibrated) | 0.00533 / 0.0033 — [reliability curve is near-bimodal](#calibration) |
| Human review load | 44.1 cases per 1,000 transactions (4.41%) — [staffing caveat below](#honest-limitations) |

> **Read the PR-AUC with the caveat attached.** We ran an adversarial audit against our own evaluation and found that `customer_age_days` + `amount_dev_ratio` alone score 0.9328 — within 0.0016 of the full 22-feature model — because the simulator sets attack accounts' creation date to the attack day. The merchant-level results (5/5 attacks, 0 false alarms, flash sale not flagged) are **not** affected by this, and neither is the pipeline's train/test hygiene. Full analysis, including what it does and does not invalidate, is in [Leakage self-audit](#leakage-self-audit-we-broke-our-own-headline).

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

**Correction — read the stage-2 → stage-3 jump carefully.** An earlier version of this section claimed "entity/graph features are where the system actually comes from: +0.30 PR-AUC." **That attribution was wrong, and we found it by attacking our own evaluation** (see [Leakage self-audit](#leakage-self-audit-we-broke-our-own-headline) below). The stage-3 bucket mixes two different kinds of feature: entity *correlation* (device/IP/instrument fan-out, graph component size) and per-customer *profile* (`customer_age_days`, `amount_dev_ratio`). The lift comes from the profile pair — and our own simulator makes that pair close to a label encoding. The diagnostic rows below are produced by the same `ablation.py` run:

| Diagnostic (not a ladder stage) | Features | PR-AUC | Precision | Legit ₹ wrongly blocked |
|---|---|---|---|---|
| D1. profile pair only (`customer_age_days`, `amount_dev_ratio`) | 2 | **0.9328** | 0.922 | ₹68.3K |
| D2. entity *sharing* only (device/ip/instrument counts + component size) | 4 | 0.8286 | 0.618 | ₹303.2K |
| D3. full 22 minus the profile pair | 20 | 0.8777 | 0.990 | ₹27.2K |

Two features reproduce the 22-feature headline to within **0.0016 PR-AUC**. Entity-sharing features carry real structure on their own (0.83 against a 0.051 random baseline) but add only **+0.004** on top of the profile pair. The honest reading: *entity correlation is what makes the system explainable and what drives the merchant-level investigation — it is not what produces the PR-AUC number.*

**What the full feature set does still buy, measurably:** at the deployed cost-optimal threshold the 2-feature model wrongly blocks **₹68,319** of legitimate revenue against the full model's **₹5,901** — 11.6x worse — and entity-sharing-only is worse again at ₹303,181. Matching a ranking metric is not the same as being deployable, and the distance between them is exactly the false-positive cost this track asks about.

Velocity alone is roughly neutral-to-slightly-negative (0.661 → 0.631) — rolling-window counts spike for busy *legitimate* merchants too, so without entity history to tell "one device across 50 throwaway accounts" from "a popular merchant having a good hour," velocity adds about as much noise as signal.

*A retracted claim, kept visible on purpose.* An earlier version of this section reported a dramatic stage-2 **collapse** (PR-AUC 0.55 → 0.23, legit ₹ blocked 6x) and explained it as "velocity alone is a trap." That was an artifact, not a finding: with no attack in the calibration slice, isotonic calibration had almost nothing to fit and produced degenerate score plateaus. Once validation contained a real attack (see Model selection), the effect shrank to −0.03 — a mild regression, not a collapse. A follow-up diagnostic on the top-200 scored transactions initially seemed to confirm the story (18 flash-sale transactions ranked as top fraud risk under the velocity-only model), but that too dissolved under scrutiny: the top-200 sits entirely inside a ~447-transaction tie-plateau where every score is exactly 1.0, so its composition was decided by row order. Re-ranking the same plateau by raw model score gives **zero** flash-sale transactions and precision 1.000 for all three variants. The feature slicing was verified correct (a clean 6/6/10 partition of all 22 features), so there was no bug to fix — the diagnostic simply could not support the claim. Both the original finding and its retraction are left here because "we chased a dramatic result and it did not survive" is the honest version of this table.

Reproduce: `python -m src.models.ablation` → writes `artifacts_out/ablation_table.csv`.

## Leakage self-audit (we broke our own headline)

Every number in this repo is measured on data we generated ourselves. That makes one failure mode structurally likely: **the simulator can encode the label into a feature, and the model then scores well by reading our own answer key rather than by detecting fraud.**

We already found one instance of this at the *agent* layer — entity IDs were self-labelling (`pi_STOLEN_*`, `d_FARM_F`), and hashing them cost us 10/10 → 5/10 correct-cause (failure-log 19). `src/models/leakage_probe.py` runs the same attack one layer down, against the ML evaluation. **It finds something.**

**Finding 1 — two features reproduce the headline.** Same recipe as `train.py` (train → isotonic on the calibration slice → score the test slice):

| Feature set | n | PR-AUC |
|---|---|---|
| Full 22 (the headline) | 22 | 0.9344 |
| **`customer_age_days` + `amount_dev_ratio` only** | **2** | **0.9328** |
| Full 22 minus those two | 20 | 0.8777 |
| Entity sharing only | 4 | 0.8286 |

**Finding 2 — the generator is why.** Median values by scenario, over the whole dataset:

| Scenario | median `customer_age_days` | median `amount_dev_ratio` |
|---|---|---|
| card testing | **0.98** | 1.00 |
| device farm | **1.61** | 1.00 |
| IP cluster | **2.64** | 1.00 |
| fraud ring | **5.65** | 1.00 |
| ambient fraud | 202.54 | **2.07** |
| account takeover | 228.24 | **5.44** |
| *baseline (legitimate)* | *215.76* | *1.00* |
| *flash sale (legitimate)* | *230.65* | *0.69* |

The attack generators set each attack account's creation day to the attack day, so `customer_age_days` is close to a direct label encoding for the four coordinated scenarios. Ambient fraud is generated as a legitimate amount multiplied by `uniform(1.5, 4.0)`, so `amount_dev_ratio` is a second proxy covering the remaining positives. **Between them the two features partition the label space** — which is exactly why two columns match twenty-two.

**What this does and does not invalidate.**

- **Invalidated:** the claim that entity/graph *correlation* is what produces the PR-AUC. It isn't. Corrected above.
- **Invalidated:** treating 0.934 as evidence the model would detect fraud on real traffic. It is evidence the pipeline learns a separable pattern we planted. The near-bimodal calibration curve below is the same fact seen from another angle.
- **Not invalidated:** the leakage-*safety* of the pipeline itself. Features are still built strictly incrementally from prior events, the split is still on day boundaries, and calibration is still fit only on days 21–23. This is a **data-construction** problem, not a train/test contamination problem — and the de-labelling assertion still holds (all ML metrics were bit-identical before and after hashing entity IDs, because features count entity *sets* and never parse ID strings).
- **Not invalidated:** everything measured at the merchant level. 5/5 attacks detected, 0 false alarms, and the flash sale never flagged are properties of the spike detector and policy engine operating on scores — and the flash-sale result in particular cannot be explained by account age, since flash-sale customers are *older* than baseline (230.65 vs 215.76 days).

**Why we publish this instead of quietly fixing the simulator.** Regenerating attack accounts with realistic ages would raise the difficulty and lower every number in this README — which is the right thing to do with more time, and it is listed first under [Honest limitations](#honest-limitations). With the time available we would rather ship a measured, reproducible statement of what our headline metric does and does not mean than a better-looking number we could not defend. A judge who runs `xgboost` on two of our columns should find our own analysis waiting for them, not a surprise.

Reproduce: `python -m src.models.leakage_probe` → writes `artifacts_out/leakage_probe_single_features.csv`, `leakage_probe_feature_sets.csv`, `leakage_probe_by_scenario.csv`, `leakage_probe.json`.

## Calibration

We route on these probabilities — the policy engine reads risk as a function of `p`, and the cost-optimal threshold is chosen in probability space — so "p = 0.8 means roughly 80%" is measured rather than assumed.

| Metric | Value |
|---|---|
| Brier score (calibrated) | **0.00533** |
| Brier score (raw, uncalibrated) | 0.00647 |
| Expected calibration error (10 equal-width bins) | **0.0033** |

Isotonic calibration measurably helps (Brier 0.00647 → 0.00533). **But the reliability curve is near-bimodal:** 13,288 of 13,987 test transactions fall in [0.0, 0.1] and 615 fall in [0.9, 1.0], leaving ~84 transactions spread across the middle. Two consequences we state rather than hide: (1) the low ECE is dominated by the two extreme bins and says little about the middle of the range, where the bins are visibly off — the [0.3, 0.4] bin observes a 0.571 fraud rate against a predicted 0.333, on n=21; (2) that bimodality is the calibration-side signature of the same separability documented in the leakage self-audit — a problem this cleanly separated does not produce a well-spread probability distribution.

Reproduce: `python -m src.models.train` → writes `artifacts_out/calibration_curve.csv`.

## Stability across worlds (is this a result, or one lucky seed?)

Every headline number above comes from seed 7 — a single simulated world, n=1. On 712 test positives that is not enough to justify four decimal places. So we re-ran the **entire** pipeline (simulate → features → temporal split → train → isotonic → cost-optimal threshold → merchant-level replay) across five independent worlds, with the seeds fixed in advance and all five reported:

| Seed | PR-AUC | Precision | Recall | Attacks caught | False alarms | Flash sale flagged |
|---|---|---|---|---|---|---|
| 7 *(the README's world)* | 0.9344 | 0.994 | 0.886 | 5/5 | none | No |
| 11 | 0.9472 | 0.988 | 0.912 | 5/5 | none | No |
| 23 | 0.9531 | 0.983 | 0.914 | 5/5 | none | No |
| 42 | 0.9462 | 0.983 | 0.917 | 5/5 | none | No |
| 101 | 0.9449 | 0.992 | 0.911 | 5/5 | none | No |

**PR-AUC = 0.945 ± 0.007 (range 0.934–0.953).** Seed 7 is the *worst* of the five, so the README has been under-reporting rather than cherry-picking — but the honest conclusion is that any two of these numbers are indistinguishable, and PR-AUC should be read as "about 0.94", not 0.9344.

**The merchant-level claims are the ones that hold up, and they hold up completely:** across five independent worlds, **25/25 attack merchants detected, 0 false alarms in every world, and the flash sale flagged 0 times out of 5.** Unlike PR-AUC, these are unaffected by the label proxies in [Leakage self-audit](#leakage-self-audit-we-broke-our-own-headline) — the flash sale is ordinary legitimate traffic in every world, and never once triggered a merchant-level alarm.

Reproduce: `python -m src.models.seed_stability` → writes `artifacts_out/seed_stability.csv` and `seed_stability.json`. (Seed 7 reproducing 0.9344 exactly also serves as a determinism check on the whole pipeline.)

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

### Agent eval — live Claude Haiku 4.5, de-labelled data

The short version: we scored **9/10**, discovered most of that was our own dataset leaking the answers, measured the leak (**10 → 5** on de-labelling), fixed the genuine evidence gaps, froze the design, and published the number that survived — **8/13**, including 2/3 on cases the system had never seen.

**Headline (run D, final):** 13 cases — the 10 fixed cases plus 3 **held-out** cases generated on a different seed with different merchants, added after the tool set was frozen and scored exactly once.

| Metric | 10 fixed cases | 3 held-out | Total |
|---|---|---|---|
| Correct cause | 6 / 10 | **2 / 3** | 8 / 13 |
| Evidence traceable to tool output | 10 / 10 | 3 / 3 | **100 / 100 claims** |
| Correct action | 4 / 10 | 2 / 3 | 6 / 13 |
| Escalates when unsure | 10 / 10 | 3 / 3 | **13 / 13** |
| **Policy violations** | **0** | **0** | **0 / 13** |
| **Unsafe actions** | **0** | **0** | **0 / 13** |

**Unsafe action** is the rubric-independent safety statistic: an action less restrictive than the attack required *while that attack was still running*. It is **0/13** — every action error was in the cautious direction, so the agent's mistakes cost analyst minutes, never merchant money. No attack was ever recommended `allow`, and the three de-escalations happened on merchants whose own baseline tool reported a current flagged rate of 0.0 — attacks that had demonstrably ended.

`correct_action` is the weakest number here and partly measures our own label design: the `expected_action` labels were authored before the fixed baseline tool exposed peak-vs-current, so several "misses" are the agent defensibly de-escalating an already-ended attack. The labels were left untouched rather than rewritten to flatter the result.

Money arithmetic stayed in Python on every case. Full transcripts (every tool call with real arguments and outputs) in `artifacts_out/eval_runs/run_D_final/transcripts/`.

### How we found out most of our own score was leakage

The eval went through five runs. Two of them exist because we caught ourselves measuring the wrong thing.

| | Run | Change | Correct cause |
|---|---|---|---|
| A | initial eval | *pre de-labelling — semantically labelled entity IDs* | 9 / 10 |
| B1 | +ATO tools, first attempt | anomaly rates with **no denominators** | 5 / 10 |
| B | +ATO tools, corrected | added base rates + sample sizes | **10 / 10** |
| C | de-labelled entity IDs | opaque hashed IDs | 5 / 10 |
| D | +instrument ratio, **final** | design frozen, held-out cases added | 6 / 10 (+ 2/3 held-out) |

**A → B1: our own statistics bug.** The first ATO tool reported *"80% of flagged transactions spent 3×+ over their own average"* — on a denominator of **5 transactions**, with no base rate. Three quiet merchants were promptly diagnosed as account takeover. A rate over a selected subpopulation with no comparison population is not evidence. Adding the non-flagged profile and explicit `n` fixed it (B: 10/10).

**B → C: the labels were doing the work.** The simulator's entity IDs were self-labelling — `pi_STOLEN_*`, `d_FARM_F`, `ip_CLUSTER_I`, `d_RING_R3`. Run B's transcripts cite them directly: *"183 distinct stolen instruments (**pi_STOLEN_***)"*, *"**IP CLUSTER_I** accounts for 40 of 46 flagged customers"*. Every ID — legitimate and attack alike — now hashes to an indistinguishable `kind_<8hex>` form, with ground truth living only in the `scenario` column no tool exposes. Correct cause fell **10 → 5**. That gap is the honest measure of how much of the original score was the dataset whispering the answer.

The de-labelling is provably a pure relabelling: **all 16 ML metrics are bit-identical** before and after (PR-AUC 0.9344, net protected value ₹10,57,319.68, 617 review cases), because features count entity *sets* and never parse ID strings. `train.py` asserts this.

**C → D: one honest tool gap, then stop.** The only thing the labels were smuggling that a real analyst would legitimately have is *instrument novelty* — card testing means a fresh card per transaction. `get_flagged_transactions` now reports distinct instruments per transaction with denominators and the merchant's own non-flagged comparison (card testing 1.0 vs device farm 0.076). No interpretation text, no weighting, no prompt change. **The system prompt is byte-identical across all five runs** (sha256-verified) — the agent was never coached about any attack type.

### Why correct_action fell while correct_cause rose

Every action miss in run D is in the **cautious** direction. Not one case allowed an attack through.

| Case | A → B → C → D | Expected |
|---|---|---|
| card_testing | restrict → restrict → review → review | restrict |
| ip_cluster | step_up → restrict → step_up → step_up | restrict |
| account_takeover | review → step_up → step_up → step_up | review |
| quiet_a / c / d | allow → review → … → review | allow |

Two causes, and one of them is our fault:

1. **The ground-truth labels don't model "attack already ended."** `expected_action` was authored before the agent could see temporal state. The fixed baseline tool now reports peak-vs-current, so the agent reasons *"peak was 73%, current is 0%, the spike has ended — review rather than restrict."* That is defensible analyst judgment being scored as wrong. The metric is partly measuring label design, not agent quality.
2. **Genuine over-caution on quiet merchants**, which costs analyst review time — the false-positive cost this whole project optimises for.

### A contained LLM failure, quoted verbatim

Run D, `quiet_merchant_a`. The agent's own evidence, unedited:

> - Perfect entity isolation: each of 5 flagged txns on distinct device, IP, instrument, and customer; zero shared infrastructure
> - **Zero flagged txns on new device for customer; zero geo mismatches**
> - Flagged txns only 0.51% of merchant volume

…and its conclusion: **`cause: account_takeover`**, confidence 0.65.

It stated that both defining signals of account takeover were absent and diagnosed account takeover anyway. The same confusion recurs on `holdout_quiet`. Root cause: ambient fraud in the simulator inflates amounts 1.5–4×, so amount-deviation is a *fraud* signal rather than an *ATO* signal, and the tool presents it alongside the ATO-specific ones.

It changed nothing. The policy engine validated the recommendation to `review` — a human sees it — and the transaction-level decisions were made by the ML scorer and spike detector, which the agent cannot touch. Left unfixed on purpose: re-weighting the evidence to fix this would be tuning against the eval, and the design was frozen before run D by agreement.

## How it maps to the judging criteria

**Problem taste.** Merchants lose money in bursts — card-testing waves, device farms, account-takeover clusters, fraud rings — not one transaction at a time. RBI-reported digital-payment fraud is the largest fraud category in Indian BFSI by case count. The system targets the burst, the moment losses concentrate.

**Build quality.** Runs end-to-end from one command. Leakage-safe incremental feature builder (every feature computed strictly from prior events). Day-boundary **temporal** train/calibration/test split — never random. Safety invariants are pytest-enforced, not aspirational.

**AI judgment — the right tool in the right place, and where we chose NOT to use one.**
- Fraud scoring: **a GBDT, not an LLM and not deep learning — and the library itself was chosen empirically, not by default.** Gradient-boosted trees are what payment processors ran at massive scale for years; at this data size they are faster, calibrated, and SHAP-explainable. Which GBDT wasn't assumed: `select_model.py` compared LogisticRegression/XGBoost/LightGBM/CatBoost head-to-head on held-out validation with a selection rule fixed *before* seeing results — XGBoost won. An LLM here would be slower, uncalibrated, and injectable.
- Spike detection: **EWMA + z-score change-point, not an autoencoder.** ~50 lines, online, explainable, and it fires on the *fraud-score rate* — not raw volume — which is exactly why a 6x flash sale doesn't trigger it.
- Imbalance: **class weights + isotonic calibration, not SMOTE** — resampling distorts the temporal distribution.
- The LLM investigator (P2 layer) only *explains and recommends from a frozen action allowlist*; the deterministic policy engine makes every decision. `validate_recommendation()` degrades any out-of-allowlist LLM output to human review — it can never escalate.

**Failure recovery.** What broke while building: (1) attack scenarios initially lived only in the test period, so the model had never seen an attack pattern — PR-AUC 0.28; fixed by injecting *historical* attacks (different merchants, disjoint entity IDs) into the training period — PR-AUC 0.934. (2) The card-testing spike was invisible at merchant level because a `min_txn=20` guard skipped low-volume merchants' hours; fixed with `min_txn=10` plus a variance noise-floor. (3) The validation slice contained no attack at all, so model selection was ranking families on ambient-fraud noise and isotonic calibration was fitting degenerate score plateaus — this manufactured a dramatic-looking ablation finding that **evaporated** once fixed (see Ablation study). (4) Risk fusion shipped with two measurement-caught bugs: a weighted average that silently overruled the calibrated model, and a graph threshold that gave 26% of legitimate transactions a risk bonus. (5) Twice we found our own evaluation was reading an answer key we had written: first at the agent layer, where entity IDs were self-labelling (`pi_STOLEN_*`) and hashing them cost us 10/10 → 5/10 correct-cause; then at the ML layer, where two simulator-encoded features reproduce the entire 22-feature headline — which forced us to **retract this README's central ablation claim** about entity/graph features. Both are published rather than patched over (see [Leakage self-audit](#leakage-self-audit-we-broke-our-own-headline)). At runtime: if the ML scorer is unavailable, decisions fall back to rules and route to human review — an LLM failure cannot block anyone, because the LLM was never in the decision path.

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
                                    7 read-only tools · ₹ math in Python · audit log
                                    explains and recommends FROM the allowlist
                                                                      │
                                    any failure ──> deterministic report ──> human review
```

## Repo layout

```
src/sim/        transaction simulator — 6 labeled scenarios incl. legitimate flash sale
src/features/   incremental leakage-safe feature builder (~22 features)
src/models/     empirical model selection, temporal-split training, isotonic
                calibration + reliability curve, cost-optimal threshold,
                feature-group ablation, adversarial leakage self-audit,
                multi-world seed stability
src/spike/      merchant-level EWMA + z-score change-point detector
src/policy/     deterministic policy engine + frozen allowlist; risk fusion
                (ML floor + spike/graph/rule lift -> risk_score + confidence);
                validation threshold sweep
src/agent/      LangGraph investigator (Claude Haiku), 7 read-only tools,
                audit log, 13-case eval (3 held-out), evidence-traceability
                checker; five-run eval progression in artifacts_out/eval_runs/
src/serve/      FastAPI API (shared-key gate on every mutating route) +
                replay driver + React SPA (vendored, no build)
run_demo.py     one-command demo: train -> serve -> replay
tests/          safety invariants (fail-safe, LLM cannot escalate, flash-sale
                no-fire, fusion floor/bounds, agent gate/audit/read-only,
                serving side-effect-freedom, analyst allowlist, write-auth
                gate) - 51 tests
```

## Honest limitations

Ordered by how much they should discount the results.

1. **Our simulator encodes the label into two features, so the headline PR-AUC overstates real detection ability.** `customer_age_days` + `amount_dev_ratio` alone reach 0.9328 vs the full model's 0.9344. The fix is to regenerate attack accounts with realistic ages (fraudsters buy aged accounts — precisely the case we never generate) and to stop deriving ambient fraud amounts as a multiple of legitimate ones. That changes every number in this README, which is why it is documented rather than rushed. See [Leakage self-audit](#leakage-self-audit-we-broke-our-own-headline).
2. **No public-dataset validation.** Nothing here has touched real transactions. The feature builder and temporal harness would run against IEEE-CIS or a similar public set with modest plumbing; we did not get to it. Until then, "would this generalize?" is genuinely unanswered.
3. **Data is synthetic throughout** (simulator parameters like "0.7%→5% fraud" are design choices, not Razorpay statistics). The simulator explicitly wires shared entities across accounts — fraud rings only exist in a graph if you construct them.
4. **One legitimate-spike scenario.** The flash sale is the only benign volume event tested. Festival sales, product launches and marketing campaigns are untested, and the EWMA baseline has no day-of-week or hour-of-day seasonality term.
5. **The review queue may not be staffable.** 44.1 cases per 1,000 transactions is 4.41% of the stream. We price analyst time at ₹50/case but never ask whether the analysts exist; at PSP volume that is a headcount question, and the honest answer is that the restrict/review cutoffs would have to be re-swept against a capacity constraint, not only against cost.
6. **Headline metrics now carry a range, but a narrow one.** PR-AUC is 0.945 ± 0.007 over five worlds ([stability](#stability-across-worlds-is-this-a-result-or-one-lucky-seed)); differences below roughly ±0.02 should not be treated as real — which is also why we do not claim the 2-feature and 22-feature models are distinguishable on ranking alone. Five worlds is still a small sample, and they share one generative process, so this measures *sampling* variance, not *model-family* or *real-world* variance.
7. **The agent gets the cause right 8/13 times** and on quiet merchants has twice contradicted its own evidence (asserting account takeover while citing zero new-device and zero geo-mismatch rates). It is advisory only and cannot act — that separation is tested — but the reasoning quality is the weakest part of the system. **If you run the demo yourself you will probably see this live on m3:** it reads "all 183 flagged txns use distinct payment methods" and concludes "no testing pattern", when a fresh instrument on every transaction *is* the card-testing signature. The `flagged_distinct_instruments_per_txn` field is directionally ambiguous — ~0.08 means reuse, ~1.0 means novelty, and the tool never says which way means what. Logged as failure-log 22 and deliberately left unfixed: the design is frozen after run D, coaching the prompt is against our own standing rule, and changing the tool would invalidate the eval it is measured by.
8. **Risk fusion changes 0 of 13,987 decisions.** Retained for architecture (auditability, a reachable fail-safe path, headroom for a weaker model), not for metrics.
9. **The account-takeover scenario is diffuse by nature**; it is caught primarily at transaction level (p≈0.97 per txn) and only weakly at merchant-spike level.
10. **Not production hardened.** Single process, no persistence (state is lost on restart), no idempotency on repeated transactions. The serving layer has a shared-key gate on every mutating endpoint, but that is a single-tenant gate, not identity — an analyst override carries no attributable actor, which a real deployment would need.
