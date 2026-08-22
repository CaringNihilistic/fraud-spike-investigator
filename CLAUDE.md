# CLAUDE.md — Fraud Spike Investigator

Razorpay Buildathon **Track 02: AI Risk Manager**. Solo builder, tight deadline.
Goal: get shortlisted. Judging criteria (structure ALL work and docs around these):
1. **Problem taste** — picked something that matters
2. **Build quality** — runs, structured, trustworthy
3. **AI judgment** — right tool in the right place, and where we chose NOT to use AI
4. **Failure recovery** — what broke and what we did about it
Track bar: honest metrics incl. false-positive cost; strictly defense-only.

## What this is
Merchant-level fraud-spike detection + entity correlation + policy-gated LLM
investigation. Sits ABOVE per-order scoring (Razorpay's Thirdwatch/Shield do
per-order; Bumblebee does onboarding review) — our wedge is real-time
transaction-stream "you are under attack" investigation with ₹ economics.

## Architecture (do not change without strong reason)
- `src/sim/` — synthetic generator. ALL entity IDs are opaque hashes via
  `oid()` — never semantic (see failure-log 19). `generate_holdout()` builds a
  second world for held-out agent eval only. 6 scenarios (card-testing, device farm, IP
  cluster, ATO, fraud ring, legitimate flash sale that must NEVER be flagged).
  Historical attacks in train period (days 6–18, own entity IDs), novel attacks
  in test period (days 24–29, fresh entity IDs + different merchants).
- `src/features/builder.py` — 22 features, STRICTLY incremental in ts order
  (features from prior events only, state updated after emission). Never break
  this leakage guarantee.
- `src/models/select_model.py` — P1a-0, empirical model selection. Trains
  LogisticRegression/XGBoost/LightGBM/CatBoost with identical class
  weighting and default hyperparameters, compares PR-AUC + speed on the
  VALIDATION slice only (test slice untouched), selects by a rule fixed in
  advance. Persists the winner to
  `artifacts_out/model_selection_decision.json`; `build_gbdt()` is the only
  place that constructs a specific library, so train.py/ablation.py never
  hardcode one. Current winner: XGBoost.
- `src/models/train.py` — day-boundary temporal split (train ≤ d20, cal d21–23,
  test ≥ d24), class-weighted GBDT (library selected empirically by
  select_model.py, NO SMOTE — distorts temporal distribution), isotonic
  calibration, cost-optimal ₹ threshold, net-protected-value economics
  (step 7), time-to-detect comparison (step 8).
- `src/models/ablation.py` — P1a, feature-group ablation (basics → +velocity
  → +entity/graph) plus a stage-4 merchant-level replay through the spike
  detector + policy engine, using the select_model.py winner. Output:
  `artifacts_out/ablation_table.csv`.
- `src/spike/detector.py` — hourly EWMA+z-score (slow path) AND
  StreamingSpikeDetector (fast path: ≥8 high-risk of last 30 txns within
  bounded span; rate+span guards give structural false-alarm immunity).
- `src/policy/engine.py` — deterministic policy engine, frozen action allowlist
  (allow/step_up/review/restrict). THE ONLY component that authorizes actions.
- `src/policy/fusion.py` — P1b risk fusion. Calibrated ML prob is the FLOOR;
  context (spike z .50 / graph .30 / rules .20, × CONTEXT_LIFT 0.6) escalates
  into the headroom above it → risk_score 0–100. Confidence = signal
  AGREEMENT (high ML + no corroboration ⇒ low confidence ⇒ human review).
  Never authorizes anything; emits (risk_score, confidence) for engine.decide().
  Graph signal measures EXCESS component size over the ordinary population
  (FLOOR 25 = train legit p99, SATURATE 120 = train fraud p90; train-derived
  ONLY). Also `evaluate_rules()` — deterministic rule SIGNALS, not actions.
- `src/policy/threshold_sweep.py` — cost-optimizes restrict/step_up cutoffs on
  the VALIDATION slice only. Adopted (85, 25): step_up 60→25 = +8.28% val NPV;
  restrict stayed 85 (surface flat/unidentified across 40–80). Per-parameter
  2% adopt margin fixed in advance.
- `src/agent/` — P2 LLM investigator. `tools.py` = 6 READ-ONLY tools (no tool
  exposes is_fraud/scenario; calculate_exposure does ALL ₹ math in Python).
  `investigator.py` = LangGraph loop (Claude Haiku 4.5, max 8 rounds) whose
  recommended_action always passes through validate_recommendation(); any
  failure → deterministic templated report → human REVIEW. `audit.py` = every
  tool call logged (tool, inputs hash, output hash, ts), degraded path too.
  `eval.py` = 10 fixed + 3 HELD-OUT ground-truth cases (new seed, unseen
  merchants) → artifacts_out/agent_eval.csv + per-case transcripts.
  `verify_evidence.py` = traceability checker: every number in evidence[] must
  be findable in a tool output the agent actually received.
- `src/serve/` — P3 dashboard. `state.py` = in-memory pipeline state (single
  writer = replay thread, RLock-guarded). `replay.py` = streams the test slice
  through the REAL fusion→policy path (not a canned animation); investigations
  fire on SPIKE, off the hot path. `api.py` = FastAPI + static SPA.
  `static/` = React via vendored UMD + htm — NO build step, NO CDN, NO npm.
  `run_demo.py` = one command: train → serve → replay.
- `tests/test_safety.py`, `tests/test_agent.py`, `tests/test_serving.py` — safety invariants. Keep
  green, extend when touching policy/detector/agent. Agent tests use a
  SCRIPTED client double — no network, no credentials needed.

## Hard constraints (non-negotiable)
- The LLM NEVER makes the fraud decision. It investigates/explains/recommends
  from the allowlist; `validate_recommendation()` degrades unknown actions to
  REVIEW, never escalates. LLM failure must never block anyone.
- Temporal evaluation only. Never random splits. Never leak future data.
- No SMOTE. No GNNs, Kafka, Neo4j, autoencoders, RAG, k8s, feature stores.
- Fail safe: ML down → rules + human review; low confidence → escalate.
- Honest metrics: report ₹ wrongly blocked and false-alarm counts, incl.
  negative results. Never present synthetic data as Razorpay data.
- Defense-only. Nothing offense-capable.

## Current verified results (test slice, synthetic data)
Model: XGBoost, selected empirically over LightGBM/CatBoost/LogisticRegression
on temporal validation (src/models/select_model.py). All 3 GBDTs within the
0.02 tie margin (LGBM .9308 / Cat .9265 / XGB .9249) → won on the PRE-DECLARED
speed tie-break (0.27s vs 1.64s vs 7.94s), NOT on raw PR-AUC.
PR-AUC 0.934 | P 0.994 / R 0.886 @ cost-optimal threshold | P@100/@500 = 1.00
Policy cutoffs (85, 25) — step_up cost-optimized on validation (+8.28% val NPV,
+3.7% on test); restrict unidentified on validation, left at 85.
₹10.93L prevented, ₹10.57L net protected value (~31x), 617 review cases
5/5 attack merchants detected, 0 false alarms, flash sale NOT flagged
TTD beats flag-counter 4/5 (fraud ring 70m vs 101m); LOSES on IP cluster
(31m vs 26m) — report it, don't smooth it.
Ablation: basics 0.661 → +velocity 0.631 (mildly negative) → +entity/graph
0.934. Entity/graph is where the system comes from (+0.30 PR-AUC, recall
.58→.89). NOTE: an earlier dramatic "velocity collapses to 0.23" finding was
an artifact of an attack-free calibration slice and has been RETRACTED in the
README — keep the retraction visible, it's the honest version.
P1b fusion changes 0/13,987 decisions vs the old p*100 shortcut. Kept for
architecture (auditability, reachable fail-safe, headroom for a weaker model),
NOT for metrics. Say so plainly; do not re-frame it as a win.
P2 agent eval (LIVE Claude Haiku 4.5, DE-LABELLED data, run D = FINAL):
8/13 correct cause (2/3 held-out), 100/100 evidence claims traceable,
escalates-when-unsure 13/13, policy violations 0/13, UNSAFE ACTIONS 0/13
(no attack ever got `allow`; all de-escalations were on attacks whose current
flagged rate was 0.0 = already ended). correct_action 6/13 partly measures
LABEL DESIGN — expected_action predates peak-vs-current visibility. Do NOT
"fix" it by editing expected_action labels. Five-run progression
A->B1->B->C->D preserved in artifacts_out/eval_runs/ — run A (9/10) is
PRE-de-labelling and must always be labelled as such; most of that score was
semantic-ID leakage. DESIGN IS FROZEN: no further tool/prompt/weighting
changes without a new held-out set, or the number stops meaning anything.
Run: `python -m src.models.select_model && python -m src.policy.threshold_sweep
&& python -m src.models.train && python -m src.models.ablation`
Demo: `python run_demo.py` (one command; ~60s replay at default 250 txn/s)
Tests: `python -m pytest tests/ -q` (46 pass, no network needed)

## Roadmap (in priority order)
- ~~P1a-0 — empirical model selection~~ DONE: `src/models/select_model.py`,
  winner XGBoost, persisted decision consumed by train.py/ablation.py.
- ~~P1a — ablation study~~ DONE: `src/models/ablation.py`, table in README
  and `artifacts_out/ablation_table.csv`.
- ~~P1b — risk fusion~~ DONE: `src/policy/fusion.py`, wired into the
  economics loop, 10 new safety tests. Honest result: 0 decisions changed.
- ~~P2 — LLM investigator~~ DONE: `src/agent/`, 7 read-only tools, 16 agent
  tests, live eval on de-labelled data with a held-out set (see results).
- **P2b — FastAPI serving**: POST /transactions, GET /merchants/{id}/risk,
  POST /merchants/{id}/investigate, GET /review-queue. Audit log table.
- ~~P3 — React dashboard~~ DONE: `src/serve/` + `run_demo.py`, 13 serving tests.
  Verified end-to-end: 5/5 attacks spike (peak flagged rate 73–93%, z 4.7–5.8),
  m11 flash sale 3% / z=1.1 / 0 restricts / empty entity graph.

## Failure-recovery log (KEEP UPDATING — it's a judging criterion)
1. Attacks initially test-period-only → model had nothing to learn → PR-AUC
   0.28. Fix: historical attacks in train period → 0.79.
2. min_txn=20 hourly guard hid the card-testing spike on a low-volume merchant.
   Fix: min_txn=10 + variance noise-floor (0.03) so stray fraud can't false-alarm.
3. StreamingSpikeDetector required a FULL 30-txn window before evaluating →
   slower than a naive flag-counter. Fix: partial-window evaluation.
4. min_rate=0.35 on a full window silently required 11 hot, overriding k_hot=8.
   Fix: min_rate=0.25 (guards only partial windows).
5. LightGBM was assumed as the default GBDT without ever comparing it.
   Fix: P1a-0 empirical selection (select_model.py) — XGBoost actually wins
   on validation PR-AUC and is ~5x faster to train; switching the winner
   raised end-to-end PR-AUC 0.79 → 0.93 (also reflects other pipeline
   changes since the 0.79 baseline, not library choice alone).
6. RETRACTED FINDING (kept deliberately). Ablation stage 2 appeared to
   REGRESS hard (0.55 → 0.23, 6x legit ₹ blocked) and was written up as
   "velocity alone is a trap." Root cause was ours, not the features: the
   calibration slice (days 21–23) contained NO attack, so isotonic
   calibration fit degenerate score plateaus. Fix: move one historical
   device-farm attack to day 22. Effect shrank to 0.661 → 0.631 — mild, not
   a collapse. A supporting top-200 diagnostic ALSO dissolved: the top-200
   sat inside a ~447-txn tie-plateau at p=1.0, so its composition was decided
   by row order; re-ranking by raw score gives 0 flash-sale txns and
   precision 1.000 for every variant. Feature slicing was verified correct
   (clean 6/6/10 partition) — there was no bug, the diagnostic just couldn't
   support the claim. Lesson: a dramatic result on a small validation slice
   is a prompt to re-measure, not to publish.
7. Fusion v1 was a plain weighted average (ml*0.6 + context*0.4) → capped a
   p=1.0 txn at risk 60, silently OVERRULING the calibrated model it was
   meant to defer to; net protected value fell ₹10.20L → ₹9.27L. Fix: ML is
   the FLOOR, context escalates into the headroom above it.
8. Fusion COMPONENT_SATURATE=15 gave full graph-risk credit to 26% of
   LEGITIMATE txns — ordinary customers already sit in ~10-node components
   via shared ISP IP pools. Fix: measure EXCESS over the ordinary population
   (floor 25 / saturate 120, both train-derived). Legit mean risk 9.02 → 0.67.
9. precision_at_k silently depended on row order: isotonic maps ~614 test
   txns to exactly 1.0, so k=100/500 fell inside a tie-plateau. Fix: explicit
   raw-score tie-break. Verified the reported 1.00 holds under index-order,
   raw-score, AND adversarial tie-breaking — it was accidentally correct, now
   it's well-defined.
10. Threshold sweep's pre-declared rule ("adopt the best PAIR if >2%") was
   under-specified for a degenerate surface: restrict NPV is EXACTLY flat from
   40–80 (no validation txn scores in that band), so "best pair" would have
   adopted restrict=80 — an arbitrary point in a dead zone — on a +0.53%
   difference. Fix: apply the same margin PER PARAMETER. Refinement was made
   after seeing the surface and is documented in the module, not hidden.
11. LangGraph silently dropped `_stop_reason` because it wasn't declared in the
   AgentState TypedDict — so `route()` never saw "tool_use", EVERY investigation
   fell through to the fallback, and it looked exactly like "the model is bad".
   Caught only because the scripted-client tests asserted on the audit log's
   tool sequence. Fix: declare `stop_reason` in the state schema. Lesson: with
   LangGraph, undeclared state keys vanish without error — assert on observable
   side effects (the audit log), not just on the final output.
12. The deterministic fallback called tools directly, bypassing the audit log —
   so the audit went quiet precisely when something had gone wrong, and eval
   scored evidence_valid 0/10. Fix: fallback records its tool calls too.
13. Dashboard risk gauge showed the LAST transaction's risk, so attack
   merchants read ~0 once their burst passed. Fix: rolling peak over last 50.
14. Dashboard invented its OWN baseline (first 200 txns), which is contaminated
   whenever the attack starts early in the slice — m3's card-testing wave is on
   day 24, the first test day, so the board read "0.3 -> 0.0" (improving!)
   during an active attack. Fix: read baseline/current/z from the spike
   detector's own slow EWMA. One definition of "normal", not two.
15. Added `peak_risk_ever` to preserve the story after a burst cools — it came
   out 100 for ALL 12 merchants incl. the flash sale, because every merchant
   has ≥1 ambient-fraud txn scoring ~100. Useless as a discriminator and would
   have shown "peak 100/100" on the flash-sale card, undermining the demo's
   central claim. Removed it; peak flagged-RATE and peak z discriminate
   properly (93%/z5.8 attack vs 3%/z1.1 quiet) because they're merchant-level.

16. LIVE AGENT RUN: the agent labelled m2 (a real ₹5.5L account-takeover) as
   `legitimate_traffic` / `allow` at 0.95 CONFIDENCE. Evidence was factually
   correct ("29 customers, 29 devices, 29 IPs, zero shared entities") and the
   conclusion was exactly wrong. Root cause is a TOOL GAP, not model quality:
   ATO = established customers on NEW devices / NEW geo / atypical amounts, and
   NO agent tool exposes is_new_device_for_customer, geo_mismatch,
   amount_dev_ratio, or customer_age_days — they exist in the feature builder
   and drive the ML scorer, but the agent reasons only from entity-sharing, and
   ATO does not share entities. The policy engine still restricted 68 txns and
   queued 8 for review on m2 because the LLM was never in the decision path —
   this is the architecture's central claim demonstrated, not asserted.
   CLOSED (run B): added 7th tool get_customer_anomalies exposing new-device /
   geo-mismatch / amount-vs-own-average / account-age, with the non-flagged
   population as comparison. m2 now diagnosed correctly and — more importantly —
   quiet merchants use the SAME fields to RULE OUT ATO ("0% new device, 0% geo
   mismatch"). Prompt unchanged (sha256-verified) — the fix was evidence, not
   coaching. Residual: agent output still varies run to run.
17. get_merchant_baseline splits at ts.quantile(0.75), so an attack that ENDS
   mid-window reads as "baseline 8.99% -> recent 0.35%", i.e. improving. The
   model then rationalises it ("flagged rate jumped from baseline 16.65% to
   recent 0.38% AFTER accounting for the farm burst" — describing a DECREASE
   as a jump). Contributed to the ATO miss and to ip_cluster being downgraded
   to step_up. CLOSED (run B): get_merchant_baseline now reports the spike
   detector's OWN slow-EWMA baseline plus the PEAK 30-txn window and explicit
   window bounds, so "attack that ended" is legible as such instead of reading
   as "improving". Side effect worth knowing: the agent became temporally aware
   and now de-escalates finished attacks (restrict -> review), which our
   expected_action labels — written before the tool existed — score as wrong.

18. EVAL-INTEGRITY, part 1 (our statistics bug). First version of
   get_customer_anomalies reported shares over FLAGGED transactions with no
   denominator and no comparison population: "80% spent 3x+ over own average"
   on n=5. Three quiet merchants were promptly diagnosed account_takeover
   (run B1, preserved in artifacts_out/eval_runs/). Fix: report the same
   profile over NON-flagged transactions of the same merchant, with explicit n
   and lift. Lesson: a rate over a selected subpopulation is not evidence
   without its base rate — and we shipped that mistake INTO a safety tool.
19. EVAL-INTEGRITY, part 2 (the dataset was whispering the answer). Simulator
   entity IDs were self-labelling: pi_STOLEN_*, d_FARM_F, ip_CLUSTER_I,
   d_RING_R3, d_ATO_*. Run B transcripts cite them verbatim as evidence
   ("183 distinct stolen instruments (pi_STOLEN_*)"). Fix: hash EVERY id —
   legitimate and attack alike — to an indistinguishable kind_<8hex> form via
   oid(); ground truth lives only in `scenario`, which no tool exposes.
   Correct-cause fell 10/10 -> 5/10. That gap IS the leakage measurement.
   De-labelling asserted to be a pure relabelling: all 16 ML metrics
   bit-identical (features count entity SETS, never parse ID strings).
   Only ONE tool was added afterwards (instrument-novelty ratio, the one
   signal a real analyst legitimately has), then the design was FROZEN and
   run D scored once, with 3 held-out cases on a new seed. Final: 8/13.
20. verify_evidence.py's entity-id regex was written through a bash heredoc,
   which turned `` into a literal 0x08 BACKSPACE character. The compiled
   pattern printed as correct in the terminal (control char invisible) but
   could never match, so 11 evidence claims were reported UNTRACEABLE when
   they were only quoting hashed entity ids. Fixed -> 100/100 traceable.
   Lesson: when a regex "looks right but never matches", check the bytes.

## Style
Plain, direct comments explaining WHY. Small modules. No cleverness that costs
explainability — every component must be defensible to a judge in one sentence.
