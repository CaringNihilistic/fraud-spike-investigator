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
- `src/models/leakage_probe.py` — P0, ADVERSARIAL SELF-AUDIT. Attacks our own
  evaluation the way a hostile judge would: single-feature PR-AUC for all 22,
  feature-SET comparisons, and suspect-feature distributions BY SCENARIO.
  Reuses ablation._fit_eval so there is exactly ONE definition of "the same
  measurement" (failure-log 14's lesson). It found failure-log 21. Re-run it
  after ANY simulator change.
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
  Every MUTATING route requires an `X-API-Key` (`FSI_API_KEY`, else an
  ephemeral key minted at startup and injected into the page same-origin, so
  the demo stays one command). Reads stay open on purpose. Single-tenant gate,
  NOT identity — an analyst override carries no attributable actor.
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
CALIBRATION: Brier 0.00533 (raw 0.00647), ECE 0.0033 - isotonic measurably
helps, but the reliability curve is near-bimodal (13,288 txns in [0,.1], 615 in
[.9,1]), so the low ECE is dominated by the extremes. Report both.
REVIEW LOAD: 44.1 cases per 1,000 txns (4.41%) - staffing question, not just
cost. Do not quote INR/case without it.
!! HEADLINE CAVEAT (failure-log 21): customer_age_days + amount_dev_ratio ALONE
score 0.9328 vs the full 22-feature 0.9344. The simulator writes the answer key.
NEVER quote 0.934 without the caveat. The README ablation claim about
entity/graph carrying the lift is RETRACTED and corrected in place.
Policy cutoffs (85, 25) — step_up cost-optimized on validation (+8.28% val NPV,
+3.7% on test); restrict unidentified on validation, left at 85.
₹10.93L prevented, ₹10.57L net protected value (~31x), 617 review cases
5/5 attack merchants detected, 0 false alarms, flash sale NOT flagged
TTD beats flag-counter 4/5 (fraud ring 70m vs 101m); LOSES on IP cluster
(31m vs 26m) — report it, don't smooth it.
Ablation: basics 0.661 → +velocity 0.631 (mildly negative) → +entity/graph
0.934. The "entity/graph is where the system comes from" reading is WRONG and
RETRACTED — see failure-log 21. Diagnostics (printed by ablation.py itself):
profile pair alone 0.9328 (n=2) | entity SHARING alone 0.8286 (n=4) | full
minus the pair 0.8777 (n=20). What entity features DO buy, and the only
defensible claim: legit INR wrongly blocked 68,319 (2-feature) → 5,901 (full). NOTE: an earlier dramatic "velocity collapses to 0.23" finding was
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
Self-audit: `python -m src.models.leakage_probe` (adversarial eval integrity)
Real data: `python -m src.models.real_data_check` (ULB/IEEE via Kaggle; data/ is
gitignored and NEVER committed - competition terms, publish metrics not data)
Demo: `python run_demo.py` (one command; ~60s replay at default 250 txn/s)
Tests: `python -m pytest tests/ -q` (57 pass, no network needed)
REAL-DATA (ULB creditcardfraud, same recipe): PR-AUC 0.731 vs 0.0017 random
baseline. Methodology transfers, the 0.934 does NOT. Merchant-level layer is
UNVALIDATED on real data - say so whenever the product claim comes up.

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

21. WE BROKE OUR OWN HEADLINE. Ran an adversarial audit against the ML eval
   (`src/models/leakage_probe.py`) — the same attack that found the semantic-ID
   leakage at the agent layer (entry 19), aimed one layer down. Result:
   `customer_age_days` + `amount_dev_ratio` ALONE score PR-AUC 0.9328 vs the
   full 22-feature 0.9344 — a gap of 0.0016. Cause is the GENERATOR, not the
   model: attack generators set customer_created_day to the attack day, so
   median account age is 0.98 (card testing) / 1.61 (device farm) / 2.64 (ip
   cluster) / 5.65 (fraud ring) days against a legitimate baseline of 215.76;
   and ambient fraud is `legit_amount * uniform(1.5, 4.0)`, making
   amount_dev_ratio a second proxy (2.07 ambient / 5.44 ATO vs 1.00 legit).
   Between them the two features partition the label space — which is why two
   columns match twenty-two.
   CONSEQUENCE: the README's central ablation claim ("entity/graph is where the
   system comes from, +0.30 PR-AUC") was MIS-ATTRIBUTED — both proxies sit
   inside the ENTITY_GRAPH bucket. Entity SHARING alone is 0.8286 and adds only
   +0.004 on top of the pair. Claim RETRACTED and corrected in place; the
   diagnostic rows now print from ablation.py itself so the table cannot be
   read without them.
   NOT INVALIDATED (each checked, not assumed): pipeline leakage hygiene
   (features still strictly incremental, day-boundary splits, calibration on
   d21-23 only) — this is DATA CONSTRUCTION, not train/test contamination; the
   de-labelling assertion (all 16 ML metrics bit-identical); and every
   merchant-level result, since flash-sale customers are OLDER than baseline
   (230.65 vs 215.76), so 5/5-attacks / 0-false-alarms / flash-sale-not-flagged
   cannot be explained by account age.
   DELIBERATELY NOT FIXED: regenerating attack accounts with realistic ages
   (real fraudsters buy AGED accounts — the case we never generate) changes
   every number in the README. Published as a measured limitation instead of
   rushed under deadline. Listed first under Honest limitations.
   LESSON: we found this by attacking our own eval a SECOND time, after
   believing we had already done that. "We audited for leakage" is not a state
   you reach once — entry 19 fixed the agent's evidence and we never
   re-pointed the same test at the model.

22. THE AGENT READ A NUMBER BACKWARDS. Observed during a pre-recording DEMO
   rehearsal, NOT an eval run: agent_eval.csv and run D are untouched and the
   headline stays 8/13. On the demo slice the agent got 4/5 causes right,
   including m2 (the account-takeover that failed catastrophically in run A),
   and missed m3: card-testing diagnosed as `fraud_ring` at 0.92 confidence.
   Its own evidence line is the tell: "Unique instrument per transaction: all
   183 flagged txns use distinct payment methods; NO INSTRUMENT REUSE OR
   TESTING PATTERN". It has the correct number and draws the inverted
   conclusion — a distinct instrument on every transaction IS the
   card-testing signature (a fraudster burning through stolen cards), not
   evidence against it.
   ROOT CAUSE is the tool, not the model. `flagged_distinct_instruments_per_txn`
   (added in run D) is DIRECTIONALLY AMBIGUOUS: ~0.08 means reuse (farm/ring),
   ~1.0 means novelty (card testing), and nothing in the tool output says which
   direction means what. The agent read m5's 0.076 correctly as a farm and
   m3's ~1.0 as "no pattern". Same field, opposite ends, only one of them
   legible.
   NOT FIXED, deliberately. The design was FROZEN after run D. The two
   available fixes are both disqualified right now: coaching the prompt is
   forbidden (the standing rule is that the fix must be evidence, never hints
   — see entry 16), and changing the tool invalidates the frozen eval, so the
   number would stop meaning anything. The honest move is to log it and let a
   NEW held-out set measure the fix, not to patch it under deadline and
   re-report the same 13 cases.
   CONTAINED: recommended action was `review`, not `allow` — the error was in
   the cautious direction, and the policy engine restricted m3 transactions on
   its own regardless, because the LLM is not in the decision path.
   LESSON: a number is not evidence until its DIRECTION is stated. Entry 18 was
   a rate without its base rate; this is a ratio without its polarity. Both
   shipped inside a tool we had already reviewed for exactly this.

23. REAL DATA FOUND A BUG OUR OWN DATA COULD NOT. Ran the unchanged recipe
   against ULB creditcardfraud (284,807 real card txns, 0.173% fraud) via
   `src/models/real_data_check.py`. PR-AUC 0.731 vs a 0.0017 random baseline
   (423x lift); our simulator scores 0.934 on the same recipe, and that gap is
   the honest measure of how much our own data was helping. Two findings, both
   structurally impossible to surface on synthetic data:
   (a) OUR AMOUNT MODEL IS BACKWARDS. We generate fraud as
   `legit_amount * uniform(1.5, 4.0)`, so fraud is 1.37x the median legit
   amount. Real card fraud is 0.42x (median $9.25 vs $22.00) - card testing
   uses TINY amounts on purpose. So amount_dev_ratio, one of the two label
   proxies from entry 21, points the WRONG WAY on real data. Independent
   real-world confirmation of entry 21.
   (b) BUG in cost_optimal_threshold: the grid was
   `quantile(p, linspace(0,1,200))`, whose max is max(p), and `p >= max(p)`
   still blocks the top-scoring rows - so ABSTAIN ("block nothing") was
   UNREACHABLE. Our data hid this completely because fraud is expensive there
   by construction, so blocking always pays and the optimum is always interior.
   On ULB, abstaining is 3.8x cheaper than the threshold the function chose
   ($7,729 vs $29,577). FIXED by appending np.inf to the grid. Every synthetic
   number is BIT-IDENTICAL after the fix (PR-AUC 0.9344, threshold 0.8333, NPV
   1,057,319.68, all six ablation rows) - which is what "latent" means, and is
   how we know the fix is safe.
   FOLLOW-ON, reported as-is: with abstain reachable, the cost-optimal action
   on ULB is to BLOCK NOTHING. The model ranks fine (ROC-AUC 0.974) and the
   economics still say do not act. That is a limit of OUR cost model, which
   prices a false negative at exactly the fraud amount and nothing else - real
   issuers also carry chargeback fees, dispute handling, regulatory exposure
   and churn. Add a fixed per-fraud penalty and the optimum leaves abstention
   at once. Do NOT let this get quoted as "fraud detection is not worth it".
   SCOPE: transaction level ONLY. ULB has no merchant column, so the spike
   detector, entity graph and policy engine are NOT exercised. IEEE-CIS has the
   entity columns to close that gap and is already wired into the same module;
   it needs its competition rules accepted (the API 403s otherwise).
   LESSON: we spent the whole project auditing our own evaluation and still
   could not see this one from the inside. Some bugs are only visible from
   outside your own data.

## Style
Plain, direct comments explaining WHY. Small modules. No cleverness that costs
explainability — every component must be defensible to a judge in one sentence.
