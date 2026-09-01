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
  second world for held-out agent eval only. 8 scenarios: 5 attacks (card
  testing, device farm, IP cluster, ATO, fraud ring) and 3 LEGITIMATE spikes
  that must NEVER be flagged — a flash sale (6x volume, no shared entities),
  a corporate buyer (40 accounts on ONE office IP) and a shared kiosk (25
  accounts through ONE device). The last two are the HARD negatives: they carry
  the exact entity signature of an attack and are honest. The flash sale alone
  never exercised the entity layer at all. See failure-log 29.
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
  the VALIDATION slice only. Adopted (85, 25): step_up 60→25 = +2.73% val NPV.
  Restrict stays 85 because moving it alone to the best pair's 55 is NOT
  INDEPENDENTLY EVALUABLE — step_up must remain the lower bar, so (55, 60) is
  not a point on the grid at all. Per-parameter 2% adopt margin fixed in
  advance; a move that cannot be measured alone is not evidence for making it.
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
  writer = replay thread, RLock-guarded). The review queue is ranked by EXPECTED
  LOSS (amount x calibrated p), never by arrival or by risk_score — the sort key
  is a policy decision under a capacity constraint, is returned on the wire as
  `ordering`, and is pytest-locked. `ReviewCase.p_fraud` exists for exactly this
  and is kept separate from `risk_score` on purpose. `MerchantState.signature()` describes
  the SHAPE of a merchant's flagged traffic by COUNTING entities ("3 devices
  shared by 60 accounts"), never by inferring and never from `scenario` — it is
  what the card shows when no LLM is running, and it is test-locked against
  leaking the answer key. `replay.py` = streams the test slice
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
0.02 tie margin (LGBM .7702 / Cat .7573 / XGB .7625) → won on the PRE-DECLARED
speed tie-break (0.40s vs 1.85s vs 8.82s), NOT on raw PR-AUC.
PR-AUC 0.825 | P 0.996 / R 0.735 @ cost-optimal threshold | P@100/@500 = 1.00
STABILITY: 0.862 +/- 0.024 over 5 SEEDS OF THE SAME GENERATOR (not independent
worlds - identical scenario definitions). Seed 7, the demo world, is the WORST
of the five. The spread TRIPLED after the hard negatives (was 0.007) - those
merchants sit near the decision boundary, so where they land moves with seed.
CALIBRATION: Brier 0.0123 (raw 0.01468), ECE 0.00736 - measurably worse than
before the hard negatives (0.0082/0.0031); near-boundary merchants are exactly
what a calibrator finds hardest. Reliability curve still near-bimodal.
REVIEW LOAD: 66.9 cases per 1,000 txns (6.69%) - UP 60% from 41.9. Teaching the
model that shared devices can be honest made it more cautious everywhere. At 1M
txns/day that is ~279 analysts, not 175. Staffing question, not just cost.
FALSE-POSITIVE COST: INR 2,575 wrongly blocked = 0.024% of the INR 1.07Cr in
legitimate value processed. ALWAYS give it that denominator.
COST-MODEL ROBUSTNESS: review cost is 5.5% of gross prevented at INR 50/case;
NPV break-even is INR 905/case, 18x the assumption. Do NOT build a full cost
sweep - fused scores are near-bimodal so cost changes rescale the headline
without moving the policy. The open cost question is the FN price (entry 23).
LEAK AUDIT (entry 21, CLOSED): the two profile proxies score 0.5970 vs the full
0.8248 - gap 0.2278. Pre-fix they were 0.9328 vs 0.9344, gap 0.0016. The 0.934
and 0.898 figures are PRE-FIX and appear only where labelled as history.
SENSITIVITY (entry 26): aged_share is NOT load-bearing across 0.3-0.9.
Policy cutoffs (85, 25) - step_up cost-optimized on validation (+2.73%).
Restrict stays 85: moving it alone is NOT INDEPENDENTLY EVALUABLE.
INR 8.61L exposure prevented, INR 8.13L net protected value (~18x), INR 452
legit revenue impacted on the policy path, 948 review cases
5/5 attack merchants detected | 25/25 across 5 seeds
!! 1 FALSE ALARM in 35 non-attack merchant-windows (2.9%, 95% CI 0.1-14.9%)
- a LEGITIMATE corporate buyer (m4) on seed 11, checked across all 5 seeds. This is NOT 0 any more and must never be quoted as 0.
It is 1 because the symmetric fix costs an entire attack class - see entry 29.
Flash sale flagged 0/5. Three legitimate spikes are now tested, two of which
share entities; the flash sale alone never exercised the entity layer at all.
TOPOLOGY GENERALISATION (entry 34): attacks rebuilt with DIFFERENT entity
graphs, everything else matched, frozen pipeline, 5 seeds. known 22/25 detected
vs unseen 21/25; 4 of the 5 unseen variants are 2.5-10x LESS concentrated, i.e.
harder, and all four were still caught 5/5. Mean flagged rate inside the attack
0.815 -> 0.709 (card testing 0.758 -> 0.436, ip cluster 0.872 -> 0.612): the
merchant layer holds while the TRANSACTION scorer loses confidence. ATO control
is weak (2/5 even known) - both its rows are weak evidence and we did NOT tune
the window until it fired.
CONTINUUM SWEEP (entry 32): those three are POINTS, so the boundary was swept -
legitimate sharing 0-100% through one entity, 5 seeds, frozen pipeline, real
device farm as positive control. Nothing fires below 80% concentration (40/40
merchant-seeds clean); 3 of 60 fire at >=80%; INR 0 legit revenue restricted
anywhere - they are ALERTS, not blocks. Control fires 5/5 at flagged 0.959 vs a
worst legit 0.067. Does NOT license 'never false-alarms' - m4 still fires on 1
seed in 5.
The STREAMING detector (what the product runs, and what 25/25 measures) catches
5/5 in all seeds; the hourly EWMA path misses the ATO on seed 7. Report both.
TTD beats flag-counter 4/5; LOSES on fraud ring (68m vs 59m). Static volume is
fastest on the device farm (a farm IS a volume event) but misses the other four
AND fires on ALL THREE legitimate spikes - that row is the whole argument.
Ablation: basics 0.629 -> +velocity 0.611 -> +entity/graph 0.825. Paired
bootstrap CIs (2,000 resamples): velocity -0.0181 [-0.0453, +0.0079] NOT
distinguishable from zero; entity/graph +0.2138 [+0.1748, +0.2517] SIGNIFICANT.
!! VELOCITY HAS BEEN MEASURED THREE TIMES AND GIVEN THREE ANSWERS (mild
regression -> +0.0317 significant -> -0.0181 not significant). Do not quote a
sign. The honest statement is that the step is small enough that the dataset
decides it. Report the interval.
Diagnostics: profile pair alone 0.5970 | entity SHARING alone 0.3950 (was
0.7877 before the hard negatives, with precision 1.000 and INR 0 blocked; it is
now precision 0.654 and INR 187.7K blocked) | full minus the pair 0.7663.
!! ENTITY SHARING IS NECESSARY, NOT SUFFICIENT. Its collapse under the hard
negatives is the quantitative twin of entry 16, where the agent reasoned only
from entity sharing and called a real INR 5.5L takeover legitimate.
P1b fusion now changes 433/14,160 decisions (3.1%). The trajectory IS the
argument: 0/13,987 on the leaky generator, 3/13,782 after the generator fix,
433/14,160 after the hard negatives. The value of corroboration scales with how
much genuine ambiguity the data holds. We kept it when it changed nothing and
said so; report the change on the same terms. Kept for
architecture (auditability, reachable fail-safe, headroom for a weaker model),
NOT for metrics. Say so plainly; do not re-frame it as a win.
P2 agent eval (LIVE Claude Haiku 4.5, DE-LABELLED data, run D = FINAL):
!! SMALL SAMPLE. n=13. The 95% CI on 8/13 is roughly +/-25 points, so 8/13 and
5/13 are NOT distinguishable at this sample size. EVERY conclusion drawn from
the agent eval - including the run A->B->C->D progression, whose deltas are
noisier than the table implies - is a small-sample result and must say so.
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
Docs-vs-artifacts: `python -m src.audit.verify_submission` (exits non-zero if
a documented headline disagrees with the artifact that produced it; also
checks engine.py's cutoffs against threshold_sweep_decision.json. Run it
BEFORE claiming any number is current - entry 31)
Config 4 NPV: `python -m src.policy.config4_npv` (the REJECTED training-data
configuration, measured over 5 seeds rather than argued about - it is worse on
both NPV and attack detection, and stays reachable via HIST_CORPORATE_BUYER)
Queue order: `python -m src.policy.queue_order` (what the review queue's sort
key is worth; arrival order cost 45%->5% of fraud value at 50 cases worked, and
ranking by risk_score measured WORSE than arrival - it is not rupees)
Ablation CIs: `python -m src.models.ablation_ci` (paired bootstrap, 2,000
resamples, so the table's deltas carry intervals rather than bare points)
Generator sensitivity: `python -m src.models.aged_share_sensitivity` (is the
aged_share choice load-bearing? no - and there is a control proving the
sweep reproduces the leak when it should)
Topology generalisation: `python -m src.models.topology_generalisation` (does
detection survive a change of attack SHAPE? each attack rebuilt with a different
entity graph, volume/window/amounts/ageing matched, frozen pipeline, 5 seeds,
known shapes as positive control. known 22/25 vs unseen 21/25 detected, but mean
flagged rate 0.815 -> 0.709 - the MERCHANT layer holds while the TRANSACTION
scorer degrades. ATO control is weak (2/5 known) and both its rows are weak
evidence; say so. See entry 34)
Sharing sensitivity: `python -m src.models.sharing_sensitivity` (where does
LEGITIMATE entity sharing start firing us? swept 0-100% over 5 seeds through the
frozen pipeline, with a real device farm as a POSITIVE CONTROL - 3/60 legit
merchant-seeds fire, all at >=80% sharing, INR 0 restricted; control 5/5 at
0.959. The first run was WRONG - see entry 32)
Merchant-data testability: `python -m src.models.merchant_data_check` (now
checks BOTH Sparkov and IBM TabFormer under one criterion, and tests two things:
CONCENTRATION (any window at an attack-like fraud rate?) and REACHABILITY (can
any merchant pack 30 txns into the detector's 6h span guard?). Sparkov fails
both, TabFormer passes reachability and fails concentration. Neither is
testable. The first TabFormer run said TESTABLE and was a pandas datetime64[us]
bug - see entry 33. Old note follows; it asks
whether a public set can evaluate the MERCHANT layer at all; Sparkov has the
merchant column and still cannot - 0 evaluable merchant-hour windows, max fraud
rate 0.273 vs our 0.70-0.93. Run BEFORE modelling, never after)
Real data: `python -m src.models.real_data_check` (ULB/IEEE via Kaggle; data/ is
gitignored and NEVER committed - competition terms, publish metrics not data)
Demo: `python run_demo.py` (one command; ~60s replay at default 250 txn/s)
Tests: `python -m pytest tests/ -q` (117 pass, no network needed)
REAL-DATA, same recipe, no tuning: ULB creditcardfraud PR-AUC 0.731 (vs 0.0017
random, ROC 0.974) | IEEE-CIS PR-AUC 0.460 (vs 0.0350 random, ROC 0.888).
Methodology transfers; the 0.898 does NOT. NEGATIVE RESULT (entry 24): our
entity fan-out features move IEEE-CIS -0.0053 and score 0.040 alone. Entity/
graph is UNPROVEN at our hands on real data. Never claim otherwise. The
merchant-level layer is wholly UNVALIDATED on real data. ULB and IEEE-CIS have
no merchant column; Sparkov HAS one and still cannot test the layer (0 evaluable
merchant-hour windows, max merchant-day fraud rate 0.273 vs our 0.70-0.93) -
public card-fraud data models stolen CARDS spent across merchants, not merchants
under attack. Different loss class. Do not claim we simply did not look.

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
   a collapse. THIRD ACT (entry 26): after the generator fix the same step is
   +0.0317 with a 95% bootstrap CI of [+0.0189, +0.0445] — positive and
   significant. Published, retracted, then revised again in the other
   direction. Do not quote the "velocity is neutral" reading; it belonged to
   the leaky generator. A supporting top-200 diagnostic ALSO dissolved: the top-200
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

16. THE LIVE AGENT RUN THAT DEMONSTRATED THE ARCHITECTURE (one incident
   demonstrates the separation works; it does not PROVE it). The agent labelled m2 (a real ₹5.5L account-takeover) as
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
   CLOSED (see entry 26). Regenerating attack accounts with realistic ages was
   first published as a measured limitation rather than rushed under deadline;
   it was then actually done, and the fix cost is reported rather than hidden.
   THIS ENTRY IS THE ONLY PLACE THE PRE-FIX NUMBERS APPEAR. 0.9344 full /
   0.9328 proxy-pair are PRE-FIX figures and are labelled as such everywhere,
   exactly as run A (9/10) is always labelled pre-de-labelling. They are the
   measurement of what the leak was worth, not a competing headline.
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
   (423x lift); our simulator scores 0.898 on the same recipe (0.934 pre-fix,
   see entry 26), and that gap is
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
   how we know the fix is safe. (Those are the PRE-entry-26 figures, quoted
   here because bit-identity is a claim about THAT moment; the current
   generator scores 0.8981.)
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

24. AN INVALID TEST OF THE CENTRAL CLAIM - AND WHY WE DO NOT REPORT IT AS
   EVIDENCE EITHER WAY. IEEE-CIS (Vesta,
   590,540 real e-commerce txns, 3.5% fraud) is the one public set here WITH
   entity columns, so it is the direct test of the thing the product is built
   on. Built shared-entity fan-out on it the same way builder.py does -
   strictly incremental, emitted from prior rows only, repo's own UnionFind.
   RESULT: adding our entity features moves PR-AUC 0.4604 -> 0.4551. That is
   MINUS 0.0053. Alone they score 0.0400 against a 0.0350 random baseline -
   essentially nothing. Entity/graph correlation, as WE compute it, does not
   show up on this real dataset.
   THE HONEST CONFOUND (state it, do not hide behind it): IEEE-CIS DeviceInfo
   is a device TYPE, not a fingerprint - "Windows" alone is 40.2% of rows,
   "iOS Device" another 16.7%. Our simulator's device_id is a real fingerprint
   (top 3 values = 0.53% of rows). So device_card_count there means "how many
   cards ever used Windows", which is not the quantity we reason about. IEEE
   also has NO account id, so "one device across fifty accounts" is literally
   inexpressible; we used card1 as an account proxy, which is crude.
   THE PART THAT SURVIVES: tier 2 is the biggest jump in the table, +0.352
   PR-AUC, and Vesta's C1-C14 ARE entity-counting features - their own
   description is "counting, such as how many addresses are found to be
   associated with the payment card". So entity fan-out counting WITH REAL
   ENTITY RESOLUTION is the single largest contributor to real-data
   performance. The concept is validated. Our implementation on this dataset's
   proxies is what adds nothing.
   VERDICT (revised after outside review - the first version of this entry
   over-stated it). Reporting a null from a test that provably cannot measure
   the thing is NOT honesty; it is presenting invalid evidence in the
   pessimistic direction. We do not claim IEEE-CIS validates the graph, and we
   do not claim it refutes it. We publish the rows and the caveat together -
   real_data_check.py prints both - and we say plainly that the merchant-level
   layer is evaluated on controlled scenarios because the two public datasets
   we EVALUATED (ULB, IEEE-CIS) do not expose the persistent entity
   relationships needed to test it directly. Scope that claim to the datasets
   we actually ran - we did not survey the field, and "no public dataset can"
   is an unbounded claim we cannot support.
   The earlier framing ("our central claim came back negative") is RETRACTED as
   over-stated. It was written under an explicit instruction not to protect the
   author's ego and overshot in the other direction - which is its own kind of
   inaccuracy. Symmetry rule going forward: we would not accept a POSITIVE
   result from this experiment either.
   Do NOT put IEEE-CIS in the demo video. Zero seconds. It tests the
   transaction model, not the product.
   ALSO: this run CORRECTED an over-claim we had made one commit earlier. On
   ULB alone we wrote "our amount model is backwards" (fraud 0.42x legit vs our
   1.37x). IEEE-CIS e-commerce fraud is 1.10x - much closer to ours. The true
   statement is narrower: fraud amount is FRAUD-TYPE dependent, we model only
   the expensive case, and ULB's card-testing case inverts it. Two datasets
   caught an over-claim that one dataset produced. Same lesson as entry 6, and
   we nearly published it again.

## Style
Plain, direct comments explaining WHY. Small modules. No cleverness that costs
explainability — every component must be defensible to a judge in one sentence.

25. A NULL FIELD SILENTLY DISABLED THE GUARD WRITTEN TO PREVENT THE EXACT BUG
   IT THEN CAUSED. The dashboard auto-selects an opening merchant, and had a
   deliberate guard: prefer a spiking merchant whose entity graph has something
   to draw, because account takeover shares no entities by construction and
   opening the demo on an empty graph is a terrible first frame. The guard
   filtered on `top_cause` — which is sourced ONLY from the LLM investigation.
   The hosted deployment runs `--no-agent`, so `top_cause` is null on every
   merchant, `.find()` matched nothing, and the fallback selected `spiking[0]`
   = m2, the account takeover. The live demo opened on the empty graph the
   guard exists to avoid, under a panel titled "entity network", every time.
   Same null also meant five cards read "UNDER ATTACK" with nothing on screen
   saying what the attack was — the `cause` row simply never rendered.
   FIX: `MerchantState.signature()`, computed server-side by counting what is
   already in `entities` — "1 device shared by 50 accounts", "183 different
   cards, none used twice", or, when nothing is shared AND the detector fired,
   "the abuse is inside the accounts, not between them". Always present,
   independent of the LLM, and pytest-locked against ever carrying `scenario`.
   Deliberately NOT a diagnosis: it reports what is shared, never what kind of
   attack it is. Two directional facts (sharing, card novelty) are stated in
   WORDS with their direction, which is failure-log 22's lesson applied at the
   point of display rather than left for a reader to infer.
   ALSO FOUND while fixing it: `action_mix` was in the API payload and shown
   nowhere, so the board never said what the system DID; and `now/baseline`
   read "0.0%" on all 12 cards after the replay ends, which under a red UNDER
   ATTACK badge reads as a contradiction rather than as "the burst is over".
   LESSON: the same shape as entry 11. A guard that depends on an optional
   field fails OPEN and silently — it does not error, it just stops guarding,
   and the failure looks like an unrelated design flaw. Assert on the guard's
   observable outcome, not on the field it happens to read.





30. THE EXPERIMENT WE DID NOT RUN, AND WHY THAT IS THE RESULT.
   Our standing limitation was "the merchant-level layer is validated only on
   our own data, because the public sets have no merchant column". That is an
   argument from ABSENCE and invites one reply: then go and find one that does.
   So we did. kartik2112/fraud-detection (Sparkov) has a merchant column, card
   ids and timestamps - everything the spike detector needs. Downloaded it and
   ran a TESTABILITY CHECK BEFORE writing any modelling code
   (src/models/merchant_data_check.py).
   IT CANNOT TEST THE LAYER, and not for want of data:
     evaluable merchant-HOUR windows (>=10 txns)      0
     highest fraud rate of any merchant-DAY           0.273
     windows at an attack-like rate (>=0.70)          0
     our own attacks, 30-txn window                   0.70 - 0.93
   Worst-affected merchant: 49 fraud txns spread over 538 DAYS, max 2 per day.
   Nothing in it is a burst.
   THE STRUCTURAL TELL: 7,506 fraud txns = 9.8 per compromised CARD but only
   11.1 per merchant across 693 merchants, of which just 14 escape entirely.
   Public card-fraud data models STOLEN CARDS SPENT ACROSS MANY MERCHANTS. This
   product models MERCHANTS UNDER COORDINATED ATTACK. Different loss class. A
   merchant column is necessary to test our layer and nowhere near sufficient -
   the burst structure has to be there too, and in three public datasets now
   examined it is not.
   WHY THIS COUNTS AS A RESULT RATHER THAN A DEAD END: running the detector
   anyway would have produced a null that measures the DATASET, not the
   detector - exactly the invalid-evidence trap entry 24 documents for
   IEEE-CIS. We nearly repeated the mistake we had already written up; the only
   thing that stopped it was checking testability first. The limitation is now
   narrower and evidenced: it is not that we did not look, it is that this loss
   class is not represented in reachable public data, and closing it needs a
   PSP's own traffic rather than another Kaggle download.
   LESSON: "we could not find data" and "we looked, and here is the measurement
   showing why the data that exists cannot answer it" are the same conclusion
   with very different evidential weight. The second costs half an hour.

29. WE BUILT A LEGITIMATE MERCHANT DESIGNED TO BREAK US, AND IT DID.
   For most of this project "0 false alarms" rested on ONE negative - the flash
   sale - and re-reading its generator shows why that was thin: every account
   gets its own device, IP and instrument by construction. It only ever tested
   "does raw volume fire us" and never touched the entity layer; m11's entity
   graph renders EMPTY. The first question a real risk person asks - "what
   about an office where 50 people share an IP, or a counter where everyone
   pays through one terminal?" - had no answer.
   Built two legitimate merchants that carry an attack's entity signature, both
   from REAL world customers (aged accounts, own amounts) so that entity
   sharing is the ONLY difference from ordinary traffic:
     s7 corporate buyer  40 accounts, 1 office IP, 2 company cards  (~ip cluster)
     s8 shared kiosk     25 accounts, 1 device, own cards, bursty   (~device farm)
   IT WORKED. The legitimate kiosk scored mean p=0.972 (97.2% flagged) against
   a real device farm's 0.985. Peak spike z was 5.34 on the honest merchant
   against 5.33 on a real IP-cluster ATTACK - the legitimate one produced a
   HIGHER z than the fraud. It was restricted, INR 1,24,285 of legitimate
   revenue was impacted, PR-AUC fell to 0.695. The corporate buyer passed
   (z 1.10): shared entities were not the problem, a shared DEVICE was.
   ROOT CAUSE was a TRAINING-DISTRIBUTION gap, not a missing feature. The
   separating signal already existed - a farm has 50 accounts on ~8 shared
   instruments, a kiosk has 25 accounts on 25 of their own - but training
   contained shared-device FRAUD and no shared-device HONEST traffic, so the
   model learned "shared device = fraud" because in this world it always was.
   Same shape as entry 1 (attacks only in test) and entry 6 (no attack in cal).
   FIX: a legitimate kiosk in the TRAINING period, different merchant, entity
   ids pytest-verified disjoint from the held-out one. Kiosk 0.972 -> 0.002,
   peak z 5.34 -> 0.00, while device farm stayed 0.985 -> 0.982 and all five
   attacks still fire. Precision IMPROVED to 0.996 and legit INR blocked fell
   to 2,575.
   NOT FULLY FIXED. The corporate buyer still false-alarms on 1 seed in 5 -
   measured across ALL five, not spot-checked. 1 in 35 non-attack
   merchant-windows is 2.9%, 95% CI [0.1%, 14.9%]; never quote the point alone.
   THE REJECTION WAS ARGUED BEFORE IT WAS MEASURED, and that was our error.
   The symmetric fix (teach it a legitimate shared IP too) removes the false
   alarm and on SEED 7 raises NPV to INR 9.47L vs our 8.13L - so this entry
   used to say we were OVERRIDING our own declared cost rule on failure-severity
   grounds. An outside reviewer pointed out that a win on one seed against a
   failure on another is not a comparison. Correct. Measured across all five
   (src/policy/config4_npv.py):
     shipped   mean NPV INR 939,179   25/25 attacks
     rejected  mean NPV INR 874,988   24/25 attacks
   The advantage was a SEED-7 ARTIFACT. Averaged, the rejected configuration is
   INR 64,192 WORSE and also loses an attack. There is no money-vs-safety trade;
   it is worse on both axes. We only believed otherwise because we compared one
   seed in a project that keeps a five-seed harness to prevent exactly that.
   We still cannot explain WHY corporate-buyer examples destabilise card
   testing, so the false alarm stays rather than be fixed by a change we do not
   understand. The rejected config is left REACHABLE (HIST_CORPORATE_BUYER) so
   it can be re-measured.
   FOUR CONFIGURATIONS WERE TRIED. That number goes next to the result.
   LESSON (second one in this entry): we made a design decision on a
   single-seed comparison while owning a five-seed harness. Reach for the
   harness you already built before reasoning about which failure you prefer.
   WHAT ELSE MOVED: entity-sharing-alone collapsed 0.7877 -> 0.3950 and now
   blocks INR 187.7K of legitimate value - which is the point, and makes the
   honest claim "necessary, not sufficient". Fusion went from a no-op to
   changing 3.1% of decisions. Review load rose 60%. Velocity flipped sign a
   third time. Every one of those is downstream of the same cause: the dataset
   finally contains genuine ambiguity.
   LESSON: a negative case that cannot fail you is not a test. We shipped
   "0 false alarms" for weeks against a merchant that shares no entities, in a
   system whose entire thesis is entity correlation.

28. THE REVIEW QUEUE WAS SORTED BY ARRIVAL TIME.
   Found by reviewing a competitor who caught the same class of bug in their
   own queue (they sorted by probability while everything else was denominated
   in money). Ours was worse: `snapshot_queue` returned cases in insert order
   and the dashboard reversed them, so an analyst worked the MOST RECENT cases.
   No relationship to money at all.
   A queue only matters if the analyst runs out of time before it runs out of
   cases - ours yields 948 on a 14,160-txn slice, so it always does. The ORDER
   is therefore a policy decision worth as much as the threshold, and we had
   never made one.
   MEASURED (src/policy/queue_order.py), share of the queue's INR 744,217 of
   fraud value put in front of an analyst:
     cases worked      arrival    by risk    by expected loss
     50                   5.7%       5.4%              47.3%
     240                 69.2%      41.6%              82.0%
   THE OBVIOUS FIX WOULD HAVE MADE IT WORSE. Ranking by risk_score is BELOW
   arrival order at every capacity. Fusion's risk_score is an escalation scale,
   not a probability: high scores cluster on cheap card-testing txns while the
   expensive account-takeover cases score lower. Multiplying rupees by it is a
   category error, so expected loss uses the CALIBRATED p - which is why
   ReviewCase carries p_fraud separately from risk_score.
   FIXED: ranked by expected loss, the wire cap now takes the TOP 200 by rank
   rather than the last 200 by arrival (capping by arrival would have hidden
   exactly the cases the ranking exists to surface), `ordering` is returned on
   the wire, and two tests lock it.
   CAVEAT WE REPORT RATHER THAN HIDE: arrival order looks respectable at 240+
   cases because this slice's biggest attacks fall near its end, so "newest
   first" accidentally surfaces them. That would not survive production. The
   low-capacity rows are the ones that generalise.
   LESSON: we costed the threshold, the cutoffs, the FN price and the review
   rate, and never once asked what ORDER the resulting queue should be worked
   in. An operational policy sitting in a list comprehension is still a policy.

27. THE ONE CALLER-SUPPLIED STRING THAT REACHED THE MODEL'S PROMPT.
   Prompted by reviewing a competitor that documents a prompt-injection
   defence we did not have. Audited our own surface rather than assuming it
   was fine, and found one real hole: `merchant_id` is taken straight off the
   URL path by `POST /api/merchants/{merchant_id}/investigate` and
   interpolated into the agent's opening message. Nothing validated it. An
   authenticated caller could have put arbitrary text in front of the model.
   WHAT WAS ALREADY TRUE, and why the impact was bounded rather than nil:
   every OTHER string the agent sees is an entity id from oid(), which is
   structurally `kind_<8 hex>` and cannot carry a payload; and
   validate_recommendation() degrades any unknown action to REVIEW, so even a
   successful injection could not have produced an unauthorised action - the
   LLM is not in the decision path. Worst case was a wasted call and a garbage
   report, not a bad decision.
   FIXED anyway: `InvestigationContext.known_merchant()` checks the id against
   real data; `investigate()` refuses an unknown id BEFORE building any prompt
   and returns a degraded REVIEW report; the route answers 404. Guarded in two
   places on purpose, so it does not depend on the caller. Three tests: an
   injection-shaped merchant id never reaches graph construction, known_merchant
   rejects near-misses, and oid() stays opaque under hostile keys including
   NUL, newlines and a 5,000-character string.
   SCOPE, stated because it matters more than the fix: this is a property of
   OUR generator. On real production traffic, device fingerprints and
   instrument ids come from the payment stream, are not oid() output, and would
   need sanitising at ingestion. The test says so in its docstring rather than
   leaving a reader to assume the guarantee travels.
   LESSON: "we probably do not have that problem" is not an audit. The surface
   took ten minutes to check and there was something in it.

26. WE FIXED THE GENERATOR, AND THE FIX FOUND THREE MORE BUGS.
   Entry 21 ended at "disclosed but not fixed", which is the weakest ending an
   audit finding can have. Closed it. Two generator edits, both grounded in
   real-world fact rather than in what would make the number look good:
   (a) attack account ages now come from a MIXTURE - a share aged exactly like
   the legitimate population, because REAL FRAUDSTERS BUY AGED ACCOUNTS, the
   case we never generated; the rest genuinely new, because throwaway guest
   checkout is also real. Shares fixed on stated rationale BEFORE measuring
   and not tuned after: rings 0.8, farms/clusters 0.6, card testing 0.5.
   (b) ambient fraud amounts are no longer uniformly expensive, because ULB's
   real card fraud runs 0.42x the median legitimate amount while IEEE-CIS
   e-commerce fraud runs 1.10x (entries 23/24): fraud amount is FRAUD-TYPE
   dependent and we had modelled only the expensive case.
   RESULT - the leak is gone: the two proxy features fall 0.9328 -> 0.5997
   while the full set falls only 0.9344 -> 0.8981. The gap goes 0.0016 ->
   0.2984. The headline dropped 0.036; the cheat dropped 0.333.
   AND THE RETRACTED CLAIM BECOMES TRUE. Entry 21 forced us to retract
   "entity/graph is where the system comes from". On data that does not hand
   out the answer, component_size is now the top feature by BOTH single-feature
   PR-AUC (0.711) and model importance (0.406). The claim is RE-ESTABLISHED on
   evidence rather than quietly restored.
   IS OUR NEW CHOICE JUST AS ARBITRARY? Measured, not argued
   (src/models/aged_share_sensitivity.py). Across aged_share 0.3-0.9 the proxy
   pair never comes within 0.02 of the full set (worst 0.7179 at 0.3) and
   full-set PR-AUC moves only 0.0056. ANY realistic ageing removes the leak;
   our specific value is not what did it. The old value (100% newborn) was a
   judgement call too - just an implicit, never-examined one, load-bearing
   enough to fake the headline.
   THE SWEEP CAUGHT ITSELF FIRST. Its initial control reverted only the ages,
   not the amounts, so it did NOT reproduce the leak - and the derived verdict
   flagged "this sweep may not be measuring what we think it is" instead of
   reporting a clean pass. Fixed by making the control revert BOTH edits, which
   then reproduced the leak (gap +0.0172) as it should.
   THREE LATENT BUGS SURFACED BY THE RE-RUN:
   (i) threshold_sweep.py CRASHED. Its per-parameter rule (entry 10) assumed
   each cutoff can move alone, but the grid requires step_up < restrict, so the
   new best pair's one-at-a-time point (55, 60) does not exist -> IndexError.
   Now reports "NOT INDEPENDENTLY EVALUABLE" and keeps the conservative
   default. A move we cannot measure in isolation is not evidence FOR making
   it. Adopted (85, 25).
   (ii)+(iii) leakage_probe.py AND ablation.py both HARDCODED their failing
   conclusions. After the fix they printed "two features reproduce the
   headline" directly above their own output showing they don't. Both verdicts
   are now DERIVED from the numbers. AN AUDIT TOOL THAT CANNOT REPORT A PASS IS
   NOT AN AUDIT TOOL - and we had two of them.
   WHAT IT COST, reported not buried: PR-AUC 0.945 -> 0.910 over 5 seeds, NPV
   INR 10.57L -> 7.95L, legit INR wrongly blocked 5,901 -> 21,728 (0.21% of
   legitimate value processed). Those two FP figures measure different DATASET
   DIFFICULTY, not different system quality: precision 0.994 was purchasable on
   data where two columns partition the label space. We did NOT retune the
   threshold to recover the FP number - (85, 25) is what the validation sweep
   adopted, and moving it because we dislike the result would be optimising
   something other than expected cost. We have refused that move four times now
   (expected_action labels, fusion re-weighting, the frozen agent design, this).
   ALSO CHANGED: TTD now WINS on IP cluster (24m vs 28m, previously our one
   loss) and LOSES on fraud ring (59m vs 48m). Still 4/5. And the two detector
   paths no longer agree - streaming catches 5/5 in all seeds, the hourly EWMA
   path misses the account takeover on seed 7. Both reported.
   FOURTH THING THE RE-RUN CHANGED, found later: the ablation's velocity step
   flipped SIGN. Under the leaky generator it was a mild regression and the
   README called velocity "roughly neutral"; on fixed data it is +0.0317 with a
   paired-bootstrap 95% CI of [+0.0189, +0.0445] - positive and significant.
   When two profile features were quietly encoding the label, every honest
   feature looked redundant next to them. We shipped the corrected tables
   without following through to the PROSE underneath, so for a while the README
   had a table saying +0.032 and a paragraph forty lines below saying
   "neutral-to-slightly-negative". Caught and fixed. Lesson: rewriting the
   numbers is half the job; the sentences that INTERPRET them are the other
   half, and they fail silently because nothing type-checks English.
      LESSON: "disclosed but not fixed" is a resting place, not a destination. The
   fix cost 0.036 of headline and bought back the ability to state what the
   number means - plus three bugs that only a re-run could expose.


31. WE SHIPPED A POLICY CONSTANT OUR OWN RULE DOES NOT AUTHORISE, AND AN
   OUTSIDE AUDIT FOUND IT BY READING OUR ARTIFACTS AGAINST OUR PROSE.
   Handed CLAUDE.md + SUBMISSION.md + the machine-generated result JSONs to an
   external model with a prompt written to make it hostile. It came back with
   six documentation findings. FIVE WERE CORRECT. Chasing the sixth - a
   complaint that CLAUDE.md said policy cutoffs (85, 25) in one place and
   (85, 20) in another - found something the auditor could not see, because it
   only had the documents and not the code:
     src/policy/engine.py          STEP_UP_CUT = 20.0
     threshold_sweep, re-run       ADOPTED (85, 25)
   The two disagreed. Re-running the sweep on current data settles which is
   right, and it is worse than a mismatch:
     step_up 20  validation NPV 78,714  lift +1.73%  legit impacted INR 1,158
     step_up 25  validation NPV 79,485  lift +2.73%  legit impacted INR   388
   Our pre-declared adopt margin is 2%. 20 scores +1.73% - BELOW OUR OWN
   MARGIN. Under our declared rule the shipped value would not be adopted at
   all. It was correct once: the sweep chose 20 on the PRE-hard-negative data
   (+15.82% there), commit 3f2bef5. Then failure-log 29 changed the data and
   nobody re-ran the sweep. The constant was one dataset generation stale and
   we never noticed because nothing checks a constant against its own
   derivation.
   FIXED: STEP_UP_CUT = 25.0, and every downstream number re-measured rather
   than edited. It is BETTER ON BOTH AXES - NPV 8.11L -> 8.13L, legit revenue
   impacted 4,266 -> 452 (-89%), protected-per-rupee 16.7x -> 18.0x, step_up
   actions 106 -> 25. We did not go looking for a number that improved; we
   applied the rule and this is what it gave. Had it come out worse we would
   have shipped it worse, which is the whole point of fixing the rule in
   advance.
   THE FIVE DOCUMENTATION FINDINGS, all real, all ours:
     (a) SUBMISSION.md headline said "False alarms: 0 in every world" while
         line 51 of the same file said "1 false alarm in 35". The retracted
         number was still the FIRST thing a judge read. This is the single
         worst thing in this entry: our differentiator is honest metrics and
         our headline carried a figure we had publicly retracted.
     (b) Calibration published as Brier 0.0053 / ECE 0.0033; current values
         0.0123 / 0.00736. The stale pair is the flattering one.
     (c) "Fraud INR prevented 8.63L" is fraud_exposure_prevented_inr, not
         fraud_inr_prevented (7.86L). Two different quantities, one label.
     (d) The FP figure quoted the classifier's INR 2,575 next to policy-path
         economics, mixing two measurement paths in one table. README already
         explained the distinction; SUBMISSION did not.
     (e) Model-selection figures (.9308/.9265/.9249, 0.27/1.64/7.94s) were
         pre-hard-negative. Current: .7702/.7625/.7573, 0.40/1.85/8.82s.
   PLUS ONE THE AUDIT MISSED: SUBMISSION said "5 independent simulated worlds"
   - a phrase README explicitly corrects, because they are five SEEDS of one
   generator. We had the correction written down and the uncorrected claim
   still shipped in the summary doc a judge reads first.
   ROOT CAUSE, and it is not carelessness: EVERY ONE of these is the same
   defect as failure-log 26's fourth finding - we re-measure, we update the
   tables, and the numbers that live somewhere else go stale silently because
   nothing type-checks English. We wrote that lesson down and then reproduced
   it across three documents and one policy constant. Writing a lesson in the
   failure log does not install it.
   WHAT WOULD ACTUALLY PREVENT IT: a check that reads the artifact JSONs and
   greps the docs for contradicting figures, run like a test.
   NOW CLOSED - src/audit/verify_submission.py, `python -m
   src.audit.verify_submission`, exits non-zero when a documented headline
   disagrees with the artifact that produced it. Three check kinds: CLAIMS (a
   number must equal its artifact field), RETIRED (a retracted phrasing must
   not appear unless the line marks itself history, and every exemption is
   PRINTED), and CODE (a shipped constant must equal the decision artifact that
   adopted it - this entry's own bug, which no documentation check could see).
   Deliberately a REGISTRY, not a number scraper: this repo carries ~22
   superseded figures on purpose, and a scraper would flag them all and create
   pressure to delete our own history.
   VALIDATED AGAINST THIS REPOSITORY'S PAST, because a checker that passes
   today proves nothing. Pointed at the docs from 5ae3a4a - before this entry's
   corrections - it flags 10 problems including the retracted "0 in every
   world" headline and the Brier 0.0053 an external audit found by hand.
   AND IT IMMEDIATELY FOUND ONE MORE, in a table BOTH audits missed: README's
   config-3-vs-4 comparison still read INR 9,43,276 / 8,69,852 while the
   artifact said 939,179 / 874,988 - with the corrected INR 64,192 gap stated
   in the paragraph directly beneath it. Sixth instance of this defect.
   THE CHECKER'S OWN BUGS ARE THE INTERESTING PART. Five, during construction:
   a precision/recall pattern that could not cross a table pipe; a Brier
   pattern that matched the deliberately-listed uncalibrated row; a PR-AUC
   pattern matching one of five phrasings; an exemption ordering that skipped
   CURRENT figures merely because the line also mentioned history; and worst,
   the "0 false alarms" pattern required the words adjacent when the real text
   splits them across table cells - so the check written for the single most
   important retracted claim silently matched NOTHING. Only running it against
   the real pre-fix document exposed that. Hence the "claim not found" report:
   a pattern that rots makes the checker blind, and blindness must fail loudly
   rather than pass quietly.
   LESSON: an audit that only reads your documents can still find your code
   bugs, because a documentation contradiction is often a code contradiction
   that has surfaced. Five doc findings were cheap to fix; chasing the sixth
   found a shipped policy value that our own decision rule forbids.

32. THE SWEEP'S FIRST ANSWER WAS ABOUT THE SWEEP, NOT THE SYSTEM.
   The audit in entry 31 left one criticism standing that no number of ours
   could answer: our three legitimate merchants are three POINTS, the detector's
   whole thesis is that entity sharing is suspicious, and we had never measured
   what happens BETWEEN "shares nothing" and "shares everything". If the
   false-positive rate has a cliff in the middle, "0 or 1 false alarms" is a
   fact about where our three examples happen to sit.
   Built src/models/sharing_sensitivity.py: legitimate merchants whose ONLY
   varying property is the share of traffic through one device/IP, swept
   0-100%, run through the FROZEN pipeline (shipped model, calibration and
   cutoffs, no retraining), with a real device farm at the same volume and
   window as a POSITIVE CONTROL.
   FIRST RUN SAID WE FALSE-ALARM ON MERCHANTS THAT SHARE NOTHING. Flagged rate
   0.144 at 0% sharing, falling to 0.011 at 100% - a backwards curve, and two
   spikes at the LOW end. Read literally it says the detector fires on ordinary
   traffic and calms down as coordination increases, which is nonsense on its
   face and would have been a spectacular thing to publish.
   IT WAS OUR HARNESS. The swept merchants drew ISP pools from `cid % 400`
   while World.__post_init__ uses 1500 - so their pools were 3.75x DENSER than
   the world the model was trained on, inflating ip_account_count and the
   entity-graph signal on every swept merchant, worst where nothing else was
   shared. One character of parameterisation, and the whole curve inverted.
   Matched the pool count to the generator and re-ran: nothing fires below 80%
   sharing at all.
   WHAT IT ACTUALLY SAYS, across 5 seeds and 60 legitimate merchant-seeds:
     sharing 0/20/40/60%   0 of 40 merchant-seeds fire
     sharing 80%           1 of 10
     sharing 100%          2 of 10
     legit INR restricted  0, everywhere in the sweep
     positive control      5/5 seeds, mean flagged 0.959 vs worst legit 0.067
   The rate does not climb through the middle; it is flat at zero until traffic
   is overwhelmingly funnelled through one entity. The three fires are ALERTS,
   not blocks - the merchants entered spike state and none of their
   transactions were restricted. And the reason the middle is safe is the hard
   negative from entry 29: training now contains an HONEST shared-device
   merchant, so heavy sharing reads as the kiosk it was taught.
   WHAT IT DOES NOT SETTLE, said before anyone else says it: our generator's
   topology, one event per merchant, 5 seeds. It retires the narrow objection
   (the three points were not hiding a cliff). It does not license "never
   false-alarms" - m4 still fires on 1 seed in 5.
   WHY THE CONTROL IS THE MOST IMPORTANT ROW: without it, a clean sweep is
   indistinguishable from a broken harness, and we would have reported a null
   that measures our own code - the exact trap entry 24 documents for IEEE-CIS
   and entry 30 avoids for Sparkov. The verdict is VOID if the control fails,
   and that branch is written and reachable rather than assumed away.
   LESSON: we caught this only because the first result was absurd in a
   DIRECTION we could reason about. A harness bug that produced a plausible
   curve would have shipped. The defence is not vigilance, it is the control
   row - build the thing that must fire, then believe the things that do not.

33. THE HARNESS BUG THAT ARRIVED DISGUISED AS GOOD NEWS.
   Both audits' surviving objection is that the merchant layer is validated
   only on a world we authored. So we went looking properly this time rather
   than asserting: Amazon Science's fraud-dataset-benchmark - the field's
   curated standard - carries NINE datasets, and exactly ONE (Sparkov) has a
   merchant identifier at all. Outside it, IBM TabFormer
   (ealtman2019/credit-card-transactions) has 24M transactions, 100,343
   merchants and a real merchant id. It is IBM-generated rather than real, but
   it is not OURS, which is the half of "self-authored synthetic world" it can
   actually remove. Downloaded it and ran the testability check first.
   IT CAME BACK TESTABLE. 204,006 evaluable merchant-hour windows, 77 of them
   at an attack-like fraud rate, and 5,074 merchants able to satisfy the
   detector's 6h span guard - against Sparkov's 0/0/0. That is precisely the
   result we wanted: a public dataset that can finally evaluate the merchant
   layer, found days before a deadline.
   IT WAS WRONG. pandas 3 returns datetime64[us] from to_datetime(dict(...)),
   not the [ns] we assumed, so `ts.astype("int64") // 10**9` divided
   MICROSECONDS by a billion and compressed 29 years of data into 11 days.
   With time squashed 1000x every merchant looks like a burst, which is exactly
   why the span guard appeared satisfiable and the windows appeared dense.
   CORRECTED, the answer inverts and gets more interesting than Sparkov's:
     span                        11 days -> 10,650 days
     merchants clearing 6h guard   5,074 -> 29 of 20,702
     max merchant-day fraud rate   1.000 -> 0.417
     windows at attack-like rate      77 -> 0
   TabFormer FAILS on concentration while PASSING reachability - the opposite
   shape to Sparkov, which fails both. So it is not a speed problem and not a
   size problem: 97,512 of its 100,343 merchants carry ZERO fraud and the rest
   average ~10 fraud txns spread across 29 years. Fraud per compromised CARD
   10.81 vs per merchant 10.51. Stolen cards spent across many merchants, once
   more, from a completely different generator.
   WHAT CAUGHT IT, and this is the uncomfortable part: NEITHER CRITERION DID.
   Both happily consumed the compressed timestamps and returned TESTABLE. What
   caught it was an internal consistency check with nothing to do with the
   verdict - a dataset documented as spanning 1991-2020 reporting 11 days.
   Sparkov acted as the control without being designed as one: it carries a
   native unix_time column with no conversion, and its numbers came back
   bit-identical to an ad-hoc check run earlier, which localised the fault to
   the TabFormer loader rather than the criterion.
   THIS IS THE SECOND HARNESS ARTIFACT IN TWO EXPERIMENTS (see entry 32), and
   the more dangerous of the pair. Entry 32's bug produced an ABSURD curve -
   false alarms falling as sharing rose - so it argued with us. This one
   produced exactly the finding we were hoping for. A harness bug that flatters
   you is not caught by scrutiny, because you are not applying any.
   NOT FIXED BY VIGILANCE, and we will not pretend otherwise. The rule we can
   actually keep: assert a loaded dataset's own documented shape - row count,
   date range, label prevalence - BEFORE measuring anything with it, and fail
   loudly on mismatch. The check now derives its firing preconditions from the
   shipped StreamingSpikeDetector rather than restating them, so at least the
   criterion cannot drift from the thing it checks.
   NET RESULT: the limitation is unchanged but far better evidenced. Nine
   benchmark datasets, one merchant identifier; that one plus IBM's 24M set
   both measured, both unable to evaluate this layer, for two DIFFERENT
   structural reasons. It is not that we did not look.

34. WE ASKED WHETHER THE DETECTOR KNOWS FRAUD OR ONLY OUR DRAWINGS.
   Two audits converged on the same objection and neither could be answered by
   more documentation: our five attacks are five TOPOLOGIES, and the model had
   only ever been tested against the shapes it was trained on. If detection
   collapses when the graph changes while the crime does not, then every
   merchant-level number in this repo is a fact about the generator.
   Built src/models/topology_generalisation.py. Each attack appears twice with
   transaction count, burst window, amount distribution, account ageing and
   fraud prevalence MATCHED; the only difference is who shares what with whom.
   Frozen pipeline, 5 seeds, and the known shapes rebuilt with FRESH entity ids
   as a positive control - so nothing is recognised by identity, only by form.
     card testing   3 dev / 2 ip     ->  25 dev / 20 ip   60 -> 7.2 txn/dev
     device farm    1 dev, 50 acct   ->  10 dev / 6 ip    130 -> 13
     ip cluster     40 acct on 1 ip  ->  6 rotating ips   6x diluted
     fraud ring     dense 15x4       ->  sparse 30x2of20  30 -> 12
     takeover       25 unique dev    ->  5 shared proxies CONCENTRATED
   FOUR OF THE FIVE UNSEEN VARIANTS ARE STRICTLY HARDER (2.5-10x less entity
   concentration, which is the signal the detector leans on). A test asserts
   this rather than trusting it, because a suite of secretly-easier variants
   would make a survival result meaningless.
   RESULT: known 22/25 detected, unseen 21/25. All four harder variants still
   5/5. But the mean flagged rate INSIDE the attack falls 0.815 -> 0.709, and
   on the two families whose fan-out was removed it falls much further - card
   testing 0.758 -> 0.436, ip cluster 0.872 -> 0.612.
   THE GAP IS THE FINDING, not the headline count. The per-transaction model
   clearly IS partly recognising shape: it loses confidence when the graph
   changes. The merchant-level layer fires anyway. That is this project's own
   thesis appearing as a measurement rather than an argument - the case for a
   merchant layer is precisely that it survives what per-order scoring finds
   ambiguous, and here we can watch that happen.
   WHAT WE DID NOT DO: tune anything until it fired. The account-takeover
   control fires only 2/5 even in its KNOWN form, because ATO is diffuse by
   construction - ~75 fraud txns over 22 hours never reaches a rate-in-a-window
   bar on a merchant doing 200/day. Compressing its window would have produced
   a clean 5/5 and measured nothing. Both ATO rows are reported as WEAK
   EVIDENCE and the verdict names them itself.
   LIMIT WE STATE FIRST: each variant degrades ONE signature and keeps the
   others - card testing still burns a novel instrument per transaction, the
   farm still shares instruments. A real adversary would degrade several at
   once, and this does not simulate that. Five topologies, one variant each,
   our generator, 5 seeds.
   THE HARNESS BUG THIS TIME WAS TRIVIAL AND THAT IS WORTH RECORDING TOO: the
   verdict string had seven format specifiers and six arguments, so the run
   crashed AFTER printing a complete and correct table. Third harness fault in
   three experiments (32, 33, 34) - but the first that could not corrupt a
   result, because it failed loudly at the point of reporting rather than
   quietly at the point of measuring. That is the difference between a bug that
   costs five minutes and one that nearly shipped a false finding.
