# Fraud Spike Investigator — Track 02: AI Risk Manager

**Merchants don't lose money one transaction at a time. They lose it in bursts.** Per-order scorers flag bad orders; nobody tells a merchant *"you are under attack right now, here's why, here's who's behind it, here's your ₹ exposure, here's the bounded action."* This system closes that loop — merchant-level spike detection, entity correlation, and a policy-gated LLM investigator that **never makes the decision**. Temporally evaluated, costed in ₹, defense-only.

📹 **Demo video:** `[TODO: paste link]` · 💻 **Repo:** `[TODO: paste link]` · ▶️ **Run it:** `pip install -r requirements.txt && python run_demo.py`

---

## 1. Problem taste

Fraud concentrates in bursts — card-testing waves, device farms, IP clusters, account-takeover clusters, fraud rings. RBI-reported digital-payment fraud is the largest fraud category in Indian BFSI by case count. This system targets the burst, the moment losses concentrate.

The wedge is deliberately narrow and sits **above** per-order scoring: Thirdwatch/Shield score individual transactions, Bumblebee reviews merchant onboarding. Nobody does real-time, transaction-stream, **merchant-level** attack investigation. We consume per-order scores as *inputs* and answer the question they can't: is this merchant under attack, and what should a human do about it?

The hardest part of the problem is not catching attacks — it's **not crying wolf**. A 6× legitimate flash sale must not be blocked. That case is a first-class scenario in the simulator and the finale of the demo.

## 2. Build quality

One command runs it: `python run_demo.py` (trains, serves, replays 13,987 transactions through the real pipeline). **51 tests pass with no network and no credentials.**

| Metric (temporal test slice, synthetic data — labeled as such) | Value |
|---|---|
| PR-AUC | **0.945 ± 0.007** across 5 independent worlds (0.934 on the README's seed) — *see failure #1: two features reproduce this, and our simulator is why* |
| Precision / Recall @ cost-optimal threshold | 0.994 / 0.886 |
| Fraud ₹ prevented | ₹10.93L |
| Legitimate ₹ impacted | ₹4,814 |
| **Net protected value** (after 617 reviews × ₹50) | **₹10.57L (~31×)** |
| Calibration — Brier / ECE | 0.00533 / 0.0033 (reliability curve near-bimodal, reported) |
| Human review load | 44.1 cases per 1,000 transactions (4.41%) |
| **Merchant-level, across 5 worlds** | **25/25 attacks caught, 0 false alarms, flash sale flagged 0/5** |
| **Same recipe on REAL data**, zero tuning | ULB (284k txns): **PR-AUC 0.731** vs 0.0017 baseline · IEEE-CIS (590k txns): **0.460** vs 0.035 baseline |

**Time-to-detect vs. two baselines** — ground-truth attack starts from the simulator:

| Scenario | Static volume | Flag counter | **Streaming spike** |
|---|---|---|---|
| Card-testing | not detected | 62m47s | **55m22s** |
| Device farm | 3m59s | 21m54s | **19m18s** |
| IP cluster | not detected | **26m15s** | 31m03s ← *we lose this one* |
| Account takeover | not detected | 53m38s | **53m30s** |
| Fraud ring | not detected | 100m49s | **69m37s** |
| **False alarms** | **5 merchants (incl. flash sale)** | 0 | **0** |

Leakage-safe incremental features (every feature from prior events only), day-boundary temporal splits — never random. Model selection, threshold tuning, and calibration all happen on **validation**; the test slice is read once, at the end.

## 3. AI judgment — including where we chose *not* to use AI

- **Fraud scoring: a GBDT, not an LLM — and the library was chosen empirically.** Four classifiers, identical class weighting, default hyperparameters, compared on validation. All three GBDTs landed inside a pre-declared 0.02 margin, so the tie-break fired: XGBoost won on **speed** (0.27s vs 1.64s vs 7.94s), not on a 0.0059 PR-AUC difference that is pure noise on 6,166 rows.
- **Spike detection: EWMA + z-score, not an autoencoder.** ~50 lines, online, explainable. It fires on the *fraud-score rate*, not volume — which is exactly why the flash sale doesn't trigger it.
- **Ablation says where the value is — and then said something we didn't want to hear.** basics 0.661 → +velocity 0.631 → **+entity/graph 0.934**. We wrote that up as "entity/graph is the system, +0.30 PR-AUC." Auditing our own evaluation showed the attribution was wrong: the lift comes from two *profile* features filed in that bucket (`customer_age_days`, `amount_dev_ratio`), which alone score 0.9328, because the simulator sets attack accounts' creation date to the attack day. Entity *sharing* alone scores 0.8286 and adds +0.004 on top. **The README claim is retracted and corrected in place.** What entity features do still buy is measurable and matters for this track: at the deployed threshold they cut legitimate ₹ wrongly blocked from ₹68,319 to ₹5,901.
- **The LLM investigates; it never decides.** Seven read-only tools, all ₹ arithmetic in Python (LLM arithmetic is unauditable), audit log of every tool call. We scored 9/10, found most of it was our own dataset leaking the answers, measured the leak (10→5 on de-labelling), fixed the evidence gaps, froze the design, and published what survived. Live eval on Claude Haiku 4.5 over **de-labelled** data, 13 cases incl. 3 held-out on a fresh seed: **correct cause 8/13 (2/3 held-out), 100/100 evidence claims traceable, escalates-when-unsure 13/13, policy violations 0/13, unsafe actions 0/13.** Every action error was in the cautious direction — the agent's mistakes cost analyst minutes, never merchant money. (`correct_action` 6/13 partly measures our label design: `expected_action` predates the tool that let the agent see an attack had already ended; labels left untouched.) Transcripts in `artifacts_out/eval_runs/run_D_final/`.
- **The allowlist binds humans too.** The analyst override endpoint rejects an invented action with HTTP 400 — the same gate the LLM gets. An analyst console accepting arbitrary action strings is the identical hole from the other side.
- **Risk fusion changes 0 of 13,987 decisions — and we say so.** It's kept for auditability, a reachable fail-safe, and headroom for a weaker model; not claimed as a metric win.

## 4. Failure recovery

**24 logged failures** in `CLAUDE.md`, every one caught by measuring a claim rather than re-reading code. The seven that matter most:

1. **We broke our own headline metric, on purpose, and published the result.** Every number here is measured on data we generated — so we ran an adversarial audit against our own evaluation (`python -m src.models.leakage_probe`). It found that **two features reproduce the entire 22-feature model**: `customer_age_days` + `amount_dev_ratio` score 0.9328 against the full pipeline's 0.9344, a gap of 0.0016. The cause is in our generator, not the model: attack accounts are created *on the attack day*, giving median account ages of 0.98–5.65 days against a legitimate baseline of 215.76, while ambient fraud amounts are literally a legitimate amount × `uniform(1.5, 4.0)`. Between them the two features partition the label space. **This forced us to retract this submission's central ablation claim.** What it does *not* touch, and we checked each one: the pipeline's train/test hygiene (features are still built strictly from prior events, splits are still on day boundaries), and every merchant-level result — 5/5 attacks, 0 false alarms, flash sale never flagged — which cannot be explained by account age, since flash-sale customers are *older* than baseline (230.65 days vs 215.76). We chose to ship a measured statement of what our headline means over a better-looking number we couldn't defend. A judge who runs XGBoost on two of our columns will find our own analysis waiting for them.

2. **A dramatic finding that was wrong — retraction left visible in the README.** The ablation appeared to show velocity features *collapsing* PR-AUC 0.55 → 0.23. Root cause was ours: the calibration slice contained **no attack at all**, so isotonic calibration fit degenerate score plateaus. Fixed, the effect shrank to 0.661 → 0.631 — mild, not a collapse. A supporting top-200 diagnostic *also* dissolved: it sat inside a 447-transaction tie-plateau where composition was decided by row order; re-ranked by raw score it gave **zero** flash-sale transactions. Both the original claim and its retraction are in the README, because "we chased a dramatic result and it didn't survive" is the honest version.

3. **A threshold sweep that would have optimized a dead zone.** The pre-declared "adopt the best pair" rule picked restrict=80 — but restrict NPV was *exactly flat* from 40–80 (no validation transaction scores in that band). Refined to a per-parameter margin: step-up moved (+8.28% validation, +3.7% test), restrict stayed at 85. The refinement was made after seeing the surface and is documented as such.

4. **A silent LangGraph bug that looked like a bad model.** An undeclared `stop_reason` state key was dropped by LangGraph, so `route()` never saw `tool_use` and **every** investigation fell through to the fallback. Caught only because a test asserted on the audit log's *tool sequence*, not the final output.

5. **A dashboard metric that would have undermined the demo.** `peak_risk_ever` came out 100 for all 12 merchants — including the flash sale — because every merchant has one ambient-fraud transaction. It would have displayed "peak 100/100" on the card whose entire purpose is showing *no attack*. Removed; peak flagged-rate (93% vs 3%) and z (5.8 vs 1.1) discriminate properly.

6. **We caught our own eval cheating — twice — and published the lower number.** The agent scored 9/10 on cause until we noticed the simulator's entity IDs were self-labelling (`pi_STOLEN_*`, `d_FARM_F`, `ip_CLUSTER_I`) and the transcripts were citing them verbatim as evidence. Hashing every ID to an opaque form dropped it to **5/10** — that gap is how much of the score was the dataset whispering the answer. Separately, our first ATO tool reported "80% spent 3×+ over own average" on a denominator of **5 transactions** with no base rate, and promptly diagnosed three quiet merchants as account takeover; adding base rates and sample sizes fixed it. The five-run progression (A → B1 → B → C → D) is preserved in `artifacts_out/eval_runs/` as the exhibit. De-labelling is provably a pure relabelling: **all 16 ML metrics are bit-identical** before and after, asserted in `train.py`. The system prompt is sha256-identical across all five runs — the agent was never coached.

7. **The agent got a ₹5.5L attack completely wrong — and it changed nothing.** In the live demo run, Claude labelled merchant m2 (a real account-takeover) as `legitimate_traffic` / `allow` at **0.95 confidence**. Its evidence was factually correct ("29 customers, 29 devices, 29 IPs, zero shared entities") and the conclusion was exactly wrong. Root cause is a *tool gap*: ATO is defined by established customers on new devices in new geos spending atypical amounts, and **no agent tool exposes any of those signals** — the agent reasons from entity-sharing, and ATO doesn't share entities. The system restricted 68 transactions and queued 8 for human review on m2 regardless, **because the agent was never in the decision path.** This is the architecture's central claim, demonstrated under a real failure rather than asserted. Closed by adding a seventh tool exposing new-device / geo-mismatch / amount-vs-own-average / account-age with the non-flagged population as the comparison base — **the fix was the missing evidence, not a prompt hint** (the system prompt is sha256-identical across all five eval runs). m2 is now diagnosed correctly, and quiet merchants use the same fields to *rule out* ATO.

**Runtime failure recovery:** ML down → rules + human review. LLM down, timing out, or uncredentialed → deterministic templated report → human review, never a block. Low confidence → escalate. The LLM was never in the decision path, so disabling it changes no decision — pytest-enforced.

---

### Honest limitations

**Our simulator encodes the label into two features, so the headline PR-AUC overstates real detection ability** (failure #1) — the fix is regenerating attack accounts with realistic ages, which changes every number here, so it is documented rather than rushed. **Public data validates the transaction-risk component only.** ULB 0.731 and IEEE-CIS 0.460 (same recipe, zero tuning) show the modelling pipeline is not an artifact of our simulator. The **merchant-level** layer — spike detection, entity resolution, the policy engine — is evaluated on controlled scenarios, because the two public datasets we evaluated do not expose the persistent entity relationships required to directly test it: IEEE-CIS `DeviceInfo` is a device *type* ("Windows" is 40.2% of rows), not a fingerprint, and it has no account identifier at all. We ran that experiment and publish the rows in the repo, but do not present them as evidence either way, since they do not measure the hypothesis. Independent support for the approach does exist: Vesta's own entity-counting features are the single largest contributor to IEEE-CIS performance (+0.352 PR-AUC). Data is synthetic throughout; simulator parameters are design choices, not Razorpay statistics. Only **one** legitimate-spike scenario (the flash sale) is tested, and the EWMA baseline has no seasonality term. Headline metrics carry a range (PR-AUC 0.945 ± 0.007 over five worlds) but five worlds share one generative process — that measures sampling variance, not real-world variance; differences below ~±0.02 PR-AUC should not be treated as real. **44.1 review cases per 1,000 transactions** is priced at ₹50 each but never checked against whether the analysts exist. The agent still confuses low-signal quiet merchants with account takeover — its own evidence says "zero new devices, zero geo mismatches" and it concludes account takeover anyway (documented verbatim in the README; contained by the policy gate, which routes it to a human). Its output varies run to run. `correct_action` is partly measuring our label design: `expected_action` predates the agent being able to see that an attack had already ended. Validation holds 125 fraud transactions dominated by one attack type. A live demo rehearsal caught a 22nd failure we chose not to fix: the agent reads instrument *novelty* (~1.0 distinct instruments per transaction) as evidence *against* card testing, when it is the signature of it — the tool reports the number without its polarity. Fixing it means either coaching the prompt (against our own rule that the fix must be evidence, not hints) or changing a tool the frozen eval depends on, so it is logged for the next held-out set instead. No production hardening: single process, in-memory state (lost on restart), no idempotency on repeated transactions. Every mutating endpoint requires a shared write key, but that is a single-tenant gate, not identity — an analyst override carries no attributable actor, which a real deployment needs.
