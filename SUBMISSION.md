# Fraud Spike Investigator — Track 02: AI Risk Manager

**Merchants don't lose money one transaction at a time. They lose it in bursts.** Per-order scorers flag bad orders; nobody tells a merchant *"you are under attack right now, here's why, here's who's behind it, here's your ₹ exposure, here's the bounded action."* This system closes that loop — merchant-level spike detection, entity correlation, and a policy-gated LLM investigator that **never makes the decision**. Temporally evaluated, costed in ₹, defense-only.

📹 **Demo video:** `[TODO: paste link]` · 💻 **Repo:** `[TODO: paste link]` · ▶️ **Run it:** `pip install -r requirements.txt && python run_demo.py`

---

## 1. Problem taste

Fraud concentrates in bursts — card-testing waves, device farms, IP clusters, account-takeover clusters, fraud rings. RBI-reported digital-payment fraud is the largest fraud category in Indian BFSI by case count. This system targets the burst, the moment losses concentrate.

The wedge is deliberately narrow and sits **above** per-order scoring: Thirdwatch/Shield score individual transactions, Bumblebee reviews merchant onboarding. Nobody does real-time, transaction-stream, **merchant-level** attack investigation. We consume per-order scores as *inputs* and answer the question they can't: is this merchant under attack, and what should a human do about it?

The hardest part of the problem is not catching attacks — it's **not crying wolf**. A 6× legitimate flash sale must not be blocked. That case is a first-class scenario in the simulator and the finale of the demo.

## 2. Build quality

One command runs it: `python run_demo.py` (trains, serves, replays 13,987 transactions through the real pipeline). **46 tests pass with no network and no credentials.**

| Metric (temporal test slice, synthetic data — labeled as such) | Value |
|---|---|
| PR-AUC | 0.934 |
| Precision / Recall @ cost-optimal threshold | 0.994 / 0.886 |
| Fraud ₹ prevented | ₹10.93L |
| Legitimate ₹ impacted | ₹4,814 |
| **Net protected value** (after 617 reviews × ₹50) | **₹10.57L (~31×)** |

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
- **Ablation says where the value is:** basics 0.661 → +velocity 0.631 → **+entity/graph 0.934**. Entity/graph features are the system (+0.30 PR-AUC, recall 0.58→0.89). Velocity alone is *mildly negative* — rolling counts spike for busy legitimate merchants too.
- **The LLM investigates; it never decides.** Six read-only tools, all ₹ arithmetic in Python (LLM arithmetic is unauditable), audit log of every tool call. Live eval on Claude Haiku 4.5, 10 ground-truth cases: **correct cause 9/10, policy violations 0/10, money-arithmetic-in-Python 10/10, and 4/4 low-signal merchants refused to invent an attack.** Full transcripts in `artifacts_out/agent_transcripts/`.
- **The allowlist binds humans too.** The analyst override endpoint rejects an invented action with HTTP 400 — the same gate the LLM gets. An analyst console accepting arbitrary action strings is the identical hole from the other side.
- **Risk fusion changes 0 of 13,987 decisions — and we say so.** It's kept for auditability, a reachable fail-safe, and headroom for a weaker model; not claimed as a metric win.

## 4. Failure recovery

**17 logged failures** in `CLAUDE.md`, every one caught by measuring a claim rather than re-reading code. The five that matter most:

1. **A dramatic finding that was wrong — retraction left visible in the README.** The ablation appeared to show velocity features *collapsing* PR-AUC 0.55 → 0.23. Root cause was ours: the calibration slice contained **no attack at all**, so isotonic calibration fit degenerate score plateaus. Fixed, the effect shrank to 0.661 → 0.631 — mild, not a collapse. A supporting top-200 diagnostic *also* dissolved: it sat inside a 447-transaction tie-plateau where composition was decided by row order; re-ranked by raw score it gave **zero** flash-sale transactions. Both the original claim and its retraction are in the README, because "we chased a dramatic result and it didn't survive" is the honest version.

2. **A threshold sweep that would have optimized a dead zone.** The pre-declared "adopt the best pair" rule picked restrict=80 — but restrict NPV was *exactly flat* from 40–80 (no validation transaction scores in that band). Refined to a per-parameter margin: step-up moved (+8.28% validation, +3.7% test), restrict stayed at 85. The refinement was made after seeing the surface and is documented as such.

3. **A silent LangGraph bug that looked like a bad model.** An undeclared `stop_reason` state key was dropped by LangGraph, so `route()` never saw `tool_use` and **every** investigation fell through to the fallback. Caught only because a test asserted on the audit log's *tool sequence*, not the final output.

4. **A dashboard metric that would have undermined the demo.** `peak_risk_ever` came out 100 for all 12 merchants — including the flash sale — because every merchant has one ambient-fraud transaction. It would have displayed "peak 100/100" on the card whose entire purpose is showing *no attack*. Removed; peak flagged-rate (93% vs 3%) and z (5.8 vs 1.1) discriminate properly.

5. **The agent got a ₹5.5L attack completely wrong — and it changed nothing.** In the live demo run, Claude labelled merchant m2 (a real account-takeover) as `legitimate_traffic` / `allow` at **0.95 confidence**. Its evidence was factually correct ("29 customers, 29 devices, 29 IPs, zero shared entities") and the conclusion was exactly wrong. Root cause is a *tool gap*: ATO is defined by established customers on new devices in new geos spending atypical amounts, and **no agent tool exposes any of those signals** — the agent reasons from entity-sharing, and ATO doesn't share entities. The system restricted 68 transactions and queued 8 for human review on m2 regardless, **because the agent was never in the decision path.** This is the architecture's central claim, demonstrated under a real failure rather than asserted. Not patched for submission: adding ATO-signal tools is a design change, and making it under deadline pressure is how the *next* failure gets introduced.

**Runtime failure recovery:** ML down → rules + human review. LLM down, timing out, or uncredentialed → deterministic templated report → human review, never a block. Low confidence → escalate. The LLM was never in the decision path, so disabling it changes no decision — pytest-enforced.

---

### Honest limitations

Data is synthetic; simulator parameters are design choices, not Razorpay statistics — and the simulator's entity IDs are partly self-labelling (`pi_STOLEN_*`, `d_FARM_*`), which makes the agent's diagnosis task easier than reality. The agent cannot currently diagnose account takeover at all (no tool exposes new-device / geo-shift / amount-deviation signals), and its output varies run to run — m2 came back `account_takeover` in the eval and `legitimate_traffic` in the demo. `get_merchant_baseline` splits at the 75th percentile, so an attack ending mid-window reads as "improved". Validation holds 125 fraud transactions dominated by one attack type. No production hardening: single process, in-memory state, no authn.
