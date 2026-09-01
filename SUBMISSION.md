# Fraud Spike Investigator — Track 02: AI Risk Manager

**Merchants don't lose money one transaction at a time. They lose it in bursts.** Per-order scorers flag bad orders; nobody tells a merchant *"you are under attack right now, here's why, here's who's behind it, here's your ₹ exposure, here's the bounded action."* This system closes that loop — merchant-level spike detection, entity correlation, and a policy-gated LLM investigator that **never makes the decision**. Temporally evaluated, costed in ₹, defense-only.

📹 **Demo video:** `[TODO: paste link]` · 💻 **Repo:** https://github.com/CaringNihilistic/fraud-spike-investigator · 🖥️ **Live console:** https://fraud-spike-investigator.onrender.com/ · ▶️ **Run it:** `pip install -r requirements.txt && python run_demo.py`

---

## At a glance

| | |
|---|---|
| **What it is** | Merchant-level fraud-spike detection, entity correlation, and policy-gated LLM investigation — one layer *above* per-order scoring |
| **Attack merchants detected** | **25 / 25** across 5 seeds of the generator (seeds, not independent worlds — the scenario definitions are identical) |
| **False alarms** | **1 in 35 non-attack merchant-windows** (2.9%, 95% CI 0.1–14.9%) — a legitimate corporate buyer, on one seed. Three legitimate spikes are tested, two of which share entities; the 6× flash sale is flagged **0 / 5** |
| **Net protected value** | **₹8.13L** on the held-out test slice, after 948 human reviews costed at ₹50 each |
| **Precision / Recall** | **0.996 / 0.735** at the cost-optimal threshold |
| **Calibration** | Brier **0.0123**, ECE **0.0074** — measured, not assumed, and measurably *worse* since the hard negatives |
| **Real public data**, same pipeline, zero tuning | ULB (284k txns): **PR-AUC 0.731** vs 0.0017 random · IEEE-CIS (590k txns): **0.460** vs 0.035 random |
| **LLM safety** | **0/13** policy violations · **0/13** unsafe actions · the LLM cannot authorize anything — pytest-enforced |
| **Engineering** | **80 tests**, no network or credentials needed · one-command demo · **32 logged failures** with root causes |

Evaluation is temporal throughout — day-boundary splits, never random. Model selection, threshold tuning and calibration all happen on a validation slice; the test slice is read once, at the end. Every number above reproduces from a clean clone.

---

## 1. Problem taste

Fraud concentrates in bursts — card-testing waves, device farms, IP clusters, account-takeover clusters, fraud rings. RBI-reported digital-payment fraud is the largest fraud category in Indian BFSI by case count. This system targets the burst, the moment losses concentrate.

The wedge is deliberately narrow and sits **above** per-order scoring: Thirdwatch/Shield score individual transactions, Bumblebee reviews merchant onboarding. Nobody does real-time, transaction-stream, **merchant-level** attack investigation. We consume per-order scores as *inputs* and answer the question they can't: is this merchant under attack, and what should a human do about it?

The hardest part of the problem is not catching attacks — it's **not crying wolf**. A 6× legitimate flash sale must not be blocked. That case is a first-class scenario in the simulator and the finale of the demo: the legitimate merchant is the *busier* of the two (2,284 transactions against 1,041) and receives **zero** restrictions, because the detector fires on fraud-score **rate**, not volume.

## 2. Build quality

One command runs it: `python run_demo.py` (trains, serves, replays 14,160 transactions through the real pipeline). **80 tests pass with no network and no credentials.**

| Metric — temporal held-out test slice, synthetic data (labeled as such) | Value |
|---|---|
| PR-AUC | **0.862 ± 0.024** across 5 seeds of the generator |
| Precision / Recall @ cost-optimal threshold | **0.996 / 0.735** |
| Precision@100 / @500 | 1.00 / 1.00 (stable under adversarial tie-breaking) |
| Fraud **exposure** prevented (policy path) | **₹8.61L** |
| Legitimate ₹ wrongly blocked (classifier, at its threshold) | **₹2,575 — 0.024%** of the ₹1.07Cr legitimate value processed |
| Legitimate ₹ actually impacted (policy path, after step-up/review routing) | **₹452** |
| Net protected value (after 948 reviews × ₹50) | **₹8.13L (~18×)** |
| Robustness of that figure | break-even at **₹905/review — 18× the assumed ₹50** |
| Legitimate spikes tested | **3**, two of which share entities — incl. one built to defeat the entity layer |
| Calibration — Brier / ECE | 0.0123 / 0.0074 |
| Human review load | 66.9 cases per 1,000 transactions (6.69%) |
| Merchant-level, across 5 seeds | **25/25 attacks caught, 1 false alarm in 35 non-attack merchant-windows, flash sale flagged 0/5** |

> These numbers survived three independent audits of our own evaluation — see **§5**. One of those audits lowered what the PR-AUC means, and we report both the number and what it does and does not measure.

**Time-to-detect vs. two baselines** — ground-truth attack starts from the simulator:

| Scenario | Static volume | Flag counter | **Streaming spike** |
|---|---|---|---|
| Card-testing | not detected | 62m47s | **55m22s** |
| Device farm | 3m59s | 21m54s | **19m18s** |
| IP cluster | not detected | **26m15s** | 31m03s ← *we lose this one* |
| Account takeover | not detected | 53m38s | **53m30s** |
| Fraud ring | not detected | 100m49s | **69m37s** |
| **False alarms** | **5 merchants (incl. flash sale)** | 0 | **0** |

Leakage-safe incremental features (every feature computed from prior events only, state updated after emission), day-boundary temporal splits — never random.

## 3. AI judgment — including where we chose *not* to use AI

- **Fraud scoring: a GBDT, not an LLM — and the library was chosen empirically.** Four classifiers, identical class weighting, default hyperparameters, compared on validation. All three GBDTs landed inside a pre-declared 0.02 margin, so the tie-break fired: XGBoost won on **speed** (0.40s vs 1.85s vs 8.82s), not on a 0.0077 PR-AUC difference that is pure noise on 6,190 rows.
- **Spike detection: EWMA + z-score, not an autoencoder.** ~50 lines, online, explainable. It fires on the *fraud-score rate*, not volume — which is exactly why the flash sale doesn't trigger it.
- **No SMOTE, no GNN, no Kafka, no vector DB.** Class weighting plus isotonic calibration instead of resampling, because resampling distorts a temporal distribution. Each omission is a decision we can defend in one sentence.
- **The LLM investigates; it never decides.** Seven read-only tools, no tool exposes ground truth, all ₹ arithmetic in Python (LLM arithmetic is unauditable), audit log of every tool call. Live eval on Claude Haiku 4.5 over **de-labelled** data, 13 cases including 3 held-out on a fresh seed: **correct cause 8/13 (2/3 held-out), 100/100 evidence claims traceable, escalates-when-unsure 13/13, policy violations 0/13, unsafe actions 0/13, attacks let through 0.** Every action error was in the cautious direction — the agent's mistakes cost analyst minutes, never merchant money.
- **The allowlist binds humans too.** The analyst override endpoint rejects an invented action with HTTP 400 — the same gate the LLM gets. An analyst console accepting arbitrary action strings is the identical hole from the other side.
- **Risk fusion went from a no-op to changing 3.1% of decisions — and we reported it at every stage.** 0 of 13,987 on the leaky generator, 3 of 13,782 after the generator fix, 433 of 14,160 once the data contained real ambiguity. Kept for auditability, a reachable fail-safe, and headroom for a weaker model; not claimed as a metric win.

## 4. Failure recovery

**32 logged failures** in `CLAUDE.md`, with root causes. Most were caught by *measuring a claim* rather than re-reading code — but not all, and the newest was not: an outside audit found it. The most important:

**We built a legitimate merchant designed to break our own system, and it did.** "0 false alarms" had rested on a single negative — a flash sale where every account has its own device, IP and card, so it never exercised the entity layer at all. We added a shared payment kiosk: 25 real customers, one terminal, entirely honest. **It scored 0.972 against a real device farm's 0.985, and produced a higher spike z than an actual attack.** Root cause was a training-distribution gap — the model had only ever seen shared devices committing fraud. Fixed by putting a legitimate kiosk in the *training* period: 0.972 → 0.002, attacks unaffected, precision up to 0.996. **A second hard negative still false-alarms on one seed in five and we left it there.** The symmetric fix looked *better* on the seed we first measured — higher net protected value — so this originally read as us overriding our own cost rule on judgment. An outside reviewer pointed out that comparing a win on one seed to a failure on another is not a comparison. Measured across all five: the rejected configuration is **₹64,192 worse on average and loses an attack.** It was not a trade at all. Four configurations were tried; the rejected one is left reachable so anyone can re-measure it.

**Then we stopped sampling that boundary at three points and swept it.** An outside audit's sharpest criticism was that our three legitimate merchants are three *points* on a continuum, and a system whose thesis is “entity sharing is suspicious” had never measured what happens between “shares nothing” and “shares everything”. So we built legitimate merchants whose only varying property is the share of traffic through one device/IP, swept **0→100%** across **5 seeds** through the **frozen pipeline**, with a real device farm as a **positive control** — because a sweep whose control cannot fire measures the harness, not the system. Result: **nothing fires below 80% concentration** (40/40 merchant-seeds clean), **3 of 60** fire at or above it, and **₹0 of legitimate revenue is restricted anywhere** — those three are *alerts, not blocks*. The control fires **5/5** at a mean flagged rate of **0.959** against a worst legitimate **0.067**. The rate does not climb through the middle; it is flat at zero until traffic is overwhelmingly funnelled through one entity, and the reason is the hard negative above — training now contains an *honest* shared-device merchant. **The first run of this sweep was wrong** and said the opposite (failure 32): our swept merchants drew ISP pools 3.75× denser than the world the model was trained on. We caught it because the curve was absurd in a direction we could reason about, which is not a defence that generalises — the control row is.

And the newest, because it is the least flattering:

**We shipped a policy constant our own rule forbids, and an outside audit found it.** We handed this document, `CLAUDE.md` and the machine-generated result JSONs to an external model with a prompt written to make it hostile. It returned six documentation findings; **five were correct**, including a headline that still read *"0 false alarms"* on a page that says *"1 false alarm in 35"* forty lines below. Chasing the sixth — a cutoff quoted as both `(85, 25)` and `(85, 20)` — found something the auditor could not see, because it had our documents and not our code: `engine.py` shipped `STEP_UP_CUT = 20.0` while the validation sweep adopts **25**. Re-run on current data, 20 is worth **+1.73%** — *below our own pre-declared 2% adopt margin*. It had been correct one dataset generation earlier; the hard negatives changed the data and nobody re-ran the sweep. Fixed to 25, and every downstream number **re-measured rather than edited**: net protected value ₹8.11L → **₹8.13L**, legitimate revenue impacted ₹4,266 → **₹452**. It came out better on both axes, which is luck — the rule was applied, not shopped. **The process gap that caused it is still open and labelled as open:** nothing checks a documented number against the artifact it came from.

Five more:

1. **The agent got a ₹5.5L attack completely wrong — and it changed nothing.** On a live run, Claude labelled merchant m2 (a real account takeover) as `legitimate_traffic` / `allow` at **0.95 confidence**. Its evidence was factually correct ("29 customers, 29 devices, 29 IPs, zero shared entities") and the conclusion was exactly wrong. Root cause was a *tool gap*: account takeover means established customers on new devices, and no tool exposed that. **The system restricted 68 transactions and queued 8 for human review anyway, because the LLM was never in the decision path.** This is the architecture's central claim, demonstrated under a real failure rather than asserted. Closed by adding evidence — a seventh tool exposing new-device / geo-mismatch / amount-vs-own-average / account-age against the non-flagged base rate — **not by coaching the prompt**, which is sha256-identical across all five eval runs.

2. **A dramatic finding that was wrong — retraction left visible in the README.** The ablation appeared to show velocity features *collapsing* PR-AUC 0.55 → 0.23. Root cause was ours: the calibration slice contained **no attack at all**, so isotonic calibration fit degenerate score plateaus. Fixed, the effect shrank to 0.661 → 0.631 — mild, not a collapse. After the generator fix it reversed again, to **+0.0317 [+0.0189, +0.0445]** — positive and significant on a paired bootstrap. Published, retracted, then revised a second time in the opposite direction. A supporting top-200 diagnostic *also* dissolved: it sat inside a 447-transaction tie-plateau where composition was decided by row order. Both the original claim and its retraction stay in the README, because "we chased a dramatic result and it didn't survive" is the honest version.

3. **A threshold sweep that would have optimized a dead zone.** The pre-declared "adopt the best pair" rule picked restrict=80 — but restrict NPV was *exactly flat* from 40–80 (no validation transaction scores in that band). Refined to a per-parameter margin: step-up moved (+2.73% validation), restrict stayed at 85. The refinement was made after seeing the surface and is documented as such rather than presented as foresight.

4. **A silent LangGraph bug that looked like a bad model.** An undeclared `stop_reason` state key was dropped by LangGraph, so `route()` never saw `tool_use` and **every** investigation fell through to the fallback. Caught only because a test asserted on the audit log's *tool sequence*, not the final output.

5. **The review queue was sorted by arrival time.** A queue only matters if the analyst runs out of time first — ours yields 948 cases — so the order is a policy decision, and we had not made one. Ranking by **expected loss** (amount × calibrated probability) puts **47%** of the queue's fraud value in front of a human in the first 50 cases against **5.7%** for arrival order. The obvious fix would have been worse: ranking by risk score scores *below arrival* at every capacity, because fusion's scale is not rupees.
6. **A dashboard metric that would have undermined the demo.** `peak_risk_ever` came out 100 for all 12 merchants — including the flash sale — because every merchant has one ambient-fraud transaction. It would have displayed "peak 100/100" on the card whose entire purpose is showing *no attack*. Removed; peak flagged-rate (93% vs 3%) and z-score (5.8 vs 1.1) discriminate properly.

**Runtime failure recovery:** ML down → rules + human review. LLM down, timing out, or uncredentialed → deterministic templated report → human review, never a block. Low confidence → escalate. The LLM was never in the decision path, so disabling it changes no decision — pytest-enforced.

## 5. How we stress-tested our own numbers

Three independent audits, each aimed at our own evaluation rather than at the model. **Each one lowered a number we had already written down, and we published the lower number.** All three are reproducible from the repo.

**Audit 1 — was the agent reading our answer key?** It scored 9/10 on cause. Then we noticed the simulator's entity IDs were self-labelling (`pi_STOLEN_*`, `d_FARM_F`, `ip_CLUSTER_I`) and the transcripts were citing them verbatim as evidence. Hashing every ID to an opaque form dropped the score to **5/10** — that gap is exactly how much was the dataset whispering. We then fixed the genuine evidence gaps, froze the design, and scored once against a held-out set on a fresh seed: **8/13**. The five-run progression A→B1→B→C→D is preserved in `artifacts_out/eval_runs/`. De-labelling is provably a pure relabelling — **all 16 ML metrics are bit-identical** before and after.

**Audit 2 — was the *model* reading our answer key?** Same attack, one layer down (`python -m src.models.leakage_probe`). It found that **two features reproduced the entire 22-feature model**: `customer_age_days` + `amount_dev_ratio` scored 0.9328 against the full pipeline's 0.9344. The cause was our generator — attack accounts were created *on the attack day*, giving median ages of 1–6 days against a legitimate baseline of 216. It forced us to retract this submission's claim that entity/graph features were the source of the lift.

**Then we fixed the generator and re-measured.** Attack accounts are now aged realistically (real fraudsters *buy* aged accounts — the case we never generated) and fraud amounts are fraud-type dependent, because ULB's real card fraud runs 0.42× the median legitimate amount while IEEE-CIS e-commerce fraud runs 1.10×.

| | before | after |
|---|---|---|
| Full 22 features | 0.9344 | **0.8981** |
| **The two proxies alone** | **0.9328** | **0.5997** |
| **Gap** | **0.0016** | **0.2984** |

**The headline fell 0.036. The shortcut fell 0.333.** And the retracted claim turned out to be true: `component_size` is now the top feature by *both* single-feature PR-AUC (0.711) and model importance (0.406). We restate it only because a harder dataset now supports it.

**What the fix cost, stated plainly:** PR-AUC 0.945 → 0.910, net protected value ₹10.57L → ₹7.95L, legitimate ₹ wrongly blocked ₹5,901 → ₹21,728. **That last pair is not a regression** — the old generator made fraud nearly linearly separable, so precision 0.994 was *purchasable*; the two figures measure dataset difficulty, not system quality. We did **not** retune the threshold to recover it. **Untouched by all of it, as measured at that time:** 25/25 attacks, 0 false alarms, flash sale 0/5. (The false-alarm count became **1 in 35** one dataset generation later, when the hard negatives were added — see the row at the top of this page.)

**"Isn't your new choice just as arbitrary?"** Measured, not argued (`python -m src.models.aged_share_sensitivity`): across the whole realistic range the proxy pair never comes within 0.02 of the full set, and full-set PR-AUC moves only **0.0056**. A control reverting *both* fixes reproduces the leak, which is how we know the sweep measures what it claims to. The old value — 100% newborn — was a judgement call too; just an implicit one, never examined, load-bearing enough to fake a headline.

**Audit 3 — does the method survive real data?** We ran the unchanged recipe on two public datasets: **ULB PR-AUC 0.731** (vs a 0.0017 random baseline — 423× lift) and **IEEE-CIS 0.460**, both with zero tuning. It also exposed a latent bug our own data structurally could not: the cost-optimal threshold grid topped out at `max(p)`, so **"block nothing" was unreachable** — and on real data abstaining was **3.8× cheaper** than the threshold the function chose. Fixed; every synthetic number is **bit-identical** afterwards, which is how we know it was latent and the fix is safe.

---

## Limitations

Ordered by how much each should discount the results.

1. **Our audit tooling was written by the person whose work it audits, and we have a documented instance of that biting us.** `leakage_probe.py` and `ablation.py` both hardcoded their *failing* conclusions, so once the generator was fixed they printed "two features reproduce the headline" above their own output showing they don't. Neither could structurally report a pass. Both verdicts are now derived from the numbers — but every "we checked this" here was checked by the person who wrote the thing being checked.
2. **We fixed the two label proxies we found. We have not proven there are no others.** A negative result from an adversarial test is only ever as strong as the test.
3. **The merchant-level system is validated on synthetic data only — and we went looking.** ULB and IEEE-CIS have no merchant column. A third dataset (Sparkov) does, and we measured that it *still* cannot test this layer: **zero evaluable merchant-hour windows, and a maximum merchant-day fraud rate of 0.273 against the 0.70–0.93 our attacks reach.** Public card-fraud data models stolen cards spent across many merchants; this product models merchants under coordinated attack. Different loss class. Checked *before* modelling, so we did not publish a null that measures the dataset rather than the detector.
4. **Our entity graph is unvalidated on public data.** The two datasets we evaluated do not expose the persistent entity relationships needed to test it — IEEE-CIS `DeviceInfo` is a device *type* ("Windows" is 40.2% of rows), not a fingerprint, and it has no account identifier. We ran the experiment and publish the rows, but do not present them as evidence either way, because they do not measure the hypothesis. Independent support does exist: Vesta's own entity-counting features are the single largest contributor to IEEE-CIS performance (+0.352).
5. **One legitimate-spike scenario.** The flash sale is the only benign volume event tested; the EWMA baseline has no seasonality term.
6. **The review queue may not be staffable, and the load rose 60%.** 66.9 cases per 1,000 transactions (up from 41.9) is priced at ₹50 each but never checked against whether the analysts exist. The ₹ conclusion is robust to that price — break-even at ₹905/review — the headcount is not.
7. **Metrics carry a range, and it widened.** PR-AUC 0.862 ± 0.024 across five *seeds of the same generator* measures sampling variance, not real-world variance. Differences below ~±0.02 should not be treated as real. **The agent eval is far smaller — n=13, giving roughly a ±25-point confidence interval, so 8/13 and 5/13 are not distinguishable.** Every conclusion drawn from it, including the A→B→C→D progression, is a small-sample result.
8. **The agent's reasoning is the weakest component.** 8/13 correct cause; on quiet merchants it has twice contradicted its own evidence. It reads instrument *novelty* as evidence against card testing when that is the signature of it (failure 22, logged and deliberately unfixed — the design is frozen and the fix must come from evidence, not prompt hints). Output varies run to run. It is advisory only and cannot act.
9. **Risk fusion changed almost nothing until the data got hard.** 0 → 3 → 433 decisions across three dataset versions. Retained for architecture throughout, and reported the same way when it did nothing and when it did something.
10. **Not production hardened.** Single process, in-memory state (lost on restart), no idempotency. Every mutating endpoint requires a shared write key, but that is a single-tenant gate, not identity — an analyst override carries no attributable actor.

**Data is synthetic throughout and labeled as such everywhere. Simulator parameters are design choices, not Razorpay statistics. Nothing here is Razorpay data.**


### Defense-only, including the obvious objection

The repo contains a synthetic generator that produces labelled attack traffic,
which is the thing to challenge on a defense-only track. The argument, not just
the assertion: it emits rows with an `is_fraud` label so a supervised model has
positives to learn from and the evaluation has ground truth. It touches no real
system, optimizes nothing (the attack shapes are fixed hand-written topologies
from the public literature), models no evasion, and discovers no weaknesses.
**The output is a labelled dataset, not attack tooling.**

The defensive side is enforced in code: the LLM has read-only tools, cannot
authorise anything, and any unrecognised recommendation is degraded to `REVIEW`
rather than escalated. Every action comes from a frozen four-item allowlist that
binds the human analyst as tightly as the model. All pytest-enforced.

---

### The rule we hold ourselves to

One of our audits (IEEE-CIS, failure 24) came back *negative* on the entity-correlation idea the whole product rests on. We do **not** report it as evidence against — because the dataset's `DeviceInfo` is a device *type*, not a fingerprint ("Windows" alone is 40.2% of rows), and it has no account identifier at all, so the quantity we reason about is literally inexpressible there. Reporting a null from a test that provably cannot measure the thing is not honesty; it is presenting invalid evidence in the pessimistic direction. So we publish the rows and the caveat together, and claim nothing in either direction.

**And the symmetry that makes that defensible: we would not have accepted a positive result from that experiment either.**
