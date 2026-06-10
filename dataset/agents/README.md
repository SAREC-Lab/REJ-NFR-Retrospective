# Agent-Based NFR Classifier — Debate and NoDebate

Two techniques are implemented here, both using the same multi-agent debate pipeline:

| Technique | Script | allresults.txt row |
|-----------|--------|-------------------|
| **Agent Debate** | `run_debate.py` | `Agent Debate` |
| **Agent Basic (NoDebate)** | `tune_nodebate.py` | `Agent Basic` |

Both require the debate pipeline to have been run first (`run_debate.py`), since NoDebate re-uses the advocate confidence scores saved by the debate run.

---

## Technique Overview

### Agent Debate

A multi-agent pipeline that processes one requirement at a time through up to six sequential stages:

1. **Initial screen** — a single LLM call classifies the requirement as FR (functional), NFR (one unambiguous type), or DEBATE (ambiguous, triggers full pipeline).
2. **Advocate screening** — each candidate NFR type screens the requirement and reports a confidence score. Run in parallel.
3. **Advocate filter** — advocates below `advocate_threshold` (0.3) are eliminated; only believers proceed.
4. **Advocate arguments** — surviving advocates make their full case for why this requirement belongs to their type. Run in parallel.
5. **Devil's advocate** — argues the requirement is functional; backs down if NFR evidence is strong. Sustained only if confidence exceeds `da_threshold` (0.7).
6. **Arbiter** — weighs all arguments and the devil's advocate, returns final label(s). Errs toward NFR if borderline (recall > precision).

Requirements classified as FR at the initial screen skip stages 2–6 (one API call). Requirements classified as unambiguous NFR at the initial screen skip stages 2–6 but accept the type directly (one API call). Only ambiguous requirements run the full pipeline.

The arbiter's direct label decisions are reported as **Agent Debate** results.

### Agent Basic (NoDebate)

Uses the advocate confidence scores saved by `run_debate.py` (file `scores_*.json`) without running the arbiter. Instead, a threshold is applied to the best available confidence at each stage:
- If the type reached the argument stage: use `argue_conf`
- If the type was screened but not argued: use `screen_conf`
- If the requirement was classified NFR directly (skipped debate): use initial `screen_conf`
- If the requirement was classified FR (skipped debate): score = 0

A single global threshold is LOPO-tuned to maximise macro F2. A per-type variant is also computed as an analytical upper bound.

---

## Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `da_threshold` | 0.7 | Devil's advocate FR confidence above this → challenge sustained to arbiter |
| `max_candidates` | 4 | Max NFR types forwarded from initial screen to advocate pipeline |
| `advocate_threshold` | 0.3 | Min screen confidence for an advocate to proceed to argument stage |
| Model | `claude-sonnet-4-6` | Default; override with `--model` |
| NoDebate θ sweep | 0.00–1.00, step 0.01 | Tuned per LOPO fold on training projects |
| NFR types | 9 | availability, legal, look-and-feel, maintainability, operational, performance, scalability, security, usability |

---

## Experimental Protocol

All three rules from `rules.txt` apply as follows:

1. **F2-based threshold optimisation on non-test data:**
   - Debate: not applicable — the arbiter makes direct label decisions with no tunable threshold.
   - NoDebate: threshold tuned inside the LOPO loop on 14 training projects per fold. The test project is never seen during tuning.
2. **Leave-one-project-out (LOPO)** — 15 folds, one per project. Both scripts implement LOPO evaluation.
3. **Requirement-level metrics** — TP/FP/FN accumulated per requirement across all folds before computing final recall, precision, and F2.

The F2 rationale: NFR detection is a retrieval task — a missed NFR (false negative) is more costly than flagging a non-NFR for review (false positive). F2 weights recall twice as heavily as precision.

---

## Results

After running, results are written to `results/` and `../allresults.txt` is updated automatically.

*(Populate after running on `promise-june2026.csv`.)*

---

## How to Run

### Step 1 — Run the debate pipeline (expensive, one-time)

```bash
cd agents/
pip install -r requirements.txt
python3 run_debate.py
```

This processes all 622 requirements through the multi-agent pipeline. Expect approximately 5,000–7,000 API calls to `claude-sonnet-4-6`. Runtime: 1–3 hours depending on rate limits.

The run is **resumable** — if interrupted, re-run the same command and it will pick up from the checkpoint. Use `--force` to restart from scratch.

Outputs written to `results/`:
- `summary_debate_da70_c4_at30.json` — aggregate metrics
- `per_type_debate_da70_c4_at30.csv` — per-type P/R/F1/F2
- `scores_debate_da70_c4_at30.json` — advocate confidence scores (needed for NoDebate)
- `transcripts_debate_da70_c4_at30.json` — full debate transcripts (citable artifact)
- `detail_debate_da70_c4_at30.csv` — per-requirement outcomes

`../allresults.txt` is updated with the `Agent Debate` row when the run completes.

**Environment variable required:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or add to a .env file in this directory
```

### Step 2 — Tune NoDebate thresholds (fast, no API calls)

```bash
python3 tune_nodebate.py --scores results/scores_debate_da70_c4_at30.json
```

This sweeps LOPO-tuned thresholds over the saved advocate scores. Runs in under a minute. Prints both global and per-type results; updates `../allresults.txt` with the `Agent Basic` (global threshold) row.

---

## Known Differences from Prior Work

**Architectural parameters are not LOPO-tuned.** `da_threshold` and `advocate_threshold` are fixed at their defaults. Tuning them would require re-running the full debate pipeline once per fold per parameter value, which is prohibitively expensive. They are treated as architectural constants, not F2-optimised thresholds.

**NoDebate bypasses the arbiter.** By applying a threshold directly to advocate scores, NoDebate uses only the advocate evidence stage (not the devil's advocate or arbiter reasoning). This is a deliberate design choice to compare a simple thresholded score against a full deliberative process.

**Dataset:** `promise-june2026.csv` (622 requirements, 15 projects) replaces the original PROMISE dataset used in prior runs.
