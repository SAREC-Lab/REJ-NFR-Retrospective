# REJ NFR Classifier

**Paper:** Cleland-Huang, Settimi, Zou, and Solc. "Automated Classification of Non-Functional Requirements." *Requirements Engineering Journal*, 12:103–120, 2007.

---

## Technique Overview

An information-retrieval approach that automatically detects and classifies non-functional requirements (NFRs) into nine quality types. The classifier learns weighted *indicator terms* from a labeled training corpus and uses them to score unseen requirements against each NFR type.

### Phase 1 — Indicator Term Mining

For each NFR type `Q`, every term `t` in the training set receives a weight `Pr_Q(t)`:

```
Pr_Q(t) = [avg normalized TF of t in type-Q docs]
         × [fraction of docs containing t that are type-Q]
         × [fraction of type-Q projects in which t appears with a type-Q doc]
```

The three factors capture (1) how often the term appears in type-Q requirements, (2) how exclusive it is to that type, and (3) whether it generalizes across projects rather than being project-specific jargon. The **top 15 terms** per type are selected as indicator terms `I_Q`.

### Phase 2 — Scoring and Classification

For a preprocessed requirement `R` and type `Q`:

```
score_Q(R) = sum of Pr_Q(t) for t in (R ∩ I_Q)
           / sum of Pr_Q(t) for t in I_Q
```

`R` is classified as type `Q` if `score_Q(R) > θ`. This is **multi-label**: a requirement can be assigned to multiple types simultaneously. If no type exceeds θ, it is classified as `functional`.

### Preprocessing

Lowercase → tokenize (alpha only) → remove stopwords → Porter stem.

Stopwords: standard English list plus requirements-domain boilerplate (`shall`, `must`, `system`, `product`, `applic`, `abil`). Without these, all nine types select high-frequency domain words as top indicator terms, collapsing cross-type discriminability.

---

## Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `TOP_N` | 15 | Indicator terms selected per NFR type |
| `θ` (theta) | tuned per fold | Swept over [0.01, 0.30] in steps of 0.01; best macro F2 on training fold selected |
| NFR types | 9 | availability, legal, look-and-feel, maintainability, operational, performance, scalability, security, usability |
| Excluded types | fault-tolerance, portability, other | Insufficient examples in dataset |

---

## Experimental Protocol

All three rules from `rules.txt` apply:

1. **F2-based threshold optimisation on non-test data** — F2 is the shared optimisation target across all techniques in this replication because NFR detection is a retrieval task: missing an NFR (false negative) is more costly than flagging a non-NFR for review (false positive), so recall is weighted twice as heavily as precision. For each LOPO fold, θ is swept over the 14 training projects' scores. The θ that maximises macro F2 on training data is applied to the held-out project. The test fold is never seen during tuning.
2. **Leave-one-project-out (LOPO)** — 15 folds; each fold holds out one project as test set and trains on the remaining 14.
3. **Requirement-level metrics** — TP/FP/FN/TN are accumulated per requirement across all folds before computing final recall, precision, and F2.

---

## Results

**Dataset:** `promise-june2026.csv` — 622 requirements across 15 projects.

### Replication vs. Original Paper

| Type | Paper R | Paper P | Paper F2 | Repl R | Repl P | Repl F2 |
|------|--------:|--------:|---------:|-------:|-------:|--------:|
| Availability | 0.889 | 0.111 | 0.370 | 0.725 | 0.254 | 0.529 |
| Legal | 0.700 | 0.163 | 0.422 | 0.333 | 0.348 | 0.336 |
| Look-and-feel | 0.514 | 0.117 | 0.306 | 0.565 | 0.236 | 0.442 |
| Maintainability | 0.882 | 0.109 | 0.365 | 0.545 | 0.346 | 0.489 |
| Operational | 0.721 | 0.114 | 0.349 | 0.593 | 0.330 | 0.512 |
| Performance | 0.625 | 0.273 | 0.497 | 0.667 | 0.368 | 0.574 |
| Scalability | 0.722 | 0.111 | 0.344 | 0.552 | 0.172 | 0.383 |
| Security | 0.807 | 0.184 | 0.481 | 0.671 | 0.300 | 0.538 |
| Usability | 0.984 | 0.144 | 0.454 | 0.682 | 0.243 | 0.501 |
| **Macro avg** | **0.760** | **0.147** | **0.399** | **0.593** | **0.289** | **0.478** |

*Paper F2 computed from reported R and P. Replication uses LOPO theta tuning (θ* ranged 0.05–0.08 across folds).*

### Key observations

- Replication **F2 exceeds the paper** (0.478 vs. 0.399) despite lower macro recall (0.593 vs. 0.760), because the paper's fixed low threshold prioritised recall at the cost of precision. LOPO theta tuning finds a better F2 operating point.
- **Precision is substantially higher** (0.289 vs. 0.147), consistent with the updated dataset having cleaner labels and the threshold being optimised for F2 rather than recall.
- **Legal and scalability are the weakest types** in both versions — sparse training examples make indicator terms less reliable.

---

## How to Run

```bash
cd rej/
pip install -r requirements.txt
python3 run_rej_classifier.py
```

Raw per-fold results are written to `results/` (created automatically). Aggregate metrics are appended to `../allresults.txt`.

---

## Known Differences from the Original Paper

**Dataset:** `promise-june2026.csv` contains 622 requirements vs. the paper's 684, with revised per-type label counts (e.g., Legal: 31 vs. 10 in paper). This affects indicator term weights and per-type metrics.

**Threshold selection:** The paper used a fixed θ = 0.04 calibrated to its original dataset. This replication tunes θ inside the LOPO loop on training data only, which is unbiased with respect to the test fold and optimises F2 rather than recall.

**Precision denominators:** The 2006 conference version of this paper erroneously excluded functional requirements from precision denominators. This replication uses the corrected formula from the 2007 journal version: a functional requirement misclassified as type Q is counted as a false positive.
