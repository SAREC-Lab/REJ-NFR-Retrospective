# P003 — Zero-Shot NFR Classification

**Paper:** Alhoshan, W., Ferrari, A., & Zhao, L. (2023). Zero-Shot Learning for Requirements Classification: An Exploratory Study. *Requirements Engineering*, 28, 15–33.

---

## Technique Overview

This technique classifies NFRs into quality types using zero-shot learning — no labelled training data, no fine-tuning, and no cross-validation. Requirements and class labels are each encoded as sentence embeddings using a pre-trained SentenceTransformer model. Each requirement is assigned the class whose label embedding has the highest cosine similarity to the requirement embedding.

The approach is purely inference-time: once label embeddings are precomputed, every requirement is classified in a single forward pass with no parameter updates. Because there is no decision threshold — the prediction is always the argmax over cosine similarities — no F2-based tuning is applied.

**Important:** This is a **single-label** classifier. Each requirement is assigned exactly one NFR type (the closest label by cosine similarity). This differs from the multi-label setup used by REJ, P011, NoRBERT, and the agent-based techniques.

### Models evaluated

Two SentenceTransformer models are evaluated:

- `all-MiniLM-L12-v2` — compact 12-layer model, fast
- `all-mpnet-base-v2` — larger model, stronger sentence representations

### Label configurations

Two label configurations are evaluated:

- **Original (MultiNFR\_A):** plain class names (e.g., "security", "usability")
- **Expert-curated (MultiNFR\_B):** descriptive phrases authored by domain experts and published in the paper's supplementary materials (e.g., "security, authorization, or protection"; "instructive, easy, helpful, useful, learnable…"). Sourced verbatim from the authors' appendix: <https://github.com/waadalhoshan/ZSL4REQ/tree/main/Appendix>

The best-performing variant — MPNet with expert-curated labels — is reported in the comparison table (`allresults.txt`).

---

## Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Models | `all-MiniLM-L12-v2`, `all-mpnet-base-v2` | Both evaluated; best reported |
| Label configuration | expert-curated (MultiNFR\_B) | Best variant per original paper |
| Decision rule | argmax cosine similarity | No threshold; no tuning needed |
| NFR types | 9 | availability, legal, look-and-feel, maintainability, operational, performance, scalability, security, usability |
| Excluded types | fault-tolerance, portability, other | Insufficient examples in dataset |
| Random seed | 42 | Reproducibility only; experiment is deterministic |

---

## Experimental Protocol

The three rules from `rules.txt` apply as follows:

1. **F2-based threshold optimisation** — not applicable. Zero-shot prediction uses argmax cosine similarity; there is no threshold parameter to tune.
2. **LOPO cross-validation** — not applicable. There are no trainable parameters, so no hold-out is needed. All requirements are classified in a single pass.
3. **Requirement-level metrics** — applied. Macro precision, recall, F1, and F2 are computed over all NFR requirements after classification.

The rationale for reporting F2 (rather than F1) remains: NFR detection is a retrieval task in which missing an NFR (false negative) is more costly than a false alarm. F2 weights recall twice as heavily as precision.

---

## Results

**Dataset:** `promise-june2026.csv` — NFR requirements only (excludes FT and PO classes). Primary label derived by first-active-column priority for any multi-label requirements.

### All four variants (macro-averaged)

| Model | Label config | Precision | Recall | F1 | F2 |
|-------|-------------|-----------|--------|-----|-----|
| MiniLM | original | — | — | 0.34 | 0.36 |
| MiniLM | expert    | — | — | 0.44 | 0.44 |
| MPNet  | original  | — | — | 0.31 | 0.33 |
| MPNet  | **expert** | — | — | **0.45** | **0.46** |

*Original paper results (364 requirements, original PROMISE dataset). Cells left blank for P/R not reported in the paper.*

### Comparison with paper (best variant: MPNet + expert)

Results for the current dataset (`promise-june2026.csv`, 622 requirements) are filled in after running:

| Metric | Paper (orig. dataset) | Replication |
|--------|----------------------|-------------|
| Precision | 0.49 | — |
| Recall | 0.48 | — |
| F1 | 0.45 | — |
| F2 | 0.46 | — |

*(Run `python3 run_zeroshot.py` to populate replication column and update `allresults.txt`.)*

### Per-class F2 (MPNet + expert, best variant)

| Class | Paper F2 | Support (orig.) |
|-------|----------|-----------------|
| Availability | 0.48 | 38 |
| Legal | 0.42 | 31 |
| Look & Feel | 0.62 | 60 |
| Maintainability | 0.51 | 26 |
| Operational | 0.30 | 49 |
| Performance | 0.19 | 48 |
| Scalability | 0.38 | 17 |
| Security | 0.88 | 59 |
| Usability | 0.37 | 36 |

---

## How to Run

```bash
cd zeroshot/
pip install -r requirements.txt
python3 run_zeroshot.py
```

The script classifies all four variants (2 models × 2 label configs), writes per-predictions CSVs and confusion matrix PNGs/CSVs to `results/`, and updates the `P003-ZeroShot` row in `../allresults.txt` with the MPNet + expert results. Runtime is a few minutes (no GPU required).

---

## Known Differences from the Original Paper

**Single-label only:** The original paper also explored binary one-vs-rest and label-generation experiments. This replication covers only multi-class NFR type classification (single-label argmax), the task directly comparable to the other techniques in this package.

**Dataset:** `promise-june2026.csv` contains 622 requirements vs. the original 364-requirement PROMISE subset used in the paper. The updated dataset has revised label counts for some types, which affects per-class metrics.

**Excluded types:** Fault Tolerance and Portability are excluded (insufficient examples). The paper likewise excluded these classes from its 9-class experiment.

**Primary-label derivation:** For any requirement with multiple active NFR labels, the first active column determines the class (availability → legal → look-and-feel → … → usability). This is consistent with all other techniques in this replication package.
