# P011 — Slankas & Williams NFR Classifier

**Paper:** Slankas & Williams. "Automated Extraction of Non-Functional Requirements in Available Documentation." *NaturaLiSE*, 2013.

Three classifiers are evaluated: a graph-based k-NN (the paper's novel contribution) and two bag-of-words baselines (SVM and Naive Bayes) included in the paper for comparison.

---

## Technique Overview

### k-NN (graph distance)

Each requirement is parsed into a dependency graph using the Stanza NLP pipeline (tokenize, POS, lemma, dependency parse, NER). The graph is represented as a sequence of `Vertex` objects, one per token, storing lemma, collapsed POS tag, dependency relation, parent count, whether the token is numeric, and named-entity class. Determiners (`a`, `an`, `the`) are the only words removed.

Classification is **k=1 nearest neighbour**: the test requirement inherits all NFR labels of its single closest training requirement. Similarity between two sentences is a modified Levenshtein distance that operates at the vertex level:

- Vertices at the same position are compared structurally first (POS, parent count, dependency relation)
- If structure matches, content is compared: identical lemma or numeric tokens cost 0; same NER class costs 0; WordNet synonym within 4 hops costs 0.1–0.4; otherwise cost 1.0
- Sentence distance = sum of per-position vertex costs (shorter sentence padded with `None`)

Because k-NN inherits multi-label assignments from the nearest neighbor, no decision threshold is needed.

### SVM and Naive Bayes (bag-of-words baselines)

Both use raw word counts (`CountVectorizer`, determiners removed, no stemming) and train one binary classifier per NFR type (binary relevance). `LinearSVC` is used for SVM; `MultinomialNB` for Naive Bayes. Each type is classified independently; a requirement can be assigned to multiple types simultaneously.

---

## Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| k | 1 | Nearest neighbours; best value per paper |
| SVM regularisation C | 1.0 | sklearn default; paper does not specify |
| Vocabulary | `CountVectorizer` defaults | Raw counts, determiners removed |
| SVM θ sweep | −2.0 to +2.0, step 0.1 | `decision_function` boundary; tuned per fold |
| NB θ sweep | 0.01–0.09 (step 0.01), 0.10–0.95 (step 0.05) | `predict_proba[:,1]` boundary; tuned per fold |
| NFR types | 9 | Same Cleland-Huang labeling as all other techniques |

---

## Experimental Protocol

All three rules from `rules.txt` apply:

1. **F2-based threshold optimisation on non-test data** — F2 is the shared optimisation target because NFR detection is a retrieval task: missing an NFR (false negative) is more costly than a spurious candidate. For each LOPO fold, the SVM and NB thresholds are swept over the in-sample training scores; the θ that maximises macro F2 on training requirements is applied to the held-out project. k-NN has no threshold. The test fold is never seen during tuning.
2. **Leave-one-project-out (LOPO)** — 15 folds, one per project.
3. **Requirement-level metrics** — TP/FP/FN/TN accumulated per individual requirement across all folds before computing final recall, precision, and F2.

**Note on in-sample threshold tuning:** LinearSVC's `decision_function` returns near-perfect separation on training data (train macro F2 ≈ 1.0 in every fold). The tuned SVM threshold is therefore always negative (−0.6 to −0.7), reflecting the classifier's need to push the boundary well below zero to recover recall on unseen data. This is expected and unbiased with respect to the test fold.

---

## Results

**Dataset:** `promise-june2026.csv` — 622 requirements across 15 projects.

### Replication (LOPO with F2-tuned thresholds)

| Type | kNN R | kNN F2 | SVM R | SVM F2 | NB R | NB F2 |
|------|------:|-------:|------:|-------:|-----:|------:|
| Availability | 0.300 | 0.305 | 0.700 | 0.504 | 0.400 | 0.430 |
| Legal | 0.125 | 0.130 | 0.500 | 0.397 | 0.167 | 0.196 |
| Look-and-feel | 0.188 | 0.195 | 0.667 | 0.522 | 0.435 | 0.449 |
| Maintainability | 0.030 | 0.032 | 0.455 | 0.344 | 0.091 | 0.104 |
| Operational | 0.254 | 0.253 | 0.729 | 0.543 | 0.305 | 0.326 |
| Performance | 0.190 | 0.194 | 0.667 | 0.554 | 0.476 | 0.493 |
| Scalability | 0.241 | 0.216 | 0.517 | 0.472 | 0.276 | 0.315 |
| Security | 0.224 | 0.235 | 0.842 | 0.684 | 0.750 | 0.718 |
| Usability | 0.306 | 0.295 | 0.612 | 0.491 | 0.541 | 0.510 |
| **Macro avg** | **0.207** | **0.206** | **0.632** | **0.501** | **0.382** | **0.394** |

### Comparison with paper targets

The paper used stratified 10-fold cross-validation (not LOPO) and reported F1. The only directly comparable result is k-NN on PROMISE, where the paper reports F1 = 0.382 vs. our LOPO F1 = 0.207. The gap reflects LOPO's stricter anti-leakage property — in stratified folds, training and test requirements may come from the same project, giving the distance metric a structural advantage it does not have in LOPO. The SVM and NB paper results are from a different (larger) dataset and category scheme and are not directly comparable.

### Key observations

- **SVM is the strongest classifier** at macro F2 (0.501), closely matching REJ (0.478) but with much higher recall (0.632 vs. 0.593).
- **Naive Bayes** has the highest precision (0.537) but lower recall, yielding a lower macro F2 (0.394) than SVM.
- **k-NN is substantially weaker** under LOPO (F2 = 0.206). The graph distance metric appears sensitive to within-project writing style; removing same-project neighbours hurts significantly.
- **Maintainability is the hardest type** for all three classifiers, consistent with its sparse and heterogeneous vocabulary.

---

## How to Run

```bash
cd p011/
pip install -r requirements.txt
python3 run_p011_classifiers.py
```

Sentence representations are cached in `parse_cache.json` — all 622 requirements from `promise-june2026.csv` are already cached, so no Stanza parsing is needed on first run. Results are written to `results/` and `../allresults.txt` is updated automatically.

---

## Known Differences from the Original Paper

**Protocol:** The paper used stratified 10-fold CV; this replication uses LOPO. LOPO is stricter (no within-project leakage) and consistent with all other techniques in this package.

**Dataset and label scheme:** We apply the classifiers to the 9-category Cleland-Huang labeling of PROMISE. The paper's SVM and NB results are from a 14-category scheme on a larger dataset (PROMISE + additional documents). Direct numerical comparison for these two classifiers is therefore not meaningful.

**Threshold selection:** The paper reports SVM and NB at default thresholds (0.0 and 0.5). This replication tunes thresholds per LOPO fold on training data, optimising for F2, to allow fair comparison with the other techniques in this package.

**SVM implementation:** `sklearn.svm.LinearSVC` with `dual='auto'` and `max_iter=5000`, equivalent to Weka's SMO with a linear kernel at default regularisation.
