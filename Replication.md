# Replication Package

This package replicates five NFR classification techniques against a shared dataset ([`promise-june2026.csv`](./promise-june2026.csv)) under a common experimental protocol (LOPO cross-validation, F2-optimised thresholds). See [`DATA_CURATION.md`](./DATA_CURATION.md) for dataset provenance and label corrections.

---

## Techniques

### REJ — Indicator-Term Classifier

Classifies requirements by scoring them against weighted indicator terms mined per NFR type from a labeled training corpus; multi-label, threshold-based.

**Paper:** Cleland-Huang, J., Settimi, R., Zou, X., and Solc, P. "Automated Classification of Non-Functional Requirements." *Requirements Engineering Journal*, 12:103–120, 2007. [[link]](PLACEHOLDER)

**Replication:** [`rej-original/README.md`](./dataset/rej-original/README.md)

---

### P001 — NoRBERT

Fine-tunes BERT on the PROMISE NFR corpus using transfer learning to classify requirements into NFR subtypes.

**Paper:** Hey, T., Keim, J., Koziolek, A., and Tichy, W. "NoRBERT: Transfer Learning for Requirements Classification." *RE*, 2020. [[link]](PLACEHOLDER)

**Replication:** `p001-norbert/README.md` *(pending)*

---

### P003 — Zero-Shot Classification

Assigns each requirement to the NFR type whose label embedding has the highest cosine similarity to the requirement embedding; requires no labeled training data.

**Paper:** Alhoshan, W., Ferrari, A., and Zhao, L. "Zero-Shot Learning for Requirements Classification: An Exploratory Study." *Requirements Engineering*, 28:15–33, 2023. [[link]](PLACEHOLDER)

**Replication:** [`p003-zeroshot/README.md`](./p003-zeroshot/README.md)

---

### P011 — Classic ML (k-NN, SVM, Naive Bayes)

Three classifiers — a graph-based k-NN using dependency-parse structure, an SVM, and a Naive Bayes — applied to NFR subtype classification using either structural or bag-of-words features.

**Paper:** Slankas, J. and Williams, L. "Automated Extraction of Non-Functional Requirements in Available Documentation." *NaturaLiSE*, 2013. [[link]](PLACEHOLDER)

**Replication:** [`p011-classicML/README.md`](./p011-classicML/README.md)

---

### Agent Debate & Agent Basic

A multi-agent LLM pipeline in which advocate agents argue for candidate NFR types and an arbiter resolves disagreements (Debate); a simpler variant applies a tuned threshold directly to advocate confidence scores without arbitration (Basic).

**Paper:** *(this work)*

**Replication:** [`agents/README.md`](./agents/README.md)
