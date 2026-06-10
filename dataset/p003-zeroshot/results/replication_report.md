# Zero-Shot NFR Classification — Replication Report

## Reference
Alhoshan, W., Ferrari, A., Zhao, L. (2023). *Zero-Shot Learning for Requirements
Classification: An Exploratory Study.*

## Scope
- Task: multi-class NFR type classification
- Dataset: promise-june2026.csv (RequirementText column)
- NFR requirements classified: 366
- Classes (9): availability, legal, look-and-feel, maintainability, operational, performance, scalability, security, usability
- No training, no fine-tuning, no cross-validation
- Single-label classifier: each requirement assigned to nearest class by cosine similarity

## Models
- MiniLM: sentence-transformers/all-MiniLM-L12-v2
- MPNet:  sentence-transformers/all-mpnet-base-v2

## Label Configurations
- **original** (Config A): plain class names (equivalent to MultiNFR_A, extended to 9 classes)
- **expert**   (Config B): expert-curated labels from authors' GitHub appendix (equivalent to MultiNFR_B)

## Class Distribution

- availability: 40
- legal: 24
- look-and-feel: 64
- maintainability: 25
- operational: 50
- performance: 48
- scalability: 17
- security: 62
- usability: 36

## Summary Results (macro-averaged)

| Model | Labels | Precision | Recall | F1 | F2 | Specificity |
| ----- | ------ | --------- | ------ | -- | -- | ----------- |
| minilm | original | 0.4198 | 0.3940 | 0.3360 | 0.3589 | 0.9212 |
| minilm | expert | 0.4869 | 0.4523 | 0.4334 | 0.4364 | 0.9327 |
| mpnet | original | 0.3986 | 0.3630 | 0.3032 | 0.3219 | 0.9196 |
| mpnet | expert | 0.5012 | 0.4853 | 0.4636 | 0.4698 | 0.9387 |

## Per-Class F2

### minilm / original

| Class | F2 | Support |
| ----- | -- | ------- |
| availability | 0.5185 | 40 |
| legal | 0.4074 | 24 |
| look-and-feel | 0.1145 | 64 |
| maintainability | 0.3929 | 25 |
| operational | 0.1142 | 50 |
| performance | 0.2045 | 48 |
| scalability | 0.4762 | 17 |
| security | 0.6667 | 62 |
| usability | 0.3352 | 36 |

### minilm / expert

| Class | F2 | Support |
| ----- | -- | ------- |
| availability | 0.4025 | 40 |
| legal | 0.5036 | 24 |
| look-and-feel | 0.4371 | 64 |
| maintainability | 0.5785 | 25 |
| operational | 0.3333 | 50 |
| performance | 0.1316 | 48 |
| scalability | 0.3333 | 17 |
| security | 0.8663 | 62 |
| usability | 0.3416 | 36 |

### mpnet / original

| Class | F2 | Support |
| ----- | -- | ------- |
| availability | 0.2865 | 40 |
| legal | 0.2273 | 24 |
| look-and-feel | 0.2230 | 64 |
| maintainability | 0.3608 | 25 |
| operational | 0.0000 | 50 |
| performance | 0.1733 | 48 |
| scalability | 0.4464 | 17 |
| security | 0.7362 | 62 |
| usability | 0.4439 | 36 |

### mpnet / expert

| Class | F2 | Support |
| ----- | -- | ------- |
| availability | 0.4605 | 40 |
| legal | 0.5172 | 24 |
| look-and-feel | 0.6487 | 64 |
| maintainability | 0.4924 | 25 |
| operational | 0.2915 | 50 |
| performance | 0.1878 | 48 |
| scalability | 0.3738 | 17 |
| security | 0.8859 | 62 |
| usability | 0.3704 | 36 |

## Environment

- Python: 3.12.3
- sentence-transformers: 5.5.1
- transformers: 5.10.2
- torch: 2.12.0+cu130
- scikit-learn: 1.9.0
- random seed: 42
