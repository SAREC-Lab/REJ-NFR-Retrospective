# Replicated Benchmarks

To investigate how automated NFR classification techniques have evolved over the past two decades, we selected representative approaches spanning five major methodological generations: indicator-term classification, traditional machine learning, transformer-based transfer learning, zero-shot learning, and modern multi-agent reasoning. Each selected paper used the PROMISE NFR dataset (or a direct derivative) and was sufficiently documented to permit replication.

For each approach, we analyzed the original publication, constructed a Python-based replication package, and reproduced the reported technique as faithfully as possible. Because prior studies differed in threshold optimization strategies, data partitioning schemes, and metric computation methods, we conducted two evaluations:

1. **Faithful Replication** – reproducing the methodology described in the original paper as closely as possible.
2. **Standardized Evaluation** – rerunning each technique under a common framework using:
   - F2-based threshold optimization
   - Leave-One-Project-Out (LOPO) validation
   - Requirement-level metric computation

This standardization reduces methodological variation unrelated to the underlying classification technique and enables more meaningful comparison across generations of approaches.

The selected techniques provide both methodological diversity and a view of the field's evolving capabilities, illustrating the progression from manually engineered features to modern language-model reasoning.

| ID | Year | Technique |
|----|------|-----------|
| P-REJ | 2007 | Indicator-term classification using TF-IDF-inspired weighting and threshold-based assignment |
| P011 | 2010 | Dependency-graph similarity with k-NN, SVM, and Naïve Bayes classifiers |
| P001 | 2020 | NoRBERT transformer-based transfer learning |
| P003 | 2023 | Zero-shot NFR classification using pretrained transformer embeddings and semantic similarity |
| P-AGENT | 2026 | Multi-agent expert-panel classification using specialized NFR agents, deliberation, and arbitration |

## Multi-Agent Expert Panel (P-AGENT)

In addition to replicating historical approaches, we developed a new multi-agent classification framework to provide a contemporary point of comparison representing current-generation LLM capabilities.

The approach combines:

- Specialized expert agents for each NFR category
- A functional-requirement (FR) advocate
- Structured NFR knowledge bases
- Selective debate for ambiguous requirements
- Independent arbitration
- Recall-oriented decision making

Unlike prior approaches based on feature engineering, supervised learning, transfer learning, or embedding similarity, this architecture explicitly reasons about competing interpretations of a requirement before assigning one or more NFR categories.

The design was inspired by recent advances in multi-agent debate and deliberation systems, which have demonstrated improvements in reasoning quality through structured interaction among specialized agents. However, unlike prior debate approaches that rely on generic debating agents, the expert-panel architecture employs domain-specific NFR specialists equipped with explicit classification criteria, boundary conditions, and category-specific knowledge. The resulting system combines expert reasoning, adversarial challenge, selective deliberation, and arbitration within a unified framework for NFR classification.

## Dataset

All experiments use the PROMISE NFR dataset family. For consistency across replications, experiments reported in this repository use the most recent Gokul et al. extension of the dataset while preserving compatibility with the original PROMISE NFR labels.

## Repository Structure

```text
replications/
├── p-rej/
├── p011/
│   ├── knn/
│   ├── svm/
│   └── naive_bayes/
├── p001-norbert/
├── p003-zeroshot/
└── p-agent/
```

Each replication directory contains:

- Source code
- Experiment configuration
- Dataset preparation scripts
- Execution instructions
- Generated results
- Replication notes

## References

- Cleland-Huang et al. (2006, 2007) – Indicator-term classification
- Casamayor et al. (2010) – Dependency-graph similarity and classical machine learning
- Hey et al. (2020) – NoRBERT transfer learning
- Alhoshan et al. (2023) – Zero-shot NFR classification
- This study (2026) – Multi-agent expert-panel classification
