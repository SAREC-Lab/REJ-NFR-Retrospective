## Impact Study: Two Decades of Research on Automated NFR Classification

This repository contains the replication package, datasets, analysis artifacts, and supporting materials for our retrospective study of *Automated Classification of Non-Functional Requirements*. The study examines the influence of the original work over the past two decades, analyzes the evolution of automated requirements classification techniques, and replicates representative approaches spanning multiple generations of machine learning and AI technology.

## Key Publications
| Year | Publication | Links |
|------|-------------|-------|
| 2026 | Cleland-Huang, J. and Vierhauser, M. *Automated Classification of Non-Functional Requirements Twenty Years Later and Why It Remains an Unsolved Problem in the Age of LLMs.* *Requirements Engineering* (Special Retrospective Edition: *30 Years of Requirements Engineering*). | Paper: Coming Soon<br>BibTeX: [REJ-Retrospective-2026.bib](references/REJ-Retrospective-2026.bib) |
| 2007 | Cleland-Huang, J., Settimi, R., Zou, X., and Solc, P. *Automated Classification of Non-Functional Requirements.* *Requirements Engineering*, 12(2), 103–120. | [Paper](https://doi.org/10.1007/s00766-007-0045-1)<br>[BibTeX](references/REJ-2007.bib) |
| 2006 | Cleland-Huang, J., Settimi, R., Zou, X., and Solc, P. *The Detection and Classification of Non-Functional Requirements with Application to Early Aspects.* In *Proceedings of the 14th IEEE International Conference on Requirements Engineering (RE 2006)*, pp. 36–45. | [Paper](https://doi.org/10.1109/RE.2006.65)<br>[BibTeX](references/RE-2006.bib) |

## Overview

The original work introduced one of the earliest machine-learning approaches for identifying and classifying non-functional requirements (NFRs) and released the PROMISE NFR dataset, which subsequently became one of the most widely used benchmarks in Requirements Engineering research. Over the following two decades, the dataset and the underlying research problem inspired a diverse body of work spanning classical machine learning, deep learning, transfer learning, transformer-based models, and, more recently, Large Language Models (LLMs).

The goal of this study is to understand how that work influenced subsequent research and how the field has evolved over the past twenty years. To achieve this, we conducted a systematic citation analysis of papers citing the original publication, investigated how researchers reused and extended the PROMISE dataset, tracked the evolution of requirements-classification techniques, and replicated representative approaches from different generations of technology.

This repository provides the artifacts used in that analysis, including:

* Metadata and references for analyzed papers
* Citation impact and influence analyses
* Thematic coding and codebooks
* PROMISE dataset lineage and extensions
* Replication scripts and experimental infrastructure
* Benchmarking results across representative approaches
* Generated prompts, intermediate artifacts, and supporting materials
* Reproduction instructions for all analyses and experiments

Together, these artifacts document how a single research contribution evolved into a foundational benchmark that shaped two decades of work in requirements classification, requirements traceability, quality requirements engineering, and AI-assisted software engineering.

## Repository Contents

| Directory          | Description                                             |
| ------------------ | ------------------------------------------------------- |
|[Analyzed Papers](papers/analysis.md)| References and metadata for analyzed papers             |
| `impact-analysis/` | Citation analysis results and influence classifications |
| `datasets/`        | PROMISE datasets and related resources                  |
| [Replicated Techniques](replication/overview.md)| Replication implementations of selected approaches      |


## Citation Corpus

The complete citation corpus analyzed in this study is provided through the `papers/` and `references/` directories. Each paper is assigned a stable identifier (`p001`, `p002`, ...), together with:

* Full bibliographic reference
* DOI or persistent URL
* Citation metadata
* Influence classification
* Associated BibTeX entry

These resources are intended to support replication, secondary analysis, and future studies of the evolution of Requirements Engineering research.
