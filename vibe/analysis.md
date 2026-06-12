# Conversational Quality Concerns Dataset (Exploratory)

This artifact contains an exploratory dataset of quality-related developer prompts extracted from the DevGPT corpus of developer–ChatGPT conversations.

## Overview

As part of our retrospective study on automated classification of non-functional requirements (NFRs), we investigated whether quality concerns continue to emerge in modern AI-assisted software development workflows. To support this exploration, we applied an experimental agent-based classifier to developer prompts contained within the DevGPT dataset and extracted prompts that appeared to discuss software quality concerns.

The resulting dataset contains conversational fragments that were automatically tagged with one or more quality categories, together with confidence scores and classifier rationale.

## Important Disclaimer

This dataset is **exploratory** and the labels should be considered **preliminary**.

The quality categories were assigned automatically using an experimental classification pipeline and **have not undergone comprehensive manual validation**. Consequently:

- Labels may contain false positives and false negatives.
- Some prompts classified as NFR-related may be better characterized as architectural discussions, implementation decisions, debugging activities, design rationale, or code review comments.
- Category assignments should not be interpreted as ground-truth annotations.

The purpose of releasing this artifact is to support transparency, reproducibility, and future research into quality concerns expressed within developer–LLM conversations.

## Motivation

Traditional NFR classification research has focused on requirements artifacts such as requirements specifications, issue reports, app reviews, and user stories [1,2]. Developer–LLM conversations present a fundamentally different type of artifact in which requirements, architecture, implementation, debugging, and design rationale frequently appear within the same conversational context [3].

Our preliminary analysis suggests that quality concerns remain pervasive in AI-assisted development environments, but are often expressed conversationally rather than as explicit requirements statements. Examples include discussions of:

- performance ("it needs to be snappy-ish")
- scalability
- reliability and resilience
- maintainability and refactoring
- portability
- usability
- security

This artifact is intended as an initial step toward understanding how quality concerns emerge and evolve within conversational software engineering workflows.

## File Format

Each entry contains:

- Conversation identifier
- Source metadata
- Developer prompt text
- Automatically assigned quality categories
- Confidence score
- Classifier rationale

The schema may evolve as additional validation and analysis are performed.

## Future Work

We are currently conducting a more detailed analysis of these conversational artifacts to better understand:

1. How quality concerns are expressed in developer–LLM interactions.
2. The relationship between conversational quality concerns, architectural decisions, and implementation activities.
3. The limitations of existing NFR classification approaches when applied to conversational artifacts.
4. Opportunities for developing NFR-aware conversational agents that can proactively identify and surface architecturally significant quality concerns.

Researchers are welcome to use this artifact for exploratory studies, replication, and follow-on research.

## References

### [1] Original REJ Paper

Cleland-Huang, J., Settimi, R., Zou, X., and Solc, P.  
**The Detection and Classification of Non-Functional Requirements with Application to Early Aspects.**  
*Requirements Engineering*, 12(2), 103–120, 2007.

```bibtex
@article{clelandhuang2007nfr,
  author = {Cleland-Huang, Jane and Settimi, Raffaella and Zou, Xuchang and Solc, Peter},
  title = {The Detection and Classification of Non-Functional Requirements with Application to Early Aspects},
  journal = {Requirements Engineering},
  year = {2007},
  volume = {12},
  number = {2},
  pages = {103--120}
}
```

### [2] Retrospective Study

Cleland-Huang, J. and Vierhauser, M.  
**Automated Classification of Non-Functional Requirements Twenty Years Later and Why This Isn't a Solved Problem in the Age of LLMs.**  
*Requirements Engineering*, 2026.

```bibtex
@article{clelandhuang2026retrospective,
  author = {Cleland-Huang, Jane and Vierhauser, Michael},
  title = {Automated Classification of Non-Functional Requirements Twenty Years Later and Why This Isn't a Solved Problem in the Age of LLMs},
  journal = {Requirements Engineering},
  year = {2026}
}
```

### [3] DevGPT Dataset

Xiao, T., Treude, C., Hata, H., and Matsumoto, K.  
**DevGPT: Studying Developer-ChatGPT Conversations.**  
Proceedings of the 21st International Conference on Mining Software Repositories (MSR), 2024, pp. 227–230.

```bibtex
@inproceedings{xiao2024devgpt,
  author = {Xiao, Tao and Treude, Christoph and Hata, Hideaki and Matsumoto, Kenichi},
  title = {DevGPT: Studying Developer-ChatGPT Conversations},
  booktitle = {Proceedings of the 21st International Conference on Mining Software Repositories},
  year = {2024},
  pages = {227--230},
  doi = {10.1145/3643991.3648400}
}
```
