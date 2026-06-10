# Dataset Curation Notes

## Overview

The PROMISE NFR dataset originates from a Requirements Engineering course taught by Jane Cleland-Huang at DePaul University. The course was aimed at Professional Masters students, approximately 80% of whom were working practitioners in industry. Using Robertson and Robertson's *Mastering the Requirements Process* as the primary text, students were required to produce a complete requirements specification as a final project, including examples of different non-functional requirement types. The resulting specifications form the basis of the PROMISE NFR corpus.

The dataset used in this replication study is `promise-june2026.csv`, a version of that corpus with a small number of label corrections applied. This document describes those corrections and the criteria used to make them.

## Dataset

**File:** [`promise-june2026.csv`](./promise-june2026.csv)

The CSV contains 622 requirements drawn from 15 projects in the PROMISE NFR corpus. Each row includes the original requirement text, a perturbed variant, binary flags for `IsFunctional` / `IsQuality`, and one binary column per NFR category:

| Column | Category |
|--------|----------|
| `Availability (A)` | Availability |
| `Fault Tolerance (FT)` | Fault Tolerance |
| `Legal (L)` | Legal / Regulatory |
| `Look & Feel (LF)` | Look & Feel |
| `Maintainability (MN)` | Maintainability |
| `Operability (O)` | Operability |
| `Performance (PE)` | Performance |
| `Portability (PO)` | Portability |
| `Scalability (SC)` | Scalability |
| `Security (SE)` | Security |
| `Usability (US)` | Usability |
| `Other (OT)` | Other |

## Label Corrections

During the course of inspecting the data and results we identified numerous ambiguous cases. We decided to keep all labels as-is unless the labeling was unambiguously incorrect. As a result, we identified 10 requirements to relabel. The two clearest sources of error were:

1. **Legal over-labeling** — requirements referencing *internal* corporate or community standards were incorrectly tagged `Legal`. The `Legal` category is reserved for external regulatory obligations (e.g., Sarbanes-Oxley, Regulation E/Z, HIPAA). Internal corporate guidelines, UI standards, and coding conventions do not qualify.

2. **Missing NFR labels** — a small number of requirements with clear non-functional characteristics (access control, fault tolerance, availability) were left unlabeled or assigned to the wrong category.

All corrections are listed in [`relabel_candidates.json`](./relabel_candidates.json) and summarized in the table below.

| ReqID | Project | Action | Rationale |
|-------|---------|--------|-----------|
| R148 | 4 | Remove `Legal` | Corporate UI standard, not an external regulation |
| R149 | 4 | Remove `Legal` | Corporate UI standard, not an external regulation |
| R188 | 4 | Add `Security` | Textbook access-control requirement (external vs. internal users) |
| R207 | 5 | Remove `Legal`, Add `Operability` | Corporate architecture guideline; governs the technical environment |
| R208 | 5 | Remove `Legal` | Internal corporate UI guideline |
| R209 | 5 | Remove `Legal` | Internal corporate color scheme |
| R445 | 8 | Remove `Legal` | PEAR is a PHP community coding standard, not a regulatory body |
| R566 | 12 | Add `Availability` | Fault tolerance / compensatory transactions define availability |
| R570 | 12 | Remove `Maintainability`, Add `Availability` | "No interruption in service during upgrades" is availability, not maintainability |
| R596 | 14 | Remove `Legal` | Role-based access control (Commissioner only); no external obligation |

## Relabel Candidates File

The full audit record, including the original text, current labels, suggested labels, action, confidence rating, and rationale for each change, is preserved in `relabel_candidates.json`:

```json
[
  {
    "req_id": "R207",
    "project": "5",
    "text": "The product shall adhere to the corporate Architecture guidelines",
    "current_labels": ["legal"],
    "suggested_labels": ["operational"],
    "action": "REMOVE legal, ADD operational",
    "confidence": "high",
    "rationale": "Explicitly an internal corporate guideline. No external regulatory or governing body involved. Corporate architecture guidelines govern the technical environment the product must fit into, which is Operational."
  },
  {
    "req_id": "R208",
    "project": "5",
    "text": "The product shall comply with corporate User Interface Guidelines",
    "current_labels": ["legal", "look-and-feel"],
    "suggested_labels": ["look-and-feel"],
    "action": "REMOVE legal",
    "confidence": "high",
    "rationale": "Explicitly an internal corporate UI guideline. Internal corporate standards are not Legal by any standard definition. Look-and-feel is correct and sufficient."
  },
  {
    "req_id": "R209",
    "project": "5",
    "text": "The product shall comply with corporate color scheme",
    "current_labels": ["legal", "look-and-feel"],
    "suggested_labels": ["look-and-feel"],
    "action": "REMOVE legal",
    "confidence": "high",
    "rationale": "Internal corporate color scheme is a visual style guideline, not a legal obligation. Look-and-feel is correct and sufficient."
  },
  {
    "req_id": "R148",
    "project": "4",
    "text": "The Disputes application shall comply with the corporate standards for user interface creation for internally and externally used applications.",
    "current_labels": ["legal", "look-and-feel"],
    "suggested_labels": ["look-and-feel"],
    "action": "REMOVE legal",
    "confidence": "high",
    "rationale": "The standard is explicitly described as a corporate (internal) standard. The phrase 'internally and externally used applications' describes the scope of the applications covered, not the source of the standard. Look-and-feel is correct."
  },
  {
    "req_id": "R149",
    "project": "4",
    "text": "All screens created as part of the Disputes application must comply with corporate standards for interface creation.",
    "current_labels": ["legal", "look-and-feel"],
    "suggested_labels": ["look-and-feel"],
    "action": "REMOVE legal",
    "confidence": "high",
    "rationale": "Same as R148 — explicitly a corporate (internal) standard for interface creation. Look-and-feel is correct."
  },
  {
    "req_id": "R596",
    "project": "14",
    "text": "The Commissioner will be the only authorized user per league to adjust league settings.",
    "current_labels": ["legal", "security"],
    "suggested_labels": ["security"],
    "action": "REMOVE legal",
    "confidence": "high",
    "rationale": "Pure access control requirement — defines who is authorised to perform an action. Security is correct. There is no reference to any law, regulation, governing body, or external obligation."
  },
  {
    "req_id": "R445",
    "project": "8",
    "text": "The PHP code will comply with PEAR standards.",
    "current_labels": ["legal", "security"],
    "suggested_labels": ["security"],
    "action": "REMOVE legal",
    "confidence": "high",
    "rationale": "PEAR (PHP Extension and Application Repository) is a PHP community coding standards project, not a regulatory or governing body. Compliance with it is a coding practice, not a legal obligation. Security may be defensible as PEAR includes security guidelines."
  },
  {
    "req_id": "R188",
    "project": "4",
    "text": "The Disputes System must prevent external users from requesting original receipts. Requests for original receipts are restricted to internal users.",
    "current_labels": [],
    "suggested_labels": ["security"],
    "action": "ADD security",
    "confidence": "high",
    "rationale": "Textbook access control requirement — restricts which users (external vs internal) may perform an action. Currently labeled as a functional requirement, but defining who is authorised to access what is the core definition of security."
  },
  {
    "req_id": "R566",
    "project": "12",
    "text": "The product shall be robust with fault tolerance. The product shall have fault tolerance by using recovery technique compensatory transaction and routing around failures.",
    "current_labels": [],
    "suggested_labels": ["availability"],
    "action": "ADD availability",
    "confidence": "high",
    "rationale": "Fault tolerance, compensatory transactions, and routing around failures are the core definition of availability. The KB explicitly lists fault tolerance and graceful degradation as availability criteria. Currently unlabeled (treated as functional), which is clearly wrong."
  },
  {
    "req_id": "R570",
    "project": "12",
    "text": "The product shall continue to operate during upgrade change or new resource addition. The product shall be able to continue to operate with no interruption in service due to new resource additions.",
    "current_labels": ["maintainability"],
    "suggested_labels": ["availability"],
    "action": "REMOVE maintainability, ADD availability",
    "confidence": "high",
    "rationale": "'No interruption in service' and 'continue to operate during upgrades' define availability, not maintainability. Maintainability is about how easy it is to make changes; availability is about staying up while changes happen. The KB explicitly draws this boundary: the existence of a maintenance window (when downtime is permitted) is Availability."
  }
]
