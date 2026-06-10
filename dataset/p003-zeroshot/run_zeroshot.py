#!/usr/bin/env python3
"""
Zero-shot NFR classification replication.

Alhoshan, Ferrari, Zhao (2023). "Zero-Shot Learning for Requirements
Classification: An Exploratory Study."

Replication scope: multi-class NFR classification on PROMISE dataset,
9 classes (excludes portability and fault-tolerance), two SentenceTransformer
models, two label configurations (original and expert-curated).

Protocol (rules.txt):
  No LOPO required — zero-shot classifiers have no trainable parameters or
  decision thresholds. All requirements are classified in a single pass.
  No threshold tuning is needed: prediction is argmax cosine similarity.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
)

# ── reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_CSV        = Path('../promise-june2026.csv')
RESULTS_DIR     = Path('results')
ALLRESULTS_PATH = Path('../allresults.txt')
RESULTS_DIR.mkdir(exist_ok=True)

# ── label column order ─────────────────────────────────────────────────────────
LABEL_COL_ORDER = [
    ("Availability (A)",     "A"),
    ("Fault Tolerance (FT)", "FT"),
    ("Legal (L)",            "L"),
    ("Look & Feel (LF)",     "LF"),
    ("Maintainability (MN)", "MN"),
    ("Operability (O)",      "O"),
    ("Performance (PE)",     "PE"),
    ("Portability (PO)",     "PO"),
    ("Scalability (SC)",     "SC"),
    ("Security (SE)",        "SE"),
    ("Usability (US)",       "US"),
]

# abbreviation → canonical class name
ABBR_TO_CLASS = {
    "A":  "availability",
    "L":  "legal",
    "LF": "look-and-feel",
    "MN": "maintainability",
    "O":  "operational",
    "PE": "performance",
    "SC": "scalability",
    "SE": "security",
    "US": "usability",
}

# classes to include in the 9-class experiment (excludes FT and PO)
TARGET_CLASSES = list(ABBR_TO_CLASS.values())

# Column order matching allresults.txt: A L LF MN O PE SC SE US
_RESULT_TYPE_ORDER = [
    'availability', 'legal', 'look-and-feel', 'maintainability',
    'operational', 'performance', 'scalability', 'security', 'usability',
]

# ── label configurations ───────────────────────────────────────────────────────
# Config A: original labels (simple class names).
# Equivalent to MultiNFR_A extended to all 9 classes.
LABELS_A = {
    "availability":    "availability",
    "legal":           "legal",
    "look-and-feel":   "look and feel",
    "maintainability": "maintainability",
    "operational":     "operational",
    "performance":     "performance",
    "scalability":     "scalability",
    "security":        "security",
    "usability":       "usability",
}

# Config B: expert-curated labels (exact text from authors' GitHub appendix).
# Equivalent to MultiNFR_B extended to all 9 classes.
# Source: https://github.com/waadalhoshan/ZSL4REQ/tree/main/Appendix
LABELS_B = {
    "availability":    "avaliable or timely achievable",
    "legal":           "legal, law, or rules",
    "look-and-feel":   "appearance, interface, look and feel",
    "maintainability": "maintaining, fixing, running or updating",
    "operational":     "working, running, connecting, interfacing, or operative environment",
    "performance":     "periodic execution or efficacy performance",
    "scalability":     "scalable, increasable or developable",
    "security":        "security, authorization, or protection",
    "usability":       "instructive, easy, helpful, useful, learnable, explainable, affordable, intuitive, or understandable",
}

LABEL_CONFIGS = {
    "original": LABELS_A,
    "expert":   LABELS_B,
}

# ── model definitions ──────────────────────────────────────────────────────────
MODEL_IDS = {
    "minilm": "sentence-transformers/all-MiniLM-L12-v2",
    "mpnet":  "sentence-transformers/all-mpnet-base-v2",
}


# ── data loading ───────────────────────────────────────────────────────────────

def derive_primary_label(row: pd.Series) -> str | None:
    """Return the first active NFR column as class name, or None to exclude."""
    for col, abbr in LABEL_COL_ORDER:
        if str(row.get(col, "0")).strip() == "1":
            if abbr in ABBR_TO_CLASS:
                return ABBR_TO_CLASS[abbr]
            return None   # FT or PO — exclude
    return None


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["requirement_text"] = df["RequirementText"].str.strip().str.strip("'")
    df["project_id"] = df["ProjectID"].astype(str)
    df["requirement_id"] = df["ReqID"].astype(str)

    nfr_mask = df["IsQuality"] == 1
    df_nfr = df[nfr_mask].copy()

    labels = df_nfr.apply(derive_primary_label, axis=1)
    df_nfr = df_nfr.assign(true_label=labels)
    df_nfr = df_nfr[df_nfr["true_label"].notna()].copy()
    df_nfr = df_nfr[df_nfr["true_label"].isin(TARGET_CLASSES)].copy()

    return df_nfr[["requirement_id", "project_id", "requirement_text", "true_label"]].reset_index(drop=True)


# ── classification ─────────────────────────────────────────────────────────────

def embed_labels(model: SentenceTransformer, label_config: dict[str, str]) -> dict[str, np.ndarray]:
    classes = TARGET_CLASSES
    texts   = [label_config[c] for c in classes]
    vecs    = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return {c: vecs[i] for i, c in enumerate(classes)}


def classify(
    model: SentenceTransformer,
    df: pd.DataFrame,
    label_config: dict[str, str],
) -> pd.DataFrame:
    label_embs = embed_labels(model, label_config)
    classes    = TARGET_CLASSES

    req_embs = model.encode(
        df["requirement_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )

    label_matrix = np.stack([label_embs[c] for c in classes])  # (n_classes, dim)
    scores       = req_embs @ label_matrix.T                    # (n_req, n_classes)

    predicted_labels  = [classes[i] for i in scores.argmax(axis=1)]
    similarity_scores = scores.max(axis=1).tolist()

    result = df.copy()
    result["predicted_label"]  = predicted_labels
    result["similarity_score"] = similarity_scores
    return result


# ── evaluation ─────────────────────────────────────────────────────────────────

def _macro_specificity(y_true: list, y_pred: list) -> float:
    """Macro-averaged specificity (TN / (TN+FP)) across all classes, one-vs-rest."""
    cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
    specs = []
    for i in range(len(TARGET_CLASSES)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        denom = tn + fp
        specs.append(tn / denom if denom > 0 else 0.0)
    return float(np.mean(specs))


def evaluate(df: pd.DataFrame, model_name: str, config_name: str) -> dict:
    y_true = df["true_label"].tolist()
    y_pred = df["predicted_label"].tolist()

    mac_p, mac_r, mac_f1, supports = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=TARGET_CLASSES, zero_division=0
    )
    mac_f2_per_class = fbeta_score(
        y_true, y_pred, beta=2, average=None, labels=TARGET_CLASSES, zero_division=0
    )

    overall_p  = float(np.mean(mac_p))
    overall_r  = float(np.mean(mac_r))
    overall_f1 = float(np.mean(mac_f1))
    overall_f2 = float(np.mean(mac_f2_per_class))
    overall_sp = _macro_specificity(y_true, y_pred)

    per_class = {
        cls: {
            "precision":   float(mac_p[i]),
            "recall":      float(mac_r[i]),
            "f1":          float(mac_f1[i]),
            "f2":          float(mac_f2_per_class[i]),
            "support":     int(supports[i]),
        }
        for i, cls in enumerate(TARGET_CLASSES)
    }

    return {
        "model":        model_name,
        "label_config": config_name,
        "precision":    overall_p,
        "recall":       overall_r,
        "f1":           overall_f1,
        "f2":           overall_f2,
        "specificity":  overall_sp,
        "per_class":    per_class,
    }


def save_confusion_matrix(df: pd.DataFrame, model_name: str, config_name: str) -> None:
    y_true = df["true_label"].tolist()
    y_pred = df["predicted_label"].tolist()

    cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
    cm_df = pd.DataFrame(cm, index=TARGET_CLASSES, columns=TARGET_CLASSES)

    stem = f"confusion_matrix_{model_name}_{config_name}"
    cm_df.to_csv(RESULTS_DIR / f"{stem}.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_df, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES,
    )
    ax.set_title(f"Confusion Matrix — {model_name} / {config_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{stem}.png", dpi=150)
    plt.close(fig)


# ── allresults.txt update ──────────────────────────────────────────────────────

def update_allresults(all_results: list[dict], path: Path = ALLRESULTS_PATH) -> None:
    """Replace the P003-ZeroShot row in allresults.txt with best-variant results (mpnet+expert)."""
    best = next(
        r for r in all_results
        if r['model'] == 'mpnet' and r['label_config'] == 'expert'
    )
    per_type = ' & '.join(
        f'{best["per_class"][t]["f2"]:.3f}' for t in _RESULT_TYPE_ORDER
    )
    new_data_line = (
        f'  & {best["recall"]:.3f} & {best["precision"]:.3f}'
        f' & {best["f1"]:.3f} & {best["f2"]:.3f}'
        f' & {per_type} \\\\\n'
    )

    lines = path.read_text().splitlines(keepends=True)
    out, replace_next = [], False
    for line in lines:
        if replace_next:
            out.append(new_data_line)
            replace_next = False
        elif line.strip() == 'P003-ZeroShot':
            out.append(line)
            replace_next = True
        else:
            out.append(line)
    path.write_text(''.join(out))
    print(f'Updated {path}  (mpnet+expert macro F2={best["f2"]:.3f})')


# ── report generation ──────────────────────────────────────────────────────────

def write_per_class_f2(all_results: list[dict]) -> None:
    rows = []
    for r in all_results:
        for cls in TARGET_CLASSES:
            pc = r["per_class"][cls]
            rows.append({
                "model":        r["model"],
                "label_config": r["label_config"],
                "class":        cls,
                "f2":           round(pc["f2"], 4),
                "support":      pc["support"],
            })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "per_class_f2.csv", index=False)


def write_metrics_summary(all_results: list[dict]) -> None:
    rows = []
    for r in all_results:
        rows.append({
            "model":        r["model"],
            "label_config": r["label_config"],
            "precision":    round(r["precision"], 4),
            "recall":       round(r["recall"], 4),
            "f1":           round(r["f1"], 4),
            "f2":           round(r["f2"], 4),
            "specificity":  round(r["specificity"], 4),
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)


def write_report(all_results: list[dict], df: pd.DataFrame) -> None:
    lines = [
        "# Zero-Shot NFR Classification — Replication Report",
        "",
        "## Reference",
        "Alhoshan, W., Ferrari, A., Zhao, L. (2023). *Zero-Shot Learning for Requirements",
        "Classification: An Exploratory Study.*",
        "",
        "## Scope",
        "- Task: multi-class NFR type classification",
        "- Dataset: promise-june2026.csv (RequirementText column)",
        f"- NFR requirements classified: {len(df)}",
        f"- Classes ({len(TARGET_CLASSES)}): {', '.join(TARGET_CLASSES)}",
        "- No training, no fine-tuning, no cross-validation",
        "- Single-label classifier: each requirement assigned to nearest class by cosine similarity",
        "",
        "## Models",
        "- MiniLM: sentence-transformers/all-MiniLM-L12-v2",
        "- MPNet:  sentence-transformers/all-mpnet-base-v2",
        "",
        "## Label Configurations",
        "- **original** (Config A): plain class names (equivalent to MultiNFR_A, extended to 9 classes)",
        "- **expert**   (Config B): expert-curated labels from authors' GitHub appendix (equivalent to MultiNFR_B)",
        "",
        "## Class Distribution",
        "",
    ]

    dist = df["true_label"].value_counts().sort_index()
    for cls, cnt in dist.items():
        lines.append(f"- {cls}: {cnt}")

    lines += [
        "",
        "## Summary Results (macro-averaged)",
        "",
        "| Model | Labels | Precision | Recall | F1 | F2 | Specificity |",
        "| ----- | ------ | --------- | ------ | -- | -- | ----------- |",
    ]
    for r in all_results:
        lines.append(
            f"| {r['model']} | {r['label_config']} "
            f"| {r['precision']:.4f} | {r['recall']:.4f} "
            f"| {r['f1']:.4f} | {r['f2']:.4f} | {r['specificity']:.4f} |"
        )

    lines += [
        "",
        "## Per-Class F2",
        "",
    ]
    for r in all_results:
        lines.append(f"### {r['model']} / {r['label_config']}")
        lines.append("")
        lines.append("| Class | F2 | Support |")
        lines.append("| ----- | -- | ------- |")
        for cls in TARGET_CLASSES:
            pc = r["per_class"][cls]
            lines.append(f"| {cls} | {pc['f2']:.4f} | {pc['support']} |")
        lines.append("")

    lines += [
        "## Environment",
        "",
        f"- Python: {sys.version.split()[0]}",
    ]
    try:
        import sentence_transformers as st
        lines.append(f"- sentence-transformers: {st.__version__}")
    except Exception:
        pass
    try:
        import transformers as tr
        lines.append(f"- transformers: {tr.__version__}")
    except Exception:
        pass
    lines.append(f"- torch: {torch.__version__}")
    try:
        import sklearn
        lines.append(f"- scikit-learn: {sklearn.__version__}")
    except Exception:
        pass
    lines.append(f"- random seed: {SEED}")

    (RESULTS_DIR / "replication_report.md").write_text("\n".join(lines) + "\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading dataset from {DATA_CSV} …")
    df = load_dataset(DATA_CSV)
    print(f"  {len(df)} NFR requirements across {len(TARGET_CLASSES)} classes")
    print(f"  Class distribution:\n{df['true_label'].value_counts().sort_index()}\n")

    all_results:  list[dict]     = []
    all_pred_dfs: dict[str, pd.DataFrame] = {}

    for model_name, model_id in MODEL_IDS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  ({model_id})")
        print(f"{'='*60}")
        model = SentenceTransformer(model_id)

        for config_name, label_config in LABEL_CONFIGS.items():
            print(f"\n  Config: {config_name}")
            print(f"  Labels: { {k: v[:40] for k, v in label_config.items()} }")

            pred_df = classify(model, df, label_config)
            pred_df["model"]               = model_name
            pred_df["label_configuration"] = config_name

            key = f"{model_name}_{config_name}"
            all_pred_dfs[key] = pred_df

            result = evaluate(pred_df, model_name, config_name)
            all_results.append(result)

            print(f"  P={result['precision']:.4f}  R={result['recall']:.4f}  "
                  f"F1={result['f1']:.4f}  F2={result['f2']:.4f}  "
                  f"Spec={result['specificity']:.4f}")

            save_confusion_matrix(pred_df, model_name, config_name)

            out_cols = [
                "requirement_id", "project_id", "requirement_text",
                "true_label", "predicted_label", "similarity_score",
                "model", "label_configuration",
            ]
            pred_df[out_cols].to_csv(
                RESULTS_DIR / f"{model_name}_{config_name}_predictions.csv",
                index=False,
            )

    print("\n\nWriting summary files …")
    write_metrics_summary(all_results)
    write_per_class_f2(all_results)
    write_report(all_results, df)
    update_allresults(all_results)

    print("\nDone. Results written to:", RESULTS_DIR)
    print("\nSummary:")
    print(f"{'Model':<10} {'Config':<10} {'P':>7} {'R':>7} {'F1':>7} {'F2':>7} {'Spec':>7}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['model']:<10} {r['label_config']:<10} "
            f"{r['precision']:>7.4f} {r['recall']:>7.4f} "
            f"{r['f1']:>7.4f} {r['f2']:>7.4f} {r['specificity']:>7.4f}"
        )


if __name__ == "__main__":
    main()
