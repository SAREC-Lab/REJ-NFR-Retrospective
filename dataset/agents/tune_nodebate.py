#!/usr/bin/env python3
"""
No-debate threshold tuner — LOPO-tuned.

Uses advocate confidence scores saved by run_debate.py (scores_*.json) to find
the best prediction threshold WITHOUT re-running the full debate pipeline.

Scoring rule per (requirement, NFR type):
  DEBATE outcome: argue_conf if type reached argument stage, else screen_conf.
  NFR outcome:    initial screen_conf (single type accepted directly, skipped debate).
  FR outcome:     0 for all types (initial screen classified as functional).

LOPO threshold tuning (per rules.txt rule 1):
  For each of 15 folds (held-out project p):
    - Train: all entries where project != p
    - Sweep threshold on train entries to find threshold* maximising macro F2
    - Apply threshold* to test entries (project == p)
  Pool test predictions across all 15 folds; compute final metrics once.

Two variants reported:
  Global   — single threshold applied uniformly to all 9 NFR types.
  Per-type — each type gets its own LOPO-tuned threshold (ceiling on performance).

allresults.txt is updated with the global variant (Agent Basic row).

Usage:
    python3 tune_nodebate.py --scores results/scores_debate_da70_c4_at30.json
    python3 tune_nodebate.py --scores results/scores_debate_da70_c4_at30.json --steps 200
"""

import argparse
import csv
import json
from pathlib import Path

NFR_TYPES = [
    'availability', 'legal', 'look-and-feel', 'maintainability',
    'operational', 'performance', 'scalability', 'security', 'usability',
]

SHORT = {
    'availability': 'A', 'legal': 'L', 'look-and-feel': 'LF',
    'maintainability': 'MN', 'operational': 'O', 'performance': 'PE',
    'scalability': 'SC', 'security': 'SE', 'usability': 'US',
}

_RESULT_TYPE_ORDER = [
    'availability', 'legal', 'look-and-feel', 'maintainability',
    'operational', 'performance', 'scalability', 'security', 'usability',
]

ALLRESULTS_PATH = Path('../allresults.txt')


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_for(entry: dict, nfr_type: str) -> float:
    """Best available advocate confidence for this type in this requirement."""
    ns = entry['nfr_scores'].get(nfr_type, {})
    if 'argue_conf' in ns:
        return ns['argue_conf']
    if 'screen_conf' in ns:
        return ns['screen_conf']
    # NFR-outcome: type was directly accepted without running advocate pipeline;
    # use the initial screen's confidence as proxy.
    if entry['screen_outcome'] == 'NFR' and ns.get('was_candidate'):
        return entry['screen_conf'] or 0.0
    return 0.0


def apply_threshold(entries: list[dict], threshold: float) -> list[dict]:
    return [{t: score_for(e, t) >= threshold for t in NFR_TYPES} for e in entries]


def apply_per_type_thresholds(entries: list[dict],
                               thresholds: dict[str, float]) -> list[dict]:
    return [{t: score_for(e, t) >= thresholds[t] for t in NFR_TYPES}
            for e in entries]


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(entries: list[dict],
                    predictions: list[dict]) -> tuple[dict, dict]:
    per_type = {}
    for t in NFR_TYPES:
        tp = sum(1 for e, p in zip(entries, predictions)
                 if t in e['true_types'] and p[t])
        fp = sum(1 for e, p in zip(entries, predictions)
                 if t not in e['true_types'] and p[t])
        fn = sum(1 for e, p in zip(entries, predictions)
                 if t in e['true_types'] and not p[t])
        tn = len(entries) - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec)     if (prec + rec)     > 0 else 0.0
        f2   = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
        per_type[t] = dict(prec=prec, rec=rec, f1=f1, f2=f2,
                           tp=tp, fp=fp, fn=fn, tn=tn)
    n  = len(NFR_TYPES)
    macro = {k: sum(per_type[t][k] for t in NFR_TYPES) / n
             for k in ('prec', 'rec', 'f1', 'f2')}
    return per_type, macro


# ── LOPO tuning — global threshold ────────────────────────────────────────────

def lopo_tune_global(scores_data: list[dict], steps: int) -> tuple[dict, dict]:
    """
    For each of 15 LOPO folds:
      - sweep threshold on training entries (14 projects) to maximise macro F2
      - apply best threshold to held-out project entries
    Pool all test predictions; compute final metrics.
    """
    projects = sorted(set(e['project'] for e in scores_data))

    all_test_entries: list[dict] = []
    all_test_preds:   list[dict] = []

    for test_project in projects:
        train = [e for e in scores_data if e['project'] != test_project]
        test  = [e for e in scores_data if e['project'] == test_project]

        best_t, best_f2 = 0.0, -1.0
        for i in range(steps + 1):
            t     = i / steps
            preds = apply_threshold(train, t)
            _, macro = compute_metrics(train, preds)
            if macro['f2'] > best_f2:
                best_f2 = macro['f2']
                best_t  = t

        all_test_entries.extend(test)
        all_test_preds.extend(apply_threshold(test, best_t))

    return compute_metrics(all_test_entries, all_test_preds)


# ── LOPO tuning — per-type thresholds ─────────────────────────────────────────

def lopo_tune_per_type(scores_data: list[dict], steps: int) -> tuple[dict, dict]:
    """
    Same as lopo_tune_global but each NFR type gets its own threshold,
    independently tuned on training data for that fold.
    """
    projects = sorted(set(e['project'] for e in scores_data))

    all_test_entries: list[dict] = []
    all_test_preds:   list[dict] = []

    for test_project in projects:
        train = [e for e in scores_data if e['project'] != test_project]
        test  = [e for e in scores_data if e['project'] == test_project]

        thresholds: dict[str, float] = {}
        for t in NFR_TYPES:
            best_thresh, best_f2 = 0.0, -1.0
            for i in range(steps + 1):
                thresh = i / steps
                tp = sum(1 for e in train
                         if t in e['true_types'] and score_for(e, t) >= thresh)
                fp = sum(1 for e in train
                         if t not in e['true_types'] and score_for(e, t) >= thresh)
                fn = sum(1 for e in train
                         if t in e['true_types'] and score_for(e, t) < thresh)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f2   = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
                if f2 > best_f2:
                    best_f2, best_thresh = f2, thresh
            thresholds[t] = best_thresh

        all_test_entries.extend(test)
        all_test_preds.extend(apply_per_type_thresholds(test, thresholds))

    return compute_metrics(all_test_entries, all_test_preds)


# ── Full-debate baseline (for comparison) ─────────────────────────────────────

def debate_baseline(scores_data: list[dict]) -> tuple[dict, dict]:
    """Metrics from the arbiter's direct decisions (no threshold applied)."""
    preds = [{t: t in e['predicted'] for t in NFR_TYPES} for e in scores_data]
    return compute_metrics(scores_data, preds)


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_per_type_table(per_type: dict) -> None:
    print(f"  {'Type':<20}  {'Rec':>6}  {'Prec':>6}  {'F1':>6}  {'F2':>6}  "
          f"{'TP':>4}  {'FP':>4}  {'FN':>4}")
    print('  ' + '-' * 68)
    for t in NFR_TYPES:
        v = per_type[t]
        print(f"  {t:<20}  {v['rec']:>6.3f}  {v['prec']:>6.3f}  "
              f"{v['f1']:>6.3f}  {v['f2']:>6.3f}  "
              f"{v['tp']:>4}  {v['fp']:>4}  {v['fn']:>4}")


def write_results_csv(per_type: dict, macro: dict, path: Path) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['type', 'recall', 'precision', 'f1', 'f2', 'tp', 'fp', 'fn'])
        for t in NFR_TYPES:
            v = per_type[t]
            w.writerow([t, round(v['rec'], 4), round(v['prec'], 4),
                        round(v['f1'], 4), round(v['f2'], 4),
                        v['tp'], v['fp'], v['fn']])
        w.writerow(['macro', round(macro['rec'], 4), round(macro['prec'], 4),
                    round(macro['f1'], 4), round(macro['f2'], 4), '', '', ''])


# ── allresults.txt update ──────────────────────────────────────────────────────

def update_allresults(per_type: dict, macro: dict,
                      path: Path = ALLRESULTS_PATH) -> None:
    """Replace the Agent Basic row in allresults.txt (global LOPO-tuned NoDebate)."""
    per_type_f2 = ' & '.join(f'{per_type[t]["f2"]:.3f}' for t in _RESULT_TYPE_ORDER)
    new_data_line = (
        f'  & {macro["rec"]:.3f} & {macro["prec"]:.3f}'
        f' & {macro["f1"]:.3f} & {macro["f2"]:.3f}'
        f' & {per_type_f2} \\\\\n'
    )
    lines = path.read_text().splitlines(keepends=True)
    out, replace_next = [], False
    for line in lines:
        if replace_next:
            out.append(new_data_line)
            replace_next = False
        elif line.strip() == 'Agent Basic':
            out.append(line)
            replace_next = True
        else:
            out.append(line)
    path.write_text(''.join(out))
    print(f'Updated {path}  (macro F2={macro["f2"]:.3f})')


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='No-debate LOPO threshold tuner — optimises macro F2 over advocate scores')
    parser.add_argument('--scores', type=Path, required=True,
                        help='scores_*.json written by run_debate.py')
    parser.add_argument('--steps', type=int, default=100,
                        help='Threshold sweep granularity (default 100 → 0.01 steps)')
    parser.add_argument('--out-dir', type=Path, default=Path('results'),
                        help='Directory for CSV output files (default: results/)')
    args = parser.parse_args()

    scores_data: list[dict] = json.loads(args.scores.read_text())
    args.out_dir.mkdir(exist_ok=True)
    stem = args.scores.stem  # e.g. scores_debate_da70_c4_at30

    print(f"Loaded {len(scores_data)} requirements  |  {args.steps} sweep steps")
    projects = sorted(set(e['project'] for e in scores_data))
    print(f"Projects: {len(projects)}  ({', '.join(projects)})\n")

    # ── Full debate baseline ───────────────────────────────────────────────────
    d_per_type, d_macro = debate_baseline(scores_data)
    print('── Full debate (arbiter decisions — reference) ' + '─' * 16)
    print(f"  Rec={d_macro['rec']:.3f}  Prec={d_macro['prec']:.3f}  "
          f"F1={d_macro['f1']:.3f}  F2={d_macro['f2']:.3f}")
    print_per_type_table(d_per_type)
    print()

    # ── LOPO global threshold ──────────────────────────────────────────────────
    print('── NoDebate — LOPO global threshold ' + '─' * 26)
    g_per_type, g_macro = lopo_tune_global(scores_data, args.steps)
    print(f"  Rec={g_macro['rec']:.3f}  Prec={g_macro['prec']:.3f}  "
          f"F1={g_macro['f1']:.3f}  F2={g_macro['f2']:.3f}")
    print_per_type_table(g_per_type)

    csv_global = args.out_dir / f"nodebate_global_{stem}.csv"
    write_results_csv(g_per_type, g_macro, csv_global)
    print(f"\n  Results → {csv_global}")
    print()

    # ── LOPO per-type thresholds ───────────────────────────────────────────────
    print('── NoDebate — LOPO per-type thresholds ' + '─' * 23)
    pt_per_type, pt_macro = lopo_tune_per_type(scores_data, args.steps)
    print(f"  Rec={pt_macro['rec']:.3f}  Prec={pt_macro['prec']:.3f}  "
          f"F1={pt_macro['f1']:.3f}  F2={pt_macro['f2']:.3f}")
    print_per_type_table(pt_per_type)

    csv_pertype = args.out_dir / f"nodebate_pertype_{stem}.csv"
    write_results_csv(pt_per_type, pt_macro, csv_pertype)
    print(f"\n  Results → {csv_pertype}")
    print()

    # ── Update allresults.txt (global variant = Agent Basic) ──────────────────
    update_allresults(g_per_type, g_macro)


if __name__ == '__main__':
    main()
