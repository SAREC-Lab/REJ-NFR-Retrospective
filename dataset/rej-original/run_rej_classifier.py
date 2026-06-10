#!/usr/bin/env python3
"""
REJ NFR Classifier — Cleland-Huang et al. (2007)
"Automated Classification of Non-Functional Requirements"
Requirements Engineering Journal, 12:103-120.

Protocol (rules.txt):
  (1) F2-based threshold optimisation learned on non-test (training) data
  (2) Leave-one-project-out (LOPO) cross-validation
  (3) Metric computation at the individual requirement level
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

for pkg in ('stopwords', 'punkt', 'punkt_tab'):
    nltk.download(pkg, quiet=True)

# ── Constants ─────────────────────────────────────────────────────────────────

TOP_N       = 15
THETA_RANGE = [round(i * 0.01, 2) for i in range(1, 31)]   # 0.01 … 0.30
DATA_PATH   = Path('../promise-june2026.csv')
RESULTS_DIR = Path('results')
ALLRESULTS_PATH = Path('../allresults.txt')

NFR_COLUMN_MAP = {
    'Availability (A)':     'availability',
    'Legal (L)':            'legal',
    'Look & Feel (LF)':     'look-and-feel',
    'Maintainability (MN)': 'maintainability',
    'Operability (O)':      'operational',
    'Performance (PE)':     'performance',
    'Scalability (SC)':     'scalability',
    'Security (SE)':        'security',
    'Usability (US)':       'usability',
}
NFR_TYPES = list(NFR_COLUMN_MAP.values())
ALL_TYPES  = NFR_TYPES + ['functional']

# Column order matching allresults.txt: A L LF MN O PE SC SE US
_RESULT_TYPE_ORDER = [
    'availability', 'legal', 'look-and-feel', 'maintainability',
    'operational', 'performance', 'scalability', 'security', 'usability',
]

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path: Path) -> list[dict]:
    records = []
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            types = [name for col, name in NFR_COLUMN_MAP.items()
                     if row[col].strip() == '1']
            records.append({
                'project': row['ProjectID'].strip(),
                'text':    row['RequirementText'].strip().strip("'"),
                'types':   types if types else ['functional'],
            })
    return records

# ── Preprocessing ─────────────────────────────────────────────────────────────

_stemmer = PorterStemmer()
_DOMAIN_STOPS = {'shall', 'must', 'system', 'product', 'applic', 'abil'}
_stop_words = set(stopwords.words('english')) | {
    _stemmer.stem(w) for w in _DOMAIN_STOPS
}

def preprocess(text: str) -> list[str]:
    """Lowercase → tokenize (alpha only) → remove stopwords → Porter stem."""
    tokens = re.findall(r'[a-zA-Z]+', text.lower())
    return [_stemmer.stem(t) for t in tokens if t not in _stop_words]

# ── Phase 1: Indicator term mining ────────────────────────────────────────────

def mine_indicator_terms(train: list[dict]) -> tuple[dict, dict]:
    """
    Compute Pr_Q(t) for every (type, term) pair in the training set.

    Returns
    -------
    all_weights : {nfr_type: {term: float}}
    top_terms   : {nfr_type: [(term, weight)]}  — top-15 per type
    """
    for req in train:
        if 'tokens' not in req:
            req['tokens'] = preprocess(req['text'])

    S_Q: dict[str, list[dict]] = defaultdict(list)
    for req in train:
        for t in req['types']:
            if t != 'functional':
                S_Q[t].append(req)

    N_t: dict[str, int] = defaultdict(int)
    for req in train:
        for term in set(req['tokens']):
            N_t[term] += 1

    all_weights: dict[str, dict[str, float]] = {}

    for nfr_type in NFR_TYPES:
        docs = S_Q[nfr_type]
        N_Q  = len(docs)
        all_weights[nfr_type] = {}
        if N_Q == 0:
            continue

        NP_Q = len({d['project'] for d in docs})
        N_Q_t:  dict[str, int]   = defaultdict(int)
        NP_Q_t: dict[str, set]   = defaultdict(set)
        tf_sum: dict[str, float] = defaultdict(float)

        for doc in docs:
            tokens  = doc['tokens']
            doc_len = len(tokens)
            if doc_len == 0:
                continue
            term_freq: dict[str, int] = defaultdict(int)
            for tok in tokens:
                term_freq[tok] += 1
            for term, cnt in term_freq.items():
                tf_sum[term]  += cnt / doc_len
                N_Q_t[term]   += 1
                NP_Q_t[term].add(doc['project'])

        weights: dict[str, float] = {}
        for term in N_Q_t:
            if N_t[term] == 0:
                continue
            w = (tf_sum[term] / N_Q) * (N_Q_t[term] / N_t[term]) * \
                (len(NP_Q_t[term]) / NP_Q)
            if w > 0:
                weights[term] = w
        all_weights[nfr_type] = weights

    top_terms = {
        nfr_type: sorted(w.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
        for nfr_type, w in all_weights.items()
    }
    return all_weights, top_terms

# ── Phase 2: Raw scoring (pre-threshold) ─────────────────────────────────────

def score_requirements(records: list[dict],
                       top_terms: dict[str, list],
                       all_weights: dict[str, dict]) -> list[dict]:
    """Return raw per-type scores (before thresholding) for each requirement."""
    scored = []
    for req in records:
        tokens    = preprocess(req['text'])
        token_set = set(tokens)
        req_scores: dict[str, float] = {}
        for nfr_type, indicator in top_terms.items():
            if not indicator:
                req_scores[nfr_type] = 0.0
                continue
            I_Q   = {t for t, _ in indicator}
            denom = sum(w for _, w in indicator)
            numer = sum(all_weights[nfr_type].get(t, 0.0)
                        for t in token_set & I_Q) if denom > 0 else 0.0
            req_scores[nfr_type] = numer / denom if denom > 0 else 0.0
        scored.append({'record': req, 'scores': req_scores, 'tokens': tokens})
    return scored

# ── Phase 2b: LOPO threshold tuning ──────────────────────────────────────────

def _f2(precision: float, recall: float) -> float:
    denom = 4 * precision + recall
    return (5 * precision * recall) / denom if denom > 0 else 0.0

def find_best_theta(scored: list[dict],
                    theta_range: list[float] = THETA_RANGE) -> tuple[float, float]:
    """
    Sweep theta over pre-scored requirements; return (theta*, macro_F2).
    Tuned on training data — never called with test-fold items.
    """
    best_theta, best_macro_f2 = theta_range[0], -1.0

    for theta in theta_range:
        tp = defaultdict(int); fp = defaultdict(int)
        fn = defaultdict(int); tn = defaultdict(int)

        for item in scored:
            req       = item['record']
            predicted = {t for t, v in item['scores'].items() if v > theta} \
                        or {'functional'}
            actual    = set(req['types'])
            for t in ALL_TYPES:
                a, p = t in actual, t in predicted
                if   a and p:     tp[t] += 1
                elif p and not a: fp[t] += 1
                elif a and not p: fn[t] += 1
                else:             tn[t] += 1

        f2_vals = []
        for t in NFR_TYPES:
            prec = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0.0
            rec  = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0.0
            f2_vals.append(_f2(prec, rec))
        macro_f2 = sum(f2_vals) / len(f2_vals)

        if macro_f2 > best_macro_f2:
            best_macro_f2, best_theta = macro_f2, theta

    return best_theta, best_macro_f2

# ── Phase 3: Evaluation ───────────────────────────────────────────────────────

def evaluate_scored(scored: list[dict],
                    theta: float) -> tuple[list[dict], dict]:
    """Apply threshold to pre-scored items and compute per-type metrics."""
    tp = defaultdict(int); fp = defaultdict(int)
    fn = defaultdict(int); tn = defaultdict(int)
    classifications = []

    for item in scored:
        req       = item['record']
        predicted = sorted(
            {t for t, v in item['scores'].items() if v > theta} or {'functional'})
        actual    = sorted(req['types'])

        classifications.append({
            'project':   req['project'],
            'text':      req['text'],
            'actual':    actual,
            'predicted': predicted,
            'tokens':    item['tokens'],
        })

        actual_set    = set(actual)
        predicted_set = set(predicted)
        for t in ALL_TYPES:
            a, p = t in actual_set, t in predicted_set
            if   a and p:     tp[t] += 1
            elif p and not a: fp[t] += 1
            elif a and not p: fn[t] += 1
            else:             tn[t] += 1

    metrics = {}
    for t in ALL_TYPES:
        recall      = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0.0
        precision   = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0.0
        specificity = tn[t] / (tn[t] + fp[t]) if (tn[t] + fp[t]) > 0 else 0.0
        metrics[t] = {
            'tp': tp[t], 'fp': fp[t], 'fn': fn[t], 'tn': tn[t],
            'recall':      round(recall, 4),
            'precision':   round(precision, 4),
            'specificity': round(specificity, 4),
        }
    return classifications, metrics

# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_indicator_terms(run_dir: Path, top_terms: dict, all_weights: dict):
    with open(run_dir / 'indicator_terms.json', 'w') as f:
        json.dump({
            nfr_type: [{'term': t, 'weight': round(w, 8)} for t, w in terms]
            for nfr_type, terms in top_terms.items()
        }, f, indent=2)
    with open(run_dir / 'term_weights.json', 'w') as f:
        json.dump({
            nfr_type: {t: round(w, 8) for t, w in sorted(
                weights.items(), key=lambda x: x[1], reverse=True)}
            for nfr_type, weights in all_weights.items()
        }, f, indent=2)

def save_classifications(run_dir: Path, classifications: list[dict]):
    with open(run_dir / 'classifications.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['project', 'text', 'actual_types', 'predicted_types',
                         'preprocessed_tokens'])
        for c in classifications:
            writer.writerow([
                c['project'], c['text'],
                '|'.join(c['actual']), '|'.join(c['predicted']),
                ' '.join(c['tokens']),
            ])

def save_metrics(run_dir: Path, metrics: dict):
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(run_dir / 'metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'tp', 'fp', 'fn', 'tn',
                         'recall', 'precision', 'specificity'])
        for t in ALL_TYPES:
            m = metrics[t]
            writer.writerow([t, m['tp'], m['fp'], m['fn'], m['tn'],
                             m['recall'], m['precision'], m['specificity']])

# ── allresults.txt update ─────────────────────────────────────────────────────

def update_allresults(agg_rows: list[dict], path: Path = ALLRESULTS_PATH):
    """Replace the REJ row in allresults.txt with computed values."""
    by_type  = {r['type']: r for r in agg_rows}
    nfr_rows = [by_type[t] for t in _RESULT_TYPE_ORDER]

    macro_R  = sum(r['recall']    for r in nfr_rows) / len(nfr_rows)
    macro_P  = sum(r['precision'] for r in nfr_rows) / len(nfr_rows)
    macro_F1 = sum(r['f1']        for r in nfr_rows) / len(nfr_rows)
    macro_F2 = sum(r['f2']        for r in nfr_rows) / len(nfr_rows)
    per_type = ' & '.join(f'{r["f2"]:.3f}' for r in nfr_rows)

    new_data_line = (
        f'  & {macro_R:.3f} & {macro_P:.3f} & {macro_F1:.3f} & {macro_F2:.3f}'
        f' & {per_type} \\\\\n'
    )

    lines = path.read_text().splitlines(keepends=True)
    out, replace_next = [], False
    for line in lines:
        if replace_next:
            out.append(new_data_line)
            replace_next = False
        elif line.strip() == 'REJ':
            out.append(line)
            replace_next = True
        else:
            out.append(line)
    path.write_text(''.join(out))
    print(f'Updated {path}  (macro F2={macro_F2:.3f})')

# ── Main LOPO experiment ──────────────────────────────────────────────────────

def run_experiment(results_dir: Path = RESULTS_DIR,
                   verbose: bool = True) -> list[dict]:
    """
    Run 15-fold LOPO cross-validation with per-fold F2-based theta tuning.
    Returns list of aggregate metric dicts (one per type).
    """
    if verbose:
        print('Loading data...')
    reqs     = load_data(DATA_PATH)
    projects = sorted({r['project'] for r in reqs}, key=int)
    assert len(projects) == 15, f'Expected 15 projects, got {len(projects)}'
    if verbose:
        print(f'  {len(reqs)} requirements across {len(projects)} projects\n')

    results_dir.mkdir(parents=True, exist_ok=True)
    agg = {t: defaultdict(int) for t in ALL_TYPES}
    all_run_summaries = []

    for run_idx, test_project in enumerate(projects, start=1):
        train = [r for r in reqs if r['project'] != test_project]
        test  = [r for r in reqs if r['project'] == test_project]

        # Phase 1 — mine indicator terms from the 14 training projects
        all_weights, top_terms = mine_indicator_terms(train)

        # Phase 2 — score training requirements, tune theta on training fold only
        train_scored           = score_requirements(train, top_terms, all_weights)
        theta_star, train_f2   = find_best_theta(train_scored)

        if verbose:
            print(f'── Run {run_idx:02d}/15  hold-out={test_project}'
                  f'  theta*={theta_star:.2f}  train-F2={train_f2:.3f}'
                  f'  (train={len(train)} test={len(test)})')

        # Phase 3 — score and evaluate held-out project with theta*
        test_scored                    = score_requirements(test, top_terms, all_weights)
        classifications, metrics       = evaluate_scored(test_scored, theta_star)

        run_dir = results_dir / f'run_{run_idx:02d}_project_{test_project}'
        run_dir.mkdir(exist_ok=True)
        with open(run_dir / 'run_info.json', 'w') as f:
            json.dump({
                'run': run_idx, 'test_project': test_project,
                'train_projects': [p for p in projects if p != test_project],
                'n_train': len(train), 'n_test': len(test),
                'theta_star': theta_star,
                'train_macro_f2': round(train_f2, 4),
            }, f, indent=2)
        save_indicator_terms(run_dir, top_terms, all_weights)
        save_classifications(run_dir, classifications)
        save_metrics(run_dir, metrics)

        if verbose:
            print(f"   {'type':<20} {'R':>6} {'P':>6} {'F2':>6}  TP  FP  FN")
            for t in NFR_TYPES:
                m = metrics[t]
                f2v = _f2(m['precision'], m['recall'])
                print(f"   {t:<20} {m['recall']:>6.3f} {m['precision']:>6.3f}"
                      f" {f2v:>6.3f}  {m['tp']:3d} {m['fp']:3d} {m['fn']:3d}")
            print()

        for t in ALL_TYPES:
            for k in ('tp', 'fp', 'fn', 'tn'):
                agg[t][k] += metrics[t][k]
        all_run_summaries.append({
            'run': run_idx, 'project': test_project,
            'theta_star': theta_star, 'metrics': metrics,
        })

    # ── Aggregate metrics ─────────────────────────────────────────────────────

    if verbose:
        print('══ AGGREGATE RESULTS (all 15 runs) ════════════════════════════════')
    agg_rows = []
    for t in ALL_TYPES:
        tp, fp, fn, tn = agg[t]['tp'], agg[t]['fp'], agg[t]['fn'], agg[t]['tn']
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) \
             if (precision + recall) > 0 else 0.0
        f2 = _f2(precision, recall)
        agg_rows.append({
            'type': t, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'recall':      round(recall, 4),
            'precision':   round(precision, 4),
            'specificity': round(specificity, 4),
            'f1':          round(f1, 4),
            'f2':          round(f2, 4),
        })

    nfr_rows = [r for r in agg_rows if r['type'] != 'functional']
    macro_R  = sum(r['recall']    for r in nfr_rows) / len(nfr_rows)
    macro_P  = sum(r['precision'] for r in nfr_rows) / len(nfr_rows)
    macro_F1 = sum(r['f1']        for r in nfr_rows) / len(nfr_rows)
    macro_F2 = sum(r['f2']        for r in nfr_rows) / len(nfr_rows)

    if verbose:
        for r in nfr_rows:
            print(f"  {r['type']:<20} R={r['recall']:.3f}  P={r['precision']:.3f}"
                  f"  F1={r['f1']:.3f}  F2={r['f2']:.3f}"
                  f"  (TP={r['tp']} FP={r['fp']} FN={r['fn']})")
        print(f"\n  {'macro-avg (NFR)':20s} R={macro_R:.3f}  P={macro_P:.3f}"
              f"  F1={macro_F1:.3f}  F2={macro_F2:.3f}")

    with open(results_dir / 'aggregate_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'type', 'tp', 'fp', 'fn', 'tn',
            'recall', 'precision', 'specificity', 'f1', 'f2'])
        writer.writeheader()
        writer.writerows(agg_rows)
        writer.writerow({
            'type': 'macro-avg-NFR', 'tp': '', 'fp': '', 'fn': '', 'tn': '',
            'recall': round(macro_R, 4), 'precision': round(macro_P, 4),
            'specificity': '', 'f1': round(macro_F1, 4), 'f2': round(macro_F2, 4),
        })

    with open(results_dir / 'all_runs.json', 'w') as f:
        json.dump(all_run_summaries, f, indent=2)

    if verbose:
        print(f'\nRaw results written to {results_dir.resolve()}/')

    if ALLRESULTS_PATH.exists():
        update_allresults(agg_rows)

    return agg_rows


if __name__ == '__main__':
    run_experiment()
