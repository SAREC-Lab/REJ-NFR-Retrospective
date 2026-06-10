#!/usr/bin/env python3
"""
P011 — Slankas & Williams (2013) NFR Classifier
"Automated Extraction of Non-Functional Requirements in Available Documentation"
NaturaLiSE 2013.

Three classifiers evaluated under identical LOPO protocol:
  - k-NN  : graph-based nearest-neighbour (no threshold)
  - SVM   : LinearSVC with per-fold F2-tuned decision boundary
  - NB    : Multinomial Naive Bayes with per-fold F2-tuned probability boundary

Protocol (rules.txt):
  (1) F2-based threshold optimisation on non-test data (SVM and NB)
  (2) Leave-one-project-out (LOPO) cross-validation
  (3) Requirement-level metrics

Usage:
    python3 run_p011_classifiers.py
    python3 run_p011_classifiers.py --data ../promise-june2026.csv --results-dir results
"""

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import time

import nltk
import numpy as np
import stanza
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_CSV  = Path('../promise-june2026.csv')
RESULTS_DIR  = Path('results')
CACHE_PATH   = Path('parse_cache.json')
ALLRESULTS_PATH = Path('../allresults.txt')

# SVM: decision_function scores; NB: predict_proba[:, 1]
SVM_THRESHOLDS = [round(t * 0.1, 1) for t in range(-20, 21)]      # -2.0 … +2.0
NB_THRESHOLDS  = (
    [round(t * 0.01, 2) for t in range(1, 10)]   # 0.01 … 0.09 (fine)
  + [round(t * 0.05, 2) for t in range(2, 20)]   # 0.10 … 0.95 (coarse)
)

DETERMINERS = {'a', 'an', 'the'}

CLELAND_NFR_TYPES = [
    'availability', 'legal', 'look-and-feel', 'maintainability',
    'operational', 'performance', 'scalability', 'security', 'usability',
]

CSV_COLUMN_MAP = {
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

# Column order matching allresults.txt: A L LF MN O PE SC SE US
_RESULT_TYPE_ORDER = [
    'availability', 'legal', 'look-and-feel', 'maintainability',
    'operational', 'performance', 'scalability', 'security', 'usability',
]

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Vertex:
    lemma: str
    pos: str
    parents: list
    parent_count: int
    is_number: bool
    ner_class: str

# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(path: Path) -> tuple[list[dict], list[str]]:
    records = []
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            types = [name for col, name in CSV_COLUMN_MAP.items()
                     if row[col].strip() == '1']
            records.append({
                'text':    row['RequirementText'].strip().strip("'"),
                'types':   types,
                'project': row['ProjectID'].strip(),
            })
    return records, CLELAND_NFR_TYPES

# ── NLP preprocessing ─────────────────────────────────────────────────────────

def _collapse_pos(xpos: str) -> str:
    if xpos in ('NN', 'NNS', 'NNP', 'NNPS'): return 'NN'
    if xpos in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'): return 'VB'
    if xpos in ('JJ', 'JJR', 'JJS'): return 'JJ'
    return xpos or ''


def build_sr(text: str, nlp) -> list[Vertex]:
    vertices = []
    for sent in nlp(text).sentences:
        for word in sent.words:
            lemma = (word.lemma or word.text).lower()
            if lemma in DETERMINERS:
                continue
            pos     = _collapse_pos(word.xpos or '')
            parents = [word.deprel] if (word.head or 0) != 0 else []
            is_num  = (word.xpos == 'CD') or bool(re.fullmatch(r'[\d,\.]+', word.text))
            ner_raw = getattr(word, 'ner', 'O') or 'O'
            ner_cls = ner_raw.split('-')[-1] if ner_raw != 'O' else ''
            vertices.append(Vertex(
                lemma=lemma, pos=pos, parents=parents,
                parent_count=len(parents), is_number=is_num, ner_class=ner_cls,
            ))
    return vertices


def parse_all(records: list[dict]) -> list[list[Vertex]]:
    cache: dict[str, list[Vertex]] = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            raw = json.load(f)
        cache = {t: [Vertex(**v) for v in vs] for t, vs in raw.items()}
        print(f'  Cache: {len(cache)} sentences loaded from {CACHE_PATH}')

    texts   = [r['text'] for r in records]
    missing = [t for t in texts if t not in cache]

    if missing:
        print(f'  Parsing {len(missing)} sentences with stanza ...')
        nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse,ner',
                              verbose=False)
        t0 = time()
        for i, text in enumerate(missing, 1):
            cache[text] = build_sr(text, nlp)
            if i % 50 == 0 or i == len(missing):
                print(f'    {i}/{len(missing)}  ({time()-t0:.0f}s)', end='\r')
        print()
        raw = {t: [{'lemma': v.lemma, 'pos': v.pos, 'parents': v.parents,
                    'parent_count': v.parent_count, 'is_number': v.is_number,
                    'ner_class': v.ner_class} for v in vs]
               for t, vs in cache.items()}
        with open(CACHE_PATH, 'w') as f:
            json.dump(raw, f)
        print(f'  Cache saved → {CACHE_PATH}')

    return [cache[r['text']] for r in records]

# ── Distance metric ───────────────────────────────────────────────────────────

@lru_cache(maxsize=500_000)
def _wn_distance(la: str, lb: str) -> float:
    min_hops = None
    for sa in wn.synsets(la):
        for sb in wn.synsets(lb):
            d = sa.shortest_path_distance(sb)
            if d is not None and (min_hops is None or d < min_hops):
                min_hops = d
    if min_hops is None or min_hops > 4:
        return 0.0
    return {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}[min_hops]


def vertex_distance(a: Vertex | None, b: Vertex | None) -> float:
    if a is None or b is None:                   return 1.0
    if a.pos != b.pos:                           return 1.0
    if a.parent_count != b.parent_count:         return 1.0
    if set(a.parents) != set(b.parents):         return 1.0
    if a.lemma == b.lemma:                       return 0.0
    if a.is_number and b.is_number:              return 0.0
    if a.ner_class and a.ner_class == b.ner_class: return 0.0
    wn_val = _wn_distance(a.lemma, b.lemma)
    if wn_val > 0:                               return wn_val
    return 1.0


def sentence_distance(sr_a: list, sr_b: list) -> float:
    n = max(len(sr_a), len(sr_b))
    return sum(
        vertex_distance(
            sr_a[i] if i < len(sr_a) else None,
            sr_b[i] if i < len(sr_b) else None,
        )
        for i in range(n)
    )

# ── Classifiers ───────────────────────────────────────────────────────────────

def knn_classify(test_sr: list, train_pairs: list) -> list[str]:
    best_dist, best_types = float('inf'), []
    for sr, types in train_pairs:
        d = sentence_distance(test_sr, sr)
        if d < best_dist:
            best_dist, best_types = d, types
    return best_types


def remove_determiners(text: str) -> str:
    return ' '.join(t for t in text.lower().split() if t not in DETERMINERS)

# ── Metrics ───────────────────────────────────────────────────────────────────

def per_type_metrics(test_recs: list, predictions, nfr_types: list,
                     classifier: str = 'knn') -> dict:
    """
    predictions: list[list[str]] for knn, or {type: list[int]} for svm/nb.
    Returns {type: {precision, recall, specificity, f1, f2, tp, fp, fn, tn}}.
    """
    n = len(test_recs)
    metrics = {}
    for t in nfr_types:
        if classifier == 'knn':
            tp = sum(1 for r, p in zip(test_recs, predictions) if t in r['types'] and t in p)
            fp = sum(1 for r, p in zip(test_recs, predictions) if t not in r['types'] and t in p)
            fn = sum(1 for r, p in zip(test_recs, predictions) if t in r['types'] and t not in p)
        else:
            preds  = predictions[t]
            actual = [1 if t in r['types'] else 0 for r in test_recs]
            tp = sum(1 for a, p in zip(actual, preds) if a == 1 and p == 1)
            fp = sum(1 for a, p in zip(actual, preds) if a == 0 and p == 1)
            fn = sum(1 for a, p in zip(actual, preds) if a == 1 and p == 0)
        tn  = n - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spc  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f2   = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
        metrics[t] = {'precision': prec, 'recall': rec, 'specificity': spc,
                      'f1': f1, 'f2': f2, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
    return metrics


def macro_avg(metrics: dict, nfr_types: list) -> dict:
    keys = ('precision', 'recall', 'specificity', 'f1', 'f2')
    return {k: sum(metrics[t][k] for t in nfr_types) / len(nfr_types) for k in keys}

# ── Threshold tuning ──────────────────────────────────────────────────────────

def find_best_threshold(records: list, scores_by_type: dict, classifier: str,
                        threshold_range: list, nfr_types: list) -> tuple[float, float]:
    """
    Sweep threshold over in-sample training scores; return (theta*, macro_F2).
    Called on training data only — test fold is never seen here.
    """
    best_thr, best_f2 = threshold_range[0], -1.0
    for thr in threshold_range:
        preds = {t: [1 if s >= thr else 0 for s in scores_by_type[t]] for t in nfr_types}
        f2 = macro_avg(per_type_metrics(records, preds, nfr_types, classifier), nfr_types)['f2']
        if f2 > best_f2:
            best_f2, best_thr = f2, thr
    return best_thr, best_f2

# ── allresults.txt update ─────────────────────────────────────────────────────

def _allresults_row(label: str, m: dict, per_type: dict) -> str:
    nfr_rows = [per_type[t] for t in _RESULT_TYPE_ORDER]
    macro_f2  = ' & '.join(f'{r["f2"]:.3f}' for r in nfr_rows)
    return (
        f'{label}\n'
        f'  & {m["recall"]:.3f} & {m["precision"]:.3f}'
        f' & {m["f1"]:.3f} & {m["f2"]:.3f}'
        f' & {macro_f2} \\\\\n'
    )


def update_allresults(label: str, m: dict, per_type: dict,
                      path: Path = ALLRESULTS_PATH):
    per_type_f2 = ' & '.join(f'{per_type[t]["f2"]:.3f}' for t in _RESULT_TYPE_ORDER)
    new_data = (f'  & {m["recall"]:.3f} & {m["precision"]:.3f}'
                f' & {m["f1"]:.3f} & {m["f2"]:.3f}'
                f' & {per_type_f2} \\\\\n')

    lines = path.read_text().splitlines(keepends=True)
    out, replace_next = [], False
    for line in lines:
        if replace_next:
            out.append(new_data)
            replace_next = False
        elif line.strip() == label:
            out.append(line)
            replace_next = True
        else:
            out.append(line)
    path.write_text(''.join(out))
    print(f'  Updated {path}  [{label}]  macro F2={m["f2"]:.3f}')

# ── Main LOPO experiment ──────────────────────────────────────────────────────

def run_experiment(data_path: Path = DEFAULT_CSV,
                   results_dir: Path = RESULTS_DIR,
                   verbose: bool = True) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f'Loading data from {data_path} ...')
    records, nfr_types = load_csv(data_path)
    projects = sorted(set(r['project'] for r in records), key=lambda x: int(x))
    if verbose:
        dist = Counter(t for r in records for t in r['types'])
        print(f'  {len(records)} records  |  {len(projects)} projects')
        print(f'  Label counts: {dict(sorted(dist.items()))}')

    if verbose:
        print('Loading sentence representations ...')
    srs = parse_all(records)

    # Per-fold accumulators
    pooled_test_recs  = []
    pooled_knn_preds  = []
    pooled_svm_preds  = {t: [] for t in nfr_types}
    pooled_nb_preds   = {t: [] for t in nfr_types}
    fold_rows         = []   # for fold_thresholds.csv

    for proj in projects:
        t_fold = time()
        test_idx  = [i for i, r in enumerate(records) if r['project'] == proj]
        train_idx = [i for i, r in enumerate(records) if r['project'] != proj]
        train_recs = [records[i] for i in train_idx]
        test_recs  = [records[i] for i in test_idx]
        train_srs  = [srs[i] for i in train_idx]
        test_srs   = [srs[i] for i in test_idx]

        if verbose:
            print(f'\n── Project {proj}  (test={len(test_recs)}, train={len(train_recs)}) ──')

        # ── k-NN ─────────────────────────────────────────────────────────────
        train_pairs = [(train_srs[i], train_recs[i]['types'])
                       for i in range(len(train_recs))]
        knn_preds = []
        for j, ts in enumerate(test_srs):
            knn_preds.append(knn_classify(ts, train_pairs))
            if verbose and (j + 1) % 10 == 0:
                print(f'   kNN {j+1}/{len(test_srs)}', end='\r')

        # ── SVM + NB: vectorize, train, score ────────────────────────────────
        train_texts = [remove_determiners(r['text']) for r in train_recs]
        test_texts  = [remove_determiners(r['text']) for r in test_recs]
        vec  = CountVectorizer()
        X_tr = vec.fit_transform(train_texts)
        X_te = vec.transform(test_texts)

        train_svm_scores = {t: [] for t in nfr_types}
        test_svm_scores  = {t: [] for t in nfr_types}
        train_nb_probas  = {t: [] for t in nfr_types}
        test_nb_probas   = {t: [] for t in nfr_types}

        for t in nfr_types:
            y_tr = [1 if t in r['types'] else 0 for r in train_recs]
            if sum(y_tr) == 0:
                train_svm_scores[t] = [0.0] * len(train_recs)
                test_svm_scores[t]  = [0.0] * len(test_recs)
                train_nb_probas[t]  = [0.0] * len(train_recs)
                test_nb_probas[t]   = [0.0] * len(test_recs)
                continue
            svm = LinearSVC(max_iter=5000, dual='auto').fit(X_tr, y_tr)
            nb  = MultinomialNB().fit(X_tr, y_tr)
            train_svm_scores[t] = list(svm.decision_function(X_tr))
            test_svm_scores[t]  = list(svm.decision_function(X_te))
            train_nb_probas[t]  = list(nb.predict_proba(X_tr)[:, 1])
            test_nb_probas[t]   = list(nb.predict_proba(X_te)[:, 1])

        # ── Tune thresholds on training data only ────────────────────────────
        svm_thr, svm_tr_f2 = find_best_threshold(
            train_recs, train_svm_scores, 'svm', SVM_THRESHOLDS, nfr_types)
        nb_thr, nb_tr_f2 = find_best_threshold(
            train_recs, train_nb_probas, 'nb', NB_THRESHOLDS, nfr_types)

        # ── Apply tuned thresholds to test fold ──────────────────────────────
        svm_preds = {t: [1 if s >= svm_thr else 0 for s in test_svm_scores[t]]
                     for t in nfr_types}
        nb_preds  = {t: [1 if s >= nb_thr  else 0 for s in test_nb_probas[t]]
                     for t in nfr_types}

        knn_m = macro_avg(per_type_metrics(test_recs, knn_preds, nfr_types, 'knn'), nfr_types)
        svm_m = macro_avg(per_type_metrics(test_recs, svm_preds, nfr_types, 'svm'), nfr_types)
        nb_m  = macro_avg(per_type_metrics(test_recs, nb_preds,  nfr_types, 'nb'),  nfr_types)

        fold_rows.append({
            'project': proj, 'n_test': len(test_recs),
            'svm_theta_star': svm_thr, 'svm_train_f2': round(svm_tr_f2, 4),
            'nb_theta_star':  nb_thr,  'nb_train_f2':  round(nb_tr_f2,  4),
            'knn_f2': round(knn_m['f2'], 4),
            'svm_f2': round(svm_m['f2'], 4),
            'nb_f2':  round(nb_m['f2'],  4),
        })

        if verbose:
            print(f'   kNN F2={knn_m["f2"]:.3f}'
                  f'  |  SVM theta*={svm_thr:+.1f} train-F2={svm_tr_f2:.3f}'
                  f' test-F2={svm_m["f2"]:.3f}'
                  f'  |  NB theta*={nb_thr:.2f} train-F2={nb_tr_f2:.3f}'
                  f' test-F2={nb_m["f2"]:.3f}'
                  f'  ({time()-t_fold:.0f}s)')

        pooled_test_recs.extend(test_recs)
        pooled_knn_preds.extend(knn_preds)
        for t in nfr_types:
            pooled_svm_preds[t].extend(svm_preds[t])
            pooled_nb_preds[t].extend(nb_preds[t])

    # ── Aggregate pooled metrics ──────────────────────────────────────────────
    print('\n══ AGGREGATE RESULTS (pooled — equal weight per requirement) ══════════')
    summary = {}
    clf_labels = [
        ('knn', pooled_knn_preds, 'P011-k-NN'),
        ('svm', pooled_svm_preds, 'P011-SVM'),
        ('nb',  pooled_nb_preds,  'P011-Na\\"{i}ve Bayes'),
    ]
    for clf, preds, label in clf_labels:
        pt = per_type_metrics(pooled_test_recs, preds, nfr_types, clf)
        agg = macro_avg(pt, nfr_types)
        summary[clf] = {'macro': agg, 'per_type': pt, 'label': label}
        print(f'  {clf.upper():3s}: R={agg["recall"]:.3f}  P={agg["precision"]:.3f}'
              f'  F1={agg["f1"]:.3f}  F2={agg["f2"]:.3f}')
        if verbose:
            print(f'       {"type":<20} {"R":>6} {"P":>6} {"F2":>6}  TP  FP  FN')
            for t in nfr_types:
                m = pt[t]
                print(f'       {t:<20} {m["recall"]:>6.3f} {m["precision"]:>6.3f}'
                      f' {m["f2"]:>6.3f}  {m["tp"]:3d} {m["fp"]:3d} {m["fn"]:3d}')

    # ── Save results ──────────────────────────────────────────────────────────
    with open(results_dir / 'aggregate_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['classifier', 'type', 'tp', 'fp', 'fn', 'tn',
                         'recall', 'precision', 'specificity', 'f1', 'f2'])
        for clf, _, _ in clf_labels:
            pt = summary[clf]['per_type']
            for t in nfr_types:
                m = pt[t]
                writer.writerow([clf, t,
                                  m['tp'], m['fp'], m['fn'], m['tn'],
                                  round(m['recall'], 4), round(m['precision'], 4),
                                  round(m['specificity'], 4),
                                  round(m['f1'], 4), round(m['f2'], 4)])
            agg = summary[clf]['macro']
            writer.writerow([clf, 'macro-avg', '', '', '', '',
                              round(agg['recall'], 4), round(agg['precision'], 4),
                              round(agg['specificity'], 4),
                              round(agg['f1'], 4), round(agg['f2'], 4)])

    with open(results_dir / 'fold_thresholds.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'project', 'n_test',
            'svm_theta_star', 'svm_train_f2',
            'nb_theta_star',  'nb_train_f2',
            'knn_f2', 'svm_f2', 'nb_f2'])
        writer.writeheader()
        writer.writerows(fold_rows)

    with open(results_dir / 'summary.json', 'w') as f:
        json.dump({clf: {'macro': summary[clf]['macro'],
                         'per_type': summary[clf]['per_type']}
                   for clf, _, _ in clf_labels}, f, indent=2)

    print(f'\nResults written to {results_dir.resolve()}/')

    # ── Update allresults.txt ─────────────────────────────────────────────────
    if ALLRESULTS_PATH.exists():
        for clf, _, label in clf_labels:
            update_allresults(label, summary[clf]['macro'],
                              summary[clf]['per_type'])

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='P011 Slankas k-NN / SVM / NB — LOPO with F2-tuned thresholds')
    parser.add_argument('--data',        type=Path, default=DEFAULT_CSV)
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--quiet',       action='store_true')
    args = parser.parse_args()
    run_experiment(data_path=args.data, results_dir=args.results_dir,
                   verbose=not args.quiet)
