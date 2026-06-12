#!/usr/bin/env python3
"""
run_nodebate_devgpt.py — NFR quality classifier for DevGPT conversations

Two-stage pipeline per User turn:

  Stage 0 — Quality gate
    Ask: does this text express a concern about, ask about, or imply a
    constraint on any software quality property?
    → NOT_REQ: skip entirely (omitted from all output)
    → REQ: proceed to Stage 1

  Stage 1 — Quality type classifier
    Ask: which of the 13 quality types in KB-New.json does this text relate to?
    Returns a list of matching types (can be multiple) with confidence.

Only turns that pass Stage 0 appear in any output.

Usage:
    python3 run_nodebate_devgpt.py                          # default snapshot + model
    python3 run_nodebate_devgpt.py --snapshot snapshot_20230803
    python3 run_nodebate_devgpt.py --model claude-sonnet-4-6
    python3 run_nodebate_devgpt.py --limit 100 --dry-run    # quick test
    python3 run_nodebate_devgpt.py --force                  # discard checkpoint, restart
    python3 run_nodebate_devgpt.py --labels labels.csv      # add ground truth → metrics
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Constants ──────────────────────────────────────────────────────────────────

DEVGPT_ROOT      = Path(__file__).parent
DEFAULT_SNAPSHOT = 'snapshot_20230727'
DEFAULT_KB       = Path.home() / 'new-dataset/KB-New.json'
DEFAULT_MODEL    = 'claude-sonnet-4-6'
DEFAULT_MAX_LEN  = 3000

RETRY_DELAYS = [2, 4, 8, 16, 32]

SYSTEM_PROMPT = (
    "You are an expert software engineer and requirements analyst. "
    "You are precise and always output valid JSON."
)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_devgpt(snapshot_dir: Path, max_len: int) -> list[dict]:
    data_path = snapshot_dir / 'conversations_extracted.json'
    with open(data_path, encoding='utf-8') as f:
        conversations = json.load(f)

    records = []
    for conv_idx, item in enumerate(conversations):
        source_type = item.get('source_type', 'unknown')
        source_url  = item.get('source_url', '')
        chatgpt_url = item.get('chatgpt_url', '')

        for turn_idx, turn in enumerate(item.get('conversations', [])):
            text = turn.get('User', '').strip()
            if not text:
                continue

            truncated = len(text) > max_len
            records.append({
                'id':          f"conv{conv_idx:04d}_t{turn_idx:02d}",
                'text':        text[:max_len] if truncated else text,
                'project':     source_type,
                'source_type': source_type,
                'source_url':  source_url,
                'chatgpt_url': chatgpt_url,
                'truncated':   truncated,
            })
    return records

# ── API call with retry ────────────────────────────────────────────────────────

def call_api(client, model: str, prompt: str,
             dry_run: bool = False, label: str = '') -> str | None:
    if dry_run:
        print(f"\n── PROMPT [{label}] {'─'*40}")
        print(prompt[:500] + ("..." if len(prompt) > 500 else ""))
        return None

    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return next((b.text for b in msg.content if b.type == 'text'), None)
        except anthropic.RateLimitError:
            if attempt < len(RETRY_DELAYS):
                print(f"    [{label}] rate limit — retrying in {RETRY_DELAYS[attempt]}s ...",
                      flush=True)
                continue
            raise
        except anthropic.APIError as e:
            print(f"    [{label}] API error: {e}", flush=True)
            return None
    return None

# ── JSON parsing ───────────────────────────────────────────────────────────────

def parse_json(text: str | None, default):
    if not text:
        return default
    for candidate in [
        text.strip(),
        re.sub(r'^```(?:json)?\s*', '', text.strip()).rstrip('`').strip(),
    ]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in ('{', '['):
            try:
                obj, _ = decoder.raw_decode(text, i)
                return obj
            except json.JSONDecodeError:
                pass
    return default

# ── Stage 0: Quality gate ──────────────────────────────────────────────────────

GATE_PROMPT = """\
You are reviewing a turn from a GitHub developer conversation with ChatGPT.

Decide whether this text expresses a concern about, asks about, or implies \
a constraint on any software quality property — such as how fast, secure, \
reliable, available, usable, maintainable, portable, legally compliant, or \
safe the software should be.

Answer YES if the text:
  - States a quality constraint or preference ("I never want downtime", \
"it must be fast", "this needs to be secure")
  - Asks a question motivated by a quality concern ("is v-html safe?", \
"which is more gas-efficient?", "how do I add a health check?")
  - Discusses a trade-off or risk related to a quality property

Answer NO if the text is:
  - A pure code or data paste with no quality intent (XML dumps, stack \
traces, JSON blobs, string arrays)
  - Conversational filler ("thank you", "ok", "continue", "I understand")
  - A purely functional task with no quality angle ("sort these alphabetically", \
"convert to French")
  - A general how-does-X-work question with no quality concern

Text:
\"\"\"{text}\"\"\"

Return JSON: {{"is_quality_relevant": true or false, \
"confidence": <0.0–1.0>, "reason": "<one sentence>"}}"""

def run_gate(client, model: str, text: str, dry_run: bool) -> dict:
    response = call_api(client, model,
                        GATE_PROMPT.format(text=text[:2000]),
                        dry_run, label='gate')
    if dry_run:
        return {'is_quality_relevant': True, 'confidence': 1.0, 'reason': '[dry-run]'}
    parsed = parse_json(response, {})
    return {
        'is_quality_relevant': bool(parsed.get('is_quality_relevant', True)),
        'confidence':          float(parsed.get('confidence', 0.5)),
        'reason':              parsed.get('reason', ''),
    }

# ── Stage 1: Quality type classifier ──────────────────────────────────────────

def build_classify_prompt(text: str, kb: dict) -> str:
    type_lines = '\n'.join(
        f"  {t}: {kb[t]['core_concern'].rstrip('.')}."
        for t in kb
    )
    type_names = ', '.join(kb.keys())
    return (
        f"A developer wrote the following to ChatGPT while working on a software project.\n\n"
        f"Text:\n\"\"\"{text}\"\"\"\n\n"
        f"Which of these software quality properties does the text relate to? "
        f"List ALL that apply — even if the concern is implicit or expressed as a question.\n\n"
        f"Quality types:\n{type_lines}\n\n"
        f"Return JSON: {{\"types\": [<zero or more names from: {type_names}>], "
        f"\"confidence\": <0.0–1.0>, "
        f"\"reason\": \"<one sentence>\"}}"
    )

def run_classify(client, model: str, text: str, kb: dict, dry_run: bool) -> dict:
    prompt   = build_classify_prompt(text, kb)
    response = call_api(client, model, prompt, dry_run, label='classify')
    if dry_run:
        return {'types': list(kb.keys())[:2], 'confidence': 1.0, 'reason': '[dry-run]'}
    parsed = parse_json(response, {})
    valid  = set(kb.keys())
    types  = [t for t in parsed.get('types', []) if t in valid]
    return {
        'types':      types,
        'confidence': float(parsed.get('confidence', 0.5)),
        'reason':     parsed.get('reason', ''),
    }

# ── Classify one turn ──────────────────────────────────────────────────────────

def classify_record(client, model: str, record: dict, kb: dict,
                    dry_run: bool) -> dict:
    base = {
        'id':          record['id'],
        'text':        record['text'],
        'source_type': record['source_type'],
        'source_url':  record['source_url'],
        'chatgpt_url': record['chatgpt_url'],
        'truncated':   record['truncated'],
    }

    gate = run_gate(client, model, record['text'], dry_run)
    if not gate['is_quality_relevant']:
        return {**base, 'outcome': 'NOT_REQ', 'gate': gate, 'types': [], 'classification': {}}

    classification = run_classify(client, model, record['text'], kb, dry_run)
    return {**base, 'outcome': 'REQ', 'gate': gate,
            'types': classification['types'], 'classification': classification}

# ── Evaluation (when labels are provided) ──────────────────────────────────────

def load_labels(path: Path, valid_types: set) -> dict[str, list[str]]:
    labels = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            item_id = row['id'].strip()
            raw     = row.get('true_types', '').strip()
            types   = [t.strip() for t in raw.split(';') if t.strip() in valid_types]
            labels[item_id] = types
    return labels

def compute_metrics(results: list[dict],
                    labels: dict[str, list[str]],
                    all_types: list[str]) -> tuple[dict, dict]:
    labelled = [r for r in results if r['id'] in labels]
    if not labelled:
        print("WARNING: no result IDs matched the labels file.")
        return {}, {}

    per_type: dict[str, dict] = {}
    for t in all_types:
        tp = sum(1 for r in labelled if t in labels[r['id']] and t in r['types'])
        fp = sum(1 for r in labelled if t not in labels[r['id']] and t in r['types'])
        fn = sum(1 for r in labelled if t in labels[r['id']] and t not in r['types'])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_type[t] = dict(prec=prec, rec=rec, f1=f1, tp=tp, fp=fp, fn=fn)

    n     = len(all_types)
    macro = {k: sum(per_type[t][k] for t in all_types) / n
             for k in ('prec', 'rec', 'f1')}
    return per_type, macro

def print_metrics(per_type: dict, macro: dict, all_types: list[str]) -> None:
    print(f"\n  {'Type':<22}  {'Rec':>6}  {'Prec':>6}  {'F1':>6}  "
          f"{'TP':>4}  {'FP':>4}  {'FN':>4}")
    print('  ' + '-' * 65)
    for t in all_types:
        if t not in per_type:
            continue
        v = per_type[t]
        print(f"  {t:<22}  {v['rec']:>6.3f}  {v['prec']:>6.3f}  {v['f1']:>6.3f}  "
              f"{v['tp']:>4}  {v['fp']:>4}  {v['fn']:>4}")
    print('  ' + '-' * 65)
    print(f"  {'macro':<22}  {macro['rec']:>6.3f}  {macro['prec']:>6.3f}  "
          f"{macro['f1']:>6.3f}")

# ── Output writers (REQ-only) ──────────────────────────────────────────────────

def write_results_csv(results: list[dict], path: Path) -> None:
    req = [r for r in results if r['outcome'] == 'REQ']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'source_type', 'source_url', 'chatgpt_url',
                    'quality_types', 'confidence', 'gate_conf',
                    'reason', 'truncated', 'text'])
        for r in req:
            cl = r['classification']
            w.writerow([
                r['id'],
                r['source_type'],
                r['source_url'],
                r['chatgpt_url'],
                ';'.join(r['types']),
                cl.get('confidence', ''),
                r['gate'].get('confidence', ''),
                cl.get('reason', ''),
                r['truncated'],
                r['text'],
            ])

def write_scores_json(results: list[dict], path: Path) -> None:
    """Write full results for all turns (including NOT_REQ) — used for re-evaluation."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='NFR quality classifier for DevGPT conversations (two-stage)')
    parser.add_argument('--snapshot',  default=DEFAULT_SNAPSHOT,
                        help='Snapshot folder name under the DevGPT repo root')
    parser.add_argument('--kb',        type=Path, default=DEFAULT_KB,
                        help='Path to KB-New.json')
    parser.add_argument('--model',     default=DEFAULT_MODEL,
                        help='Claude model ID (default: claude-sonnet-4-6)')
    parser.add_argument('--max-len',   type=int, default=DEFAULT_MAX_LEN,
                        help='Truncate User turns beyond this length (default 3000)')
    parser.add_argument('--labels',    type=Path, default=None,
                        help='CSV id,true_types for Recall/Precision evaluation')
    parser.add_argument('--limit',     type=int, default=None,
                        help='Process only the first N turns')
    parser.add_argument('--force',     action='store_true',
                        help='Delete existing checkpoint and restart')
    parser.add_argument('--dry-run',   action='store_true',
                        help='Print prompts without API calls')
    args = parser.parse_args()

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    snapshot_dir = DEVGPT_ROOT / args.snapshot
    if not snapshot_dir.exists():
        print(f"ERROR: snapshot directory not found: {snapshot_dir}")
        sys.exit(1)

    results_dir = snapshot_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    with open(args.kb) as f:
        kb = json.load(f)
    all_types = list(kb.keys())

    records = load_devgpt(snapshot_dir, args.max_len)
    if args.limit:
        records = records[:args.limit]

    model_slug = args.model.replace('/', '-')
    tag = f"devgpt_{model_slug}"
    if args.limit:
        tag = f"{tag}_top{args.limit}"

    print(f"\n{'='*60}")
    print(f"DevGPT NFR Classifier  |  model: {args.model}")
    print(f"Snapshot:  {args.snapshot}  |  Turns: {len(records)}")
    print(f"KB: {args.kb.name}  ({len(all_types)} types)")
    print(f"{'='*60}\n")

    client = None if args.dry_run else anthropic.Anthropic(api_key=api_key)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    checkpoint_path = results_dir / f"checkpoint_{tag}.jsonl"
    done_ids: set[str]      = set()
    all_results: list[dict] = []

    if checkpoint_path.exists():
        if args.force:
            checkpoint_path.unlink()
            print("--force: checkpoint deleted, restarting.\n")
        else:
            with open(checkpoint_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        all_results.append(d)
                        done_ids.add(d['id'])
            print(f"Resuming — {len(done_ids)} turns already done.")

    checkpoint_file = open(checkpoint_path, 'a')
    req_count = sum(1 for r in all_results if r['outcome'] == 'REQ')

    try:
        pending = [r for r in records if r['id'] not in done_ids]
        print(f"Pending: {len(pending)}\n")

        for rec in pending:
            result = classify_record(client, args.model, rec, kb, args.dry_run)
            all_results.append(result)
            checkpoint_file.write(json.dumps(result) + '\n')
            checkpoint_file.flush()

            if result['outcome'] == 'NOT_REQ':
                # brief skip indicator so you can track progress
                print(f"  {rec['id']}  [skip]  [{rec['source_type']}]  "
                      f"\"{rec['text'][:60].replace(chr(10),' ')}\"",
                      flush=True)
            else:
                req_count += 1
                cl = result['classification']
                types_str = ', '.join(result['types']) or '(none assigned)'
                conf = f"{cl.get('confidence', 0):.2f}"
                print(f"  {rec['id']}  [REQ conf={conf}]  [{rec['source_type']}]")
                print(f"    types: {types_str}")
                print(f"    \"{rec['text'][:100].replace(chr(10),' ')}\"")
                print(f"    → {cl.get('reason', '')}", flush=True)
    finally:
        checkpoint_file.close()

    if args.dry_run:
        print("\nDry run complete — no API calls made.")
        return

    req_results = [r for r in all_results if r['outcome'] == 'REQ']
    not_req_count = len(all_results) - len(req_results)

    # ── Write outputs (REQ only) ──────────────────────────────────────────────
    csv_path = results_dir / f"results_{tag}.csv"
    write_results_csv(all_results, csv_path)
    print(f"\nResults CSV  → {csv_path}  ({len(req_results)} REQ turns)")

    scores_path = results_dir / f"scores_{tag}.json"
    write_scores_json(all_results, scores_path)
    print(f"Scores JSON  → {scores_path}  ({len(all_results)} total turns)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Gate summary {'─'*44}")
    print(f"  NOT_REQ (filtered): {not_req_count:>5}")
    print(f"  REQ (passed):       {len(req_results):>5}")

    from collections import Counter
    type_counts = Counter(t for r in req_results for t in r['types'])
    print(f"\n── Quality types in REQ turns {'─'*30}")
    for t in all_types:
        print(f"  {t:<22}: {type_counts.get(t, 0):>4}")

    src_counts: dict[str, dict] = {}
    for r in all_results:
        st = r['source_type']
        src_counts.setdefault(st, {'req': 0, 'total': 0})
        src_counts[st]['total'] += 1
        if r['outcome'] == 'REQ':
            src_counts[st]['req'] += 1
    print(f"\n── REQ rate by source type {'─'*33}")
    for st, c in sorted(src_counts.items()):
        pct = 100 * c['req'] / c['total'] if c['total'] else 0
        print(f"  {st:<14}: {c['req']:>4} / {c['total']:<4}  ({pct:.1f}%)")

    # ── Evaluation ────────────────────────────────────────────────────────────
    if args.labels:
        print(f"\n── Evaluation vs {args.labels.name} {'─'*30}")
        labels = load_labels(args.labels, set(all_types))
        per_type, macro = compute_metrics(req_results, labels, all_types)
        if per_type:
            print_metrics(per_type, macro, all_types)
            metrics_path = results_dir / f"metrics_{tag}.csv"
            with open(metrics_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['type', 'recall', 'precision', 'f1', 'tp', 'fp', 'fn'])
                for t in all_types:
                    v = per_type[t]
                    w.writerow([t, round(v['rec'],4), round(v['prec'],4),
                                round(v['f1'],4), v['tp'], v['fp'], v['fn']])
                w.writerow(['macro', round(macro['rec'],4), round(macro['prec'],4),
                            round(macro['f1'],4), '', '', ''])
            print(f"\n  Metrics → {metrics_path}")

    checkpoint_path.unlink(missing_ok=True)
    print()


if __name__ == '__main__':
    main()
