#!/usr/bin/env python3
"""
P5 — Debate-Based LLM NFR Classifier

Multi-agent debate architecture (one requirement at a time):
  1. Initial screen  — single lightweight call; returns one of:
                         FR     → accepted as functional requirement (done, 1 call)
                         NFR    → one unambiguous type accepted directly (done, 1 call)
                         DEBATE → two or more plausible types; triggers full pipeline
  2. Advocate screening   — each candidate answers "do you see evidence? confidence?"
                            (no argument yet; parallel calls)
  3. Advocate filter      — drops low-confidence screeners; only believers argue
  4. Advocate arguments   — surviving advocates make their full case (parallel)
  5. Devil's advocate     — sees all arguments, then argues this is a Functional
                            Requirement; backs down if NFR evidence is strong
  6. Arbiter              — weighs all arguments, returns final label(s);
                            errs toward NFR if borderline (recall > precision)

Protocol: leave-one-project-out cross-validation (15 folds), matching REJ baseline.
Debate transcripts saved as a citable JSON artifact.

Usage:
    python3 run_debate.py                        # all requirements, default model
    python3 run_debate.py --dry-run              # print prompts, no API calls
    python3 run_debate.py --model claude-opus-4-8
    python3 run_debate.py --da-threshold 0.6       # lower = devil's advocate backs down sooner
    python3 run_debate.py --max-candidates 3       # fewer advocate agents per requirement
    python3 run_debate.py --advocate-threshold 0.4 # raise to require stronger screen confidence
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_DATA    = Path('../promise-june2026.csv')
DEFAULT_RESULTS = Path('results')
DEFAULT_KB      = Path('KB.json')
DEFAULT_MODEL   = 'claude-sonnet-4-6'
DEFAULT_DA_THRESHOLD        = 0.7  # devil's advocate confidence above this → FR challenge sustained
DEFAULT_MAX_CANDIDATES      = 4    # advocate identifier returns at most this many types
DEFAULT_ADVOCATE_THRESHOLD  = 0.3  # screen confidence below this → advocate sits out

ALLRESULTS_PATH    = Path('../allresults.txt')
_RESULT_TYPE_ORDER = [
    'availability', 'legal', 'look-and-feel', 'maintainability',
    'operational', 'performance', 'scalability', 'security', 'usability',
]

NFR_TYPES = [
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

RETRY_DELAYS = [2, 4, 8, 16, 32]

SYSTEM_PROMPT = (
    "You are an expert requirements engineer specialising in classifying software "
    "requirements by Non-Functional Requirement (NFR) type. You are precise, "
    "systematic, and always output valid JSON."
)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(path: Path) -> list[dict]:
    records = []
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            text  = row['RequirementText'].strip().strip("'")
            types = [name for col, name in CSV_COLUMN_MAP.items()
                     if row.get(col, '0').strip() == '1']
            records.append({
                'id':      row['ReqID'].strip(),
                'text':    text,
                'types':   types,
                'project': row['ProjectID'].strip(),
            })
    return records

# ── API call with retry ────────────────────────────────────────────────────────

def call_api(client: anthropic.Anthropic, model: str, prompt: str,
             dry_run: bool = False, label: str = '', max_tokens: int = 1024) -> str | None:
    if dry_run:
        print(f"\n── PROMPT [{label}] {'─' * 40}")
        print(prompt[:600] + ("..." if len(prompt) > 600 else ""))
        print("─" * 60)
        return None

    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except anthropic.RateLimitError:
            print(f"    [{label}] rate limit — retrying in {RETRY_DELAYS[attempt]}s ...", flush=True)
            if attempt < len(RETRY_DELAYS):
                continue
            raise
        except anthropic.APIError as e:
            print(f"    [{label}] error: {e}", flush=True)
            return None
    return None

# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_conf(c: float | None) -> str:
    """Format a confidence value. None means dry-run placeholder — show [?] instead."""
    return f"{c:.2f}" if c is not None else "[?]"

# ── JSON parsing ───────────────────────────────────────────────────────────────

def parse_json_response(text: str | None, default):
    """Extract the first valid JSON object or array from model output."""
    if not text:
        return default
    # Try whole text (model returned pure JSON)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences and retry
    stripped = re.sub(r'^```(?:json)?\s*', '', text.strip())
    stripped = re.sub(r'\s*```$', '', stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Scan forward for the first parseable JSON object or array
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in ('{', '['):
            try:
                obj, _ = decoder.raw_decode(text, i)
                return obj
            except json.JSONDecodeError:
                pass
    return default

# ── Stage 1: Advocate Identifier ──────────────────────────────────────────────

def build_initial_screen_prompt(requirement: str, kb: dict, max_candidates: int) -> str:
    type_summaries = '\n'.join(
        f"  - {t}: {kb[t]['core_concern'].split('.')[0]}."
        for t in NFR_TYPES
    )
    return (
        f"Classify the following software requirement.\n\n"
        f"Requirement: \"{requirement}\"\n\n"
        f"NFR type summaries:\n{type_summaries}\n\n"
        f"Decide — pick exactly one outcome:\n"
        f"  FR     — this is clearly a functional requirement (what the system does), "
        f"not a quality constraint\n"
        f"  NFR    — this is clearly one specific NFR type with no genuine ambiguity\n"
        f"  DEBATE — this could plausibly belong to more than one NFR type\n\n"
        f"Only return NFR if you are confident a single type fits and no other type "
        f"is a serious contender. When in doubt, return DEBATE.\n"
        f"Only return FR if you are certain the requirement describes purely functional "
        f"behaviour with absolutely no quality constraint. A missed NFR is a worse error "
        f"than triggering an unnecessary debate. When in any doubt between FR and DEBATE, "
        f"return DEBATE.\n\n"
        f"Return JSON: {{\"outcome\": \"FR\" | \"NFR\" | \"DEBATE\", "
        f"\"types\": [<empty for FR, one name for NFR, 2–{max_candidates} names for DEBATE>], "
        f"\"confidence\": <0.0–1.0>, "
        f"\"reason\": \"<one sentence>\"}}"
    )

def initial_screen(client, model: str, requirement: str, kb: dict,
                   max_candidates: int, dry_run: bool) -> dict:
    prompt   = build_initial_screen_prompt(requirement, kb, max_candidates)
    response = call_api(client, model, prompt, dry_run, label='initial-screen')
    if dry_run:
        return {'outcome': 'DEBATE', 'types': NFR_TYPES[:max_candidates],
                'confidence': None, 'reason': '[dry-run]'}
    parsed  = parse_json_response(response, {})
    outcome = parsed.get('outcome', 'DEBATE')
    types   = [t for t in parsed.get('types', []) if t in NFR_TYPES]

    if outcome == 'FR':
        types = []
    elif outcome == 'NFR':
        if len(types) != 1:
            outcome = 'DEBATE'          # malformed → fall through to debate
    elif outcome == 'DEBATE':
        if not types:
            types = list(NFR_TYPES)     # no candidates returned → debate everything

    return {
        'outcome':    outcome,
        'types':      types[:max_candidates],
        'confidence': float(parsed.get('confidence', 0.5)),
        'reason':     parsed.get('reason', ''),
    }

# ── Stage 2a: Advocate Screening ──────────────────────────────────────────────

def build_screen_prompt(requirement: str, nfr_type: str, kb: dict) -> str:
    type_kb = kb[nfr_type]
    items   = '\n'.join(f"  {i+1}. {item}"
                        for i, item in enumerate(type_kb['classify_if']))
    return (
        f"You are being considered as an advocate for the NFR type '{nfr_type}'.\n\n"
        f"Requirement: \"{requirement}\"\n\n"
        f"Definition of '{nfr_type}':\n"
        f"Core concern: {type_kb['core_concern']}\n\n"
        f"Classify as '{nfr_type}' if the requirement:\n{items}\n\n"
        f"Do NOT argue yet. Simply assess: is there genuine evidence in this requirement "
        f"that it belongs to the '{nfr_type}' category?\n\n"
        f"Return JSON: {{\"evidence_seen\": true or false, "
        f"\"confidence\": <0.0–1.0>, "
        f"\"brief_reason\": \"<one sentence>\"}}"
    )

def screen_advocates(client, model: str, requirement: str,
                     candidates: list[str], kb: dict, dry_run: bool) -> list[dict]:
    def screen(nfr_type: str) -> dict:
        prompt   = build_screen_prompt(requirement, nfr_type, kb)
        response = call_api(client, model, prompt, dry_run, label=f'screen-{nfr_type}')
        if dry_run:
            return {'type': nfr_type, 'evidence_seen': True,
                    'confidence': None, 'brief_reason': '[dry-run]'}
        parsed = parse_json_response(response, {})
        return {
            'type':         nfr_type,
            'evidence_seen': bool(parsed.get('evidence_seen', True)),
            'confidence':   float(parsed.get('confidence', 0.5)),
            'brief_reason': parsed.get('brief_reason', ''),
        }

    if dry_run:
        return [screen(t) for t in candidates]
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        return list(pool.map(screen, candidates))

# ── Stage 2b: Advocate Filter ──────────────────────────────────────────────────

def filter_advocates(screen_results: list[dict], threshold: float) -> list[dict]:
    """Keep screened advocates above threshold. Fall back to top-1 if none qualify.
    If confidence is None (dry-run placeholders), keep all candidates unfiltered."""
    if any(r['confidence'] is None for r in screen_results):
        return screen_results
    qualified = [r for r in screen_results if r['confidence'] >= threshold]
    if not qualified:
        qualified = [max(screen_results, key=lambda r: r['confidence'])]
    return qualified

# ── Stage 2c: Advocate Arguments ──────────────────────────────────────────────

def build_argue_prompt(requirement: str, nfr_type: str, kb: dict) -> str:
    type_kb = kb[nfr_type]
    items   = '\n'.join(f"  {i+1}. {item}"
                        for i, item in enumerate(type_kb['classify_if']))
    return (
        f"You are an advocate arguing that the following software requirement "
        f"should be classified as '{nfr_type}'.\n\n"
        f"Requirement: \"{requirement}\"\n\n"
        f"Definition of '{nfr_type}':\n"
        f"Core concern: {type_kb['core_concern']}\n\n"
        f"Classify as '{nfr_type}' if the requirement:\n{items}\n\n"
        f"Boundary notes: {type_kb['boundary_notes']}\n\n"
        f"Make the strongest possible case FOR this classification. "
        f"Quote specific phrases from the requirement and map them to the criteria above. "
        f"Be honest about weaknesses in your case.\n\n"
        f"Return JSON: {{\"type\": \"{nfr_type}\", "
        f"\"argument\": \"<your argument>\", "
        f"\"confidence\": <0.0–1.0>}}"
    )

def argue_advocates(client, model: str, requirement: str,
                    qualified: list[dict], kb: dict, dry_run: bool) -> list[dict]:
    def argue(nfr_type: str) -> dict:
        prompt   = build_argue_prompt(requirement, nfr_type, kb)
        response = call_api(client, model, prompt, dry_run, label=f'argue-{nfr_type}')
        if dry_run:
            return {'type': nfr_type, 'argument': '[dry-run]', 'confidence': None}
        parsed = parse_json_response(response, {})
        return {
            'type':       nfr_type,
            'argument':   parsed.get('argument', response or ''),
            'confidence': float(parsed.get('confidence', 0.5)),
        }

    types = [r['type'] for r in qualified]
    if dry_run:
        return [argue(t) for t in types]
    with ThreadPoolExecutor(max_workers=len(types)) as pool:
        return list(pool.map(argue, types))

# ── Stage 3: Devil's Advocate ──────────────────────────────────────────────────

def build_devils_advocate_prompt(requirement: str, advocate_results: list[dict]) -> str:
    advocate_block = '\n\n'.join(
        f"Advocate for '{r['type']}' (confidence {_fmt_conf(r['confidence'])}):\n{r['argument']}"
        for r in advocate_results
    )
    return (
        f"You are the devil's advocate in a classification debate.\n\n"
        f"Requirement: \"{requirement}\"\n\n"
        f"NFR advocates have argued:\n\n{advocate_block}\n\n"
        f"Your role: argue that this is a FUNCTIONAL REQUIREMENT — "
        f"a statement of what the system should DO, not a quality constraint on HOW it does it. "
        f"Functional requirements describe specific behaviours, features, or outputs. "
        f"NFRs constrain quality attributes such as speed, safety, security, or availability.\n\n"
        f"The burden of proof is on you: back down and set fr_verdict to 'NFR' unless "
        f"you are highly confident the requirement is purely functional with no quality "
        f"constraint whatsoever. If the advocates have provided any reasonable evidence "
        f"of a quality concern, concede — a missed NFR is a worse error than a false "
        f"positive. Only sustain your FR challenge if the case for functional behaviour "
        f"is overwhelming and the NFR arguments are clearly a stretch.\n\n"
        f"Return JSON: {{\"fr_verdict\": \"FR\" or \"NFR\", "
        f"\"confidence\": <0.0–1.0, where 1.0 = certain it is FR>, "
        f"\"argument\": \"<your reasoning>\"}}"
    )

def run_devils_advocate(client, model: str, requirement: str,
                         advocate_results: list[dict], dry_run: bool) -> dict:
    prompt   = build_devils_advocate_prompt(requirement, advocate_results)
    response = call_api(client, model, prompt, dry_run, label="devil's-advocate")
    if dry_run:
        return {'fr_verdict': 'NFR', 'confidence': None, 'argument': '[dry-run]'}
    parsed = parse_json_response(response, {})
    return {
        'fr_verdict': parsed.get('fr_verdict', 'NFR'),
        'confidence': float(parsed.get('confidence', 0.5)),
        'argument':   parsed.get('argument', response or ''),
    }

# ── Stage 4: Arbiter ───────────────────────────────────────────────────────────

def build_arbiter_prompt(requirement: str, advocate_results: list[dict],
                          da_result: dict, da_threshold: float) -> str:
    advocate_block = '\n\n'.join(
        f"Advocate for '{r['type']}' (confidence {_fmt_conf(r['confidence'])}):\n{r['argument']}"
        for r in advocate_results
    )
    da_conf = da_result['confidence']
    da_challenged = (da_result['fr_verdict'] == 'FR'
                     and da_conf is not None and da_conf >= da_threshold)
    da_status = (
        f"CHALLENGE SUSTAINED (FR confidence {_fmt_conf(da_conf)}) — "
        f"devil's advocate believes this is a functional requirement"
        if da_challenged else
        f"BACKED DOWN (FR confidence {_fmt_conf(da_conf)}) — "
        f"devil's advocate concedes this is an NFR"
    )
    return (
        f"You are the arbiter in an NFR classification debate.\n\n"
        f"Requirement: \"{requirement}\"\n\n"
        f"NFR advocate arguments:\n\n{advocate_block}\n\n"
        f"Devil's advocate — {da_status}:\n{da_result['argument']}\n\n"
        f"Your task: decide which NFR types (if any) apply. "
        f"A requirement may receive multiple labels.\n\n"
        f"IMPORTANT — err toward NFR if the case is borderline. "
        f"Favour recall over precision: a missed NFR is a worse error than a false positive. "
        f"Classify as FR (empty labels list) only if you are confident the requirement "
        f"describes purely functional behaviour with no quality constraint.\n\n"
        f"Return JSON: {{\"labels\": [<NFR type strings, or empty list for FR>], "
        f"\"reasoning\": \"<explanation>\", "
        f"\"fr_probability\": <0.0–1.0 probability this is FR>}}"
    )

def run_arbiter(client, model: str, requirement: str,
                advocate_results: list[dict], da_result: dict,
                da_threshold: float, dry_run: bool) -> dict:
    prompt   = build_arbiter_prompt(requirement, advocate_results, da_result, da_threshold)
    response = call_api(client, model, prompt, dry_run, label='arbiter', max_tokens=1536)
    if dry_run:
        return {
            'labels':         [r['type'] for r in advocate_results[:1]],
            'reasoning':      '[dry-run]',
            'fr_probability': 0.1,
        }
    parsed = parse_json_response(response, {})
    labels = [t for t in parsed.get('labels', []) if t in NFR_TYPES]
    return {
        'labels':         labels,
        'reasoning':      parsed.get('reasoning', response or ''),
        'fr_probability': float(parsed.get('fr_probability', 0.5)),
    }

# ── Full debate pipeline for one requirement ───────────────────────────────────

def debate_requirement(client, model: str, record: dict, kb: dict,
                        da_threshold: float, max_candidates: int,
                        advocate_threshold: float, dry_run: bool) -> dict:
    req    = record['text']
    base   = {'id': record['id'], 'text': req,
               'true_types': record['types'], 'project': record['project']}

    screen = initial_screen(client, model, req, kb, max_candidates, dry_run)

    if screen['outcome'] == 'FR':
        return {**base, 'initial_screen': screen,
                'screened': [], 'debating': [], 'advocates': [],
                'devils_advocate': None, 'arbiter': None, 'predicted': []}

    if screen['outcome'] == 'NFR':
        return {**base, 'initial_screen': screen,
                'screened': [], 'debating': screen['types'], 'advocates': [],
                'devils_advocate': None, 'arbiter': None, 'predicted': screen['types']}

    # DEBATE path — two or more contenders
    screen_results   = screen_advocates(client, model, req, screen['types'], kb, dry_run)
    qualified        = filter_advocates(screen_results, advocate_threshold)
    advocate_results = argue_advocates(client, model, req, qualified, kb, dry_run)
    da_result        = run_devils_advocate(client, model, req, advocate_results, dry_run)
    arbiter_result   = run_arbiter(client, model, req, advocate_results,
                                   da_result, da_threshold, dry_run)
    return {**base, 'initial_screen': screen,
            'screened':        screen_results,
            'debating':        [r['type'] for r in qualified],
            'advocates':       advocate_results,
            'devils_advocate': da_result,
            'arbiter':         arbiter_result,
            'predicted':       arbiter_result['labels']}

# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(records: list[dict], predictions: list[dict[str, bool]],
                    types: list[str]) -> dict:
    per_type = {}
    n = len(records)
    for t in types:
        tp = sum(1 for r, p in zip(records, predictions) if t in r['types'] and p[t])
        fp = sum(1 for r, p in zip(records, predictions) if t not in r['types'] and p[t])
        fn = sum(1 for r, p in zip(records, predictions) if t in r['types'] and not p[t])
        tn = n - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec)   if (prec+rec)   > 0 else 0.0
        f2   = 5*prec*rec / (4*prec+rec) if (4*prec+rec) > 0 else 0.0
        per_type[t] = dict(precision=prec, recall=rec, specificity=spec,
                           f1=f1, f2=f2, tp=tp, fp=fp, fn=fn, tn=tn)
    keys  = ('precision', 'recall', 'specificity', 'f1', 'f2')
    macro = {k: sum(per_type[t][k] for t in types) / len(types) for k in keys}
    return {'per_type': per_type, 'macro': macro}

# ── Experiment runner ──────────────────────────────────────────────────────────

def run_single(data_path: Path, kb_path: Path, model: str,
               da_threshold: float, max_candidates: int, advocate_threshold: float,
               req_ids: list[str], dry_run: bool, api_key: str | None) -> None:
    """Run and print the full debate for one or more specific requirement IDs."""
    with open(kb_path) as f:
        kb = json.load(f)

    records = load_data(data_path)
    targets = {r['id']: r for r in records}

    missing = [rid for rid in req_ids if rid not in targets]
    if missing:
        print(f"ERROR: unknown IDs: {missing}")
        sys.exit(1)

    client = None if dry_run else anthropic.Anthropic(api_key=api_key)

    for rid in req_ids:
        rec = targets[rid]
        print(f"\n{'='*60}")
        print(f"Requirement {rec['id']}  (project {rec['project']})")
        print(f"Text:       {rec['text']}")
        print(f"True types: {rec['types'] or ['FR']}")
        print(f"{'='*60}")

        result = debate_requirement(
            client, model, rec, kb, da_threshold, max_candidates,
            advocate_threshold, dry_run)

        sc = result['initial_screen']
        print(f"\n── Initial screen → {sc['outcome']}  (conf={_fmt_conf(sc['confidence'])}) ──")
        print(f"  {sc['reason']}")

        if not dry_run and sc['outcome'] == 'DEBATE':
            print(f"\n── Advocate screening ──")
            for s in result['screened']:
                status = 'IN ' if s['type'] in result['debating'] else 'OUT'
                print(f"  [{status}] {s['type']:<20} conf={_fmt_conf(s['confidence'])}  {s['brief_reason']}")

            print(f"\n── Arguments ──")
            for a in result['advocates']:
                print(f"  {a['type']:<20} conf={_fmt_conf(a['confidence'])}")
                print(f"    {a['argument']}")

            da = result['devils_advocate']
            print(f"\n── Devil's advocate → {da['fr_verdict']} (conf={_fmt_conf(da['confidence'])}) ──")
            print(f"  {da['argument']}")

            arb = result['arbiter']
            print(f"\n── Arbiter → {arb['labels'] or ['FR']}  (fr_prob={arb['fr_probability']:.2f}) ──")
            print(f"  {arb['reasoning']}")

        print(f"\n  Predicted: {result['predicted'] or ['FR']}")
        print(f"  True:      {rec['types'] or ['FR']}")


# ── allresults.txt update ──────────────────────────────────────────────────────

def update_allresults(summary: dict, path: Path = ALLRESULTS_PATH) -> None:
    """Replace the Agent Debate row in allresults.txt with computed values."""
    m  = summary['macro']
    pt = summary['per_type']
    per_type_f2 = ' & '.join(f'{pt[t]["f2"]:.3f}' for t in _RESULT_TYPE_ORDER)
    new_data_line = (
        f'  & {m["recall"]:.3f} & {m["precision"]:.3f}'
        f' & {m["f1"]:.3f} & {m["f2"]:.3f}'
        f' & {per_type_f2} \\\\\n'
    )
    lines = path.read_text().splitlines(keepends=True)
    out, replace_next = [], False
    for line in lines:
        if replace_next:
            out.append(new_data_line)
            replace_next = False
        elif line.strip() == 'Agent Debate':
            out.append(line)
            replace_next = True
        else:
            out.append(line)
    path.write_text(''.join(out))
    print(f'Updated {path}  (macro F2={m["f2"]:.3f})')


def run_agent_debate(data_path: Path, results_dir: Path, kb_path: Path,
                     model: str, da_threshold: float, max_candidates: int,
                     advocate_threshold: float, dry_run: bool,
                     api_key: str | None, limit: int | None = None,
                     force: bool = False) -> dict:

    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"debate_da{int(da_threshold * 100)}_c{max_candidates}_at{int(advocate_threshold * 100)}"
    if limit:
        tag = f"{tag}_top{limit}"

    with open(kb_path) as f:
        kb = json.load(f)

    records = load_data(data_path)
    if limit:
        records = records[:limit]

    print(f"\n{'='*60}")
    print(f"P5 Agent Debate  |  model: {model}")
    print(f"DA threshold: {da_threshold}  |  advocate threshold: {advocate_threshold}"
          f"  |  max candidates: {max_candidates}")
    print(f"Records: {len(records)}" + (f"  |  limit: {limit}" if limit else ''))
    print(f"{'='*60}")

    client = None if dry_run else anthropic.Anthropic(api_key=api_key)

    # ── Checkpoint: resume from prior partial run ──────────────────────────────
    checkpoint_path = results_dir / f"checkpoint_{tag}.jsonl"
    done_ids: set[str] = set()
    all_debates: list[dict] = []

    if checkpoint_path.exists():
        if force:
            checkpoint_path.unlink()
            print("--force: checkpoint deleted, restarting from scratch.")
        else:
            with open(checkpoint_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        all_debates.append(d)
                        done_ids.add(d['id'])
            print(f"Resuming — {len(done_ids)} requirements already done.")

    checkpoint_file = open(checkpoint_path, 'a')

    try:
        pending = [r for r in records if r['id'] not in done_ids]
        print(f"Pending: {len(pending)}")
        for rec in pending:
            result = debate_requirement(
                client, model, rec, kb, da_threshold, max_candidates,
                advocate_threshold, dry_run)
            all_debates.append(result)
            checkpoint_file.write(json.dumps(result) + '\n')
            checkpoint_file.flush()

            sc      = result['initial_screen']
            outcome = sc['outcome']
            pred    = ', '.join(result['predicted']) or 'FR'
            current = ', '.join(rec['types']) or 'FR'
            text_preview = rec['text'][:120] + ('...' if len(rec['text']) > 120 else '')

            print(f"\n{'─' * 64}")
            print(f"  {rec['id']}  |  current: {current}")
            print(f"  \"{text_preview}\"")
            if outcome == 'FR':
                print(f"  Screen: FR  (conf={_fmt_conf(sc['confidence'])})")
                print(f"  Decision: FR")
            elif outcome == 'NFR':
                print(f"  Screen: NFR → {pred}  (conf={_fmt_conf(sc['confidence'])})")
                print(f"  Decision: {pred}")
            else:
                screened_str = ', '.join(
                    f"{r['type']}({_fmt_conf(r['confidence'])})" for r in result['screened'])
                advocates = result.get('advocates', [])
                adv_str = ', '.join(
                    f"{a['type']}({_fmt_conf(a['confidence'])})" for a in advocates)
                da_conf = result['devils_advocate']['confidence']
                print(f"  Screen: DEBATE  candidates=[{screened_str}]")
                print(f"  Advocates: [{adv_str}]")
                print(f"  DA FR-confidence: {_fmt_conf(da_conf)}")
                print(f"  Decision: {pred}")
    finally:
        checkpoint_file.close()

    # Re-align all_debates with the original record order for consistent metrics
    record_index = {r['id']: r for r in records}
    debate_index = {d['id']: d for d in all_debates}
    all_records = [record_index[d['id']] for d in all_debates]
    all_preds   = [{t: (t in d['predicted']) for t in NFR_TYPES}
                   for d in all_debates]

    final    = compute_metrics(all_records, all_preds, NFR_TYPES)
    means    = final['macro']
    per_type = final['per_type']

    print(f"\n{'='*60}")
    if dry_run:
        print(f"DRY-RUN COMPLETE  ({tag})")
        print(f"  Metrics suppressed — dry-run predictions are placeholder values.")
    else:
        print(f"FINAL  ({tag})")
        print(f"  R={means['recall']:.3f}  P={means['precision']:.3f}  "
              f"S={means['specificity']:.3f}  F1={means['f1']:.3f}  F2={means['f2']:.3f}")
    print(f"{'='*60}")

    if dry_run:
        return {}

    summary = {
        'model':               model,
        'da_threshold':        da_threshold,
        'advocate_threshold':  advocate_threshold,
        'max_candidates':      max_candidates,
        'macro':          means,
        'per_type': {
            t: {k: per_type[t][k]
                for k in ('precision', 'recall', 'specificity', 'f1', 'f2',
                          'tp', 'fp', 'fn', 'tn')}
            for t in NFR_TYPES
        },
    }

    json_path = results_dir / f"summary_{tag}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    csv_path = results_dir / f"per_type_{tag}.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['type', 'recall', 'precision', 'specificity', 'f1', 'f2'])
        for t in NFR_TYPES:
            pt = per_type[t]
            w.writerow([t, round(pt['recall'], 4), round(pt['precision'], 4),
                        round(pt['specificity'], 4), round(pt['f1'], 4),
                        round(pt['f2'], 4)])
        w.writerow(['macro',
                    round(means['recall'], 4),      round(means['precision'], 4),
                    round(means['specificity'], 4), round(means['f1'], 4),
                    round(means['f2'], 4)])

    # Per-requirement results table
    detail_path = results_dir / f"detail_{tag}.csv"
    with open(detail_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['req_id', 'project', 'true_types', 'screen_outcome',
                    'screened_types', 'debating_types',
                    'da_verdict', 'da_confidence', 'predicted', 'outcome'])
        for d in all_debates:
            true         = '; '.join(d['true_types']) or 'FR'
            screen_out   = d['initial_screen']['outcome']
            screened     = '; '.join(
                f"{r['type']}({_fmt_conf(r['confidence'])})" for r in d['screened'])
            debating     = '; '.join(d['debating'])
            da           = d['devils_advocate'] or {}
            pred   = '; '.join(d['predicted']) or 'FR'
            true_set = set(d['true_types'])
            pred_set = set(d['predicted'])
            if true_set == pred_set:
                outcome = 'exact'
            elif pred_set & true_set:
                outcome = 'partial'
            elif not true_set and not pred_set:
                outcome = 'exact'     # both FR
            elif true_set and not pred_set:
                outcome = 'FN'        # missed entirely
            else:
                outcome = 'FP'        # predicted NFR(s) that are all wrong
            w.writerow([d['id'], d['project'], true, screen_out, screened, debating,
                        da.get('fr_verdict', ''), _fmt_conf(da.get('confidence')),
                        pred, outcome])

    # Debate transcripts: citable artifact for the paper
    transcripts_path = results_dir / f"transcripts_{tag}.json"
    with open(transcripts_path, 'w') as f:
        json.dump(all_debates, f, indent=2)

    # Advocate scores: structured per-requirement scores for ablation / threshold tuning.
    # Each entry captures every NFR type's confidence at each pipeline stage so the
    # full debate need not be re-run to explore different decision rules.
    scores_path = results_dir / f"scores_{tag}.json"
    scores = []
    for d in all_debates:
        screen     = d['initial_screen']
        screened   = {r['type']: {'screen_evidence': r['evidence_seen'],
                                   'screen_conf':     r['confidence']}
                      for r in d.get('screened', [])}
        argued     = {a['type']: a['confidence']
                      for a in d.get('advocates', [])}
        da         = d.get('devils_advocate') or {}
        arbiter    = d.get('arbiter') or {}
        nfr_scores = {}
        for t in NFR_TYPES:
            entry = {'was_candidate': t in screen.get('types', [])}
            if t in screened:
                entry['screen_evidence'] = screened[t]['screen_evidence']
                entry['screen_conf']     = screened[t]['screen_conf']
            if t in argued:
                entry['argue_conf'] = argued[t]
            nfr_scores[t] = entry
        scores.append({
            'req_id':              d['id'],
            'project':             d['project'],
            'true_types':          d['true_types'],
            'screen_outcome':      screen['outcome'],
            'screen_conf':         screen['confidence'],
            'nfr_scores':          nfr_scores,
            'da_fr_conf':          da.get('confidence'),
            'arbiter_fr_prob':     arbiter.get('fr_probability'),
            'predicted':           d['predicted'],
        })
    with open(scores_path, 'w') as f:
        json.dump(scores, f, indent=2)

    # Proposed label changes for human arbitration
    changes_path = write_proposed_changes(all_debates, results_dir, tag)

    checkpoint_path.unlink(missing_ok=True)  # clean up now that final files are written

    print(f"Summary     → {json_path}")
    print(f"Per-type    → {csv_path}")
    print(f"Detail      → {detail_path}")
    print(f"Transcripts → {transcripts_path}")
    print(f"Scores      → {scores_path}  (advocate confidences for ablation/tuning)")
    print(f"Changes     → {changes_path}  (open with curation_gui.py)")
    update_allresults(summary)
    return summary


# ── Proposed-changes generator ─────────────────────────────────────────────────

def _rationale(d: dict, action: str, label: str) -> str:
    """Extract the most relevant reasoning for a proposed ADD or REMOVE."""
    parts = []
    if action == 'ADD':
        for a in d.get('advocates', []):
            if a['type'] == label:
                parts.append(f"[Advocate for '{label}']\n{a['argument']}")
                break
        arb = d.get('arbiter') or {}
        if arb.get('reasoning'):
            parts.append(f"[Arbiter]\n{arb['reasoning']}")
        if not parts:
            sc = d.get('initial_screen') or {}
            if sc.get('reason'):
                parts.append(f"[Initial screen]\n{sc['reason']}")
    else:  # REMOVE
        matched = next((s for s in d.get('screened', []) if s['type'] == label), None)
        if matched:
            conf = _fmt_conf(matched.get('confidence'))
            parts.append(f"[Screener for '{label}'] confidence={conf}\n"
                         f"{matched.get('brief_reason', '')}")
        else:
            sc  = d.get('initial_screen') or {}
            cands = sc.get('types', [])
            parts.append(f"[Initial screen] '{label}' was not selected as a candidate.\n"
                         f"Candidates identified: {', '.join(cands) or 'none'}\n"
                         f"Reason: {sc.get('reason', '')}")
        arb = d.get('arbiter') or {}
        if arb.get('reasoning'):
            parts.append(f"[Arbiter]\n{arb['reasoning']}")
    return '\n\n'.join(parts) or 'No detailed rationale available.'


def write_proposed_changes(all_debates: list[dict],
                            results_dir: Path, tag: str) -> Path:
    entries = []
    for d in all_debates:
        true_set = set(d['true_types'])
        pred_set = set(d['predicted'])
        if true_set == pred_set:
            continue

        proposed = []
        for label in sorted(pred_set - true_set):   # ADD proposals
            proposed.append({
                'id':        f"{d['id']}_ADD_{label}",
                'action':    'ADD',
                'label':     label,
                'rationale': _rationale(d, 'ADD', label),
                'decision':  None,
            })
        for label in sorted(true_set - pred_set):   # REMOVE proposals
            proposed.append({
                'id':        f"{d['id']}_REMOVE_{label}",
                'action':    'REMOVE',
                'label':     label,
                'rationale': _rationale(d, 'REMOVE', label),
                'decision':  None,
            })

        entries.append({
            'req_id':           d['id'],
            'project':          d['project'],
            'text':             d['text'],
            'current_labels':   d['true_types'],
            'predicted_labels': d['predicted'],
            'proposed_changes': proposed,
            'debate_transcript': {
                'initial_screen':  d.get('initial_screen'),
                'screened':        d.get('screened', []),
                'advocates':       d.get('advocates', []),
                'devils_advocate': d.get('devils_advocate'),
                'arbiter':         d.get('arbiter'),
            },
        })

    out = {
        'tag':                           tag,
        'total_requirements_with_changes': len(entries),
        'total_proposed_changes':          sum(len(e['proposed_changes']) for e in entries),
        'changes':                         entries,
    }
    path = results_dir / f"proposed_changes_{tag}.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    return path

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='P5 — Debate-Based LLM NFR Classifier')
    parser.add_argument('--data',           type=Path, default=DEFAULT_DATA)
    parser.add_argument('--results-dir',    type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--kb',             type=Path, default=DEFAULT_KB)
    parser.add_argument('--model',          default=DEFAULT_MODEL)
    parser.add_argument('--da-threshold',   type=float, default=DEFAULT_DA_THRESHOLD,
                        help='Devil\'s advocate FR-confidence threshold (default 0.7). '
                             'Above this the FR challenge is sustained to the arbiter.')
    parser.add_argument('--max-candidates',    type=int,   default=DEFAULT_MAX_CANDIDATES,
                        help='Max NFR types the identifier forwards to screening (default 4)')
    parser.add_argument('--advocate-threshold', type=float, default=DEFAULT_ADVOCATE_THRESHOLD,
                        help='Min screen confidence for an advocate to argue (default 0.3). '
                             'Raise to require stronger evidence before an advocate joins.')
    parser.add_argument('--dry-run',           action='store_true',
                        help='Print prompts without calling the API')
    parser.add_argument('--id',                nargs='+', metavar='REQID',
                        help='Debate specific requirement IDs only (e.g. R010 R042). '
                             'Prints full transcript; does not save files.')
    parser.add_argument('--limit',             type=int,  default=None,
                        help='Process only the first N requirements and save output files. '
                             'Appends _topN to the output tag so it does not collide with a full run.')
    parser.add_argument('--force',             action='store_true',
                        help='Delete any existing checkpoint and restart from scratch.')
    args = parser.parse_args()

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to a .env file or export it.")
        sys.exit(1)

    if args.id:
        run_single(
            data_path=args.data,
            kb_path=args.kb,
            model=args.model,
            da_threshold=args.da_threshold,
            max_candidates=args.max_candidates,
            advocate_threshold=args.advocate_threshold,
            req_ids=args.id,
            dry_run=args.dry_run,
            api_key=api_key,
        )
        sys.exit(0)

    run_agent_debate(
        data_path=args.data,
        results_dir=args.results_dir,
        kb_path=args.kb,
        model=args.model,
        da_threshold=args.da_threshold,
        max_candidates=args.max_candidates,
        advocate_threshold=args.advocate_threshold,
        dry_run=args.dry_run,
        api_key=api_key,
        limit=args.limit,
        force=args.force,
    )
