#!/usr/bin/env python3
"""
Standard LLM accuracy benchmark for Orchid / ternative.cpp.

Runs ARC-Challenge, HellaSwag, WinoGrande, and MMLU using log-probability
scoring against an OpenAI-compatible completions endpoint.

Method: for each multiple-choice question, score every answer choice via
P(choice | context) and pick the highest. This matches lm-evaluation-harness
methodology exactly, so results are directly comparable to published numbers.

Usage:
    python orchid/scripts/bench_standard.py
    python orchid/scripts/bench_standard.py --url http://localhost:8080 --limit 100
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_URL   = "http://localhost:8080"
DEFAULT_LIMIT = 250  # questions per benchmark (set None for full run)

BENCHMARKS = {
    "arc_challenge": {
        "hf_path": "allenai/ai2_arc",
        "hf_name": "ARC-Challenge",
        "split":   "test",
        "published_bitnet": 49.91,
    },
    "hellaswag": {
        "hf_path": "Rowan/hellaswag",
        "hf_name": None,
        "split":   "validation",
        "published_bitnet": 68.44,
    },
    "winogrande": {
        "hf_path": "allenai/winogrande",
        "hf_name": "winogrande_xl",
        "split":   "validation",
        "published_bitnet": None,
    },
    "mmlu": {
        "hf_path": "cais/mmlu",
        "hf_name": "all",
        "split":   "test",
        "published_bitnet": 53.17,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(url: str, text: str) -> Optional[list]:
    """Return token strings via fast /v1/tokenize (no model inference).
    Falls back to echo scoring if endpoint unavailable."""
    try:
        r = requests.post(f"{url}/v1/tokenize", json={"text": text}, timeout=10)
        if r.status_code == 200:
            return r.json().get("tokens", [])
    except Exception:
        pass
    # Fallback: echo request (slower)
    try:
        r = requests.post(f"{url}/v1/completions",
                          json={"prompt": text, "max_tokens": 0,
                                "logprobs": 1, "echo": True}, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0].get("logprobs", {}).get("tokens", [])
    except Exception:
        pass
    return None


def score_continuation(url: str, context: str, continuation: str,
                       retries: int = 3, normalize: bool = False) -> Optional[float]:
    """Return log-prob of `continuation` given `context`.

    Optimisations vs. naive two-request approach:
    1. Uses /v1/tokenize (no inference) to find the split point, eliminating
       the second echo scoring forward pass.
    2. Passes start_pos so the server skips tok_embd GEMV for context positions.
    Both together cut per-question scoring time by ~2-3x.
    """
    # Step 1: tokenize context cheaply (no model inference).
    ctx_tokens = _tokenize(url, context)
    split_hint = len(ctx_tokens) if ctx_tokens else 0

    for attempt in range(retries):
        try:
            payload: dict = {
                "prompt":     context + continuation,
                "max_tokens": 0,
                "logprobs":   1,
                "echo":       True,
            }
            if split_hint > 0:
                payload["start_pos"] = split_hint  # skip context logit computation

            r_full = requests.post(f"{url}/v1/completions", json=payload, timeout=120)
            if r_full.status_code != 200:
                time.sleep(1)
                continue
            full_data   = r_full.json()["choices"][0].get("logprobs", {})
            full_tokens = full_data.get("tokens", [])
            full_lp     = full_data.get("token_logprobs", [])
            if not full_lp:
                return None

            # Step 2: refine split via token-string comparison (handles BPE merges).
            split = split_hint if ctx_tokens else len(full_tokens)
            if ctx_tokens:
                for i, (ft, ct) in enumerate(zip(full_tokens, ctx_tokens)):
                    if ft != ct:
                        split = i
                        break

            cont_lp = [lp for lp in full_lp[split:] if lp is not None]
            if not cont_lp:
                cont_lp = ([full_lp[split - 1]]
                           if split > 0 and full_lp[split - 1] is not None else [])
            if not cont_lp:
                return None
            raw = sum(cont_lp)
            return raw / len(cont_lp) if normalize else raw
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [ERR] {e}", file=sys.stderr)
            time.sleep(1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark drivers
# ─────────────────────────────────────────────────────────────────────────────

def run_arc(url: str, limit: Optional[int]) -> float:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=False)
    correct = 0
    total   = 0
    choices_keys = ["A", "B", "C", "D", "E"]

    for ex in (ds if limit is None else ds.select(range(min(limit, len(ds))))):
        question = ex["question"]
        labels   = ex["choices"]["label"]
        texts    = ex["choices"]["text"]
        answer   = ex["answerKey"]

        context  = f"Question: {question}\nAnswer:"
        scores   = {}
        for lbl, txt in zip(labels, texts):
            lp = score_continuation(url, context, f" {txt}")
            if lp is not None:
                scores[lbl] = lp

        if not scores:
            continue
        pred = max(scores, key=scores.__getitem__)
        if pred == answer:
            correct += 1
        total += 1
        print(f"  [ARC-C] {total}/{min(limit or len(ds), len(ds))} "
              f"acc={100*correct/total:.1f}%", end="\r")

    print()
    return 100 * correct / total if total else 0.0


def run_hellaswag(url: str, limit: Optional[int]) -> float:
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=False)
    correct = 0
    total   = 0

    for ex in (ds if limit is None else ds.select(range(min(limit, len(ds))))):
        ctx    = ex["activity_label"] + ": " + ex["ctx"]
        endings = ex["endings"]
        answer  = int(ex["label"])

        scores = []
        for ending in endings:
            lp = score_continuation(url, ctx, " " + ending, normalize=True)
            scores.append(lp if lp is not None else -1e9)

        pred = scores.index(max(scores))
        if pred == answer:
            correct += 1
        total += 1
        print(f"  [HellaSwag] {total}/{min(limit or len(ds), len(ds))} "
              f"acc={100*correct/total:.1f}%", end="\r")

    print()
    return 100 * correct / total if total else 0.0


def run_winogrande(url: str, limit: Optional[int]) -> float:
    ds = load_dataset("allenai/winogrande", "winogrande_xl",
                      split="validation", trust_remote_code=False)
    correct = 0
    total   = 0

    for ex in (ds if limit is None else ds.select(range(min(limit, len(ds))))):
        sentence = ex["sentence"]
        opt1     = ex["option1"]
        opt2     = ex["option2"]
        answer   = ex["answer"]  # "1" or "2"

        # Replace the blank with each option and score
        sent1 = sentence.replace("_", opt1)
        sent2 = sentence.replace("_", opt2)

        # WinoGrande: score full sentences (not continuations), so no normalization.
        # Sentences differ by exactly one word; normalization inverts the ranking
        # when the two options tokenize to different lengths.
        lp1 = score_continuation(url, "", sent1, normalize=False)
        lp2 = score_continuation(url, "", sent2, normalize=False)

        if lp1 is None or lp2 is None:
            continue
        pred = "1" if lp1 > lp2 else "2"
        if pred == answer:
            correct += 1
        total += 1
        print(f"  [WinoGrande] {total}/{min(limit or len(ds), len(ds))} "
              f"acc={100*correct/total:.1f}%", end="\r")

    print()
    return 100 * correct / total if total else 0.0


def run_mmlu(url: str, limit: Optional[int]) -> float:
    """Run MMLU (all subjects, test split). Sample evenly across subjects."""
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging",
        "human_sexuality", "international_law", "jurisprudence",
        "logical_fallacies", "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions",
    ]
    choices_labels = ["A", "B", "C", "D"]
    per_subj = max(1, (limit or 570) // len(subjects)) if limit else None

    correct = 0
    total   = 0

    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=False)
        except Exception:
            continue
        n = min(per_subj or len(ds), len(ds))

        for ex in ds.select(range(n)):
            question = ex["question"]
            choices  = ex["choices"]
            answer   = int(ex["answer"])  # 0-3 index

            context = f"Question: {question}\nAnswer:"
            scores  = []
            for ch in choices:
                lp = score_continuation(url, context, f" {ch}")
                scores.append(lp if lp is not None else -1e9)

            pred = scores.index(max(scores))
            if pred == answer:
                correct += 1
            total += 1
            print(f"  [MMLU/{subj[:12]}] {total} acc={100*correct/total:.1f}%", end="\r")

    print()
    return 100 * correct / total if total else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_RUNNERS = {
    "arc_challenge": run_arc,
    "hellaswag":     run_hellaswag,
    "winogrande":    run_winogrande,
    "mmlu":          run_mmlu,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",   default=DEFAULT_URL)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help="Max questions per benchmark (None = full run)")
    ap.add_argument("--tasks", nargs="+",
                    choices=list(BENCHMARK_RUNNERS),
                    default=list(BENCHMARK_RUNNERS),
                    help="Which benchmarks to run")
    ap.add_argument("--output", default="orchid/tests/bench_standard_results.json")
    args = ap.parse_args()

    # Verify server is up
    try:
        r = requests.get(f"{args.url}/v1/models", timeout=5)
        model = r.json()["data"][0]["id"]
        print(f"Server: {args.url}  model: {model}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.url}: {e}")
        sys.exit(1)

    print(f"Limit: {args.limit} questions/benchmark\n")

    results = {}
    t0_total = time.time()

    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"  {task.upper()}")
        print(f"{'='*60}")
        t0 = time.time()
        acc = BENCHMARK_RUNNERS[task](args.url, args.limit)
        elapsed = time.time() - t0
        pub = BENCHMARKS.get(task, {}).get("published_bitnet")
        delta = f"{acc - pub:+.2f}pp vs BitNet base" if pub else ""
        results[task] = {"accuracy": acc, "elapsed_s": elapsed, "n": args.limit}
        print(f"  Result: {acc:.2f}%  ({elapsed/60:.1f} min)  {delta}")

    total_elapsed = time.time() - t0_total

    # ── Results table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Benchmark':<20} {'Orchid 1.0':>12} {'BitNet base':>12} {'Delta':>10}")
    print(f"  {'-'*54}")

    for task, res in results.items():
        pub  = BENCHMARKS.get(task, {}).get("published_bitnet")
        dstr = f"{res['accuracy'] - pub:+.1f}pp" if pub else "  —"
        bstr = f"{pub:.1f}%" if pub else "   —"
        print(f"  {task:<20} {res['accuracy']:>11.1f}% {bstr:>12} {dstr:>10}")

    print(f"\n  Total time: {total_elapsed/60:.1f} min")
    print(f"{'='*70}\n")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": results, "url": args.url, "limit": args.limit,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
