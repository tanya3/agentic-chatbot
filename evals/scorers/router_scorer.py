# evals/scorers/router_scorer.py

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from configs.agent_config import VALID_TARGET_CATEGORIES, DEFAULT_TARGET_CATEGORY

RESULTS_DIR = project_root / "evals" / "results"


def classify_triage(raw_llm_output: Optional[str], error: Optional[str]) -> str:
    """Replays supervisor.py::classify_query's fallback branching (exact match ->
    substring scan -> default) against the raw output already captured in the trace,
    so this never needs its own LLM call. Kept in sync with the branching order in
    app/agents/supervisor.py -- if that logic changes, update this too.
    """
    if error:
        return "errored"
    if raw_llm_output is None:
        # supervisor.py short-circuits on an empty/whitespace query before calling the LLM
        return "empty_query"
    if raw_llm_output in VALID_TARGET_CATEGORIES:
        return "clean"
    for valid_cat in VALID_TARGET_CATEGORIES:
        if valid_cat in raw_llm_output:
            return "rescued"
    return "defaulted"


def score_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    error = trace.get("error")
    triage = classify_triage(trace.get("raw_llm_output"), error)

    correct: Optional[bool]
    if error:
        correct = None
    else:
        correct = trace.get("classification") == trace.get("expected")

    label = triage if correct is None else f"{triage}_{'correct' if correct else 'wrong'}"

    return {
        **trace,
        "correct": correct,
        "triage": triage,
        "triage_label": label,
    }


def score_traces(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [score_trace(t) for t in traces]


def load_traces(run_dir: Path) -> List[Dict[str, Any]]:
    traces = []
    with open(run_dir / "traces.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def write_scored(run_dir: Path, scored: List[Dict[str, Any]]) -> None:
    with open(run_dir / "scored.jsonl", "w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_summary(scored: List[Dict[str, Any]]) -> None:
    total = len(scored)
    scoreable = [s for s in scored if s["correct"] is not None]
    n_correct = sum(1 for s in scoreable if s["correct"])

    print(f"Total trace rows: {total}")
    print(f"Errored rows:     {total - len(scoreable)}")
    if scoreable:
        print(f"Exact-match accuracy (post-fallback): {n_correct}/{len(scoreable)} "
              f"({100 * n_correct / len(scoreable):.1f}%)")

    print("\nTriage breakdown (model-wrong vs. malformed-but-rescued vs. code-defaulted):")
    for label, count in Counter(s["triage_label"] for s in scored).most_common():
        print(f"  {label:22s} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a router-target eval run.")
    parser.add_argument("--run-id", required=True, help="Directory name under evals/results/")
    args = parser.parse_args()

    run_dir = RESULTS_DIR / args.run_id
    traces = load_traces(run_dir)
    if not traces:
        print(f"No traces found in {run_dir}/traces.jsonl", file=sys.stderr)
        sys.exit(1)

    scored = score_traces(traces)
    write_scored(run_dir, scored)
    print_summary(scored)
    print(f"\nWrote {len(scored)} scored rows to {run_dir / 'scored.jsonl'}")


if __name__ == "__main__":
    main()
