# evals/runner.py

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from app.agents.supervisor import classify_query
from configs.agent_config import CLASSIFICATION_PROMPT_TEMPLATE
from configs.api_config import load_env

load_env()

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_PATH = project_root / "evals" / "datasets" / "router.jsonl"
RESULTS_DIR = project_root / "evals" / "results"

# Same default as app/core/llm.py::get_llm, kept explicit here so config.json
# always records which model a run actually used.
DEFAULT_MODEL_NAME = "claude-sonnet-5"


def load_dataset(split: str) -> List[Dict[str, Any]]:
    cases = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if split == "all" or case["split"] == split:
                cases.append(case)
    return cases


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_router_trial(case: Dict[str, Any], trial: int) -> Dict[str, Any]:
    start = time.monotonic()
    error = None
    result = {}
    try:
        result = classify_query({"query": case["query"]})
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    latency_s = time.monotonic() - start

    return {
        "case_id": case["id"],
        "trial": trial,
        "query": case["query"],
        "expected": case["expected"],
        "bucket": case["bucket"],
        "split": case["split"],
        "lang": case["lang"],
        "classification": result.get("classification"),
        "raw_llm_output": result.get("raw_llm_output"),
        "usage": result.get("usage"),
        "latency_s": latency_s,
        "error": error,
    }


def run_router_target(cases: List[Dict[str, Any]], trials: int, max_workers: int) -> List[Dict[str, Any]]:
    jobs = [(case, trial) for case in cases for trial in range(trials)]
    traces = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_router_trial, case, trial): (case, trial) for case, trial in jobs}
        for future in as_completed(futures):
            traces.append(future.result())
    traces.sort(key=lambda t: (t["case_id"], t["trial"]))
    return traces


def write_config(run_dir: Path, model_name: str, trials: int, split: str) -> None:
    config = {
        "run_id": run_dir.name,
        "target": "router",
        "model_name": model_name,
        "trials": trials,
        "split": split,
        "git_sha": get_git_sha(),
        "classification_prompt_hash": hashlib.sha256(
            CLASSIFICATION_PROMPT_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "notes": {
            "temperature": "get_llm() accepts temperature/top_p/top_k but never forwards them to "
                            "ChatAnthropic (Claude runs extended thinking by default, which rejects "
                            "sampling params) -- these fields are recorded for traceability only and "
                            "have no effect on model behavior.",
        },
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def write_traces(run_dir: Path, traces: List[Dict[str, Any]]) -> None:
    with open(run_dir / "traces.jsonl", "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval runner for the supervisor-based multi-agent chatbot.")
    parser.add_argument("--target", choices=["router"], default="router",
                        help="Only 'router' is implemented so far (graph target lands in a later phase).")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev",
                        help="Defaults to 'dev' -- holdout is meant to be run once, at the end.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="Recorded in config.json. classify_query calls get_llm() internally, "
                             "which defaults to this same model.")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    cases = load_dataset(args.split)
    if not cases:
        print(f"No cases found for split={args.split!r} in {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    run_id = args.run_id or f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running target=router split={args.split} cases={len(cases)} trials={args.trials} "
          f"-> {len(cases) * args.trials} total calls")

    traces = run_router_target(cases, trials=args.trials, max_workers=args.max_workers)
    write_traces(run_dir, traces)
    write_config(run_dir, model_name=args.model_name, trials=args.trials, split=args.split)

    errors = sum(1 for t in traces if t["error"])
    print(f"Done. Wrote {len(traces)} trace rows ({errors} errors) to {run_dir}")


if __name__ == "__main__":
    main()
