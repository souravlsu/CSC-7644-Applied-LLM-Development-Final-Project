"""Run baseline vs RAG on the curated prompt set and print a summary table.

Usage:
    python -m scripts.evaluate
    python -m scripts.evaluate --prompts data/eval_prompts.json --out artifacts/eval.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabulate import tabulate  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assistant import build_default_assistant  # noqa: E402
from src.config import ARTIFACTS_DIR, DATA_DIR, INDEX_DIR  # noqa: E402
from src.evaluate import (  # noqa: E402
    load_prompts,
    run_evaluation,
    save_records,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs RAG.")
    parser.add_argument("--prompts", type=Path, default=DATA_DIR / "eval_prompts.json")
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "eval_records.json")
    parser.add_argument(
        "--modes", nargs="+", default=["baseline", "rag"], choices=["baseline", "rag"]
    )
    return parser.parse_args()


def _fmt_ratio(value: float | None) -> str:
    """Format optional ratio values as percentages."""
    return "n/a" if value is None else f"{100 * value:.1f}%"


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts)
    assistant = build_default_assistant(args.index)

    print(f"Running {len(prompts)} prompts across modes: {args.modes}")
    records = run_evaluation(assistant, prompts, modes=args.modes)
    save_records(records, args.out)

    summary = summarize(records)
    rows = []
    for mode in args.modes:
        info = summary.get(mode)
        if not info:
            continue
        rows.append(
            [
                mode,
                info["prompts"],
                _fmt_ratio(info["macro_accuracy"]),
                _fmt_ratio(info["structural_pass_rate"]),
                _fmt_ratio(info["citation_rate"]),
            ]
        )
    print()
    print(
        tabulate(
            rows,
            headers=[
                "mode",
                "prompts",
                "macro_accuracy",
                "structural_pass_rate",
                "citation_rate",
            ],
            tablefmt="github",
        )
    )
    print(f"\nFull records saved to {args.out}")


if __name__ == "__main__":
    main()
