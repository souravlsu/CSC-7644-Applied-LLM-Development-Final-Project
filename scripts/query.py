"""Ask the UDF Copilot a single question from the command line.

Usage:
    python -m scripts.query "write a parabolic inlet velocity UDF"
    python -m scripts.query --mode baseline "explain when to use DEFINE_SOURCE"
    python -m scripts.query --retrieve-only "DEFINE_PROFILE example"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assistant import build_default_assistant  # noqa: E402
from src.config import CONFIG, INDEX_DIR  # noqa: E402
from src.retriever import Retriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse query CLI arguments."""
    parser = argparse.ArgumentParser(description="Query the Fluent UDF Copilot.")
    parser.add_argument("question", type=str, nargs="+")
    parser.add_argument("--mode", choices=["rag", "baseline"], default="rag")
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=CONFIG.top_k)
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Skip the LLM, just show retrieved passages.",
    )
    parser.add_argument(
        "--expected-family", type=str, default=None, help="For structural scoring."
    )
    return parser.parse_args()


def _print_retrieved(retriever: Retriever, question: str, top_k: int) -> None:
    """Print retrieval-only results in a compact CLI-friendly format."""
    results = retriever.query(question, top_k=top_k, use_hybrid=CONFIG.use_hybrid)
    print(f"\nTop {len(results)} retrieved chunks for: {question!r}\n")
    for idx, item in enumerate(results, start=1):
        chunk = item.chunk
        print(
            f"[C{idx}] score={item.score:.3f} {chunk.source} | {chunk.section} | pp.{chunk.page_start}-{chunk.page_end}"
        )
        macros = ", ".join(chunk.macro_families) if chunk.macro_families else "-"
        print(f"      macros={macros}")
        snippet = chunk.text.strip().replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:280] + "..."
        print(f"      {snippet}\n")


def main() -> None:
    """Run query flow in retrieval-only or full assistant mode."""
    args = parse_args()
    question = " ".join(args.question).strip()

    if args.retrieve_only:
        retriever = Retriever.load(
            args.index,
            embed_model=CONFIG.embed_model,
            embed_device=CONFIG.embed_device,
            embed_batch_size=CONFIG.embed_batch_size,
        )
        _print_retrieved(retriever, question, args.top_k)
        return

    assistant = build_default_assistant(args.index)
    result = assistant.ask(
        question,
        mode=args.mode,
        top_k=args.top_k,
        expected_family=args.expected_family,
    )

    print("=" * 80)
    print(f"Mode: {result.mode}")
    print(f"Question: {result.question}")
    print("=" * 80)
    if result.retrieved:
        print("\nRetrieved context:")
        for idx, item in enumerate(result.retrieved, start=1):
            chunk = item.chunk
            print(
                f"  [C{idx}] {chunk.source} | {chunk.section} | "
                f"pp.{chunk.page_start}-{chunk.page_end} | score={item.score:.3f}"
            )
    print("\n--- Answer ---\n")
    print(result.answer)
    print("\n--- Static check ---")
    for key, value in result.check.to_dict().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
