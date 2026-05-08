"""Ingest the Fluent UDF manual + example UDFs and build a FAISS retrieval index.

Usage (from project root):
    python -m scripts.build_index
    python -m scripts.build_index --manual path\to\manual.pdf --examples data\examples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG, EXAMPLES_DIR, INDEX_DIR, MANUAL_PDF  # noqa: E402
from src.ingest import chunk_pages, chunk_text_file, extract_pages  # noqa: E402
from src.retriever import Retriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for index build configuration."""
    parser = argparse.ArgumentParser(
        description="Build the Fluent UDF retrieval index."
    )
    parser.add_argument(
        "--manual", type=Path, default=MANUAL_PDF, help="Path to the Fluent UDF PDF."
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=EXAMPLES_DIR,
        help="Directory of example UDFs (.c/.txt).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=INDEX_DIR,
        help="Directory to save the FAISS index into.",
    )
    parser.add_argument("--chunk-tokens", type=int, default=CONFIG.chunk_tokens)
    parser.add_argument("--chunk-overlap", type=int, default=CONFIG.chunk_overlap)
    parser.add_argument("--embed-model", type=str, default=CONFIG.embed_model)
    parser.add_argument(
        "--embed-device",
        type=str,
        default=CONFIG.embed_device,
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--embed-batch-size", type=int, default=CONFIG.embed_batch_size)
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        help="Optional cap for a quick smoke build.",
    )
    return parser.parse_args()


def main() -> None:
    """Ingest sources, build vector index, and persist artifacts."""
    args = parse_args()

    all_chunks = []

    if args.manual and args.manual.exists():
        pages = extract_pages(args.manual)
        if args.limit_pages:
            pages = pages[: args.limit_pages]
        manual_chunks = chunk_pages(
            pages,
            source=args.manual.name,
            chunk_tokens=args.chunk_tokens,
            chunk_overlap=args.chunk_overlap,
        )
        print(
            f"[manual] {args.manual.name}: {len(pages)} pages -> {len(manual_chunks)} chunks"
        )
        all_chunks.extend(manual_chunks)
    else:
        print(f"[manual] skipped (not found): {args.manual}")

    if args.examples and args.examples.exists():
        for path in sorted(args.examples.rglob("*")):
            if path.is_file() and path.suffix.lower() in {
                ".c",
                ".cpp",
                ".h",
                ".txt",
                ".md",
            }:
                ex_chunks = chunk_text_file(
                    path,
                    source=str(path.relative_to(args.examples.parent)),
                    chunk_tokens=args.chunk_tokens,
                    chunk_overlap=args.chunk_overlap,
                )
                print(f"[example] {path.name}: {len(ex_chunks)} chunks")
                all_chunks.extend(ex_chunks)

    if not all_chunks:
        raise SystemExit(
            "No chunks produced. Check that the manual and/or examples exist."
        )

    print(
        f"Total chunks: {len(all_chunks)}. Building embeddings with `{args.embed_model}` "
        f"on device `{args.embed_device}` (batch_size={args.embed_batch_size})..."
    )
    retriever = Retriever(
        embed_model=args.embed_model,
        chunks=all_chunks,
        embed_device=args.embed_device,
        embed_batch_size=args.embed_batch_size,
    )
    retriever.build_index()
    retriever.save(args.out)
    print(f"Saved index to {args.out}")


if __name__ == "__main__":
    main()
