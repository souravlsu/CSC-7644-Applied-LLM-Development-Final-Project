"""Embedding-based retrieval backed by FAISS with optional BM25 hybrid reranking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import faiss  # type: ignore[import-untyped]
import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
from sentence_transformers import SentenceTransformer

from .ingest import Chunk, load_chunks, save_chunks

torch_module: Any = None
try:
    import torch as torch_module
except Exception:  # pragma: no cover
    pass


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 lexical scoring."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    source_scores: dict[str, float]


class Retriever:
    """Loads a FAISS vector index plus chunks and serves nearest-neighbor queries."""

    def __init__(
        self,
        embed_model: str,
        chunks: list[Chunk],
        index: faiss.Index | None = None,
        embed_device: str = "auto",
        embed_batch_size: int = 64,
    ) -> None:
        self.embed_model_name = embed_model
        self.embed_device = self._resolve_device(embed_device)
        self.embed_batch_size = max(1, embed_batch_size)
        self._model: SentenceTransformer | None = None
        self.chunks = chunks
        self.index = index
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[list[str]] | None = None

    @staticmethod
    def _resolve_device(requested: str) -> str:
        choice = (requested or "auto").strip().lower()
        cuda_available = bool(
            torch_module is not None and torch_module.cuda.is_available()
        )
        if choice == "auto":
            return "cuda" if cuda_available else "cpu"
        if choice == "cuda" and not cuda_available:
            print(
                "[retriever] Requested EMBED_DEVICE=cuda but CUDA is unavailable; falling back to cpu."
            )
            return "cpu"
        if choice not in {"cpu", "cuda"}:
            print(f"[retriever] Unknown EMBED_DEVICE={requested!r}; using auto.")
            return "cuda" if cuda_available else "cpu"
        return choice

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self.embed_model_name, device=self.embed_device
            )
            print(
                f"[retriever] Loaded embedding model `{self.embed_model_name}` "
                f"on device `{self.embed_device}` (batch_size={self.embed_batch_size})."
            )
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into normalized float32 vectors."""
        vectors = self.model.encode(
            texts,
            batch_size=self.embed_batch_size,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    def build_index(self) -> None:
        """Build an in-memory FAISS inner-product index for all chunks."""
        if not self.chunks:
            raise RuntimeError("No chunks loaded; cannot build index.")
        texts = [chunk.text for chunk in self.chunks]
        vectors = self._encode(texts)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def _ensure_bm25(self) -> None:
        """Initialize BM25 index lazily for hybrid retrieval."""
        if self._bm25 is None:
            self._bm25_corpus = [
                _tokenize(c.text + " " + " ".join(c.macro_families))
                for c in self.chunks
            ]
            self._bm25 = BM25Okapi(self._bm25_corpus)

    def query(
        self, text: str, top_k: int = 5, use_hybrid: bool = True
    ) -> list[RetrievedChunk]:
        """Retrieve top-k relevant chunks for a query."""
        if self.index is None:
            raise RuntimeError("Retriever has no FAISS index. Build or load one first.")
        query_vec = self._encode([text])
        scores, indices = self.index.search(query_vec, min(top_k * 4, len(self.chunks)))
        dense = {int(i): float(s) for i, s in zip(indices[0], scores[0]) if int(i) >= 0}

        ranking: dict[int, dict[str, float]] = {
            i: {"dense": s} for i, s in dense.items()
        }

        if use_hybrid:
            self._ensure_bm25()
            assert self._bm25 is not None
            bm25_scores = self._bm25.get_scores(_tokenize(text))
            if bm25_scores.max() > 0:
                bm25_scores = bm25_scores / bm25_scores.max()
            bm25_top = np.argsort(bm25_scores)[::-1][: top_k * 4]
            for idx in bm25_top:
                idx_int = int(idx)
                score = float(bm25_scores[idx_int])
                ranking.setdefault(idx_int, {"dense": 0.0})["bm25"] = score

            for idx, parts in ranking.items():
                parts.setdefault("bm25", 0.0)
                parts["combined"] = 0.65 * parts["dense"] + 0.35 * parts["bm25"]
            score_key = "combined"
        else:
            for parts in ranking.values():
                parts["combined"] = parts["dense"]
            score_key = "combined"

        ordered = sorted(
            ranking.items(), key=lambda kv: kv[1][score_key], reverse=True
        )[:top_k]
        return [
            RetrievedChunk(
                chunk=self.chunks[i], score=parts[score_key], source_scores=parts
            )
            for i, parts in ordered
        ]

    def save(self, index_dir: Path) -> None:
        """Save FAISS index and chunk metadata to disk."""
        if self.index is None:
            raise RuntimeError("Nothing to save: no index built.")
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "faiss.index"))
        save_chunks(self.chunks, index_dir / "chunks.jsonl")
        (index_dir / "embed_model.txt").write_text(
            self.embed_model_name, encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        index_dir: Path,
        embed_model: str | None = None,
        embed_device: str = "auto",
        embed_batch_size: int = 64,
    ) -> "Retriever":
        """Load a retriever from a persisted index directory."""
        chunks = load_chunks(index_dir / "chunks.jsonl")
        index = faiss.read_index(str(index_dir / "faiss.index"))
        model_file = index_dir / "embed_model.txt"
        saved_model = (
            model_file.read_text(encoding="utf-8").strip()
            if model_file.exists()
            else None
        )
        resolved = embed_model or saved_model
        if resolved is None:
            raise RuntimeError(
                "Embedding model name not provided and not stored with the index."
            )
        return cls(
            embed_model=resolved,
            chunks=chunks,
            index=index,
            embed_device=embed_device,
            embed_batch_size=embed_batch_size,
        )


def build_retriever_from_chunks(
    embed_model: str,
    chunks: Iterable[Chunk],
    embed_device: str = "auto",
    embed_batch_size: int = 64,
) -> Retriever:
    retriever = Retriever(
        embed_model=embed_model,
        chunks=list(chunks),
        embed_device=embed_device,
        embed_batch_size=embed_batch_size,
    )
    retriever.build_index()
    return retriever
