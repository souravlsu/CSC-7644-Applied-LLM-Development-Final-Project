"""Top-level assistant: ties retrieval, prompting, LLM, and the static checker together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .checker import CheckResult, check_answer, collect_known_tokens
from .config import CONFIG, Config
from .llm_client import ChatMessage, LLMClient
from .prompts import build_baseline_prompt, build_rag_prompt
from .retriever import RetrievedChunk, Retriever

Mode = Literal["rag", "baseline"]


@dataclass
class AssistantResult:
    mode: Mode
    question: str
    answer: str
    retrieved: list[RetrievedChunk]
    check: CheckResult

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "question": self.question,
            "answer": self.answer,
            "retrieved": [
                {
                    "id": r.chunk.id,
                    "source": r.chunk.source,
                    "section": r.chunk.section,
                    "pages": [r.chunk.page_start, r.chunk.page_end],
                    "macros": r.chunk.macro_families,
                    "score": r.score,
                }
                for r in self.retrieved
            ],
            "check": self.check.to_dict(),
        }


class Assistant:
    """Main orchestrator for retrieval, prompting, generation, and checking."""

    def __init__(
        self, retriever: Retriever, llm: LLMClient, config: Config = CONFIG
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.config = config
        self._known_defines, self._known_c = collect_known_tokens(retriever.chunks)

    def ask(
        self,
        question: str,
        mode: Mode = "rag",
        top_k: int | None = None,
        expected_family: str | None = None,
    ) -> AssistantResult:
        """Answer a user question in baseline or RAG mode."""
        retrieved: list[RetrievedChunk] = []
        if mode == "rag":
            retrieved = self.retriever.query(
                question,
                top_k=top_k or self.config.top_k,
                use_hybrid=self.config.use_hybrid,
            )
            prompt = build_rag_prompt(question, retrieved)
        else:
            prompt = build_baseline_prompt(question)

        messages = [
            ChatMessage(role="system", content=prompt.system),
            ChatMessage(role="user", content=prompt.user),
        ]
        answer = self.llm.chat(messages)

        check = check_answer(
            answer=answer,
            known_define_macros=self._known_defines,
            known_c_macros=self._known_c,
            expected_family=expected_family,
        )
        return AssistantResult(
            mode=mode,
            question=question,
            answer=answer,
            retrieved=retrieved,
            check=check,
        )


def build_default_assistant(index_dir, config: Config = CONFIG) -> Assistant:
    """Construct an Assistant from config and a saved retrieval index."""
    retriever = Retriever.load(
        index_dir,
        embed_model=config.embed_model,
        embed_device=config.embed_device,
        embed_batch_size=config.embed_batch_size,
    )
    llm = LLMClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
    )
    return Assistant(retriever=retriever, llm=llm, config=config)
