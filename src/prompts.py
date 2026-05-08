"""Prompt templates for baseline and retrieval-grounded UDF assistance."""

from __future__ import annotations

from dataclasses import dataclass

from .retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are the Fluent UDF Copilot, an assistant that helps engineers write, explain, "
    "and debug ANSYS Fluent User-Defined Functions (UDFs).\n\n"
    "Non-negotiable rules:\n"
    "1. Only use Fluent macros and APIs that are supported by the provided CONTEXT. "
    "If the context does not cover a needed macro, say so instead of inventing one.\n"
    '2. Every generated UDF must include `#include "udf.h"` and a correct DEFINE_* signature.\n'
    "3. Respond in this exact structure:\n"
    "   - Task type: <generation|explanation|debug>\n"
    '   - Chosen macro family: <DEFINE_XXX or "n/a">\n'
    "   - Code (fenced ```c block) when generating or fixing code; otherwise omit.\n"
    "   - Short explanation (<=6 bullet points) of what the code does.\n"
    "   - Citations: list the context tags you relied on (e.g. [C2], [C4]).\n"
    "4. If retrieval confidence looks low or context is unrelated, abstain and explain why.\n"
    "5. Remind the user that generated UDFs require human review and solver-side testing."
)


BASELINE_SYSTEM_PROMPT = (
    "You are an assistant that helps engineers write ANSYS Fluent User-Defined Functions (UDFs). "
    "Respond with a clearly structured answer that includes, when relevant:\n"
    "- Task type (generation/explanation/debug)\n"
    "- Chosen DEFINE_* macro family\n"
    '- Code in a fenced ```c block with `#include "udf.h"` when code is produced\n'
    "- A short bullet-point explanation\n"
    "Remind the user that output must be reviewed before use in Fluent."
)


@dataclass
class PromptBundle:
    system: str
    user: str


def format_context(retrieved: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as tagged context blocks."""
    parts: list[str] = []
    for idx, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        header = (
            f"[C{idx}] source={chunk.source} | section={chunk.section} | "
            f"pages={chunk.page_start}-{chunk.page_end} | "
            f"macros={', '.join(chunk.macro_families) if chunk.macro_families else 'n/a'} | "
            f"score={item.score:.3f}"
        )
        parts.append(header + "\n" + chunk.text.strip())
    return "\n\n---\n\n".join(parts)


def build_rag_prompt(question: str, retrieved: list[RetrievedChunk]) -> PromptBundle:
    """Build a grounded prompt that includes retrieved context."""
    context = format_context(retrieved) if retrieved else "(no context retrieved)"
    user = (
        "CONTEXT (retrieved from the Ansys Fluent Customization Manual and curated example UDFs):\n"
        f"{context}\n\n"
        "USER REQUEST:\n"
        f"{question.strip()}\n\n"
        "Follow every rule in the system prompt. Cite only context tags you actually used."
    )
    return PromptBundle(system=SYSTEM_PROMPT, user=user)


def build_baseline_prompt(question: str) -> PromptBundle:
    """Build a plain prompt without retrieved context."""
    user = (
        "USER REQUEST:\n"
        f"{question.strip()}\n\n"
        "Answer based on your own knowledge of Fluent. Do not hallucinate macro names."
    )
    return PromptBundle(system=BASELINE_SYSTEM_PROMPT, user=user)
