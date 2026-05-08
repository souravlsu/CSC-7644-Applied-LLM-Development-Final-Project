"""Evaluation harness: compare baseline vs RAG on a curated prompt set.

Metrics implemented:
  1. Macro-selection accuracy  - chose the correct DEFINE_* family.
  2. Structural pass rate      - static checks on includes, signature, and hallucinations.
  3. Grounding quality (RAG)   - whether the answer cites any retrieved context tag.
  4. Usefulness                - not scored here; records the outputs for human rubric scoring.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .assistant import Assistant, AssistantResult

CITATION_RE = re.compile(r"\[C\d+\]")


@dataclass
class EvalPrompt:
    id: str
    task_type: str
    prompt: str
    expected_family: str | None = None
    notes: str = ""


@dataclass
class EvalRecord:
    prompt_id: str
    task_type: str
    prompt: str
    expected_family: str | None
    mode: str
    answer: str
    macro_family: str | None
    macro_family_correct: bool | None
    structural_pass: bool
    cites_context: bool | None
    warnings: list[str]


def load_prompts(path: Path) -> list[EvalPrompt]:
    """Load evaluation prompts from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalPrompt(**item) for item in data]


def _record_from_result(prompt: EvalPrompt, result: AssistantResult) -> EvalRecord:
    """Convert a model run into a normalized evaluation record."""
    check = result.check
    macro_family_correct: bool | None
    if prompt.expected_family is None:
        macro_family_correct = None
    elif check.macro_family is None:
        macro_family_correct = False
    else:
        macro_family_correct = check.macro_family == prompt.expected_family

    cites_context: bool | None = None
    if result.mode == "rag":
        cites_context = bool(CITATION_RE.search(result.answer))

    return EvalRecord(
        prompt_id=prompt.id,
        task_type=prompt.task_type,
        prompt=prompt.prompt,
        expected_family=prompt.expected_family,
        mode=result.mode,
        answer=result.answer,
        macro_family=check.macro_family,
        macro_family_correct=macro_family_correct,
        structural_pass=check.passed,
        cites_context=cites_context,
        warnings=check.warnings,
    )


def run_evaluation(
    assistant: Assistant,
    prompts: list[EvalPrompt],
    modes: Iterable[str] = ("baseline", "rag"),
) -> list[EvalRecord]:
    """Run all prompts in all selected modes and collect records."""
    records: list[EvalRecord] = []
    for prompt in prompts:
        for mode in modes:
            result = assistant.ask(
                prompt.prompt,
                mode=mode,  # type: ignore[arg-type]
                expected_family=prompt.expected_family,
            )
            records.append(_record_from_result(prompt, result))
    return records


def summarize(records: list[EvalRecord]) -> dict:
    """Aggregate per-mode metrics from raw evaluation records."""
    summary: dict[str, dict] = {}
    for mode in sorted({r.mode for r in records}):
        subset = [r for r in records if r.mode == mode]
        total = len(subset)
        if total == 0:
            continue
        macro_scored = [r for r in subset if r.macro_family_correct is not None]
        macro_correct = sum(1 for r in macro_scored if r.macro_family_correct)
        structural_pass = sum(1 for r in subset if r.structural_pass)
        citation_scored = [r for r in subset if r.cites_context is not None]
        citation_ok = sum(1 for r in citation_scored if r.cites_context)
        summary[mode] = {
            "prompts": total,
            "macro_accuracy": (
                macro_correct / len(macro_scored) if macro_scored else None
            ),
            "structural_pass_rate": structural_pass / total,
            "citation_rate": (
                (citation_ok / len(citation_scored)) if citation_scored else None
            ),
        }
    return summary


def save_records(records: list[EvalRecord], out_path: Path) -> None:
    """Persist evaluation records to a JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)
