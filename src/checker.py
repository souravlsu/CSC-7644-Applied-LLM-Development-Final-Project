"""Lightweight static validator for generated Fluent UDF code.

This does NOT compile UDFs. It checks structural signals such as:
  * `#include "udf.h"` present when code is produced.
  * A plausible `DEFINE_*` macro signature.
  * Macro family consistency with the retrieval context / task.
  * Flags identifiers that look like hallucinated Fluent APIs (unknown DEFINE_* or C_* tokens).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CODE_BLOCK_RE = re.compile(r"```(?:c|cpp)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
DEFINE_SIGNATURE_RE = re.compile(
    r"\bDEFINE_[A-Z0-9_]+\s*\([^)]*\)",
    re.MULTILINE,
)
DEFINE_NAME_RE = re.compile(r"\bDEFINE_[A-Z0-9_]+")
C_MACRO_RE = re.compile(r"\bC_[A-Z0-9_]+")
INCLUDE_UDF_RE = re.compile(r'#\s*include\s*[<"]\s*udf\.h\s*[>"]')


@dataclass
class CheckResult:
    has_code: bool
    include_udf: bool
    define_signature: bool
    macro_family: str | None
    macro_family_matches_context: bool | None
    unknown_define_tokens: list[str] = field(default_factory=list)
    unknown_c_macros: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if not self.has_code:
            return True
        structural = self.include_udf and self.define_signature
        clean = not self.unknown_define_tokens and not self.unknown_c_macros
        context_ok = self.macro_family_matches_context in (None, True)
        return structural and clean and context_ok

    def to_dict(self) -> dict:
        return {
            "has_code": self.has_code,
            "include_udf": self.include_udf,
            "define_signature": self.define_signature,
            "macro_family": self.macro_family,
            "macro_family_matches_context": self.macro_family_matches_context,
            "unknown_define_tokens": self.unknown_define_tokens,
            "unknown_c_macros": self.unknown_c_macros,
            "warnings": self.warnings,
            "passed": self.passed,
        }


def extract_code(answer: str) -> str | None:
    """Extract the first fenced code block from an answer."""
    match = CODE_BLOCK_RE.search(answer)
    if match:
        return match.group(1).strip()
    return None


def check_answer(
    answer: str,
    known_define_macros: set[str],
    known_c_macros: set[str],
    expected_family: str | None = None,
) -> CheckResult:
    """Run structural checks on a model answer and return validation details."""
    code = extract_code(answer)
    if not code:
        return CheckResult(
            has_code=False,
            include_udf=False,
            define_signature=False,
            macro_family=None,
            macro_family_matches_context=None,
        )

    include_udf = bool(INCLUDE_UDF_RE.search(code))
    signature_match = DEFINE_SIGNATURE_RE.search(code)
    macro_family = None
    if signature_match:
        macro_name_match = DEFINE_NAME_RE.search(signature_match.group(0))
        if macro_name_match:
            macro_family = macro_name_match.group(0)

    define_tokens = set(DEFINE_NAME_RE.findall(code))
    c_tokens = set(C_MACRO_RE.findall(code))
    unknown_defines = sorted(
        tok for tok in define_tokens if tok not in known_define_macros
    )
    unknown_c = sorted(tok for tok in c_tokens if tok not in known_c_macros)

    macro_family_matches_context: bool | None
    if expected_family and macro_family:
        macro_family_matches_context = macro_family == expected_family
    else:
        macro_family_matches_context = None

    warnings: list[str] = []
    if not include_udf:
        warnings.append('Missing `#include "udf.h"`.')
    if not signature_match:
        warnings.append("No DEFINE_* signature detected in code block.")
    if unknown_defines:
        warnings.append(
            f"Unknown DEFINE_* tokens (possible hallucinations): {unknown_defines}."
        )
    if unknown_c:
        warnings.append(f"Unknown C_* macros (possible hallucinations): {unknown_c}.")
    if expected_family and macro_family and macro_family_matches_context is False:
        warnings.append(
            f"Macro family mismatch: expected {expected_family} but code uses {macro_family}."
        )

    return CheckResult(
        has_code=True,
        include_udf=include_udf,
        define_signature=bool(signature_match),
        macro_family=macro_family,
        macro_family_matches_context=macro_family_matches_context,
        unknown_define_tokens=unknown_defines,
        unknown_c_macros=unknown_c,
        warnings=warnings,
    )


def collect_known_tokens(chunks) -> tuple[set[str], set[str]]:
    """Harvest the DEFINE_* and C_* tokens that appear anywhere in the corpus.

    These form the allow-list used for hallucination detection.
    """
    defines: set[str] = set()
    c_macros: set[str] = set()
    for chunk in chunks:
        defines.update(DEFINE_NAME_RE.findall(chunk.text))
        c_macros.update(C_MACRO_RE.findall(chunk.text))
    return defines, c_macros
