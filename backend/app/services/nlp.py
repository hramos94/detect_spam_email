"""Email classification service (zero‑shot) based on Hugging Face transformers.

Currently uses `facebook/bart-large-mnli` for multilingual NLI, mapping the
closest entailment label to a macro-category (Produtivo / Improdutivo).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from transformers import pipeline

__all__ = ["Category", "classify"]

Category = Literal["Produtivo", "Improdutivo"]

# --- Zero‑shot setup
_ZS_MODEL = "facebook/bart-large-mnli"

_LABEL_MAP: dict[Category, list[str]] = {
    "Produtivo": [
        "support request",
        "status update",
        "technical question",
    ],
    "Improdutivo": [
        "greetings",
        "thank you",
        "non-urgent",
    ],
}


@lru_cache
def _get_pipeline():  # pragma: no cover (heavy)
    """Load the transformers pipeline once per process."""
    return pipeline("text-classification", model=_ZS_MODEL, tokenizer=_ZS_MODEL)


def classify(text: str) -> Category:
    """Classify *text* as "Produtivo" or "Improdutivo" using zero‑shot NLI."""
    clf = _get_pipeline()

    # Flatten sublabels for the zero‑shot API
    candidate_labels = [lbl for sub in _LABEL_MAP.values() for lbl in sub]

    result = clf(text, candidate_labels=candidate_labels, multi_label=False)[0]

    # Map the chosen sublabel back to our macro category
    for macro, sub in _LABEL_MAP.items():
        if result["label"] in sub:
            return macro  # type: ignore [return-value]

    return "Improdutivo"  # fallback; shouldn't normally happen
