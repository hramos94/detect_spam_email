"""Email classification service using Hugging Face zero-shot pipeline.

Usa `facebook/bart-large-mnli` para decidir se o e-mail é
**Produtivo** (exige ação) ou **Improdutivo** (cordial / irrelevante).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from transformers import pipeline

Category = Literal["Produtivo", "Improdutivo"]

# ---------------------------- Modelo & rótulos -----------------------------

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

# --------------------------- Pipeline singleton ----------------------------

@lru_cache
def _get_pipeline():
    """Carrega uma única instância do pipeline zero-shot por processo."""
    return pipeline("zero-shot-classification", model=_ZS_MODEL, tokenizer=_ZS_MODEL)

# --------------------------- Função pública -------------------------------

def classify(text: str) -> Category:
    """Devolve **Produtivo** ou **Improdutivo** para o e-mail fornecido."""
    if not text.strip():
        return "Improdutivo"

    clf = _get_pipeline()
    candidate_labels = [lbl for sub in _LABEL_MAP.values() for lbl in sub]

    result = clf(text, candidate_labels=candidate_labels, multi_label=False)
    chosen_label = result["labels"][0]

    for macro, sub in _LABEL_MAP.items():
        if chosen_label in sub:
            return macro  # type: ignore[return-value]

    return "Improdutivo"  # fallback defensivo
