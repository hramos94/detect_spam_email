"""Generate an automatic reply for an e‑mail based on its category.

Combines our internal classifier (`services.nlp`) with OpenAI's Chat Completions
API to craft a short, polite answer in Portuguese.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

from openai import OpenAI

from ..core.config import settings
from .nlp import Category, classify

__all__ = ["suggest"]

# OpenAI client (singleton)

@lru_cache
def _get_client() -> OpenAI:  # pragma: no cover
    return OpenAI(api_key=settings.openai_api_key)


# Prompts & templates

_SYSTEM_PROMPT = (
    "Você é um atendente cordial de uma instituição financeira. "
    "Responda sempre em português, de forma clara, objetiva e profissional."
)

_TEMPLATE = {
    "Produtivo": (
        "Olá!\n\n{assistant}\n\nFico à disposição para qualquer dúvida."
    ),
    "Improdutivo": (
        "Olá!\n\n{assistant}\n\nTenha um ótimo dia!"
    ),
}


# Public API

def suggest(text: str) -> Tuple[str, Category]:
    """Return a tuple (reply, category) for the given e‑mail *text*."""

    category = classify(text)

    client = _get_client()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # lighter, cheaper, fast
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=200,
        temperature=0.3,
    )

    assistant_reply = completion.choices[0].message.content.strip()

    reply = _TEMPLATE[category].format(assistant=assistant_reply)
    return reply, category
