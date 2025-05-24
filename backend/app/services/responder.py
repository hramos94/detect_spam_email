"""Generate an automatic reply for an e‑mail based on its category.
Adds graceful degradation when the OpenAI API is unavailable (e.g., quota
exceeded) so the backend never returns HTTP 500.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

from openai import OpenAI, OpenAIError, RateLimitError

from ..core.config import settings
from .nlp import Category, classify

__all__ = ["suggest"]

# ---------------------------------------------------------------------------
# OpenAI client (singleton)
# ---------------------------------------------------------------------------

@lru_cache
def _get_client() -> OpenAI:  # pragma: no cover
    """Create a single OpenAI client per process."""
    return OpenAI(api_key=settings.openai_api_key)


# ---------------------------------------------------------------------------
# Prompts & templates
# ---------------------------------------------------------------------------

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

_FALLBACK_ASSISTANT = (
    "Desculpe, não consegui gerar uma resposta automática no momento. "
    "Encaminhei sua mensagem para a equipe responsável."
)


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def suggest(text: str) -> Tuple[str, Category]:
    """Return ``(reply, category)`` for the given e‑mail *text*.

    * Classifica o e‑mail usando zero‑shot (``classify``).
    * Gera resposta com OpenAI; se falhar (quota, auth, rede), usa fallback.
    """

    category = classify(text)

    try:
        client = _get_client()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        assistant_reply = completion.choices[0].message.content.strip()
    except (OpenAIError, RateLimitError):  # quota, auth, network…
        assistant_reply = _FALLBACK_ASSISTANT

    reply = _TEMPLATE[category].format(assistant=assistant_reply)
    return reply, category