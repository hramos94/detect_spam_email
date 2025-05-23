"""I/O utilities for extracting raw text from email files.
Supports plain text and PDF for now. Extend as needed.
"""

from pathlib import Path
from typing import Union

import textract

__all__ = ["extract_text"]


def extract_text(path: Union[str, Path]) -> str:
    """Return the textual content of an e‑mail file.

    Parameters
    ----------
    path : str | Path
        Path to a `.txt` or `.pdf` file. For PDF, uses *textract* under the hood.

    Returns
    -------
    str
        Content decoded as UTF‑8. Empty string if the file is unreadable.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() == ".txt":
        return p.read_text(encoding="utf-8", errors="ignore")

    # Fallback: let textract handle the heavy lifting (PDF, DOCX, etc.)
    try:
        return textract.process(str(p)).decode("utf-8", errors="ignore")
    except Exception:  # pragma: no cover
        # In production, log the error details (Loguru, Sentry, etc.)
        return ""  # graceful degradation
