"""FastAPI entrypoint for the e‑mail classifier backend.
Run locally with:
    uvicorn backend.app.main:app --reload --port 8000
"""

from pathlib import Path

import aiofiles
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
# StaticFiles fica comentado até termos um frontend buildado
# from fastapi.staticfiles import StaticFiles

from .core.io import extract_text
from .services.responder import suggest

app = FastAPI(title="Email Classifier")

# Se desejar servir o frontend dentro do mesmo container, basta descomentar:
# FRONT_DIST = Path(__file__).parent.parent / ".." / "frontend" / "dist"
# if FRONT_DIST.exists():
#     app.mount("/", StaticFiles(directory=FRONT_DIST, html=True), name="frontend")


@app.post("/api/classify")
async def classify_endpoint(
    file: UploadFile | None = None,
    text: str | None = Form(None),
):
    """Classifica o e‑mail como Produtivo/Improdutivo e gera resposta sugerida.

    Aceita  um arquivo (.txt/.pdf) via multipart ou um campo
    `text` com o corpo do e‑mail.
    """
    if not (file or text):
        raise HTTPException(status_code=400, detail="Envie arquivo ou texto.")

    # --- Obtém o conteúdo do e‑mail ---------------------------------------
    if file:
        temp_path = Path("/tmp") / file.filename
        async with aiofiles.open(temp_path, "wb") as out:
            await out.write(await file.read())
        email_text = extract_text(temp_path)
    else:
        email_text = text or ""

    # --- Chama pipeline NLP ----------------------------------------------
    reply, category = suggest(email_text)

    return JSONResponse({
        "category": category,
        "suggested_reply": reply,
    })