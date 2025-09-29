import os
import logging
import re
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .db import get_engine


app = FastAPI(title="Excel-to-SQL Q&A API", version="0.1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file")
    lower = file.filename.lower()
    if not lower.endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
        raise HTTPException(status_code=400, detail="Please upload an Excel file (.xls, .xlsx, .xlsm, .xlsb)")

    logger = logging.getLogger("uvicorn.error")
    logger.info("/upload received file: %s", file.filename)

    # Sanitize filename and save to /tmp (ephemeral on Vercel)
    def _safe_name(name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        return safe or "upload.xlsx"

    target_dir = "/tmp"
    try:
        os.makedirs(target_dir, exist_ok=True)
    except Exception:
        pass

    target_path = os.path.join(target_dir, f"{uuid4().hex}_{_safe_name(file.filename)}")
    try:
        contents = await file.read()
        with open(target_path, "wb") as f:
            f.write(contents)
        logger.info("Saved uploaded file to: %s (%d bytes)", target_path, len(contents))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed saving upload: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    return {"filename": file.filename}


# Keep existing routers
from .routers import upload, ask  # noqa: E402

# Avoid conflicting with the simple /upload endpoint
app.include_router(upload.router, prefix="/upload-full", tags=["upload-full-flow"])
app.include_router(ask.router, prefix="/ask", tags=["ask"])

