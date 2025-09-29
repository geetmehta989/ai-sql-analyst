import os
import logging
import re
import io
from uuid import uuid4
from typing import Any, List
import sqlite3

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import openai

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


# Global in-memory SQLite connection for demo
SQLITE_CONN = sqlite3.connect(":memory:", check_same_thread=False)
SQLITE_CONN.row_factory = sqlite3.Row


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

    # Load into in-memory SQLite as table "sales"
    try:
        # Prefer reading from bytes to avoid FS dependency too much
        excel_bytes = None
        with open(target_path, "rb") as f:
            excel_bytes = f.read()
        df = pd.read_excel(io.BytesIO(excel_bytes))
        if df.empty:
            raise ValueError("Uploaded Excel has no rows")
        # Normalize columns
        df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)).strip("_") or "col" for c in df.columns]
        df.to_sql("sales", SQLITE_CONN, if_exists="replace", index=False)
        cols: List[str] = [str(c) for c in df.columns]
        logger.info("Loaded 'sales' table with columns: %s", cols)
        return {"filename": file.filename, "table": "sales", "columns": cols}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load Excel into SQLite: %s", exc)
        raise HTTPException(status_code=400, detail=f"Failed to parse Excel: {exc}")


@app.post("/ask")
async def ask(request: Request):
    logger = logging.getLogger("uvicorn.error")
    try:
        body = await request.json()
        question = (body or {}).get("question", "")
        logger.info("/ask received question: %s", question)
        # Build schema context from SQLite
        cursor = SQLITE_CONN.cursor()
        cursor.execute("PRAGMA table_info(sales)")
        cols = [row[1] for row in cursor.fetchall()]
        if not cols:
            return {"error": "No 'sales' table loaded yet. Upload an Excel file first.", "answer": "", "sql": "", "columns": [], "data": []}

        schema_text = "Table sales with columns: " + ", ".join(cols)

        # OpenAI client via LiteLLM proxy
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://proxyllm.ximplify.id")
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        messages = [
            {"role": "system", "content": "You generate safe single SELECT SQL queries for SQLite. Only SELECT; no comments; reference only the table 'sales'."},
            {"role": "user", "content": f"{schema_text}. Question: {question}. Return only SQL."},
        ]
        resp = client.chat.completions.create(model="azure/gpt-5-mini", messages=messages)
        sql_text = resp.choices[0].message.content.strip().strip(";")
        if not re.match(r"^\s*select\b", sql_text, flags=re.I):
            sql_text = f"SELECT * FROM sales LIMIT 20"

        # Execute generated SQL
        try:
            cur = SQLITE_CONN.cursor()
            cur.execute(sql_text)
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description] if cur.description else []
            data = [list(r) for r in rows]
            answer = f"Returned {len(data)} rows."
            result: dict[str, Any] = {"answer": answer, "sql": sql_text, "columns": col_names, "data": data}
            logger.info("/ask response: %s", {"sql": sql_text, "rows": len(data)})
            return result
        except Exception as exec_exc:  # noqa: BLE001
            logger.exception("SQL execution failed: %s", exec_exc)
            return {"error": str(exec_exc), "answer": "", "sql": sql_text, "columns": [], "data": []}
    except Exception as exc:  # noqa: BLE001
        logger.exception("/ask error: %s", exc)
        return {"error": str(exc), "answer": "", "sql": "", "columns": [], "data": []}


# Keep existing routers
from .routers import upload, ask  # noqa: E402

# Avoid conflicting with the simple /upload endpoint
app.include_router(upload.router, prefix="/upload-full", tags=["upload-full-flow"])
app.include_router(ask.router, prefix="/ask", tags=["ask"])

