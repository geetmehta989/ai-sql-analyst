import os
import sqlite3
import io
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File-based SQLite DB under /tmp to survive multiple calls on warm instances
DB_PATH = "/tmp/sales.db"
LAST_UPLOAD_FILE = "/tmp/last_upload.xlsx"


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))
    # Save to DB file
    with sqlite3.connect(DB_PATH, check_same_thread=False) as c:
        df.to_sql("sales", c, if_exists="replace", index=False)
        c.commit()
    # Save the last uploaded Excel to /tmp for possible reloads on cold start
    with open(LAST_UPLOAD_FILE, "wb") as f:
        f.write(contents)
    return {"table": "sales", "columns": df.columns.tolist(), "rows": len(df)}


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(req: AskRequest):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    # Verify that the 'sales' table exists (requires a prior upload)
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sales'")
        exists = cursor.fetchone()
        if not exists:
            # Try to reload last uploaded file if present on this instance
            if os.path.exists(LAST_UPLOAD_FILE):
                try:
                    with open(LAST_UPLOAD_FILE, "rb") as f:
                        df = pd.read_excel(io.BytesIO(f.read()))
                    df.to_sql("sales", conn, if_exists="replace", index=False)
                    conn.commit()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sales'")
                    exists = cursor.fetchone()
                except Exception as reload_exc:  # noqa: BLE001
                    conn.close()
                    return {"answer": "Could not process query", "error": str(reload_exc), "sql": "", "columns": [], "rows": []}
        if not exists:
            conn.close()
            return {"answer": "No data uploaded yet. Please upload an Excel file first.", "sql": "", "columns": [], "rows": []}
    except Exception as e:  # noqa: BLE001
        return {"answer": "Could not process query", "error": str(e), "sql": "", "columns": [], "rows": []}

    # Build a minimal schema hint for the LLM
    cursor.execute("PRAGMA table_info(sales)")
    schema_cols = [row[1] for row in cursor.fetchall()]
    schema_hint = "Columns: " + ", ".join(schema_cols)

    # Ask the LLM to generate a safe SELECT SQL query
    prompt = (
        "You write a single SQLite SELECT query (no comments, no DDL/DML).\n"
        "Table: sales. " + schema_hint + "\n"
        f"Question: {req.question}\n"
        "Only output SQL."
    )

    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "azure/gpt-5-mini")
    sql = ""
    llm_error: str | None = None
    if api_key:
        try:
            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            # Try to generate SQL via LLM
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            sql = (completion.choices[0].message.content or "").strip().strip(";")
        except (APIStatusError, APIConnectionError, RateLimitError, Exception) as e:  # noqa: BLE001
            llm_error = f"LLM error: {str(e)}"

    # If no SQL from LLM, attempt heuristic SQL generation
    if not sql:
        # Determine aggregation intent and target column
        agg = "SUM"
        q = (req.question or "").lower()
        if any(k in q for k in ["average", "avg", "mean"]):
            agg = "AVG"
        elif any(k in q for k in ["count", "how many"]):
            agg = "COUNT"
        elif "max" in q:
            agg = "MAX"
        elif "min" in q:
            agg = "MIN"

        cursor.execute("PRAGMA table_info(sales)")
        cols_info = cursor.fetchall()
        col_names = [c[1] for c in cols_info]
        types = {c[1]: (c[2] or "").upper() for c in cols_info}

        # Choose best matching column by name and numeric type preference
        keywords = ["expense", "expenses", "amount", "revenue", "sale", "sales", "price", "cost", "total"]
        candidates: list[str] = []
        for name in col_names:
            lname = name.lower()
            if any(k in lname for k in keywords):
                candidates.append(name)
        # Prefer numeric columns
        def is_numeric(sqlite_type: str) -> bool:
            t = sqlite_type.upper()
            return any(x in t for x in ["INT", "REAL", "NUM", "DECIMAL", "DOUBLE", "FLOAT"])

        numeric_candidates = [c for c in candidates if is_numeric(types.get(c, ""))]
        target_col = (numeric_candidates or candidates or [col_names[0]])[0] if col_names else None
        if not target_col:
            return {"answer": "No columns found in table.", "sql": "", "columns": [], "rows": []}
        sql = f"SELECT {agg}({target_col}) AS value FROM sales"

    # Execute the generated SQL and format the response
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description] if cursor.description else []
        json_rows = [list(r) for r in rows]

        # Try to craft a concise answer when it looks like a scalar result
        answer = "Here is the result of your query."
        if len(json_rows) == 1 and len(cols) == 1:
            answer = f"{cols[0]} = {json_rows[0][0]}"
        elif len(json_rows) == 0:
            answer = "No rows returned."

        res = {"answer": answer, "sql": sql, "columns": cols, "rows": json_rows}
        conn.close()
        return res
    except Exception as e:  # noqa: BLE001
        err = f"{str(e)}"
        if llm_error:
            err = f"{llm_error}; {err}"
        try:
            conn.close()
        except Exception:
            pass
        return {"answer": "Could not process query", "error": err, "sql": sql, "columns": [], "rows": []}

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

