import os
import io
import sqlite3
import logging
from typing import Any, List, Optional

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
try:
    from openai import OpenAI  # Optional; fallback heuristics if not configured
except Exception:  # noqa: BLE001
    OpenAI = None  # type: ignore


app = FastAPI(title="AI SQL Analyst Backend")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DB_PATH = "/tmp/sales.db"
LAST_UPLOAD_FILE = "/tmp/last_upload.xlsx"


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _get_logger() -> logging.Logger:
    return logging.getLogger("uvicorn.error")


def _ensure_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None


def _load_excel_to_db(bytes_data: bytes) -> dict[str, Any]:
    df = pd.read_excel(io.BytesIO(bytes_data))
    with _ensure_connection() as c:
        df.to_sql("sales", c, if_exists="replace", index=False)
        c.commit()
    return {"table": "sales", "columns": df.columns.tolist(), "rows": len(df)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    contents = await file.read()
    # Persist uploaded bytes for potential reload
    with open(LAST_UPLOAD_FILE, "wb") as f:
        f.write(contents)
    meta = _load_excel_to_db(contents)
    _get_logger().info("Uploaded and loaded '%s' with columns: %s", file.filename, meta.get("columns"))
    return {"filename": file.filename, **meta}


class AskRequest(BaseModel):
    question: str


def _detect_numeric(sqlite_decl_type: str) -> bool:
    t = (sqlite_decl_type or "").upper()
    return any(k in t for k in ["INT", "REAL", "NUM", "DECIMAL", "DOUBLE", "FLOAT"])


def _pick_agg_and_column(conn: sqlite3.Connection, question: str) -> tuple[str, Optional[str], list[str], dict[str, str]]:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(sales)")
    cols_info = cur.fetchall()
    col_names = [c[1] for c in cols_info]
    types = {c[1]: (c[2] or "").upper() for c in cols_info}

    q = (question or "").lower()
    agg = "SUM"
    if any(k in q for k in ["average", "avg", "mean"]):
        agg = "AVG"
    elif any(k in q for k in ["count", "how many"]):
        agg = "COUNT"
    elif "max" in q:
        agg = "MAX"
    elif "min" in q:
        agg = "MIN"

    if "total" in q and "column" in q:
        return agg, None, col_names, types

    keywords = ["expense", "expenses", "amount", "revenue", "sale", "sales", "price", "cost", "total"]
    candidates: list[str] = []
    for name in col_names:
        lname = name.lower()
        if any(k in lname for k in keywords):
            candidates.append(name)
    numeric_candidates = [c for c in candidates if _detect_numeric(types.get(c, ""))]
    target_col = (numeric_candidates or candidates or [col_names[0]])[0] if col_names else None
    return agg, target_col, col_names, types


def _try_llm_sql(question: str, schema_cols: list[str]) -> tuple[str, Optional[str]]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "azure/gpt-5-mini")
    if not api_key or OpenAI is None:
        return "", None
    try:
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        prompt = (
            "You write a single SQLite SELECT query (no comments, no DDL/DML).\n"
            f"Table: sales. Columns: {', '.join(schema_cols)}\n"
            f"Question: {question}\n"
            "Only output SQL."
        )
        comp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        sql = (comp.choices[0].message.content or "").strip().strip(";")
        return sql, None
    except Exception as e:  # noqa: BLE001
        return "", str(e)


def _execute_sql(conn: sqlite3.Connection, sql: str) -> tuple[list[str], list[list[Any]]]:
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    return cols, [list(r) for r in rows]


@app.post("/ask")
async def ask(req: AskRequest) -> dict[str, Any]:
    logger = _get_logger()
    conn = _ensure_connection()
    try:
        if not _table_exists(conn, "sales"):
            if os.path.exists(LAST_UPLOAD_FILE):
                with open(LAST_UPLOAD_FILE, "rb") as f:
                    _load_excel_to_db(f.read())
                conn = _ensure_connection()
            if not _table_exists(conn, "sales"):
                return {"answer": "No data uploaded yet. Please upload an Excel file first.", "sql": "", "columns": [], "rows": []}

        # Schema and target selection
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(sales)")
        schema_cols = [row[1] for row in cur.fetchall()]
        agg, target_col, all_cols, types = _pick_agg_and_column(conn, req.question)

        # Special: total columns
        if target_col is None and "total" in (req.question or "").lower() and "column" in (req.question or "").lower():
            return {"answer": f"Total columns = {len(all_cols)}", "sql": "", "columns": ["total_columns"], "rows": [[len(all_cols)]]}

        # Try LLM SQL first (optional)
        sql, llm_err = _try_llm_sql(req.question, schema_cols)
        if not sql:
            sql = f"SELECT {agg}({target_col}) AS value FROM sales"

        # Ensure SELECT-only
        if not sql.lower().startswith("select"):
            sql = f"SELECT {agg}({target_col}) AS value FROM sales"

        try:
            cols, rows = _execute_sql(conn, sql)
            answer = "Here is the result of your query."
            if len(rows) == 1 and len(cols) == 1:
                answer = f"{cols[0]} = {rows[0][0]}"
            elif len(rows) == 0:
                answer = "No rows returned."
            return {"answer": answer, "sql": sql, "columns": cols, "rows": rows, **({"error": llm_err} if llm_err else {})}
        except Exception as exec_err:  # noqa: BLE001
            # Fallback to pandas computation tolerating dirty data
            try:
                df = pd.read_sql_query("SELECT * FROM sales", conn)
                series = pd.to_numeric(df.get(target_col), errors="coerce")
                value: Any = None
                if agg == "SUM":
                    value = float(series.sum()) if series.notna().any() else None
                elif agg == "AVG":
                    value = float(series.mean()) if series.notna().any() else None
                elif agg == "COUNT":
                    value = int(series.count())
                elif agg == "MAX":
                    value = float(series.max()) if series.notna().any() else None
                elif agg == "MIN":
                    value = float(series.min()) if series.notna().any() else None
                if value is None:
                    raise ValueError("No numeric data to aggregate")
                return {"answer": f"{agg}({target_col}) = {value}", "sql": sql, "columns": [f"{agg.lower()}"], "rows": [[value]], **({"error": str(exec_err)} if llm_err or exec_err else {})}
            except Exception as pandas_err:  # noqa: BLE001
                err_msg = f"{llm_err + '; ' if llm_err else ''}{str(exec_err)}; pandas fallback failed: {str(pandas_err)}"
                return {"answer": "Could not process query", "error": err_msg, "sql": sql, "columns": [], "rows": []}
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


@app.post("/ask_with_file")
async def ask_with_file(question: str = Form(...), file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        contents = await file.read()
        _load_excel_to_db(contents)
        with open(LAST_UPLOAD_FILE, "wb") as f:
            f.write(contents)
    except Exception as e:  # noqa: BLE001
        return {"answer": "Could not process query", "error": f"Failed to read Excel: {str(e)}", "sql": "", "columns": [], "rows": []}
    return await ask(AskRequest(question=question))

import os
import sqlite3
import io
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
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


@app.post("/ask_with_file")
async def ask_with_file(question: str = Form(...), file: UploadFile = File(...)):
    # Load provided Excel into persistent DB for this request, then delegate to /ask logic
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        with sqlite3.connect(DB_PATH, check_same_thread=False) as c:
            df.to_sql("sales", c, if_exists="replace", index=False)
            c.commit()
        # Also update last upload cache for warm instances
        with open(LAST_UPLOAD_FILE, "wb") as f:
            f.write(contents)
    except Exception as e:  # noqa: BLE001
        return {"answer": "Could not process query", "error": f"Failed to read Excel: {str(e)}", "sql": "", "columns": [], "rows": []}

    # Reuse the JSON /ask flow by constructing an AskRequest-like object
    return await ask(AskRequest(question=question))

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

