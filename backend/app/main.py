import os
import io
import sqlite3
import logging
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AI SQL Analyst Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistence under /tmp for serverless contexts
DB_PATH = "/tmp/sales.db"
LAST_UPLOAD_FILE = "/tmp/last_upload.xlsx"


def _logger() -> logging.Logger:
    return logging.getLogger("uvicorn.error")


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None


def _load_excel_bytes_to_db(bytes_data: bytes) -> dict[str, Any]:
    df = pd.read_excel(io.BytesIO(bytes_data))
    with _conn() as c:
        df.to_sql("sales", c, if_exists="replace", index=False)
        c.commit()
    return {"table": "sales", "columns": df.columns.tolist(), "rows": len(df)}


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    contents = await file.read()
    # persist the file for warm reloads
    with open(LAST_UPLOAD_FILE, "wb") as f:
        f.write(contents)
    meta = _load_excel_bytes_to_db(contents)
    _logger().info("Uploaded '%s' with columns: %s", file.filename, meta.get("columns"))
    return {"filename": file.filename, **meta}


class AskRequest(BaseModel):
    question: str


def _pick_target_column(conn: sqlite3.Connection, question: str) -> tuple[str, Optional[str], list[str], dict[str, str]]:
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

    def _is_numeric(decl: str) -> bool:
        t = (decl or "").upper()
        return any(k in t for k in ["INT", "REAL", "NUM", "DECIMAL", "DOUBLE", "FLOAT"])

    numeric_candidates = [c for c in candidates if _is_numeric(types.get(c, ""))]
    target_col = (numeric_candidates or candidates or ([col_names[0]] if col_names else []))
    target = target_col[0] if target_col else None
    return agg, target, col_names, types


def _aggregate_numeric(conn: sqlite3.Connection, agg: str, target_col: str) -> Optional[float | int]:
    # Use pandas with aggressive cleaning to tolerate dirty data (e.g., "3000rs")
    df = pd.read_sql_query("SELECT * FROM sales", conn)
    if target_col not in df.columns:
        return None
    s = df[target_col].astype(str).str.replace(r"[^0-9.+-]", "", regex=True)
    s_num = pd.to_numeric(s, errors="coerce")
    if agg == "SUM":
        return float(s_num.sum()) if s_num.notna().any() else None
    if agg == "AVG":
        return float(s_num.mean()) if s_num.notna().any() else None
    if agg == "COUNT":
        return int(s_num.count())
    if agg == "MAX":
        return float(s_num.max()) if s_num.notna().any() else None
    if agg == "MIN":
        return float(s_num.min()) if s_num.notna().any() else None
    return None


@app.post("/ask")
async def ask(req: AskRequest) -> dict[str, Any]:
    conn = _conn()
    try:
        # Ensure table
        if not _table_exists(conn, "sales"):
            if os.path.exists(LAST_UPLOAD_FILE):
                with open(LAST_UPLOAD_FILE, "rb") as f:
                    _load_excel_bytes_to_db(f.read())
                conn = _conn()
            if not _table_exists(conn, "sales"):
                return {"answer": "", "sql": "", "columns": [], "rows": [], "error": "No 'sales' table loaded yet. Upload an Excel file first."}

        agg, target, all_cols, _types = _pick_target_column(conn, req.question)
        # Special question: total columns
        if target is None and "total" in (req.question or "").lower() and "column" in (req.question or "").lower():
            return {"answer": str(len(all_cols)), "sql": "", "columns": ["total_columns"], "rows": [[len(all_cols)]]}

        if not target:
            return {"answer": "", "sql": "", "columns": [], "rows": [], "error": "Could not determine target column"}

        value = _aggregate_numeric(conn, agg, target)
        if value is None:
            return {"answer": "", "sql": "", "columns": [], "rows": [], "error": "No numeric data to aggregate"}

        # Return numeric-only answer as requested
        return {"answer": str(value), "sql": "", "columns": [f"{agg.lower()}"], "rows": [[value]]}
    except Exception as e:  # noqa: BLE001
        return {"answer": "", "sql": "", "columns": [], "rows": [], "error": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


@app.post("/ask_with_file")
async def ask_with_file(question: str = Form(...), file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        contents = await file.read()
        _load_excel_bytes_to_db(contents)
        with open(LAST_UPLOAD_FILE, "wb") as f:
            f.write(contents)
    except Exception as e:  # noqa: BLE001
        return {"answer": "", "sql": "", "columns": [], "rows": [], "error": f"Failed to read Excel: {str(e)}"}
    return await ask(AskRequest(question=question))

