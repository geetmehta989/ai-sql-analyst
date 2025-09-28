import io
import re
from typing import List

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy import inspect

from ..db import get_engine
from ..schemas import UploadResponse

router = APIRouter()


def _slugify(value: str) -> str:
    s = (value or "").strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s or "col"


def _dedupe_columns(columns: List[str]) -> List[str]:
    seen = {}
    result = []
    for col in columns:
        base = _slugify(str(col) if col is not None else "col")
        if base not in seen:
            seen[base] = 0
            result.append(base)
        else:
            seen[base] += 1
            result.append(f"{base}_{seen[base]}")
    return result


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = _dedupe_columns(list(df.columns))
    # Strip whitespace for object columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    # Replace NaN with None for SQL compatibility
    df = df.where(pd.notnull(df), None)
    return df


@router.post("/", response_model=UploadResponse)
async def upload_excel(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
        raise HTTPException(status_code=400, detail="Please upload an Excel file")

    content = await file.read()
    try:
        xls = pd.ExcelFile(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to read Excel: {exc}")

    engine = get_engine()
    inspector = inspect(engine)

    created_tables: List[str] = []

    for sheet_name in xls.sheet_names:
        try:
            df = xls.parse(sheet_name=sheet_name, dtype_backend="pyarrow")
        except Exception:  # noqa: BLE001
            df = xls.parse(sheet_name=sheet_name)

        if df.empty:
            continue

        df = _clean_dataframe(df)
        table_name = _slugify(sheet_name or "sheet")

        try:
            df.to_sql(table_name, con=engine, if_exists="replace", index=False)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to write table {table_name}: {exc}")

        if table_name not in created_tables:
            created_tables.append(table_name)

    if not created_tables:
        raise HTTPException(status_code=400, detail="No non-empty sheets found in the Excel file")

    # Verify tables exist
    missing = [t for t in created_tables if t not in inspector.get_table_names()]
    if missing:
        raise HTTPException(status_code=500, detail=f"Tables not created: {missing}")

    return UploadResponse(tables=created_tables, message="Upload and load complete")

