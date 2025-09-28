import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import openai
import sqlalchemy as sa
from sqlalchemy.engine import Engine


def _get_openai_client() -> openai.OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://proxyllm.ximplify.id")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    return client


def _reflect_schema(engine: Engine) -> str:
    insp = sa.inspect(engine)
    lines: List[str] = []
    for table_name in insp.get_table_names():
        cols = []
        for col in insp.get_columns(table_name):
            col_name = col.get("name")
            col_type = str(col.get("type"))
            cols.append(f"{col_name} {col_type}")
        lines.append(f"TABLE {table_name} (" + ", ".join(cols) + ")")
    return "\n".join(lines)


def _guard_sql(sql_text: str) -> str:
    text = sql_text.strip().rstrip(";")
    if re.search(r"--|/\*|\*/", text):
        raise ValueError("SQL must not contain comments")
    forbidden = [r"\\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|commit|rollback)\\b"]
    if any(re.search(p, text, flags=re.I) for p in forbidden):
        raise ValueError("Only SELECT queries are permitted")
    if not re.match(r"^\s*select\b", text, flags=re.I):
        raise ValueError("Query must be a SELECT")
    if ";" in sql_text.strip():
        raise ValueError("Only a single statement is allowed")
    return text


def _infer_chart(columns: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    if not columns or not rows:
        return {"type": "table", "xKey": None, "yKeys": []}
    def is_number(val: Any) -> bool:
        try:
            float(val)
            return True
        except Exception:
            return False
    num_cols: List[int] = []
    for idx in range(1, min(len(columns), 4)):
        if all((r[idx] is None) or is_number(r[idx]) for r in rows[:10]):
            num_cols.append(idx)
    if num_cols:
        return {"type": "bar", "xKey": columns[0], "yKeys": [columns[i] for i in num_cols[:3]]}
    return {"type": "table", "xKey": None, "yKeys": []}


class SQLAgent:
    def __init__(self, engine: Engine, model: str = "azure/gpt-5-mini") -> None:
        self.engine = engine
        self.model = model
        self.client = _get_openai_client()

    def _build_prompt(self, question: str, schema_ddl: str, table_hint: Optional[str]) -> List[Dict[str, str]]:
        system = (
            "You are a senior data analyst. Generate a single safe SELECT SQL query for the given question. "
            "The database may have messy table and column names; match by semantics. "
            "Rules: Only SELECT; no comments; no DDL/DML; single statement; limit rows if large; prefer explicit columns. "
            "Return JSON with keys sql and reasoning."
        )
        user = (
            f"SCHEMA:\n{schema_ddl}\n\n"
            f"QUESTION: {question}\n"
            + (f"TABLE_HINT: {table_hint}\n" if table_hint else "")
            + "Respond as JSON: {\"sql\": \"...\", \"reasoning\": \"...\"}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _generate_sql(self, question: str, table_hint: Optional[str]) -> Tuple[str, str]:
        schema = _reflect_schema(self.engine)
        messages = self._build_prompt(question, schema, table_hint)
        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        content = resp.choices[0].message.content if hasattr(resp.choices[0], "message") else resp.choices[0].text
        try:
            data = json.loads(content)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                raise ValueError("LLM did not return JSON")
            data = json.loads(match.group(0))
        sql_text: str = _guard_sql(data["sql"])  # type: ignore[index]
        reasoning: str = data.get("reasoning", "")
        return sql_text, reasoning

    def _execute_sql(self, sql_text: str, top_k: int) -> Tuple[List[str], List[List[Any]]]:
        limited_sql = sql_text
        if re.search(r"\blimit\b", sql_text, flags=re.I) is None:
            limited_sql = f"{sql_text} LIMIT {max(10, min(top_k, 1000))}"
        with self.engine.connect() as conn:
            result = conn.execute(sa.text(limited_sql))
            columns = list(result.keys())
            rows = [list(r) for r in result.fetchall()]
        return columns, rows

    def _summarize(self, question: str, columns: List[str], rows: List[List[Any]]) -> str:
        if not columns or not rows:
            return "No data found for the query."
        preview = [dict(zip(columns, r)) for r in rows[:5]]
        return (
            f"Answer to: {question}\n"
            f"Columns: {', '.join(columns)}. Returned {len(rows)} rows. Preview: {json.dumps(preview)[:400]}"
        )

    def answer_question(self, question: str, table_hint: Optional[str], top_k: int) -> Dict[str, Any]:
        sql_text, _ = self._generate_sql(question, table_hint)
        columns, rows = self._execute_sql(sql_text, top_k)
        chart = _infer_chart(columns, rows)
        answer = self._summarize(question, columns, rows)
        return {
            "answer": answer,
            "sql": sql_text,
            "columns": columns,
            "rows": rows,
            "chart": chart,
        }

