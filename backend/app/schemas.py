from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    tables: List[str] = Field(default_factory=list, description="List of created/updated table names")
    message: str = ""


class AskRequest(BaseModel):
    question: str
    top_k: int = 100
    table_hint: Optional[str] = Field(
        default=None,
        description="Optional table name hint to scope the query",
    )


class ChartSpec(BaseModel):
    type: Literal["table", "bar", "line", "pie", "scatter", "area"] = "table"
    xKey: Optional[str] = None
    yKeys: List[str] = []


class TabularData(BaseModel):
    columns: List[str]
    rows: List[List[Any]]


class AskResponse(BaseModel):
    answer: str
    sql: str
    chart: ChartSpec
    data: TabularData

