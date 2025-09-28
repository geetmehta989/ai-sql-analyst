from fastapi import APIRouter, HTTPException

from ..db import get_engine
from ..schemas import AskRequest, AskResponse, ChartSpec, TabularData
from agent.sql_agent import SQLAgent

router = APIRouter()


@router.post("/", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    engine = get_engine()
    agent = SQLAgent(engine)

    try:
        result = agent.answer_question(
            question=request.question,
            table_hint=request.table_hint,
            top_k=request.top_k,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to answer question: {exc}")

    return AskResponse(
        answer=result["answer"],
        sql=result["sql"],
        chart=ChartSpec(**result["chart"]),
        data=TabularData(columns=result["columns"], rows=result["rows"]),
    )

