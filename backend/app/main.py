import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import get_engine


def _allowed_origins() -> list[str]:
    cors = os.getenv("CORS_ORIGINS")
    if cors:
        return [o.strip() for o in cors.split(",") if o.strip()]
    # Default to allowing local dev ports and Vercel previews
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://*.vercel.app",
    ]


app = FastAPI(title="Excel-to-SQL Q&A API", version="0.1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_origin_regex=os.getenv("CORS_ORIGIN_REGEX", r"^https://.*\\.vercel\\.app$"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routers
from .routers import upload, ask  # noqa: E402

app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(ask.router, prefix="/ask", tags=["ask"])


@app.get("/healthz")
def healthz():
    # Touch engine to ensure DB is reachable
    _ = get_engine()
    return {"status": "ok"}

