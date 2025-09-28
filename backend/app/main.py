import os
from fastapi import FastAPI, File, UploadFile
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
    return {"filename": file.filename}


# Keep existing routers
from .routers import upload, ask  # noqa: E402

app.include_router(upload.router, prefix="/upload", tags=["upload-full-flow"])
app.include_router(ask.router, prefix="/ask", tags=["ask"])

