import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


def _build_database_url() -> str:
    # Prefer DATABASE_URL; fallback to ephemeral SQLite for serverless
    default_sqlite_path = "/tmp/app.db"
    return os.getenv("DATABASE_URL", f"sqlite:///{default_sqlite_path}")


DATABASE_URL = _build_database_url()


def _sqlite_connect_args(url: str) -> dict:
    if url.startswith("sqlite"):  # allow SQLite usage in multi-threaded contexts (FastAPI)
        return {"check_same_thread": False}
    return {}


engine: Engine = create_engine(
    DATABASE_URL,
    connect_args=_sqlite_connect_args(DATABASE_URL),
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_engine() -> Engine:
    return engine


def get_session() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

