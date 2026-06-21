"""Minimal request logging. Best-effort and off the hot path: if the DB is
unconfigured or down, the API still serves results. Later this table seeds the
merchant memo and the accreting ground-truth dataset.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .config import settings

logger = logging.getLogger("transclassify.db")


class Base(DeclarativeBase):
    pass


class RequestLog(Base):
    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    route: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    transaction_count: Mapped[int] = mapped_column(Integer)
    # Compact snapshot of inputs/outputs for later mining (not PII-scrubbed in v1).
    payload: Mapped[dict] = mapped_column(JSON)


_engine = None


def init_db() -> None:
    """Create the engine + tables if a DATABASE_URL is configured. Never raises."""
    global _engine
    if not settings.database_url:
        logger.info("No DATABASE_URL set — request logging disabled.")
        return
    try:
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
        Base.metadata.create_all(_engine)
        logger.info("Database logging enabled.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("DB init failed, logging disabled: %s", exc)
        _engine = None


def log_request(route: str, model: str, transaction_count: int, payload: dict) -> None:
    """Best-effort insert. Swallows all errors — logging must never break a request."""
    if _engine is None:
        return
    try:
        # Round-trip through JSON so non-serializable bits surface here, not mid-insert.
        safe_payload = json.loads(json.dumps(payload, default=str))
        with Session(_engine) as session:
            session.add(
                RequestLog(
                    created_at=datetime.now(timezone.utc),
                    route=route,
                    model=model,
                    transaction_count=transaction_count,
                    payload=safe_payload,
                )
            )
            session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("log_request failed (ignored): %s", exc)
