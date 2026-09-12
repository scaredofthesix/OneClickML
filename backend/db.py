from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./oneclickml.db")
SEED_DIR = Path(os.getenv("SEED_DIR", "../database/seed"))

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True)
    title_en: Mapped[str] = mapped_column(String(255))
    title_ru: Mapped[str] = mapped_column(String(255))
    target: Mapped[str] = mapped_column(String(255))
    task: Mapped[str] = mapped_column(String(32))
    source_name: Mapped[str] = mapped_column(String(255), default="")
    source_url: Mapped[str] = mapped_column(String(500), default="")
    why_en: Mapped[str] = mapped_column(Text)
    why_ru: Mapped[str] = mapped_column(Text)
    rows: Mapped[int] = mapped_column(Integer)
    cols: Mapped[int] = mapped_column(Integer)
    csv: Mapped[str] = mapped_column(Text)

    def as_dict(self, with_csv: bool = False) -> dict:
        data = {
            "id": self.id,
            "slug": self.slug,
            "title": {"en": self.title_en, "ru": self.title_ru},
            "why": {"en": self.why_en, "ru": self.why_ru},
            "target": self.target,
            "task": self.task,
            "source": {"name": self.source_name, "url": self.source_url},
            "rows": self.rows,
            "cols": self.cols,
        }
        if with_csv:
            data["csv"] = self.csv
        return data


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    filename: Mapped[str] = mapped_column(String(255))
    target: Mapped[str] = mapped_column(String(255))
    task: Mapped[str] = mapped_column(String(32))
    best_feature: Mapped[str] = mapped_column(String(255))
    best_score: Mapped[float] = mapped_column(Float)
    score_metric: Mapped[str] = mapped_column(String(64))
    rows: Mapped[int] = mapped_column(Integer)
    cols: Mapped[int] = mapped_column(Integer)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "filename": self.filename,
            "target": self.target,
            "task": self.task,
            "best_feature": self.best_feature,
            "best_score": round(self.best_score, 4),
            "score_metric": self.score_metric,
            "rows": self.rows,
            "cols": self.cols,
        }


def init_db(retries: int = 10, delay: float = 2.0) -> None:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            Base.metadata.create_all(engine)
            ensure_columns()
            seed_datasets()
            return
        except OperationalError as exc:
            last_error = exc
            time.sleep(delay)
    raise RuntimeError(f"database unavailable after {retries} attempts: {last_error}")


def ensure_columns() -> None:
    """Дописывает колонки, появившиеся после того, как таблицу уже создали.

    Полноценные миграции проекту пока избыточны, а база у разработчика и в CI
    живёт между запусками: без этого шага старая таблица datasets осталась бы
    без источника датасета.
    """
    added = {
        "source_name": "VARCHAR(255) DEFAULT '' NOT NULL",
        "source_url": "VARCHAR(500) DEFAULT '' NOT NULL",
    }
    existing = {column["name"] for column in inspect(engine).get_columns(Dataset.__tablename__)}
    with engine.begin() as connection:
        for name, definition in added.items():
            if name not in existing:
                statement = f"ALTER TABLE {Dataset.__tablename__} ADD COLUMN {name} {definition}"
                connection.execute(text(statement))


def seed_datasets() -> int:
    catalog_path = SEED_DIR / "catalog.json"
    if not catalog_path.exists():
        return 0

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    touched = 0
    with Session(engine) as session:
        existing = {row.slug: row for row in session.scalars(select(Dataset)).all()}
        for item in catalog:
            csv_path = SEED_DIR / item["file"]
            if not csv_path.exists():
                continue
            content = csv_path.read_text(encoding="utf-8")
            lines = [line for line in content.splitlines() if line.strip()]
            source = item.get("source", {})
            fields = {
                "title_en": item["title"]["en"],
                "title_ru": item["title"]["ru"],
                "why_en": item["why"]["en"],
                "why_ru": item["why"]["ru"],
                "target": item["target"],
                "task": item["task"],
                "source_name": source.get("name", ""),
                "source_url": source.get("url", ""),
                "rows": max(len(lines) - 1, 0),
                "cols": len(lines[0].split(",")) if lines else 0,
                "csv": content,
            }
            row = existing.get(item["slug"])
            if row is None:
                session.add(Dataset(slug=item["slug"], **fields))
            else:
                for key, value in fields.items():
                    setattr(row, key, value)
            touched += 1
        session.commit()
    return touched


def list_datasets() -> list[dict]:
    with Session(engine) as session:
        rows = session.scalars(select(Dataset).order_by(Dataset.id)).all()
        return [row.as_dict() for row in rows]


def get_dataset(slug: str) -> dict | None:
    with Session(engine) as session:
        row = session.scalar(select(Dataset).where(Dataset.slug == slug))
        return row.as_dict(with_csv=True) if row else None


def count_datasets() -> int:
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(Dataset)) or 0


def save_run(**fields) -> dict:
    with Session(engine) as session:
        run = Run(**fields)
        session.add(run)
        session.commit()
        session.refresh(run)
        return run.as_dict()


def list_runs(limit: int = 20) -> list[dict]:
    with Session(engine) as session:
        rows = session.scalars(select(Run).order_by(Run.id.desc()).limit(limit)).all()
        return [row.as_dict() for row in rows]


def count_runs() -> int:
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(Run)) or 0


def clear_runs() -> int:
    with Session(engine) as session:
        deleted = session.execute(delete(Run)).rowcount
        session.commit()
        return deleted
