"""agent_console/wiki_usage_store.py — 경량 SQLite WAL 기반 위키 사용량·피드백 사이드카 저장소.

매 채팅 턴마다 `useCount` 1을 올리기 위해 수십 MB짜리 `events.jsonl` 전체를 lock 걸고
재작성 + 전체 아티팩트(`index.md`/`lint.md`)를 재빌드하던 병목을 제거한다.

저장 위치는 `shared_memory.shared_memory_dir() / "wiki_usage.sqlite"` 이므로 기존 테스트 격리
(`_isolate` -> `AGENT_CONSOLE_SHARED_MEMORY_DIR`)와 완전히 연동된다.
기존 `events.jsonl` 에 남아 있던 레거시 `useCount`/`feedback` 은 읽기 시 합산(overlay)된다.
"""
from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from . import shared_memory

_VALID_RATINGS = frozenset({"helpful", "not_helpful", "neutral"})
_DB_FILENAME = "wiki_usage.sqlite"
_WRITE_LOCK = threading.RLock()
_INITIALIZED_INODES: dict[str, tuple[int, int]] = {}


def db_path() -> Path:
    base = Path(shared_memory.shared_memory_dir())
    base.mkdir(parents=True, exist_ok=True)
    return base / _DB_FILENAME


def _inode_if_nonempty(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    if st.st_size <= 0:
        return None
    return (st.st_dev, st.st_ino)


@contextlib.contextmanager
def _connect(*, write: bool = False):
    path = db_path()
    key = str(path)
    pre_inode = _inode_if_nonempty(path)
    needs_init = pre_inode is None or _INITIALIZED_INODES.get(key) != pre_inode
    conn = sqlite3.connect(key, timeout=15.0, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=15000")
        if write:
            conn.execute("PRAGMA synchronous=NORMAL")
        if needs_init:
            with _WRITE_LOCK:
                cur_inode = _inode_if_nonempty(path)
                if cur_inode is None or _INITIALIZED_INODES.get(key) != cur_inode:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS page_usage (
                            page_id TEXT PRIMARY KEY,
                            use_delta INTEGER NOT NULL DEFAULT 0,
                            last_used_at TEXT NOT NULL DEFAULT '',
                            last_query TEXT NOT NULL DEFAULT '',
                            helpful_delta INTEGER NOT NULL DEFAULT 0,
                            not_helpful_delta INTEGER NOT NULL DEFAULT 0,
                            neutral_delta INTEGER NOT NULL DEFAULT 0
                        )
                        """
                    )
                    post_inode = _inode_if_nonempty(path)
                    if post_inode is not None:
                        _INITIALIZED_INODES[key] = post_inode
        yield conn
    finally:
        conn.close()


def _run_write(fn) -> None:
    """인프로세스 락 + 프로세스 간 SQLITE_BUSY/LOCKED 재시도로 원자적 쓰기를 보장한다."""
    with _WRITE_LOCK:
        for attempt in range(5):
            try:
                with _connect(write=True) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        fn(conn)
                        conn.execute("COMMIT")
                        return
                    except Exception:
                        with contextlib.suppress(sqlite3.Error):
                            conn.execute("ROLLBACK")
                        raise
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if ("locked" in msg or "busy" in msg) and attempt < 4:
                    time.sleep(0.02 * (attempt + 1))
                    continue
                raise


def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    page_id, use_delta, last_used_at, last_query, helpful, not_helpful, neutral = row
    fb: dict[str, int] = {}
    if helpful:
        fb["helpful"] = int(helpful)
    if not_helpful:
        fb["not_helpful"] = int(not_helpful)
    if neutral:
        fb["neutral"] = int(neutral)
    return {
        "page_id": str(page_id),
        "use_delta": int(use_delta or 0),
        "last_used_at": str(last_used_at or ""),
        "last_query": str(last_query or ""),
        "feedback_delta": fb,
    }


def increment_usage(page_id: str, query: str, used_at: str) -> None:
    """원자적으로 page_id의 사용 횟수를 +1 하고 최신 질의·시각을 기록한다."""
    if not page_id:
        return

    def _op(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO page_usage (page_id, use_delta, last_used_at, last_query)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(page_id) DO UPDATE SET
                use_delta = page_usage.use_delta + 1,
                last_used_at = CASE
                    WHEN excluded.last_used_at >= page_usage.last_used_at THEN excluded.last_used_at
                    ELSE page_usage.last_used_at
                END,
                last_query = CASE
                    WHEN excluded.last_used_at >= page_usage.last_used_at THEN excluded.last_query
                    ELSE page_usage.last_query
                END
            """,
            (page_id, used_at, query),
        )

    _run_write(_op)


def increment_feedback(page_id: str, rating: str) -> None:
    """원자적으로 page_id의 피드백 카운트를 +1 한다."""
    if not page_id or rating not in _VALID_RATINGS:
        return
    col = f"{rating}_delta"

    def _op(conn: sqlite3.Connection) -> None:
        conn.execute(
            f"""
            INSERT INTO page_usage (page_id, {col})
            VALUES (?, 1)
            ON CONFLICT(page_id) DO UPDATE SET
                {col} = page_usage.{col} + 1
            """,
            (page_id,),
        )

    _run_write(_op)


def get_usage(page_id: str) -> dict[str, Any] | None:
    if not page_id:
        return None
    path = db_path()
    if not path.exists():
        return None
    try:
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT page_id, use_delta, last_used_at, last_query,
                       helpful_delta, not_helpful_delta, neutral_delta
                FROM page_usage WHERE page_id = ?
                """,
                (page_id,),
            ).fetchone()
        return _row_to_dict(row) if row else None
    except sqlite3.Error:
        return None


def get_all_usage() -> dict[str, dict[str, Any]]:
    path = db_path()
    if not path.exists():
        return {}
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT page_id, use_delta, last_used_at, last_query,
                       helpful_delta, not_helpful_delta, neutral_delta
                FROM page_usage
                """
            ).fetchall()
        return {str(row[0]): _row_to_dict(row) for row in rows}
    except sqlite3.Error:
        return {}


def delete_usage(page_id: str) -> None:
    if not page_id:
        return
    path = db_path()
    if not path.exists():
        return
    with contextlib.suppress(sqlite3.Error):
        _run_write(lambda conn: conn.execute("DELETE FROM page_usage WHERE page_id = ?", (page_id,)))
