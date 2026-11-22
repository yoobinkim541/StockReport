"""agent_console/wiki_index.py — SQLite FTS5 기반 하이브리드 1차 위키 검색 인덱스.

QMD CLI가 AND 매칭 특성·롤아웃 비활성(`AGENT_CONSOLE_QMD_ENABLED=0`)으로 0건을 반환할 때
단순 Python 부분문자열 합산 대신 **SQLite FTS5 BM25 + 한국어 복합어 바이그램 + 필드 가중치 +
최신성 감쇠**로 고리콜 1차 후보를 반환한다.

원본 정본은 `events.jsonl`이며 이 파일은 `storage_signature` 변경 시 자동 재구축되는
파생 캐시(`wiki_search.sqlite`)다. 손상되거나 FTS5 미지원 환경이면 조용히 빈 결과를 반환해
호출부의 폴백 경로를 깨뜨리지 않는다.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import re
import sqlite3
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import shared_memory

_DB_FILENAME = "wiki_search.sqlite"
_SCHEMA_VERSION = "2"
_SYNC_LOCK = threading.RLock()
_TOKEN_RE = re.compile(r"[0-9a-zA-Z가-힣]+")
_HANGUL_RUN_RE = re.compile(r"[가-힣]{3,}")
_STOPWORDS = frozenset({
    "그리고", "그러면", "어떻게", "지금", "대한", "관련", "정리", "알려줘",
    "the", "and", "for", "with", "about", "from", "that", "this",
})


def db_path() -> Path:
    base = Path(shared_memory.shared_memory_dir())
    base.mkdir(parents=True, exist_ok=True)
    return base / _DB_FILENAME


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).lower()


def _extract_tokens(text: str, *, include_shingles: bool = True) -> list[str]:
    """원시 텍스트에서 안전한 영숫자/한글 토큰을 먼저 모두 모으고, 필요 시 한국어 복합어 2-gram을 뒤에 덧붙인다."""
    norm = _normalize_text(text)
    raw_tokens = _TOKEN_RE.findall(norm)
    primaries: list[str] = []
    seen: set[str] = set()
    for tok in raw_tokens:
        if len(tok) < 2 or tok in _STOPWORDS:
            continue
        if tok not in seen:
            seen.add(tok)
            primaries.append(tok)
    if not include_shingles:
        return primaries
    shingles: list[str] = []
    for tok in primaries:
        for run in _HANGUL_RUN_RE.findall(tok):
            # 한국어 복합어 및 혼합어(예: '국채금리', 'SK하이닉스') 매칭용 2-gram 보조 토큰
            max_len = min(len(run), 12)
            for i in range(max_len - 1):
                bi = run[i:i + 2]
                if bi not in _STOPWORDS and bi not in seen:
                    seen.add(bi)
                    shingles.append(bi)
    return primaries + shingles


def _indexable_text(text: str, *, max_chars: int = 16000) -> str:
    clipped = _normalize_text(text)[:max_chars]
    tokens = _extract_tokens(clipped, include_shingles=True)
    if not tokens:
        return clipped
    # 원문 정규화 텍스트 + 복합어 분해 토큰을 함께 색인
    return clipped + " " + " ".join(tokens)


def _storage_signature(records: list[dict[str, Any]]) -> str:
    events_path = Path(shared_memory._paths()["events"])
    try:
        st = events_path.stat()
        file_sig = f"{st.st_ino}:{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        file_sig = "missing"
    # 인메모리/모킹된 records나 동일 크기 수정도 정확히 감지하기 위해 레코드 핵심 요약 다이제스트 병기
    h = hashlib.blake2b(digest_size=8)
    for rec in records:
        tags = rec.get("tags") or []
        artifacts = rec.get("artifacts") or []
        line = (
            f"{rec.get('id', '')}|{rec.get('updatedAt') or rec.get('createdAt') or ''}|"
            f"{rec.get('title', '')}|{len(str(rec.get('summary') or ''))}|"
            f"{len(str(rec.get('body') or ''))}|{','.join(str(t) for t in tags)}|"
            f"{len(artifacts)}\n"
        )
        h.update(line.encode("utf-8", errors="ignore"))
    return f"v{_SCHEMA_VERSION}:{file_sig}:{len(records)}:{h.hexdigest()}"


def _unlink_db_files(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        with contextlib.suppress(OSError):
            Path(f"{path}{suffix}").unlink()


@contextlib.contextmanager
def _connect():
    path = db_path()
    conn = sqlite3.connect(str(path), timeout=10.0, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
            page_id UNINDEXED,
            surface UNINDEXED,
            status UNINDEXED,
            updated_at UNINDEXED,
            title,
            summary,
            tags,
            body,
            tokenize='unicode61 remove_diacritics 2'
        )
        """
    )


def _knowledge_text(knowledge: object) -> str:
    if not isinstance(knowledge, dict):
        return ""
    parts: list[str] = []
    for claim in knowledge.get("claims") or []:
        if isinstance(claim, dict) and isinstance(claim.get("text"), str):
            parts.append(claim["text"])
    for item in knowledge.get("limitations") or []:
        if isinstance(item, str):
            parts.append(item)
    return " ".join(parts)


def _build_rows(records: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    from . import wiki

    rows_to_insert: list[tuple[Any, ...]] = []
    for rec in records:
        page_id = str(rec.get("id") or "").strip()
        if not page_id:
            continue
        tags_list = wiki._dedupe_tags(rec.get("tags") or [])
        source_refs = shared_memory.normalize_artifact_refs(rec.get("artifacts") or [])
        status = wiki.normalize_trust_status(wiki._status_from_tags(tags_list), source_refs)
        surface = wiki._surface_from_record(rec)
        updated_at = str(rec.get("updatedAt") or rec.get("createdAt") or "")
        title = str(rec.get("title") or "")
        summary = str(rec.get("summary") or "")
        tags_text = " ".join(tags_list)
        extra_parts = [
            str(rec.get("body") or ""),
            _knowledge_text(rec.get("source_knowledge")),
            " ".join(str(x) for x in (rec.get("decisions") or []) if x),
            " ".join(str(x) for x in (rec.get("openQuestions") or []) if x),
        ]
        body_text = " ".join(p for p in extra_parts if p)
        rows_to_insert.append((
            page_id,
            surface,
            status,
            updated_at,
            _indexable_text(title, max_chars=1000),
            _indexable_text(summary, max_chars=2000),
            _indexable_text(tags_text, max_chars=2000),
            _indexable_text(body_text, max_chars=16000),
        ))
    return rows_to_insert


def _sync_once(records: list[dict[str, Any]], sig: str, *, force: bool) -> bool:
    with _connect() as conn:
        _ensure_schema(conn)
        if not force:
            row = conn.execute("SELECT value FROM index_meta WHERE key = 'signature'").fetchone()
            if row and row[0] == sig:
                return False
        rows_to_insert = _build_rows(records)
        conn.execute("BEGIN IMMEDIATE")
        try:
            if not force:
                row = conn.execute("SELECT value FROM index_meta WHERE key = 'signature'").fetchone()
                if row and row[0] == sig:
                    conn.execute("COMMIT")
                    return False
            conn.execute("DELETE FROM pages_fts")
            conn.executemany(
                """
                INSERT INTO pages_fts (page_id, surface, status, updated_at, title, summary, tags, body)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            conn.execute(
                "INSERT INTO index_meta (key, value) VALUES ('signature', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (sig,),
            )
            conn.execute("COMMIT")
            return True
        except Exception:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise


_CORRUPT_SQLITE_CODES = frozenset({11, 26})  # SQLITE_CORRUPT=11, SQLITE_NOTADB=26


def _is_corrupt_sqlite_error(exc: sqlite3.Error) -> bool:
    """실제 DB 파일 손상(CORRUPT/NOTADB)만 판별한다. lock/busy/readonly 같은 일시 OperationalError 는 절대 False."""
    code = getattr(exc, "sqlite_errorcode", None)
    if code in _CORRUPT_SQLITE_CODES:
        return True
    if isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return any(tok in msg for tok in ("malformed", "not a database", "file is encrypted", "corrupt"))


def sync_index(records: list[dict[str, Any]] | None = None, *, force: bool = False) -> bool:
    """현재 위키 레코드 스냅샷과 FTS5 인덱스를 동기화한다. 재구축 시 True, 최신이면 False."""
    from . import wiki

    if records is None:
        records = wiki._wiki_records()
    sig = _storage_signature(records)
    with _SYNC_LOCK:
        try:
            return _sync_once(records, sig, force=force)
        except sqlite3.Error as exc:
            if not _is_corrupt_sqlite_error(exc):
                # 일시 lock/busy/권한 오류에서는 살아 있는 WAL 파일을 절대 삭제하지 않고 폴백에 위임한다.
                return False
            # 파생 캐시 파일이 실제로 손상된 경우에만 삭제 후 1회 자가 복구한다.
            _unlink_db_files(db_path())
            try:
                return _sync_once(records, sig, force=True)
            except sqlite3.Error:
                return False


def _recency_boost(updated_at: str, now: datetime) -> float:
    if not updated_at:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
        return 1.5 * math.exp(-age_days / 45.0)
    except Exception:
        return 0.0


def search_pages(
    query: str,
    *,
    records: list[dict[str, Any]] | None = None,
    surface: str = "all",
    status: str = "all",
    include_archived: bool = False,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """SQLite FTS5 BM25 + 최신성/상태 가중 하이브리드 검색. 반환: [{'id', 'score', ...}]."""
    raw_query = str(query or "").strip()
    if not raw_query:
        return []
    primary_tokens = _extract_tokens(raw_query, include_shingles=False)
    all_tokens = _extract_tokens(raw_query, include_shingles=True)
    if not all_tokens:
        return []

    # 원시 키워드를 모두 우선 배치하고 남는 슬롯(총 24개 한도)만 한국어 2-gram으로 채워
    # 긴 질의의 뒤쪽 핵심 키워드가 밀려나지 않게 한다.
    budget = max(24, len(primary_tokens))
    query_tokens = all_tokens[:budget]
    # FTS5 구문 인젝션 방지: 모든 토큰은 영숫자·한글만 허용하고 큰따옴표로 안전하게 감싼다.
    match_expr = " OR ".join(f'"{tok}"' for tok in query_tokens)
    sync_index(records)

    surface_norm = str(surface or "all").strip().lower()
    status_norm = str(status or "all").strip().lower()
    fetch_cap = max(20, int(limit or 20) * 4)

    where = ["pages_fts MATCH ?"]
    params: list[Any] = [match_expr]
    if not include_archived:
        where.append("status != 'archived'")
    if surface_norm and surface_norm != "all":
        where.append("surface = ?")
        params.append(surface_norm)
    if status_norm and status_norm != "all":
        where.append("status = ?")
        params.append(status_norm)
    params.append(fetch_cap)

    sql = f"""
        SELECT page_id, surface, status, updated_at, title, summary, tags,
               bm25(pages_fts, 0.0, 0.0, 0.0, 0.0, 10.0, 6.0, 5.0, 1.5) AS rank_score
        FROM pages_fts
        WHERE {' AND '.join(where)}
        ORDER BY rank_score ASC
        LIMIT ?
    """
    try:
        with _connect() as conn:
            _ensure_schema(conn)
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        if not _is_corrupt_sqlite_error(exc):
            return []
        # index_meta 는 살아 있으나 pages_fts B-tree 가 손상된 경우 1회 강제 재구축 후 재시도
        with _SYNC_LOCK:
            _unlink_db_files(db_path())
        if not sync_index(records, force=True):
            return []
        try:
            with _connect() as conn:
                _ensure_schema(conn)
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            return []

    now = datetime.now(timezone.utc)
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for page_id, row_surface, row_status, updated_at, title_idx, summary_idx, tags_idx, bm25_raw in rows:
        # SQLite bm25()는 낮을수록(음수 절대값이 클수록) 더 잘 매칭됨
        bm25_pos = max(0.0, -float(bm25_raw or 0.0))
        header_tokens = set(_TOKEN_RE.findall(f"{title_idx} {summary_idx} {tags_idx}"))
        header_hits = sum(1 for tok in primary_tokens if tok in header_tokens)
        coverage = (header_hits / len(primary_tokens)) if primary_tokens else 0.0
        status_bonus = 0.8 if row_status == "stable" else 0.4 if row_status == "reviewed" else 0.0
        total = bm25_pos + coverage * 6.0 + status_bonus + _recency_boost(updated_at, now)
        scored.append((
            total,
            str(updated_at or ""),
            {"id": str(page_id), "score": round(total, 4), "surface": row_surface, "status": row_status},
        ))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:max(1, int(limit or 20))]]
