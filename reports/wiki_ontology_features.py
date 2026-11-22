#!/usr/bin/env python3
"""wiki_ontology_features.py — 온톨로지 기반 Point-in-Time(PIT) 종목별 섀도 ML 피처 추출기.

위키 온톨로지(`ontology.json`)의 관계 중 반드시 `last_seen <= as_of` 인 관계만 필터링해
미래 정보 누수(look-ahead bias) 없는 일별 종목 피처를 산출하고, append-only JSONL 스냅샷으로
축적해 향후 랭커/정책 모델의 구조화 감성·합의도 피처로 활용할 수 있게 한다.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import safe_io

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_PATH = Path(os.path.expanduser("~/reports/ml-data/wiki_ontology_features.jsonl"))
_KR_TICKER_RE = re.compile(r"^\d{6}\.(?:KS|KQ)$")


def _known_listed_tickers() -> set[str]:
    try:
        import ticker_names
        return set(getattr(ticker_names, "_ALL_TICKERS", ()))
    except Exception:
        return set()


def _parse_day(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()[:10]
    if len(text) != 10:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _entity_ticker(entity_id: str, entity: dict[str, Any], known_tickers: set[str]) -> str:
    raw = str(entity.get("ticker") or "").strip().upper()
    if raw:
        return raw
    if str(entity.get("type") or "").strip().lower() == "company" and ":" in entity_id:
        candidate = entity_id.split(":", 1)[1].strip().upper()
        # 비상장 슬러그(company:openai 등) 오염을 막기 위해 상장 티커 사전 또는 KR 코드만 폴백 허용
        if candidate and (_KR_TICKER_RE.match(candidate) or candidate in known_tickers):
            return candidate
    return ""


def _is_superseded_as_of(
    relation: dict[str, Any],
    ref_day: date,
    rel_day_by_id: dict[str, date | None],
) -> bool:
    """as_of 시점 기준으로 이미 대체된 관계인지 판별한다(미래 시점 대체로 인한 역방향 룩어헤드 차단)."""
    sup_by = str(relation.get("superseded_by") or "").strip()
    if sup_by:
        sup_day = rel_day_by_id.get(sup_by)
        if sup_day is not None:
            return sup_day <= ref_day
    return relation.get("state") == "superseded"


def _effective_state(relation: dict[str, Any]) -> str:
    """미래 대체로 되살린 과거 관계의 대체 이전 유효 상태(corroborated/asserted)를 복원한다."""
    state = str(relation.get("state") or "")
    if state == "superseded":
        try:
            return "corroborated" if int(relation.get("publisher_count") or 0) >= 2 else "asserted"
        except (TypeError, ValueError):
            return "asserted"
    return state


def _perspective_sentiment(relations: list[dict[str, Any]], entity_id: str, role: str) -> float:
    pos = 0
    neg = 0
    for rel in relations:
        if rel.get("to") != entity_id:
            continue
        perspectives = rel.get("perspectives") or []
        if role not in perspectives:
            continue
        rtype = rel.get("type")
        if rtype == "positive_for":
            pos += 1
        elif rtype == "negative_for":
            neg += 1
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0


def build_pit_ticker_features(
    ontology: dict[str, Any] | None,
    *,
    as_of: str | date,
) -> dict[str, dict[str, Any]]:
    """`last_seen <= as_of` 인 비대체 관계만으로 종목별 PIT 섀도 피처를 산출한다.

    미래 날짜(`last_seen > as_of`), 관측일 미상 관계, 미래 발견 개체(`first_seen > as_of`),
    그리고 미래 대체(`superseded_by` 의 `last_seen > as_of`)에 의한 역방향 누수를 모두 차단한다.
    """
    ref_day = _parse_day(as_of)
    if ref_day is None or not isinstance(ontology, dict):
        return {}

    entities = ontology.get("entities") or {}
    relations = ontology.get("relations") or []
    if not isinstance(entities, dict) or not isinstance(relations, list):
        return {}

    rel_day_by_id: dict[str, date | None] = {}
    for rel in relations:
        if isinstance(rel, dict) and rel.get("id"):
            rel_day_by_id[str(rel["id"])] = _parse_day(rel.get("last_seen"))

    # 1) 모든 관계를 PIT 기준(last_seen <= ref_day) 및 as_of 시점 비대체 조건으로 선필터링
    pit_relations: list[tuple[dict[str, Any], date]] = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        rel_day = _parse_day(rel.get("last_seen"))
        if rel_day is None or rel_day > ref_day:
            continue
        if _is_superseded_as_of(rel, ref_day, rel_day_by_id):
            continue
        pit_relations.append((rel, rel_day))

    known_tickers = _known_listed_tickers()
    out: dict[str, dict[str, Any]] = {}
    as_of_iso = ref_day.isoformat()
    for eid, ent in entities.items():
        if not isinstance(ent, dict):
            continue
        first_seen = _parse_day(ent.get("first_seen"))
        if first_seen is not None and first_seen > ref_day:
            continue
        ticker = _entity_ticker(str(eid), ent, known_tickers)
        if not ticker:
            continue
        related = [
            (rel, rday) for rel, rday in pit_relations
            if rel.get("from") == eid or rel.get("to") == eid
        ]
        rels_only = [rel for rel, _ in related]
        rel_count = len(related)
        corroborated_pos = sum(
            1 for rel in rels_only
            if rel.get("to") == eid and _effective_state(rel) == "corroborated" and rel.get("type") == "positive_for"
        )
        corroborated_neg = sum(
            1 for rel in rels_only
            if rel.get("to") == eid and _effective_state(rel) == "corroborated" and rel.get("type") == "negative_for"
        )
        contested_count = sum(1 for rel in rels_only if _effective_state(rel) == "contested")
        fresh_7d = sum(1 for _rel, rday in related if 0 <= (ref_day - rday).days <= 7)

        sell_sent = _perspective_sentiment(rels_only, str(eid), "sell_side")
        comm_sent = _perspective_sentiment(rels_only, str(eid), "community")
        corr_total = corroborated_pos + corroborated_neg

        out[ticker] = {
            "as_of": as_of_iso,
            "ticker": ticker,
            "entity_id": str(eid),
            "relation_count": rel_count,
            "corroborated_pos_count": corroborated_pos,
            "corroborated_neg_count": corroborated_neg,
            "net_corroborated_sentiment": round((corroborated_pos - corroborated_neg) / corr_total, 4) if corr_total else 0.0,
            "contested_count": contested_count,
            "contested_ratio": round(contested_count / rel_count, 4) if rel_count else 0.0,
            "sell_side_vs_community_spread": round(sell_sent - comm_sent, 4),
            "fresh_relation_count_7d": fresh_7d,
        }
    return out


def export_daily_snapshot(
    ontology: dict[str, Any] | None,
    *,
    as_of: str | date,
    path: Path = DEFAULT_SNAPSHOT_PATH,
) -> int:
    """종목별 PIT 온톨로지 피처를 append-only JSONL 에 `(as_of, ticker)` 멱등 키로 적재한다."""
    feats = build_pit_ticker_features(ontology, as_of=as_of)
    if not feats:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with safe_io.file_write_lock(str(path), timeout=5.0):
        existing: set[tuple[str, str]] = set()
        if path.exists():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(row, dict):
                        existing.add((str(row.get("as_of") or ""), str(row.get("ticker") or "")))
            except OSError:
                pass
        to_write = [
            row for tk, row in sorted(feats.items())
            if (row["as_of"], tk) not in existing
        ]
        if to_write:
            with path.open("a", encoding="utf-8") as fh:
                for row in to_write:
                    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return len(to_write)


def main() -> int:
    from agent_console import wiki_ontology

    ontology = wiki_ontology.load_ontology()
    today = datetime.now(timezone.utc).date().isoformat()
    written = export_daily_snapshot(ontology, as_of=today)
    logger.info("PIT 온톨로지 섀도 피처 스냅샷 적재: %d종목 (%s)", written, today)
    return 0


if __name__ == "__main__":
    sys.exit(main())
