#!/usr/bin/env python3
"""wiki_insight_eval.py — LLM 위키 인사이트(kind=insight) 폐루프 사후 성과 검증기.

온톨로지 기반 인사이트가 제시한 horizon(`days`=5거래일, `weeks`=20거래일, `months`=60거래일)이
성숙한 뒤, 실제 주가 수익률 및 벤치마크(QQQ 등) 대비 초과수익을 point-in-time 으로 측정해
append-only JSONL(`wiki_insight_outcomes.jsonl`)에 영구 기록한다.

사용법::
    uv run python crons/wiki_insight_eval.py
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import safe_io

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HORIZON_DAYS: dict[str, int] = {
    "days": 5,
    "weeks": 20,
    "months": 60,
}
DEFAULT_LEDGER_PATH = Path(os.path.expanduser("~/reports/ml-data/wiki_insight_outcomes.jsonl"))
_MAX_START_GAP_DAYS = 5
_KR_TICKER_RE = re.compile(r"^\d{6}\.(?:KS|KQ)$")

_BULLISH_TOKENS = frozenset({"bullish", "positive", "long", "up", "buy", "상승", "호재", "강세"})
_BEARISH_TOKENS = frozenset({"bearish", "negative", "short", "down", "sell", "하락", "악재", "약세"})


def _known_listed_tickers() -> set[str]:
    try:
        import ticker_names
        return set(getattr(ticker_names, "_ALL_TICKERS", ()))
    except Exception:
        return set()


def _normalize_direction(raw: object) -> str:
    val = str(raw or "").strip().lower()
    if val in _BULLISH_TOKENS:
        return "bullish"
    if val in _BEARISH_TOKENS:
        return "bearish"
    return "neutral"


def _extract_ticker(page: dict, insight: dict) -> str:
    direct = str(insight.get("ticker") or "").strip().upper()
    if direct:
        return direct
    for tag in page.get("tags") or []:
        t = str(tag or "").strip()
        if t.lower().startswith("ticker:"):
            sym = t.split(":", 1)[1].strip().upper()
            if sym:
                return sym
    entity_id = str(insight.get("entity_id") or "").strip()
    if entity_id.lower().startswith("company:"):
        sym = entity_id.split(":", 1)[1].strip().upper()
        # 비상장 슬러그(company:openai 등)가 잘못 티커로 변환되지 않도록 상장 티커 사전/KR 코드만 허용
        if sym and (_KR_TICKER_RE.match(sym) or sym in _known_listed_tickers()):
            return sym
    return ""


def _clean_close_series(series: pd.Series | None) -> pd.Series | None:
    if series is None or not isinstance(series, pd.Series) or series.empty:
        return None
    s = pd.to_numeric(series, errors="coerce")
    vals = s.to_numpy(dtype=float)
    valid_val_mask = np.isfinite(vals) & (vals > 0.0)
    s = s.loc[valid_val_mask]
    if s.empty:
        return None
    try:
        idx = pd.to_datetime(s.index, errors="coerce")
    except ValueError:
        idx = pd.to_datetime(s.index, errors="coerce", utc=True, format="mixed")
    valid_mask = ~idx.isna()
    s = s.loc[valid_mask]
    idx = idx[valid_mask]
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(s.to_numpy(dtype=float), index=idx.normalize())
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s if not s.empty else None


def _common_forward_returns(
    stock_s: pd.Series,
    bench_s: pd.Series | None,
    start_date: str,
    horizon_days: int,
) -> tuple[float, float | None, str, str] | None:
    """start_date 이후 첫 거래일 종가 → horizon_days 거래일 후 종가 수익률 (무룩어헤드)."""
    try:
        start_ts = pd.Timestamp(str(start_date)[:10]).normalize()
    except (TypeError, ValueError):
        return None
    if pd.isna(start_ts):
        return None

    s_clean = _clean_close_series(stock_s)
    if s_clean is None:
        return None
    b_clean = _clean_close_series(bench_s) if bench_s is not None else None

    if b_clean is not None:
        common_idx = s_clean.index.intersection(b_clean.index).sort_values()
        if common_idx.empty:
            return None
        clock = common_idx
    else:
        clock = s_clean.index

    pos = next((i for i, ts in enumerate(clock) if ts >= start_ts), None)
    if pos is None or pos + horizon_days >= len(clock):
        return None
    begin, end = clock[pos], clock[pos + horizon_days]
    if (begin - start_ts).days > _MAX_START_GAP_DAYS:
        return None

    p0, p1 = float(s_clean.loc[begin]), float(s_clean.loc[end])
    if not (math.isfinite(p0) and math.isfinite(p1) and p0 > 0 and p1 > 0):
        return None
    stock_ret = p1 / p0 - 1.0

    bench_ret: float | None = None
    if b_clean is not None:
        b0, b1 = float(b_clean.loc[begin]), float(b_clean.loc[end])
        if not (math.isfinite(b0) and math.isfinite(b1) and b0 > 0 and b1 > 0):
            return None
        bench_ret = b1 / b0 - 1.0

    return stock_ret, bench_ret, begin.date().isoformat(), end.date().isoformat()


def _read_ledger_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _outcome_key(page_id: str, generated_at: str, horizon: str) -> tuple[str, str, str]:
    """인사이트 페이지는 개체별 결정적 ID 로 주기 갱신되므로 `(page_id, generated_at_day, horizon)` 을 멱등 키로 쓴다."""
    return (str(page_id or ""), str(generated_at or "")[:10], str(horizon or ""))


def evaluate_insights(
    pages: list[dict[str, Any]],
    closes: dict[str, pd.Series],
    *,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    benchmark: str = "QQQ",
) -> dict[str, int]:
    """성숙한 위키 인사이트의 실현 수익·초과수익·적중 여부를 append-only 원장에 멱등 기록한다."""
    ledger_path = Path(ledger_path)
    existing_keys = {
        _outcome_key(r.get("page_id", ""), r.get("generated_at", ""), r.get("horizon", ""))
        for r in _read_ledger_rows(ledger_path)
    }
    closes_norm = {
        str(k).strip().upper(): v
        for k, v in (closes or {}).items()
        if str(k).strip()
    }
    default_bench_key = str(benchmark or "QQQ").strip().upper()

    recorded = 0
    pending = 0
    already_recorded = 0
    skipped = 0
    new_rows: list[dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for page in pages or []:
        if not isinstance(page, dict):
            continue
        if page.get("kind") != "insight" or page.get("status") == "archived":
            continue
        page_id = str(page.get("id") or "").strip()
        if not page_id:
            skipped += 1
            continue
        knowledge = page.get("source_knowledge") if isinstance(page.get("source_knowledge"), dict) else {}
        insight = knowledge.get("insight") if isinstance(knowledge.get("insight"), dict) else {}
        horizon = str(insight.get("horizon") or "weeks").strip().lower()
        horizon_days = HORIZON_DAYS.get(horizon)
        if horizon_days is None:
            skipped += 1
            continue

        start_raw = str(insight.get("generated_at") or page.get("created_at") or page.get("createdAt") or "").strip()
        gen_day = start_raw[:10]
        if not gen_day:
            skipped += 1
            continue
        okey = _outcome_key(page_id, gen_day, horizon)
        if okey in existing_keys:
            already_recorded += 1
            continue

        ticker = _extract_ticker(page, insight)
        if not ticker or ticker not in closes_norm:
            skipped += 1
            continue

        # KR 종목(.KS/.KQ)은 ^KS11 이 없을 때 절대 USD/미국 달력 QQQ 로 폴백하지 않는다(통화·휴장일 오염 차단).
        if ticker.endswith((".KS", ".KQ")) and default_bench_key == "QQQ":
            bench_key = "^KS11"
        else:
            bench_key = default_bench_key
        bench_series = closes_norm.get(bench_key)

        fwd = _common_forward_returns(closes_norm[ticker], bench_series, start_raw, horizon_days)
        if fwd is None:
            pending += 1
            continue
        stock_ret, bench_ret, entry_date, exit_date = fwd
        excess_ret = (stock_ret - bench_ret) if bench_ret is not None else None
        direction = _normalize_direction(insight.get("direction"))
        metric = excess_ret if excess_ret is not None else stock_ret
        hit: bool | None = None
        if direction == "bullish":
            hit = bool(metric > 0)
        elif direction == "bearish":
            hit = bool(metric < 0)

        conf_raw = insight.get("confidence", page.get("confidence"))
        confidence: float | None = None
        if conf_raw is not None and not isinstance(conf_raw, bool):
            try:
                cv = float(conf_raw)
                if math.isfinite(cv):
                    confidence = round(cv, 4)
            except (TypeError, ValueError):
                pass

        row = {
            "page_id": page_id,
            "entity_id": str(insight.get("entity_id") or ""),
            "ticker": ticker,
            "horizon": horizon,
            "horizon_days": horizon_days,
            "direction": direction,
            "confidence": confidence,
            "generated_at": gen_day,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "benchmark": bench_key if bench_ret is not None else "",
            "stock_ret": round(stock_ret, 6),
            "bench_ret": round(bench_ret, 6) if bench_ret is not None else None,
            "excess_ret": round(excess_ret, 6) if excess_ret is not None else None,
            "hit": hit,
            "evaluated_at": now_iso,
        }
        new_rows.append(row)
        existing_keys.add(okey)
        recorded += 1

    if new_rows:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with safe_io.file_write_lock(str(ledger_path), timeout=5.0):
            # 락 내부에서 최신 기존 키를 재확인해 동시 프로세스 중복 append 방지
            disk_keys = {
                _outcome_key(r.get("page_id", ""), r.get("generated_at", ""), r.get("horizon", ""))
                for r in _read_ledger_rows(ledger_path)
            }
            to_write = [
                r for r in new_rows
                if _outcome_key(r["page_id"], r["generated_at"], r["horizon"]) not in disk_keys
            ]
            if to_write:
                with ledger_path.open("a", encoding="utf-8") as fh:
                    for row in to_write:
                        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            recorded = len(to_write)

    return {
        "recorded": recorded,
        "pending": pending,
        "already_recorded": already_recorded,
        "skipped": skipped,
    }


def summarize_outcomes(ledger_path: Path = DEFAULT_LEDGER_PATH) -> dict[str, Any]:
    """기록된 인사이트 사후 검증 원장의 집계 스코어카드."""
    rows_raw = _read_ledger_rows(Path(ledger_path))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for r in rows_raw:
        key = _outcome_key(r.get("page_id", ""), r.get("generated_at", ""), r.get("horizon", ""))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    def _agg(subset: list[dict[str, Any]]) -> dict[str, Any]:
        rets = [float(r["stock_ret"]) for r in subset if isinstance(r.get("stock_ret"), (int, float)) and not isinstance(r.get("stock_ret"), bool) and math.isfinite(float(r["stock_ret"]))]
        excs = [float(r["excess_ret"]) for r in subset if isinstance(r.get("excess_ret"), (int, float)) and not isinstance(r.get("excess_ret"), bool) and math.isfinite(float(r["excess_ret"]))]
        hits = [bool(r["hit"]) for r in subset if isinstance(r.get("hit"), bool)]
        return {
            "n": len(subset),
            "directional_n": len(hits),
            "hit_rate": round(sum(hits) / len(hits), 4) if hits else None,
            "mean_ret": round(sum(rets) / len(rets), 6) if rets else None,
            "mean_excess_ret": round(sum(excs) / len(excs), 6) if excs else None,
        }

    overall = _agg(deduped)
    by_horizon = {
        h: _agg([r for r in deduped if r.get("horizon") == h])
        for h in HORIZON_DAYS
        if any(r.get("horizon") == h for r in deduped)
    }
    return {**overall, "by_horizon": by_horizon}


def main() -> int:
    from agent_console import wiki
    from ml.data_pipeline import fetch_prices

    pages = [p for p in wiki._all_wiki_pages() if p.get("kind") == "insight" and p.get("status") != "archived"]
    tickers = {"QQQ"}
    has_kr = False
    for p in pages:
        insight = (p.get("source_knowledge") or {}).get("insight") or {}
        tk = _extract_ticker(p, insight if isinstance(insight, dict) else {})
        if tk:
            tickers.add(tk)
            if tk.endswith((".KS", ".KQ")):
                has_kr = True
    if has_kr:
        tickers.add("^KS11")
    prices = fetch_prices(sorted(tickers), days=200) if len(tickers) > 1 else {}
    closes = {
        t: df["Close"]
        for t, df in (prices or {}).items()
        if df is not None and not getattr(df, "empty", True) and "Close" in df.columns
    }
    stats = evaluate_insights(pages, closes)
    summary = summarize_outcomes()
    logger.info("위키 인사이트 사후검증 완료: %s · 누적 요약=%s", stats, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
