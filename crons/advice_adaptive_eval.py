#!/usr/bin/env python3
"""
advice_adaptive_eval.py — 포트폴리오 advice(MetaAllocator) 적응 평가 (Phase 4, 최종).

기존 paper_track A/B(meta 배분 vs phase 규칙 배분)의 **실현 성과**로 meta 의 신뢰도를
★목적함수 챔피언/챌린저로 평가한다. meta(챌린저)가 rule(챔피언)을 실현수익에서 이기고
(아웃퍼폼) 하방이 rule 이하(MDD 제약)일 때만, DCA blend 를 meta 쪽으로 올리도록
**shadow 권고**(clamped 0~0.6). 기본 off → 평가·권고만, 라이브 blend 무변경.

meta/rule 은 사전 정의된 두 전략의 실현수익 비교라 학습 파라미터 과적합 위험이 없어
(train/OOS 분할 없이) 누적 성숙 표본 전체로 직접 비교한다.

안전: 순수 평가 + shadow. 라이브 DCA/advice 무변경(_phase_blend_factor 그대로).
크론 (토 04:30 UTC = 13:30 KST):
    30 4 * * 6 cd <repo> && uv run python crons/advice_adaptive_eval.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
TRACK_PATH = Path.home() / ".local" / "share" / "stock-report" / "paper_track.json"
SHADOW_PATH = Path(os.path.expanduser("~/reports/ml-cache/advice_blend_shadow.json"))
MIN_SAMPLES = 15          # paper_track 은 희소 → 보수적 최소 표본
BLEND_MAX = 0.6           # blend 상한(기존 _phase_blend_factor 최대와 일치)


def _horizon_calendar_step(horizon: str) -> int:
    """거래일 horizon 을 공휴일 1~2일 포함 비중첩 달력일 간격으로 환산한다."""
    import math
    import re
    m = re.match(r"^(\d+)", str(horizon or "").strip())
    td = int(m.group(1)) if m else 5
    # 5거래일 -> 8일(MLK/공휴일 낀 1주 overlap 차단), 20거래일 -> 30일
    return max(7, int(math.ceil(td * 7 / 5)) + (1 if td <= 10 else 2))


def _finite_real(val) -> float | None:
    import math
    if val is None or isinstance(val, bool):
        return None
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def _samples(
    track: dict,
    horizon: str = "5d",
    *,
    non_overlapping: bool = False,
) -> list[tuple[float, float]]:
    """성숙 항목 → [(ret_meta, ret_rule)].

    동일 자산군 공정 비교를 위해 `meta_equity` 필드가 존재하는 신형 엔트리에서는
    반드시 주식 전용 슬리브(`ret_meta_equity_{horizon}`)만 사용한다(100% 현금/채권일에
    `ret_meta_{horizon}` 로 폴백해 현금 쿠션 편향이 재유입되는 현상 차단).
    `meta_equity` 자체가 없는 구형 엔트리에서만 `ret_meta_{horizon}` 으로 폴백한다.
    `non_overlapping=True` 이면 ISO 날짜 파싱 실패 행을 제외(fail-closed)하고
    날짜순 정렬 뒤 유효 행이 채택될 때만 horizon 간격 커서를 전진시킨다.
    """
    eq_mk, mk, rk = f"ret_meta_equity_{horizon}", f"ret_meta_{horizon}", f"ret_rule_{horizon}"
    step_days = _horizon_calendar_step(horizon)

    parsed_rows: list[tuple[datetime | None, str, dict]] = []
    for date_str, e in track.items():
        if not isinstance(e, dict):
            continue
        cur_dt: datetime | None = None
        try:
            cur_dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        except ValueError:
            if non_overlapping:
                continue   # fail-closed: 비표준 날짜가 비중첩 게이트를 우회하지 못하게 차단
        parsed_rows.append((cur_dt, str(date_str), e))

    if non_overlapping:
        parsed_rows.sort(key=lambda item: (item[0] or datetime.min, item[1]))
    else:
        parsed_rows.sort(key=lambda item: item[1])

    out: list[tuple[float, float]] = []
    last_dt: datetime | None = None
    for cur_dt, _date_str, e in parsed_rows:
        if "meta_equity" in e:
            m_raw = e.get(eq_mk)
        else:
            m_raw = e.get(eq_mk) if e.get(eq_mk) is not None else e.get(mk)
        r_raw = e.get(rk)
        m_val = _finite_real(m_raw)
        r_val = _finite_real(r_raw)
        if m_val is None or r_val is None:
            continue
        if non_overlapping and cur_dt is not None:
            if last_dt is not None and (cur_dt - last_dt).days < step_days:
                continue
            last_dt = cur_dt
        out.append((m_val, r_val))
    return out


def _downside(vals: list[float]) -> float:
    negs = [v for v in vals if v < 0]
    return abs(sum(negs) / len(negs)) if negs else 0.0


def evaluate(samples: list[tuple[float, float]]) -> dict:
    """meta(챌린저) vs rule(챔피언) ★목적함수 평가."""
    from ml.adaptive import reward as _reward
    n = len(samples)
    if n < MIN_SAMPLES:
        return {"adopt": False, "reason": f"표본 {n}/{MIN_SAMPLES} 미달 — 보류", "n": n}
    meta = [m for m, _ in samples]
    rule = [r for _, r in samples]
    excess = sum(m - r for m, r in samples) / n            # meta 의 rule 대비 평균 우위
    meta_dd, rule_dd = _downside(meta), _downside(rule)
    # 챌린저(meta) excess(=meta-rule 평균)>0 + 하방 ≤ rule + 표본 충분
    adopt = _reward.should_adopt({"excess": excess, "mdd": meta_dd, "n": n},
                                 {"excess": 0.0, "mdd": rule_dd, "n": n},
                                 index_mdd=rule_dd, min_samples=MIN_SAMPLES)
    meta_mean, rule_mean = sum(meta) / n, sum(rule) / n
    return {"adopt": adopt, "n": n, "excess": round(excess, 4),
            "meta_mean": round(meta_mean, 4), "rule_mean": round(rule_mean, 4),
            "meta_dd": round(meta_dd, 4), "rule_dd": round(rule_dd, 4),
            "reason": ("채택 가능 ✅ meta 우위" if adopt else "보류 ⏸️ meta 우위 불충분/하방 초과")}


def _recommended_blend(ev: dict) -> float:
    """meta 우위 크기에 비례한 blend 권고(0.3~0.6 범위, 보수적)."""
    edge = max(0.0, ev.get("excess", 0.0))
    return round(min(BLEND_MAX, 0.3 + edge * 5.0), 3)      # +1%p 우위당 +0.05


def _save_shadow(blend: float, ev: dict) -> None:
    SHADOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SHADOW_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"blend": blend, "_meta": {
        "at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"), **ev}}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    os.replace(tmp, SHADOW_PATH)


from lib.cron_common import send_cron_telegram


def main() -> int:
    logger.info("=== advice_adaptive_eval 시작 [%s] ===", datetime.now(KST).strftime("%Y-%m-%d %H:%M"))
    if not TRACK_PATH.exists():
        logger.info("paper_track 없음 — 보류")
        return 0
    try:
        track = json.loads(TRACK_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("paper_track 로드 실패: %s", e)
        return 0

    ev = evaluate(_samples(track, non_overlapping=True))
    enabled = os.getenv("ADAPTIVE_ADVICE_ENABLED", "false").lower() == "true"
    lines = ["🧭 포트폴리오 advice 적응 평가 (MetaAllocator A/B)", "━━━━━━━━━━━━━━"]
    if ev.get("n", 0) < MIN_SAMPLES:
        lines.append(f"  {ev['reason']}")
    else:
        lines += [
            f"  표본 {ev['n']}  meta {ev['meta_mean']*100:+.2f}% vs rule {ev['rule_mean']*100:+.2f}% (우위 {ev['excess']*100:+.2f}%p)",
            f"  하방 meta {ev['meta_dd']*100:.2f}% vs rule {ev['rule_dd']*100:.2f}%",
            f"  {ev['reason']}",
        ]
        if ev["adopt"]:
            blend = _recommended_blend(ev)
            if enabled:
                _save_shadow(blend, ev)
                lines.append(f"  → blend 권고 {blend:.2f} (shadow 기록)")
            else:
                lines.append(f"  → blend 권고 {blend:.2f} (shadow OFF — 미기록)")
    lines.append("  ⚠️ 평가·권고만 — 라이브 blend 무변경")
    logger.info(" / ".join(lines))
    send_cron_telegram("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
