"""TDD suite for Milestone 2 ML & investment logic P1 structural upgrades."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "crons") not in sys.path:
    sys.path.insert(0, str(ROOT / "crons"))


# ── 1. Paper Track Equity-Only Sleeve + Advice Adaptive Non-Overlapping Eval ──

def test_paper_track_records_equity_only_meta_sleeve_and_fills_realized(monkeypatch):
    """paper_track은 SGOV/QQQI를 제거한 meta_equity 슬리브를 별도 기록해 rule과 공정 비교해야 한다."""
    from crons import paper_track as pt
    from ml.meta_allocator import MetaAllocation

    alloc = MetaAllocation(
        weights={"SGOV": 0.40, "QQQI": 0.10, "QLD": 0.10, "MSFT": 0.20, "NVDA": 0.20},
        signal_summary={}, regime="neutral", confidence=0.7, note="",
    )
    monkeypatch.setattr(pt, "_current_phase", lambda: ("neutral", 0))
    monkeypatch.setattr("ml.meta_allocator.get_meta_allocation", lambda mt, pk: alloc)
    monkeypatch.setattr("barbell_strategy.load_dca_weights", lambda: ({"MSFT": 0.5, "NVDA": 0.5}, {"MSFT": 0.5, "NVDA": 0.5}))
    monkeypatch.setattr(pt, "_regime_core_weights", lambda: {"QQQ": 1.0})

    track: dict = {}
    pt.record_today(track)
    entry = next(iter(track.values()))

    assert "meta_equity" in entry
    assert set(entry["meta_equity"].keys()) == {"MSFT", "NVDA"}
    assert sum(entry["meta_equity"].values()) == pytest.approx(1.0, abs=1e-3)
    assert entry["meta_equity"]["MSFT"] == pytest.approx(0.5, abs=1e-3)


def test_advice_adaptive_prefers_meta_equity_and_deduplicates_overlapping_windows():
    """advice_adaptive_eval은 ret_meta_equity를 우선 사용하고, 겹치는 일별 5d 창을 비중첩으로 걸러야 한다."""
    import advice_adaptive_eval as A

    # SGOV 포함 ret_meta_5d 는 +2%p 우위지만, 공정 비교 ret_meta_equity_5d 는 -1%p 열위
    dates = pd.bdate_range("2026-01-05", periods=25)
    track = {
        d.strftime("%Y-%m-%d"): {
            "ret_meta_5d": 0.03,
            "ret_meta_equity_5d": 0.00,
            "ret_rule_5d": 0.01,
        }
        for d in dates
    }
    samples = A._samples(track, horizon="5d")
    # ret_meta_equity_5d 우선 사용 확인
    assert all(m == 0.00 and r == 0.01 for m, r in samples)

    # 25거래일 연속 5d 창 → 비중첩(5거래일 간격)으로 줄이면 5개 독립 블록 (15개 미만 → 보류!)
    non_ov = A._samples(track, horizon="5d", non_overlapping=True)
    assert len(non_ov) <= 6
    ev = A.evaluate(non_ov)
    assert ev["adopt"] is False


# ── 2. Multivariate Non-Negative Ridge Axis Weighting ────────────────────────

def test_multivariate_ridge_stops_double_counting_collinear_axes():
    """완전 공선인 모멘텀 축 2개(mom1, mom2)와 직교 축(flow)이 같은 진실 기여도를 가질 때,
    다변량 비음수 Ridge는 공선 축 합이 직교 축을 2배로 압도하지 않도록 분산한다."""
    from ml.adaptive import learner

    rng = np.random.default_rng(42)
    n = 120
    mom = rng.normal(0, 1, n)
    flow = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.2, n)
    # mom과 flow의 진실 기여도는 동일(1.0 vs 1.0). 단 mom1, mom2는 동일 복사본(공선성)
    y = 1.0 * mom + 1.0 * flow + noise
    rows = [
        {"features": {"mom1": float(mom[i]), "mom2": float(mom[i]), "flow": float(flow[i])},
         "fwd_excess": float(y[i])}
        for i in range(n)
    ]

    w = learner.fit_multivariate_axis_weights(rows, ("mom1", "mom2", "flow"), l2=1.0)
    assert set(w.keys()) == {"mom1", "mom2", "flow"}
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= 0.0 for v in w.values())
    # 공선 축 2개 합 ≈ flow (0.5 vs 0.5 근처). 단변량 피어슨이면 (mom1+mom2)≈0.67, flow≈0.33 으로 2배 쏠림
    assert abs((w["mom1"] + w["mom2"]) - w["flow"]) < 0.12


# ── 3. Leverage Optimizer: True Peak Trailing Stop, Purged WF & Save Gate ───

def test_leverage_backtester_trailing_stop_tracks_peak_since_entry_next_bar():
    """진입가 100 → 고점 150 → 132(-12% from peak, +32% from entry) 시
    trailing_stop=-0.10 은 진입 후 고점 기준으로 익일 청산해야 한다."""
    from ml.leverage_optimizer import BacktestEngine

    idx = pd.bdate_range("2024-01-01", periods=260)
    # 215일 낙폭(-12%)으로 진입(216일 체결), 216일부터 QQQ는 MA200 위(105)로 복귀해 exit_ma 미발동
    qqq = np.full(260, 100.0)
    qqq[215] = 88.0   # -12% 낙폭으로 진입 신호 (216일 체결)
    qqq[216:] = 105.0 # MA200(≈100) 위 유지 → exit_ma 청산 불가

    qld = np.full(260, 100.0)
    qld[216] = 100.0
    qld[217] = 125.0
    qld[218] = 150.0  # 진입 후 최고점 150
    qld[219] = 132.0  # 고점 150 대비 -12% (평균단가 100 대비 +32%) -> 트레일링 스탑 트리거!
    qld[220:] = 130.0 # 익일(220) 청산

    sgov = np.full(260, 100.0)
    vix = np.full(260, 20.0)
    prices = {
        "QQQ": pd.Series(qqq, index=idx),
        "QLD": pd.Series(qld, index=idx),
        "SGOV": pd.Series(sgov, index=idx),
        "^VIX": pd.Series(vix, index=idx),
    }
    params = {
        "instrument": "QLD",
        "min_dd": -0.10,
        "max_vix_entry": 40.0,
        "min_rsi_entry": 100.0,
        "lev_weight": 0.5,
        "sgov_floor": 0.2,
        "exit_ma": 200,
        "exit_vix": 80.0,
        "trailing_stop": -0.10,
        "hold_days_max": 120,
        "cost_bps": 0.0,
    }
    res = BacktestEngine(params, prices).run()
    # 트레일링 스톱이 고점 150→132에서 발동해 익일(130) 청산(n_trades == 1)되어야 함
    # (기존 코드는 avg_entry=100 대비 +32%라 미발동 → n_trades == 0)
    assert res.n_trades == 1


def test_leverage_save_result_if_better_blocks_oos_regression(tmp_path, monkeypatch):
    """기존 챔피언보다 WF median Calmar가 낮은 후보는 leverage_best_params.json을 덮어쓰면 안 된다."""
    from ml import leverage_optimizer as lo

    out_path = tmp_path / "leverage_best_params.json"
    monkeypatch.setattr(lo, "RESULTS_PATH", out_path)

    champ = lo.OptimizationResult(
        best_params={"instrument": "QLD"}, best_calmar=1.2, best_sharpe=1.0,
        best_cagr=0.18, best_max_dd=-0.15, wf_results=[],
        wf_mean_calmar=1.1, wf_std_calmar=0.2, wf_median_calmar=1.15,
        all_trials=pd.DataFrame(), optimized_at="2026-11-01",
    )
    weak = lo.OptimizationResult(
        best_params={"instrument": "TQQQ"}, best_calmar=3.0, best_sharpe=2.0,
        best_cagr=0.40, best_max_dd=-0.13, wf_results=[],
        wf_mean_calmar=0.4, wf_std_calmar=0.5, wf_median_calmar=0.45,
        all_trials=pd.DataFrame(), optimized_at="2026-11-15",
    )

    adopted1, _ = lo.save_result_if_better(champ)
    assert adopted1 is True
    adopted2, reason = lo.save_result_if_better(weak)
    assert adopted2 is False
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["best_params"]["instrument"] == "QLD"
    assert saved["wf_median_calmar"] == pytest.approx(1.15)


# ── 4. Earnings Predictor PIT Snapshot Join & Full-Data Refit ────────────────

def test_earnings_pit_snapshot_join_and_full_refit(tmp_path):
    """PIT 스냅샷 조인은 event_date 이전 스냅샷만 허용(미래 차단)하고, train()은 OOS 평가 후 전량 재적합한다."""
    from ml import earnings_move_predictor as emp
    from ml import earnings_predictor as ep

    snaps = [
        {"date": "2026-04-20", "ticker": "NVDA", "revision_momentum": 0.12, "iv_expected_move": 0.08},
        {"date": "2026-05-01", "ticker": "NVDA", "revision_momentum": 0.99, "iv_expected_move": 0.50},  # 당일=누수 차단
        {"date": "2026-05-05", "ticker": "NVDA", "revision_momentum": -0.50, "iv_expected_move": 0.01}, # 미래=차단
    ]
    assert ep.lookup_pit_snapshot(snaps, "NVDA", "2026-05-01", "revision_momentum") == pytest.approx(0.12)
    assert emp.lookup_pit_snapshot(snaps, "NVDA", "2026-05-01", "iv_expected_move") == pytest.approx(0.08)
    # 45일 초과 낡은 스냅샷은 None
    assert ep.lookup_pit_snapshot(snaps, "NVDA", "2026-08-01", "revision_momentum") is None

    rng = np.random.default_rng(7)
    rows, labels, mags = [], [], []
    for _ in range(180):
        pbr = float(rng.random())
        vol = abs(float(rng.normal(0.02, 0.01)))
        rows.append({"features": {
            "prior_n": 8.0, "prior_surprise_mean": pbr * 5 - 2, "prior_surprise_std": 2.0,
            "prior_beat_rate": pbr, "last_surprise": pbr * 5 - 2,
            "mom_20d": 0.02, "vol_20d": vol, "revision_momentum": pbr - 0.5,
            "hist_avg_abs_move": 0.04, "hist_drift_persist": 0.5,
            "beat_prob": pbr, "iv_expected_move": vol * 2.0,
        }})
        labels.append(1 if pbr + rng.normal(0, 0.05) > 0.5 else 0)
        mags.append(0.02 + vol + abs(float(rng.normal(0, 0.005))))

    res_ep = ep.train(rows, labels)
    assert res_ep["model"] is not None
    assert res_ep.get("fit_scope") == "all_labeled"
    assert res_ep["oos_auc"] > 0.65

    res_emp = emp.train(rows, mags, labels)
    assert res_emp["mag_model"] is not None
    assert res_emp.get("fit_scope") == "all_labeled"


# ── 5. Scale-Free Ranker Feature Sanitizer (Split/Price-Level Invariance) ────

def test_sanitize_ranker_features_is_scale_and_split_invariant():
    """가격·거래량을 10배 스케일해도 sanitize_ranker_features 결과는 완전히 동일해야 한다."""
    from ml.features import compute_features, sanitize_ranker_features

    rng = np.random.default_rng(13)
    idx = pd.bdate_range("2024-01-01", periods=280)
    close = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, 280)), index=idx)
    df1 = pd.DataFrame({
        "open": close * 0.998, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.uniform(1e6, 3e6, 280),
    }, index=idx)
    df10 = pd.DataFrame({
        "open": df1["open"] * 10.0, "high": df1["high"] * 10.0, "low": df1["low"] * 10.0,
        "close": df1["close"] * 10.0, "volume": df1["volume"] * 10.0,
    }, index=idx)

    f1 = sanitize_ranker_features(compute_features(df1), close=df1["close"])
    f10 = sanitize_ranker_features(compute_features(df10), close=df10["close"])

    # 원시 가격/거래량 레벨 컬럼이 모두 제거됐는지 확인
    for banned in ("sma_5", "sma_20", "sma_200", "ema_12", "macd", "bb_mid", "atr_14", "vol_sma_20", "obv"):
        assert banned not in f1.columns
    # 스케일 불변 비율 피처 존재 확인
    assert "atr_pct_14" in f1.columns
    assert "macd_hist_pct" in f1.columns

    tail1 = f1.iloc[-20:].dropna()
    tail10 = f10.iloc[-20:].dropna()
    pd.testing.assert_frame_equal(tail1, tail10, atol=1e-6, rtol=1e-5)


# ── 6. Mock Ledger Regime Isolation & Pre-Mutation KR Shadow Logging ─────────

def test_mock_ledger_surface_isolation():
    """스위트스팟/장점 앙상블 주문은 종목정책 원장(us_mock/kr_mock)과 분리된 표면에 기록돼야 한다."""
    from crons import kiwoom_mock_track as kr
    from crons import us_mock_track as us

    assert us.resolve_ledger_surface(sweet_spot_active=True) == "us_sweet_spot"
    assert us.resolve_ledger_surface(sweet_spot_active=False) == "us_mock"

    assert kr.resolve_ledger_surface(advantage_active=True, sweet_spot_active=False) == "kr_advantage"
    assert kr.resolve_ledger_surface(advantage_active=False, sweet_spot_active=True) == "kr_sweet_spot"
    assert kr.resolve_ledger_surface(advantage_active=False, sweet_spot_active=False) == "kr_mock"

    from crons import kr_mock_learn, us_mock_learn
    us_rows = [
        {"ticker": "SGOV", "strategy_regime": "us_sweet_spot"},
        {"ticker": "MSFT", "strategy_regime": "us_mock"},
        {"ticker": "NVDA"},  # 레거시 행(strategy_regime 없음) 호환 유지
    ]
    assert [r["ticker"] for r in us_mock_learn.filter_stock_policy_rows(us_rows)] == ["MSFT", "NVDA"]
    kr_rows = [
        {"ticker": "005930.KS", "strategy_regime": "kr_advantage"},
        {"ticker": "000660.KS", "strategy_regime": "kr_sweet_spot"},
        {"ticker": "035420.KS", "strategy_regime": "kr_mock"},
    ]
    assert [r["ticker"] for r in kr_mock_learn.filter_stock_policy_rows(kr_rows)] == ["035420.KS"]


def test_calculate_rebalancing_never_sells_sgov_in_bull_or_neutral_and_uses_live_price(monkeypatch):
    """calculate_rebalancing은 calculate_sgov_target과 정합되게 Bull/Neutral/Bear0~1에서 SGOV 매도를 권고하지 않고
    하드코딩 $100.67 대신 portfolio['prices']['SGOV'] 라이브 가격을 써야 한다."""
    import barbell_strategy

    monkeypatch.setattr(barbell_strategy, "load_leverage_state", lambda: {})
    monkeypatch.setattr(barbell_strategy, "load_dca_weights", lambda: ({"MSFT": 1.0}, {"MSFT": 1.0}))

    # Neutral에서 SGOV가 목표(8%)보다 많은 20% → 현금 비축 유지(hold), 매도(sell) 금지!
    port_excess = {"total_usd": 10000.0, "sgov_usd": 2000.0, "qqqi_usd": 500.0, "prices": {"SGOV": 100.5}}
    res_excess = barbell_strategy.calculate_rebalancing("neutral", 0, port_excess)
    assert res_excess["sgov_dir"] == "hold"

    # SGOV 부족 시 라이브 가격 $200 기준으로 주수 계산 ($500 / $200 = 2주, $100.67이면 4주)
    port_low = {"total_usd": 10000.0, "sgov_usd": 300.0, "qqqi_usd": 500.0, "prices": {"SGOV": 200.0}}
    res_low = barbell_strategy.calculate_rebalancing("neutral", 0, port_low)
    assert res_low["sgov_dir"] == "buy"
    assert "(2주 SGOV)" in res_low["sgov_action"]


# ── 7. Reviewer-Driven M2 Regression Tests (B1–B6) ───────────────────────────

def test_earnings_pit_snapshot_jsonl_paths_schema_and_resilience(tmp_path, monkeypatch):
    """B1: 실제 크론 경로(ml-cache), expected_move_pct 스키마, 깨진 JSONL 복구, features_now 일관성 검증."""
    from ml import earnings_move_predictor as emp
    from ml import earnings_predictor as ep

    assert ep.SNAP_PATH == Path.home() / "reports" / "ml-cache" / "earnings_snapshots.jsonl"
    assert emp.OPT_SNAP_PATH == Path.home() / "reports" / "ml-cache" / "options_snapshots.jsonl"

    snap_file = tmp_path / "earnings_snapshots.jsonl"
    snap_file.write_text(
        json.dumps({"date": "2026-04-20", "ticker": "NVDA", "revision_momentum": 0.15}) + "\n"
        + "{corrupt truncated json\n"
        + "null\n"
        + json.dumps({"date": "2026-05-01", "ticker": "NVDA", "revision_momentum": 0.99}) + "\n",
        encoding="utf-8",
    )
    loaded = ep._load_pit_snapshots(snap_file)
    assert len(loaded) == 2
    assert ep.lookup_pit_snapshot(loaded, "NVDA", "2026-05-01", "revision_momentum") == pytest.approx(0.15)
    # non-dict row passed directly never crashes
    assert ep.lookup_pit_snapshot([None, "bad", loaded[0]], "NVDA", "2026-05-01", "revision_momentum") == pytest.approx(0.15)

    opt_file = tmp_path / "options_snapshots.jsonl"
    opt_file.write_text(
        json.dumps({"date": "2026-04-28", "ticker": "NVDA", "expected_move_pct": 0.075}) + "\n"
        + json.dumps({"date": "2026-05-01", "ticker": "NVDA", "expected_move_pct": 0.50}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(emp, "OPT_SNAP_PATH", opt_file)
    opt_loaded = ep._load_pit_snapshots(opt_file)
    assert emp._lookup_opt_expected_move(opt_loaded, "NVDA", "2026-05-01") == pytest.approx(0.075)

    # features_now train/serve 정합성: 로컬 PIT 옵션 스냅샷에서 expected_move_pct 자동 주입
    monkeypatch.setattr("lib.price_utils.fetch_closes", lambda tk: None)
    fnow = emp.features_now("NVDA", today="2026-04-29")
    assert fnow["features"]["iv_expected_move"] == pytest.approx(0.075)


def test_kiwoom_mock_track_preserves_recommendation_v2_shadow(tmp_path, monkeypatch):
    """B2: kiwoom_mock_track이 base_signals_for_shadow로 분리한 뒤에도 recommendation_v2_shadow를 주문 원장에 보존한다."""
    import sys
    from crons import kiwoom_mock_track as kr
    from ml.adaptive import ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(kr, "_record_snapshot", lambda bal: None)
    monkeypatch.setattr(kr, "_append_history", lambda row: None)
    monkeypatch.setattr(kr, "_notify", lambda *a, **k: None)
    monkeypatch.setattr(kr, "advantage_book", lambda: {"ok": False, "error": "off"})
    monkeypatch.setenv("MOCK_SWEET_SPOT", "0")
    monkeypatch.setattr(kr.time, "sleep", lambda _: None)

    fake_kiwoom = type("K", (), {
        "is_enabled": staticmethod(lambda: True),
        "get_balance": staticmethod(lambda: {"ok": True, "positions": {}, "cash_krw": 10_000_000, "nav": 10_000_000}),
        "place_order": staticmethod(lambda code, qty, side: {"ok": True, "ord_no": "1", "msg": "ok"}),
    })()
    monkeypatch.setattr(kr, "kiwoom_mock", fake_kiwoom, raising=False)
    monkeypatch.setitem(sys.modules, "kiwoom_mock", fake_kiwoom)
    monkeypatch.setattr(kr, "compute_kr_signals", lambda u: [
        {"code": "005930", "ticker": "005930.KS", "name": "삼성전자", "price": 70000.0,
         "score": 82.0, "base_score": 82.0, "policy_score": 82.0, "selection_score": 82.0,
         "features": {"mom": 0.8}, "data_quality": "fresh"},
    ])
    monkeypatch.setattr(kr, "plan_rebalance", lambda *a, **k: [
        {"side": "BUY", "code": "005930", "qty": 10, "reason": "test"},
    ])
    monkeypatch.setattr(kr, "TRANCHES", 1)
    monkeypatch.setattr(kr.trade_events, "record_trade", lambda **k: None)
    monkeypatch.setattr(kr.llm_exec, "build_order_review_payload", lambda **k: {})
    monkeypatch.setattr(kr.llm_exec, "run_order_review", lambda p: ([], "skip"))
    monkeypatch.setattr(kr.llm_exec, "log_shadow_reviews", lambda *a, **k: 0)
    monkeypatch.setattr(kr.llm_exec, "mode", lambda: "off")
    monkeypatch.setattr(kr.llm_exec, "apply_reviews", lambda plan, rev, **k: (plan, []))
    monkeypatch.setattr(kr.llm_exec, "review_execution_record", lambda **k: None)
    monkeypatch.setattr(kr.llm_exec, "fallback_note", lambda r: None)

    monkeypatch.setattr(sys, "argv", ["kiwoom_mock_track.py"])
    rc = kr.main()
    assert rc == 0

    dec_path = ledger_mod.Ledger("kr_mock").decisions_path
    rows = [json.loads(line) for line in dec_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["strategy_regime"] == "kr_mock"
    assert rows[0]["selection_rank"] == 1
    assert rows[0]["data_quality"] != "degraded:missing_snapshot"


def test_advice_adaptive_and_paper_track_edge_cases(monkeypatch):
    """B3: 깨진 행이 last_dt를 망치지 않고, 비ISO 차단, 100% 비주식일 폴백 차단, 빈 regime_core 스턱 방지."""
    import advice_adaptive_eval as A
    from crons import paper_track as pt

    # 1) 깨진 행(2026-01-05) 때문에 유효 행(2026-01-06)이 밀리면 안 됨 + NaN 거부 + 비ISO 차단
    track = {
        "2026-01-05": {"meta_equity": {"MSFT": 1.0}, "ret_meta_equity_5d": "corrupt", "ret_rule_5d": 0.01},
        "2026-01-06": {"meta_equity": {"MSFT": 1.0}, "ret_meta_equity_5d": 0.02, "ret_rule_5d": 0.01},
        "2026-01-07": {"meta_equity": {"MSFT": 1.0}, "ret_meta_equity_5d": float("nan"), "ret_rule_5d": 0.01},
        "2026/01/20": {"meta_equity": {"MSFT": 1.0}, "ret_meta_equity_5d": 0.05, "ret_rule_5d": 0.01},
        # 100% SGOV 일(meta_equity == {}): ret_meta_5d 로 폴백 금지!
        "2026-01-25": {"meta_equity": {}, "ret_meta_5d": 0.05, "ret_rule_5d": -0.02},
        # 레거시 엔트리(meta_equity 키 자체가 없음): ret_meta_5d 폴백 허용
        "2026-02-10": {"ret_meta_5d": 0.03, "ret_rule_5d": 0.01},
    }
    res = A._samples(track, horizon="5d", non_overlapping=True)
    assert res == [(0.02, 0.01), (0.03, 0.01)]

    # 2) paper_track: CASH/NaN 제거 + 빈 regime_core 가 pending 에 영원히 남지 않음
    assert pt._equity_only_weights({"SGOV": 0.3, "CASH": 0.2, "cash": 0.1, "MSFT": 0.4, "BAD": float("nan")}) == {"MSFT": 1.0}
    fetch_calls = []
    monkeypatch.setattr("ml.data_pipeline.fetch_prices", lambda *a, **k: fetch_calls.append(1) or {})
    pt_track = {
        "2026-01-05": {
            "meta": {"MSFT": 1.0}, "rule": {"MSFT": 1.0},
            "meta_equity": {}, "regime_core": {},
            "ret_meta_5d": 0.01, "ret_meta_20d": 0.02,
        }
    }
    assert pt.fill_realized(pt_track) == 0
    assert fetch_calls == []


def test_calculate_rebalancing_malformed_prices_safe(monkeypatch):
    """B4: prices=None / 'N/A' / inf / bool 에서도 크래시나 0주·100배 주수 없이 SGOV_FALLBACK_PRICE로 폴백한다."""
    import barbell_strategy

    monkeypatch.setattr(barbell_strategy, "load_leverage_state", lambda: {})
    monkeypatch.setattr(barbell_strategy, "load_dca_weights", lambda: ({"MSFT": 1.0}, {"MSFT": 1.0}))
    expected_shares = int(500.0 / barbell_strategy.SGOV_FALLBACK_PRICE)

    for bad_prices in (None, {"SGOV": "N/A"}, {"SGOV": float("inf")}, {"SGOV": True}, {"SGOV": -5.0}):
        port = {"total_usd": 10000.0, "sgov_usd": 300.0, "qqqi_usd": 500.0, "prices": bad_prices}
        res = barbell_strategy.calculate_rebalancing("neutral", 0, port)
        assert res["sgov_dir"] == "buy"
        assert f"({expected_shares}주 SGOV)" in res["sgov_action"]


def test_leverage_no_same_bar_reentry_and_save_gate_hardening(tmp_path, monkeypatch):
    """B5: 트레일링 스톱 청산 당일 같은 봉 재진입 금지 + save_result_if_better 타입/0-fold 방어."""
    from ml import leverage_optimizer as lo

    idx = pd.bdate_range("2024-01-01", periods=260)
    # QQQ가 계속 -15% 낙폭 유지 → enter_signal이 청산 당일에도 True인 극한 시나리오
    qqq = np.full(260, 100.0)
    qqq[215:] = 85.0
    qld = np.full(260, 100.0)
    qld[216] = 100.0
    qld[217] = 150.0  # peak
    qld[218] = 125.0  # -16.7% from peak → 219일 청산 트리거
    qld[219] = 125.0  # 청산 당일 재진입하면 220일 폭락(100)을 그대로 맞음
    qld[220:] = 100.0
    prices = {
        "QQQ": pd.Series(qqq, index=idx),
        "QLD": pd.Series(qld, index=idx),
        "SGOV": pd.Series(np.full(260, 100.0), index=idx),
        "^VIX": pd.Series(np.full(260, 20.0), index=idx),
    }
    params = {
        "instrument": "QLD", "min_dd": -0.10, "max_vix_entry": 40.0, "min_rsi_entry": 100.0,
        "lev_weight": 0.5, "sgov_floor": 0.2, "exit_ma": 200, "exit_vix": 80.0,
        "trailing_stop": -0.10, "hold_days_max": 120, "cost_bps": 0.0,
    }
    res = lo.BacktestEngine(params, prices).run()
    # 청산일(219)에 재진입하지 않고 220일 이후 다시 진입하므로 219→220일(-20%) 낙폭을 피해야 함
    # 진입(216:100)→청산(219:125, +25% on 50% weight → 포트 ~1.125) 후 220일 폭락(100) 무피해
    assert res.n_trades >= 1
    assert res.equity.iloc[-1] > 1.10

    # save_result_if_better 타입 방어 및 0-fold 챌린저 차단
    out_path = tmp_path / "leverage_best_params.json"
    monkeypatch.setattr(lo, "RESULTS_PATH", out_path)
    out_path.write_text("[]", encoding="utf-8")  # non-dict JSON
    fold = lo.BacktestResult(params={"instrument": "QLD"}, cagr=-0.02, sharpe=-0.1,
                             calmar=-0.2, max_dd=-0.10, n_trades=2, win_rate=0.5, period="2024")
    neg_champ = lo.OptimizationResult(
        best_params={"instrument": "QLD"}, best_calmar=0.5, best_sharpe=0.5,
        best_cagr=0.05, best_max_dd=-0.1, wf_results=[fold, fold],
        wf_mean_calmar=-0.2, wf_std_calmar=0.0, wf_median_calmar=-0.2,
        all_trials=pd.DataFrame(), optimized_at="2026-11-01",
    )
    ok_champ, _ = lo.save_result_if_better(neg_champ)
    assert ok_champ is True

    # 0-fold default 0.0 후보는 실제 검증된 기존 챔피언(n_wf_folds=2, calmar=-0.2)을 덮어쓰면 안 됨
    empty_fold_cand = lo.OptimizationResult(
        best_params={"instrument": "TQQQ"}, best_calmar=5.0, best_sharpe=2.0,
        best_cagr=0.5, best_max_dd=-0.1, wf_results=[],
        wf_mean_calmar=0.0, wf_std_calmar=0.0, wf_median_calmar=0.0,
        all_trials=pd.DataFrame(), optimized_at="2026-11-15",
    )
    ok_empty, reason = lo.save_result_if_better(empty_fold_cand)
    assert ok_empty is False
    assert "no_wf_folds" in reason

    # 문자열 wf_median_calmar 가 들어있는 손상 챔피언에서도 TypeError 없이 동작
    out_path.write_text(json.dumps({"wf_median_calmar": "1.5", "n_wf_folds": 2}), encoding="utf-8")
    ok_str, _ = lo.save_result_if_better(neg_champ)
    assert ok_str is False


def test_sanitize_ranker_features_preserves_ema_bull_short_no_dups_and_pipeline_wired():
    """B6: ema_bull_short 유지, close_vs_sma 중복 제거, build_stock_features scale_free 배선 및 learner 수치 안정성."""
    from ml.adaptive import learner
    from ml.data_pipeline import build_stock_features
    from ml.features import compute_features, sanitize_ranker_features

    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2024-01-01", periods=260)
    close = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, 260)), index=idx)
    df = pd.DataFrame({
        "open": close * 0.998, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.uniform(1e6, 3e6, 260),
    }, index=idx)
    sanitized = sanitize_ranker_features(compute_features(df), close=df["close"])
    assert "ema_bull_short" in sanitized.columns
    assert "close_vs_sma20" in sanitized.columns
    assert "close_vs_sma_20" not in sanitized.columns
    assert "close_vs_sma50" in sanitized.columns
    assert "close_vs_sma_50" not in sanitized.columns
    assert "close_vs_sma_200" in sanitized.columns

    ohlcv = pd.DataFrame({
        "Open": df["open"], "High": df["high"], "Low": df["low"],
        "Close": df["close"], "Volume": df["volume"],
    }, index=idx)
    mkt = pd.DataFrame({"vix": 20.0}, index=idx)
    sf = build_stock_features("TEST", ohlcv, mkt, scale_free=True)
    leg = build_stock_features("TEST", ohlcv, mkt, scale_free=False)
    assert "sma_20" not in sf.columns and "obv" not in sf.columns and "atr_pct_14" in sf.columns
    assert "ema_bull_short" in sf.columns
    assert "sma_20" in leg.columns and "obv" in leg.columns

    # learner 수치 안정성: 중복 축명, huge float(1e200), inf l2
    huge_rows = [
        {"features": {"a": float((i + 1) * 1e200), "b": float(i % 3)}, "fwd_excess": float(i * 0.01)}
        for i in range(20)
    ]
    w_huge = learner.fit_multivariate_axis_weights(huge_rows, ["a", "a", "b"], l2=float("inf"))
    assert set(w_huge.keys()) == {"a", "b"}
    assert sum(w_huge.values()) == pytest.approx(1.0, abs=1e-5)


def test_barbell_ml_dca_blend_supports_both_legacy_and_scale_free_rankers_on_300d_fetch(monkeypatch):
    """_ml_dca_blend는 300일 페치(idx_rsi_m=NaN 포함)에서 레거시·scale-free 랭커 모두 정상 추론해야 한다."""
    import barbell_strategy
    from ml import data_pipeline as dp
    from ml import ranker as rk

    rng = np.random.default_rng(77)
    idx = pd.bdate_range("2025-01-02", periods=280)

    def _mk_ohlcv(scale: float) -> pd.DataFrame:
        c = pd.Series(scale * np.cumprod(1.0 + rng.normal(0.0006, 0.015, len(idx))), index=idx)
        return pd.DataFrame({
            "Open": c * 0.998, "High": c * 1.01, "Low": c * 0.99,
            "Close": c, "Volume": rng.uniform(1e6, 2e6, len(idx)),
        }, index=idx)

    prices = {
        "MSFT": _mk_ohlcv(400.0),
        "NVDA": _mk_ohlcv(120.0),
        "QQQ": _mk_ohlcv(500.0),
        "^VIX": _mk_ohlcv(18.0),
    }
    monkeypatch.setattr(dp, "fetch_prices", lambda tickers, days=300: {t: prices[t] for t in tickers if t in prices})
    monkeypatch.setattr(dp, "build_fear_greed_proxy", lambda days=300: pd.Series(55.0, index=idx))
    monkeypatch.setattr(dp, "_get_sector_map", lambda tickers: {t: 1 for t in tickers})
    monkeypatch.setattr(rk, "ranker_gate", lambda res: {"passed": True, "reason": "ok"})

    class _FakeModel:
        def predict(self, X):
            return np.array([float(X.iloc[0].sum())])

    for feat_names in (
        ["sma_20", "atr_14", "obv", "rsi_14", "mom_21d"],            # 레거시 캐시 모델
        ["atr_pct_14", "ema_bull_short", "rsi_14", "mom_21d"],       # 신형 scale-free 모델
    ):
        fake_res = type("R", (), {"feature_names": feat_names, "model": _FakeModel()})()
        monkeypatch.setattr(rk, "load_ranker", lambda r=fake_res: r)
        blended, scores, _breadth = barbell_strategy._ml_dca_blend(
            {"MSFT": 0.5, "NVDA": 0.5}, use_meta=False,
        )
        assert set(scores.keys()) == {"MSFT", "NVDA"}, f"failed for {feat_names}"
        assert sum(blended.values()) == pytest.approx(1.0, abs=1e-4)



