"""TDD regression suite for Milestone 1 ML & investment logic P0 fixes."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "crons") not in sys.path:
    sys.path.insert(0, str(ROOT / "crons"))
if str(ROOT / "backtest") not in sys.path:
    sys.path.insert(0, str(ROOT / "backtest"))


# ── 1. Barbell DCA ↔ MetaAllocator & Ranker Gate ──────────────────────────────

def test_ml_dca_blend_normalizes_meta_over_equity_keys_only(monkeypatch):
    """MetaAllocator 전체자산 비중(SGOV/QQQI 포함)이 주식 전용 DCA에 섞여 비하드코딩 종목을 부풀리면 안 된다."""
    import barbell_strategy
    from ml.meta_allocator import MetaAllocation

    alloc = MetaAllocation(
        weights={"SGOV": 0.40, "QQQI": 0.10, "MSFT": 0.25, "NVDA": 0.25},
        signal_summary={},
        regime="neutral",
        confidence=0.8,
        note="",
    )
    monkeypatch.setattr("ml.meta_allocator.get_meta_allocation", lambda mt, pk: alloc)
    monkeypatch.setattr(barbell_strategy, "_phase_blend_factor", lambda mt, pk: 0.5)

    # base_weights는 주식 전용(합 1.0). AMZN은 MetaAllocator에 없는 종목.
    base_weights = {"MSFT": 0.30, "NVDA": 0.30, "AMZN": 0.40}
    blended, _, _ = barbell_strategy._ml_dca_blend(base_weights, "neutral", 0, use_meta=True)

    # MSFT/NVDA는 동일 50:50 → equity 슬리브 0.60 내에서 각 0.30, AMZN은 그대로 0.40 유지
    assert blended["AMZN"] == pytest.approx(0.40, abs=0.01)
    assert blended["MSFT"] == pytest.approx(0.30, abs=0.01)
    assert blended["NVDA"] == pytest.approx(0.30, abs=0.01)
    assert sum(blended.values()) == pytest.approx(1.0, abs=0.01)


def test_ml_dca_blend_blocks_ranker_when_gate_fails(monkeypatch):
    """ranker_gate 미통과(stale/음수 IC) 랭커는 라이브 DCA 비중에 반영되면 안 된다."""
    import barbell_strategy
    from ml.ranker import RankerResult

    class _DummyModel:
        def predict(self, X):
            return np.array([10.0])

    fake = RankerResult(
        model=_DummyModel(),
        feature_names=["f0"],
        train_end_date="2025-01-01",
        oos_ic=-0.02,
        oos_icir=-0.3,
        oos_top_decile_ret=-0.01,
        oos_hit_rate=0.45,
        feature_importance=pd.Series([1.0], index=["f0"]),
    )
    idx = pd.bdate_range("2026-01-01", periods=100)
    px = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": np.linspace(100, 120, 100), "Volume": 1e6}, index=idx)

    monkeypatch.setattr("ml.ranker.load_ranker", lambda *a, **k: fake)
    monkeypatch.setattr("ml.ranker.ranker_gate", lambda r: {"passed": False, "reasons": ["ic_low"]})
    monkeypatch.setattr("ml.data_pipeline.fetch_prices", lambda tickers, days=300: {t: px for t in tickers})
    monkeypatch.setattr("ml.data_pipeline.build_fear_greed_proxy", lambda days=300: pd.Series(50.0, index=idx))
    monkeypatch.setattr("ml.data_pipeline.build_stock_features",
                        lambda t, df, mkt, qqq_close=None, sector_id=0: pd.DataFrame({"f0": 1.0 if t == "NVDA" else -1.0}, index=df.index))

    base = {"MSFT": 0.5, "NVDA": 0.5}
    blended, scores, breadth = barbell_strategy._ml_dca_blend(base, "neutral", 0, use_meta=False)

    assert blended == base
    assert scores == {}
    assert breadth == 0.0


def test_ml_dca_blend_ranker_passes_multitf_rsi_and_sector_id(monkeypatch):
    """Ranker fallback 추론이 학습과 동일하게 index_multitf_rsi 와 sector_id 를 공급해야 한다."""
    import barbell_strategy
    from ml.ranker import RankerResult

    class _CaptureModel:
        def predict(self, X):
            return np.array([float(X["sector_id"].iloc[0]) + float(X["idx_rsi_d"].iloc[0]) * 0.01])

    fake = RankerResult(
        model=_CaptureModel(),
        feature_names=["idx_rsi_d", "sector_id"],
        train_end_date="2026-11-01",
        oos_ic=0.08,
        oos_icir=0.8,
        oos_top_decile_ret=0.03,
        oos_hit_rate=0.60,
        feature_importance=pd.Series([0.5, 0.5], index=["idx_rsi_d", "sector_id"]),
    )
    idx = pd.bdate_range("2026-01-01", periods=100)
    px = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": np.linspace(100, 120, 100), "Volume": 1e6}, index=idx)
    seen_calls = []

    def _fake_build(ticker, df, mkt, qqq_close=None, sector_id=0):
        seen_calls.append((ticker, "idx_rsi_d" in mkt.columns, sector_id))
        return pd.DataFrame({"idx_rsi_d": mkt["idx_rsi_d"], "sector_id": float(sector_id)}, index=df.index)

    monkeypatch.setattr("ml.ranker.load_ranker", lambda *a, **k: fake)
    monkeypatch.setattr("ml.ranker.ranker_gate", lambda r: {"passed": True})
    monkeypatch.setattr("ml.data_pipeline.fetch_prices", lambda tickers, days=300: {t: px for t in tickers})
    monkeypatch.setattr("ml.data_pipeline.build_fear_greed_proxy", lambda days=300: pd.Series(50.0, index=idx))
    monkeypatch.setattr("ml.data_pipeline._get_sector_map", lambda tickers: {"MSFT": 1, "NVDA": 2})
    monkeypatch.setattr("ml.data_pipeline.build_stock_features", _fake_build)

    blended, scores, _ = barbell_strategy._ml_dca_blend({"MSFT": 0.5, "NVDA": 0.5}, "neutral", 0, use_meta=False)

    assert len(seen_calls) == 2
    assert all(has_rsi for _, has_rsi, _ in seen_calls)
    assert dict((t, sid) for t, _, sid in seen_calls) == {"MSFT": 1, "NVDA": 2}
    assert set(scores.keys()) == {"MSFT", "NVDA"}
    assert blended["NVDA"] > blended["MSFT"]


def test_phase_blend_factor_conservative_cap_in_bear_crash(monkeypatch):
    """하락장 Phase 3~5 기본 ML 블렌딩 상한은 미검증 알파 과신 방지를 위해 <= 0.25 이어야 한다."""
    import barbell_strategy

    monkeypatch.delenv("ADAPTIVE_ADVICE_ENABLED", raising=False)
    for phase in (3, 4, 5):
        assert barbell_strategy._phase_blend_factor("bear", phase) <= 0.25


def test_smart_rebalancing_preserves_calculate_dca_weights(monkeypatch):
    """calculate_smart_rebalancing은 calculate_dca의 bear/ML 비중을 버리고 w_normal로 덮어쓰면 안 된다."""
    import barbell_strategy

    monkeypatch.setattr(
        barbell_strategy,
        "calculate_dca",
        lambda *a, **k: {
            "total_krw": 100_000,
            "weights": {"NVDA": 0.70, "MSFT": 0.30},
            "by_ticker": {"NVDA": 70_000, "MSFT": 30_000},
            "multiplier": 1.0,
        },
    )
    monkeypatch.setattr(barbell_strategy, "load_dca_weights", lambda: ({"NVDA": 0.20, "MSFT": 0.80}, {"NVDA": 0.70, "MSFT": 0.30}))
    monkeypatch.setattr(barbell_strategy, "calculate_safety_margin", lambda *a, **k: {"multiplier": 1.0, "score": 70})
    monkeypatch.setattr(barbell_strategy, "calculate_position_analysis", lambda *a, **k: [])
    monkeypatch.setattr(barbell_strategy, "calculate_sgov_target", lambda *a, **k: {})

    res = barbell_strategy.calculate_smart_rebalancing({"total_usd": 10000.0, "sgov_usd": 1000.0}, "bear", 3)
    assert res["adj_weights"]["NVDA"] == pytest.approx(0.70, abs=0.01)
    assert res["adj_weights"]["MSFT"] == pytest.approx(0.30, abs=0.01)


def test_safety_margin_hhi_normalizes_over_equity_sleeve_only(tmp_path, monkeypatch):
    """SGOV 50% + 종목 2개 각 25% 포트폴리오의 주식 HHI는 0.50(집중도 과다)이어야 한다."""
    import barbell_strategy

    monkeypatch.setattr(barbell_strategy, "PORTFOLIO_PATH", str(tmp_path / "missing.json"))
    port = {
        "total_usd": 10000.0,
        "sgov_usd": 5000.0,
        "holdings": {"SGOV": 50, "AAPL": 25, "MSFT": 25},
        "prices": {"SGOV": 100.0, "AAPL": 100.0, "MSFT": 100.0},
    }
    res = barbell_strategy.calculate_safety_margin(port, "neutral", 0)
    assert "집중도 과다" in res["factors"]
    assert "분산 양호" not in res["factors"]
    assert "0.50" in res["factors"]["집중도 과다"]


# ── 2. MetaAllocator Lookahead Purge, Phase Keys & Weights ───────────────────

def test_meta_excess_signal_purges_20d_label_overlap(monkeypatch):
    """_get_excess_signal WF 루프는 shift(-20) 라벨과 테스트 창이 겹치지 않도록 20일 purge 해야 한다."""
    import ml.meta_allocator as ma

    n = 160
    idx = pd.bdate_range("2025-01-01", periods=n)
    feat_df = pd.DataFrame({"row_idx": np.arange(n, dtype=float), "f1": np.sin(np.arange(n))}, index=idx)
    qqq = pd.Series(np.linspace(100, 130, n), index=idx)
    spy = pd.Series(np.linspace(100, 115, n), index=idx)

    fit_max_indices: list[tuple[int, int]] = []

    class _SpyExcess:
        def fit(self, X, y):
            self._max_train = int(np.max(X[:, 0]))
            return self

        def predict(self, X):
            min_test = int(np.min(X[:, 0]))
            fit_max_indices.append((self._max_train, min_test))
            return np.zeros(len(X))

    monkeypatch.setattr(
        "ml.data_pipeline.build_real_sweetspot_data",
        lambda asset="QQQ", days=504: {"features": feat_df, "close": qqq, "spy_close": spy},
    )
    monkeypatch.setattr("ml.models.ExcessReturnModel", _SpyExcess)

    ma._get_excess_signal()
    assert len(fit_max_indices) >= 2
    for max_train, min_test in fit_max_indices:
        # row max_train 의 20일 forward label은 close[max_train + 20]을 사용 → min_test > max_train + 20 (차이 >= 21) 이어야 무누수
        assert min_test - max_train >= 21, f"룩어헤드 누수: max_train={max_train}, min_test={min_test}"


def test_meta_phase_signal_handles_underscore_keys_and_defensive_bull2():
    """bull_1/bull_2 키 매칭 + bull_2(과열)는 방어(음수)여야 한다."""
    from ml.meta_allocator import _get_phase_signal

    assert _get_phase_signal("bull", "bull_1") > 0.0
    assert _get_phase_signal("bull", "bull_2") < 0.0
    assert _get_phase_signal("bull", "bull_2") < _get_phase_signal("bull", "bull_1")


def test_meta_determine_regime_respects_signal_weights():
    """SignalWeights 가중치가 _determine_regime 판정에 실제 반영되어야 한다."""
    from ml.meta_allocator import SignalWeights, _determine_regime

    sw_excess = SignalWeights(excess=0.80, fg_proxy=0.05, phase=0.05, leverage=0.10)
    sw_fg = SignalWeights(excess=0.05, fg_proxy=0.80, phase=0.05, leverage=0.10)

    r1, _ = _determine_regime(excess=1.0, fg=-1.0, phase=0.0, lev_sgov=0.5, sw=sw_excess)
    r2, _ = _determine_regime(excess=1.0, fg=-1.0, phase=0.0, lev_sgov=0.5, sw=sw_fg)
    assert r1 == "risk_on"
    assert r2 == "risk_off"


# ── 3. Entry Analyzer, Leverage Signal, US Policy Bugs ───────────────────────

def test_entry_score_negative_expected_return_has_zero_rr():
    """기대수익이 음수일 때 abs(exp/p25)로 양수 손익비가 되면 안 된다."""
    from ml.entry_analyzer import DEFAULT_SCORE_PARAMS, compute_entry_score

    neg_score, neg_rr = compute_entry_score(0.40, -0.10, -0.05, 45.0, -0.10, params=DEFAULT_SCORE_PARAMS)
    pos_score, pos_rr = compute_entry_score(0.40, +0.05, -0.05, 45.0, -0.10, params=DEFAULT_SCORE_PARAMS)
    assert neg_rr == 0.0
    assert pos_rr == pytest.approx(1.0)
    assert neg_score < pos_score


def test_leverage_signal_negative_expected_return_has_zero_rr():
    """leverage_signal.build_entry_signal 도 음수 기대수익에 양수 손익비를 부여하면 안 된다."""
    from ml.leverage_backtester import InstrumentStats
    from ml.leverage_signal import build_entry_signal

    st = InstrumentStats(
        name="QLD",
        n_entries=10,
        median_ret={21: -0.08, 42: -0.05, 63: -0.04, 126: 0.02},
        p25_ret={21: -0.15, 42: -0.12, 63: -0.10, 126: -0.08},
        p75_ret={21: 0.01, 42: 0.02, 63: 0.04, 126: 0.10},
        hit_rate={21: 0.30, 42: 0.35, 63: 0.40, 126: 0.55},
        max_drawdown=-0.04,
        calmar={21: -2.0, 42: -1.25, 63: -1.0, 126: 0.5},
    )
    sig = build_entry_signal({
        "current_drawdown": -0.12,
        "current_feats": {"vix": 22.0, "rsi": 40.0, "fg_proxy": 35.0},
        "current_stats": {"QLD": st},
        "current_bucket": (-0.15, -0.10),
        "n_similar": 10,
    })
    assert sig.instruments["QLD"].risk_reward_30d == 0.0


def test_leverage_crash_drawdown_bucket_does_not_fallback_to_normal():
    """-55% 극단 폭락은 정상 구간(-0.05, 0.0)이 아니라 최심 낙폭 구간으로 매핑돼야 한다."""
    from ml.leverage_backtester import select_drawdown_bucket

    lo, hi = select_drawdown_bucket(-0.55)
    assert hi <= -0.30 and lo <= -0.55
    assert select_drawdown_bucket(-0.02) == (-0.05, 0.0)
    assert select_drawdown_bucket(0.0) == (-0.05, 0.0)


def test_portfolio_stocks_reads_fractional_holdings_key(tmp_path, monkeypatch):
    """실제 holding_manager 스키마(overseas_fractional.holdings)의 소수점 종목도 읽어야 한다."""
    from ml.entry_analyzer import portfolio_stocks

    snap = {
        "overseas_general": {"holdings_usd": [{"ticker": "MSFT", "shares": 5}]},
        "overseas_fractional": {"holdings": [{"ticker": "AMZN", "shares": 0.4}]},
    }
    (tmp_path / "portfolio_snapshot.json").write_text(json.dumps(snap), encoding="utf-8")
    monkeypatch.setenv("STOCK_REPORT_PROJECT_DIR", str(tmp_path))

    assert portfolio_stocks() == ["MSFT", "AMZN"]


def test_us_policy_penalizes_negative_per_and_clips_pead_asof():
    """적자기업(PER<=0)은 흑자기업보다 낮은 밸류 점수여야 하고, pead_axis는 asof 이후 종가를 쓰면 안 된다."""
    from ml import us_policy as up

    unprofitable = up.extract_features({}, {"per": -20.0, "pbr": 2.0, "roe": -0.10}, {})
    profitable = up.extract_features({}, {"per": 30.0, "pbr": 2.0, "roe": 0.20}, {})
    assert unprofitable["value"] < profitable["value"]

    idx = pd.bdate_range("2026-05-20", periods=15)
    # 실적발표일이 asof와 동일(idx[8]) — 다음날 idx[9] 종가는 미래 데이터!
    closes_past_only = pd.Series([100.0] * 9, index=idx[:9])
    closes_with_future = pd.Series([100.0] * 9 + [130.0] * 6, index=idx)
    ev = [{"date": str(idx[8].date()), "surprise_pct": 5.0}]
    asof = str(idx[8].date())

    v_past = up.pead_axis(ev, closes_past_only, asof=asof)
    v_future = up.pead_axis(ev, closes_with_future, asof=asof)
    assert v_past == v_future


# ── 4. Ranker Sector Skew, Bagging & Metadata Contract ───────────────────────

def test_ranker_passes_sector_id_records_surface_meta_and_enables_bagging(monkeypatch):
    """make_ranker 배깅 활성화 + train_ranker 표면 메타 + rank_today sector_id 전달."""
    from ml import data_pipeline as dp
    from ml import gbdt_adapters as gbdt
    from ml import ranker as rk

    model = gbdt.make_ranker("lightgbm")
    assert model.get_params().get("subsample_freq", 0) >= 1

    # train_ranker surface/horizon/model_version 메타
    dates = pd.bdate_range("2025-01-01", periods=360)
    tickers = [f"T{i:02d}" for i in range(8)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    f0 = np.repeat(np.linspace(-1.0, 1.0, 360), 8) + np.tile(np.linspace(-0.2, 0.2, 8), 360)
    feats = pd.DataFrame({"f0": f0, "f1": np.sin(np.arange(len(index)) / 7.0)}, index=index)
    excess = pd.Series(0.03 * f0, index=index)

    class _FakeReg:
        feature_importances_ = np.array([2.0, 1.0])
        def fit(self, X, y, **kwargs): return self
        def predict(self, X): return np.asarray(X, dtype=float)[:, 0]

    monkeypatch.setattr(rk.gbdt, "make_regressor", lambda **k: _FakeReg())
    res = rk.train_ranker({"features": feats, "excess": excess, "meta": {"forward_days": 20, "mode": "nasdaq100"}}, use_ranker=False)
    assert res.meta.get("surface")
    assert res.meta.get("horizon") == "20d"
    assert res.meta.get("model_version")

    # rank_today sector_id 전달
    seen_sectors = {}
    idx = pd.bdate_range("2026-08-01", periods=70)
    ohlc = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1e5}, index=idx)

    class _SecModel:
        def predict(self, frame):
            return frame["sector_id"].to_numpy(dtype=float)

    sec_result = rk.RankerResult(
        model=_SecModel(), feature_names=["sector_id"], train_end_date="2026-11-01",
        oos_ic=0.05, oos_icir=0.5, oos_top_decile_ret=0.02, oos_hit_rate=0.6,
        feature_importance=pd.Series([1.0], index=["sector_id"]),
    )

    def _build(ticker, df, market_feat, qqq_close=None, sector_id=0):
        seen_sectors[ticker] = sector_id
        return pd.DataFrame({"sector_id": float(sector_id)}, index=df.index)

    monkeypatch.setattr(rk, "load_ranker", lambda *a, **k: sec_result)
    monkeypatch.setattr(rk, "ranker_gate", lambda _: {"passed": True})
    monkeypatch.setattr(rk, "_fundamental_scores", lambda *a, **k: {})
    monkeypatch.setattr(dp, "fetch_universe", lambda mode="nasdaq100": ["AAA", "BBB"])
    monkeypatch.setattr(dp, "fetch_prices", lambda tickers, days=300: {t: ohlc for t in tickers})
    monkeypatch.setattr(dp, "_get_sector_map", lambda tickers: {"AAA": 1, "BBB": 5})
    monkeypatch.setattr(dp, "build_stock_features", _build)
    monkeypatch.setattr(dp, "build_fear_greed_proxy", lambda days=300: pd.Series(50.0, index=idx))
    import ml.technical_rating as tr
    monkeypatch.setattr(tr, "compute_technical_rating", lambda df: None)

    df_rank = rk.rank_today(mode="nasdaq100", top_n=5)
    assert seen_sectors == {"AAA": 1, "BBB": 5}
    assert list(df_rank["ticker"]) == ["BBB", "AAA"]


# ── 5. Evaluation Hygiene: Longterm TWR, Intraday OOS MDD, Entry Calib Purge ─

def test_longterm_scorecard_uses_twr_immune_to_deposits():
    """입금으로 total_usd가 2배가 돼도 실제 TWR이 -5%면 아웃퍼폼으로 오판하면 안 된다."""
    import longterm_adaptive_eval as LT

    dates = pd.bdate_range("2026-01-01", periods=40)
    # 입금으로 total_usd가 1000 -> 2000(+100%), QQQ는 +10%, 실제 TWR은 100 -> 95(-5%)
    records = [
        {"date": d.strftime("%Y-%m-%d"), "total_usd": 1000.0 + i * 25.0, "qqq_price": 100.0 + i * 0.25}
        for i, d in enumerate(dates)
    ]
    account = {"index": dates, "twr": [100.0 - i * (5.0 / 39.0) for i in range(40)]}

    sc = LT.scorecard(records, account=account)
    assert sc is not None
    assert sc["basis"] == "twr"
    assert sc["strat_ret"] == pytest.approx(-5.0, abs=0.2)
    assert sc["excess"] < 0
    assert sc["meets"] is False


def test_intraday_index_mdd_bounds_oos_end_date_and_fails_closed(monkeypatch):
    """_index_mdd는 OOS 마지막 날짜까지만 다운로드하고 실패 시 0.10 대신 None(fail-closed)을 반환해야 한다."""
    from crons import intraday_mock_learn as learn

    seen = {}

    def _fake_download(sym, start=None, end=None, progress=False):
        seen["start"] = start
        seen["end"] = end
        idx = pd.bdate_range("2026-03-01", periods=10)
        return pd.DataFrame({"Close": [100, 102, 95, 96, 98, 100, 101, 102, 103, 104]}, index=idx)

    import yfinance as yf
    monkeypatch.setattr(yf, "download", _fake_download)

    mdd = learn._index_mdd("kr", ["2026-03-01", "2026-03-10"])
    assert seen["start"] == "2026-03-01"
    assert seen["end"] is not None and seen["end"] >= "2026-03-10" and seen["end"] <= "2026-03-12"
    assert mdd == pytest.approx((102 - 95) / 102, abs=1e-4)

    # 에러 시 fail-closed None (임의의 0.10 허용 금지)
    monkeypatch.setattr(yf, "download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("net down")))
    assert learn._index_mdd("kr", ["2026-03-01", "2026-03-10"]) is None


def test_entry_calibration_purged_split_enforces_20d_gap():
    """캘리브레이션/검증 분할은 20거래일 선행수익 겹침을 막기 위해 거래일 기준 >20일 purge 를 강제해야 한다."""
    import entry_calibration as ec

    dates = pd.bdate_range("2024-01-01", periods=200)
    samples = pd.DataFrame({"date": dates, "fwd_20": np.zeros(200)})
    cal, val = ec.purged_cal_val_split(samples, frac=0.6, embargo_days=20)

    assert len(cal) > 0 and len(val) > 0
    trading_gap = len(pd.bdate_range(cal["date"].max(), val["date"].min())) - 1
    assert trading_gap > 20, f"거래일 purge 간격 부족: {trading_gap}거래일"

    # label_end 컬럼이 있으면 cal 마지막 라벨 종점 이후로 val 시작
    samples_with_end = pd.DataFrame({
        "date": dates[:-20],
        "label_end": dates[20:],
        "fwd_20": np.zeros(180),
    })
    cal2, val2 = ec.purged_cal_val_split(samples_with_end, frac=0.6, embargo_days=20)
    assert len(cal2) > 0 and len(val2) > 0
    assert val2["date"].min() > cal2["label_end"].max()


def test_reviewer_edge_cases_m1(tmp_path, monkeypatch):
    """리뷰어 지적 엣지케이스 일괄 회귀: KR benchmark 키, NaN/inf TWR, tz-aware asof, 1행 MDD, 비유한 비중."""
    import barbell_strategy
    import longterm_adaptive_eval as LT
    from crons import intraday_mock_learn as learn
    from ml import entry_analyzer as ea
    from ml import ranker as rk
    from ml import us_policy as up
    from ml.meta_allocator import MetaAllocation

    # 1. ranker benchmark="^KS11" (mode가 kr로 시작하지 않아도 kr_ranker 인식) + 명시 surface 존중
    dates = pd.bdate_range("2025-01-01", periods=360)
    tickers = [f"T{i:02d}" for i in range(8)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    f0 = np.repeat(np.linspace(-1.0, 1.0, 360), 8) + np.tile(np.linspace(-0.2, 0.2, 8), 360)
    feats = pd.DataFrame({"f0": f0, "f1": np.sin(np.arange(len(index)) / 7.0)}, index=index)
    excess = pd.Series(0.03 * f0, index=index)

    class _FakeReg:
        feature_importances_ = np.array([2.0, 1.0])
        def fit(self, X, y, **kwargs): return self
        def predict(self, X): return np.asarray(X, dtype=float)[:, 0]

    monkeypatch.setattr(rk.gbdt, "make_regressor", lambda **k: _FakeReg())
    kr_res = rk.train_ranker({"features": feats, "excess": excess, "meta": {"forward_days": 20, "benchmark": "^KS11", "mode": "custom"}}, use_ranker=False)
    assert kr_res.meta["surface"] == "kr_ranker"
    rep_res = rk.train_ranker({"features": feats, "excess": excess, "meta": {"forward_days": 20, "surface": "investment_report"}}, use_ranker=False)
    assert rep_res.meta["surface"] == "investment_report"

    # 2. longterm_adaptive_eval NaN / inf / 문자열 오염 방어
    bd = pd.bdate_range("2026-01-01", periods=35)
    recs = [{"date": d.strftime("%Y-%m-%d"), "total_usd": 1000.0 + i, "qqq_price": 100.0 + i * 0.1,
             "twr": ("N/A" if i == 0 else float("inf") if i == 1 else float("nan") if i == 2 else 100.0 + i * 0.2)}
            for i, d in enumerate(bd)]
    sc = LT.scorecard(recs)
    assert sc is not None and sc["basis"] == "twr" and np.isfinite(sc["strat_ret"])

    # 3. pead_axis tz-aware asof & 장중 타임스탬프
    idx_tz = pd.date_range("2026-05-20 16:00", periods=15, freq="B", tz="America/New_York")
    closes_tz = pd.Series([100.0] * 9 + [106.0] * 6, index=idx_tz)
    ev = [{"date": "2026-05-28", "surprise_pct": 5.0}]
    v = up.pead_axis(ev, closes_tz, asof=pd.Timestamp("2026-06-05", tz="America/New_York"))
    assert v is not None and v > 0.5

    # 4. intraday _index_mdd 1행 DataFrame 붕괴 없이 fail-closed None
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **k: pd.DataFrame({"Close": [100.0]}, index=pd.bdate_range("2026-03-01", periods=1)))
    assert learn._index_mdd("kr", ["2026-03-01"]) is None

    # 5. entry_analyzer NaN exp_20 -> rr == 0.0
    _, nan_rr = ea.compute_entry_score(0.5, float("nan"), -0.05, 45.0, -0.10, params=ea.DEFAULT_SCORE_PARAMS)
    assert nan_rr == 0.0

    # 6. _ml_dca_blend 비유한/0 meta_weights + SGOV 포함 base_weights + 100% SGOV safety_margin
    alloc_bad = MetaAllocation(weights={"SGOV": 0.5, "MSFT": float("inf"), "NVDA": 0.0}, signal_summary={}, regime="neutral", confidence=0.5, note="")
    monkeypatch.setattr("ml.meta_allocator.get_meta_allocation", lambda mt, pk: alloc_bad)
    blended, _, _ = barbell_strategy._ml_dca_blend({"SGOV": 0.2, "MSFT": 0.4, "NVDA": 0.4}, "neutral", 0, use_meta=True)
    assert all(np.isfinite(w) for w in blended.values()) and sum(blended.values()) == pytest.approx(1.0, abs=0.01)

    monkeypatch.setattr(barbell_strategy, "PORTFOLIO_PATH", str(tmp_path / "none.json"))
    sm = barbell_strategy.calculate_safety_margin({"total_usd": 10000.0, "sgov_usd": 10000.0, "holdings": {"SGOV": 100}, "prices": {"SGOV": 100.0}}, "neutral", 0)
    assert "집중도 과다" not in sm["factors"] and "분산 양호" not in sm["factors"]

