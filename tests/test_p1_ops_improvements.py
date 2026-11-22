from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_defines_requires_python_and_non_package_uv() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data.get("project", {}).get("requires-python") == ">=3.11"
    assert data.get("tool", {}).get("uv", {}).get("package") is False


def test_etf_data_and_earnings_skip_known_etfs_without_network(monkeypatch, tmp_path) -> None:
    from providers import earnings_data as ed
    from providers import etf_data

    for sym in ("GPIQ", "SPMO", "UPRO", "QLD"):
        assert etf_data.is_etf(sym) is True

    monkeypatch.setattr(ed, "_CACHE_DIR", tmp_path)

    hits = {"n": 0}

    class _CountingTicker:
        def __getattr__(self, name: str):
            hits["n"] += 1
            raise RuntimeError(f"ETF should skip Yahoo fundamentals call: .{name}")

    boom = _CountingTicker()
    for sym in ("GPIQ", "SPMO", "UPRO", "QLD"):
        cons = ed.consensus(sym, _t=boom)
        assert cons["eps_fwd_avg"] is None
        assert cons["target_mean"] is None
        qf = ed.quarterly_fundamentals(sym, force=True, _t=boom)
        assert qf == {"ticker": sym, "quarterly": [], "annual": []}
    assert hits["n"] == 0, f"expected 0 Yahoo attribute hits for known ETFs, got {hits['n']}"


def test_earnings_snapshot_universe_excludes_etfs(monkeypatch) -> None:
    import crons.earnings_snapshot as es
    import ml.data_pipeline as dp
    import ml.entry_analyzer as ea

    monkeypatch.setattr(ea, "portfolio_stocks", lambda: ["NVDA", "GPIQ", "SPMO", "UPRO", "QLD", "005930.KS"])
    monkeypatch.setattr(dp, "US_TOP50", ["AAPL", "QQQ"])
    monkeypatch.setattr(dp, "KR_TOP10_META", {"005930.KS": "삼성전자", "069500.KS": "KODEX 200"})

    uni = es._universe()
    assert "NVDA" in uni
    assert "AAPL" in uni
    assert "005930.KS" in uni
    for etf in ("GPIQ", "SPMO", "UPRO", "QLD", "QQQ"):
        assert etf not in uni


def test_wiki_debounced_rebuild_absorbs_lock_timeout(monkeypatch) -> None:
    import safe_io
    from agent_console import wiki

    called = {"n": 0}

    def _boom(*args, **kwargs):
        called["n"] += 1
        raise safe_io.LockTimeout("publication.lock busy")

    monkeypatch.setattr(wiki, "rebuild_artifacts", _boom)
    wiki._REBUILD_RUNNING = False
    wiki._REBUILD_PENDING = False

    # Should NOT raise LockTimeout into daemon thread stderr.
    wiki._run_debounced_rebuild()
    assert called["n"] == 1
    assert wiki._REBUILD_RUNNING is False


def test_kis_stream_idle_flush_throttling() -> None:
    import kis_stream as ks
    from providers import realtime_quotes as rq

    assert 5.0 <= ks.IDLE_FLUSH_SECS < rq.HEARTBEAT_STALE_S / 2
    assert ks.should_flush_quotes(now=101.0, last_flush=100.0, dirty=True, flush_secs=1.0, idle_flush_secs=15.0) is True
    assert ks.should_flush_quotes(now=100.5, last_flush=100.0, dirty=True, flush_secs=1.0, idle_flush_secs=15.0) is False
    assert ks.should_flush_quotes(now=105.0, last_flush=100.0, dirty=False, flush_secs=1.0, idle_flush_secs=15.0) is False
    assert ks.should_flush_quotes(now=115.0, last_flush=100.0, dirty=False, flush_secs=1.0, idle_flush_secs=15.0) is True


def test_ranker_champion_status_and_gate_diagnostics(tmp_path) -> None:
    import pickle
    from ml import ranker

    # 1. Legacy/stale negative-IC champion on disk -> active=False!
    legacy_path = tmp_path / "kr_ranker_model.pkl"
    legacy_champ = SimpleNamespace(
        oos_ic=-0.047,
        oos_top_decile_ret=-0.001,
        train_end_date="2025-07-08",
        meta={},
    )
    legacy_path.write_bytes(pickle.dumps(legacy_champ))

    st = ranker.champion_status(legacy_path)
    assert st["exists"] is True
    assert st["active"] is False
    assert st["oos_ic"] == pytest.approx(-0.047)

    line = ranker.format_adopt_line(adopted=False, champ_status=st)
    assert "활성 챔피언 없음" in line
    assert "챔피언 IC -0.047 유지" not in line

    # 2. Save shadow gate diagnostics JSON cleanly alongside model cache.
    challenger = SimpleNamespace(
        oos_ic=0.0306,
        oos_top_decile_ret=-0.0092,
        train_end_date="2026-10-23",
        meta={
            "trained_at": "2026-11-21T03:33:00+00:00",
            "validation_metrics": {
                "mean_ic": 0.0306,
                "fold_ics": [0.034, 0.049, 0.009],
                "n_folds": 3,
                "purged": True,
                "top_decile_excess": -0.0092,
                "transaction_cost": 0.0011,
                "mdd": 0.358,
                "brier": 0.2492,
                "ece": 0.0314,
                "clean_mean_ic": 0.0306,
                "clean_top_decile_excess": -0.0103,
            },
            "validation_baseline": {
                "feature": "mom_126d",
                "mean_ic": 0.0675,
                "mdd": 0.329,
                "brier": 0.2482,
                "ece": 0.0314,
            },
        },
    )
    diag_path = ranker.save_gate_diagnostics(challenger, legacy_path, adopted=False, champ_status=st)
    assert diag_path.is_file()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert payload["adopted"] is False
    assert payload["champion_active"] is False
    assert payload["gate_passed"] is False
    assert payload["wf_mean_ic"] == pytest.approx(0.0306)
    assert "ic_improvement" in payload["gate_reasons"]


def test_crontab_hygiene_and_wiki_insight_eval_scheduled() -> None:
    from scripts import check_crontab_drift as drift

    text = (REPO_ROOT / "deploy" / "crontab.stock-report").read_text(encoding="utf-8")
    active_lines = [
        line.split("#", 1)[0].strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any("cloudflared_watchdog.sh" in line for line in active_lines)
    assert not any("kiwoom_mock_renewal_reminder.py" in line for line in active_lines)
    assert any("crons/wiki_insight_eval.py" in line for line in active_lines)
    assert any("wiki_insight_eval" in m for m in drift.RELEVANT_MARKERS)
