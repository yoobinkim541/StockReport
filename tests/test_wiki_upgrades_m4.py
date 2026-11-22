"""TDD suite for Milestone 4 LLM Wiki upgrades:
1. Legal direct Gemini API provider (GEMINI_API_KEY, x-goog-api-key header, JSON mode, independent_providers)
2. Quota-aware maintenance LLM runner + dedup suppression memory + entity-shared pair discovery
3. Closed-loop wiki insight outcome scorecard (crons/wiki_insight_eval.py)
4. Point-in-time ontology shadow ML features (reports/wiki_ontology_features.py)
5. Dashboard browser pagination + retrieval telemetry model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "crons") not in sys.path:
    sys.path.insert(0, str(ROOT / "crons"))


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT_CONSOLE_DB", str(tmp_path / "agent_console.sqlite3"))
    monkeypatch.setenv("AGENT_CONSOLE_SHARED_MEMORY_DIR", str(tmp_path / "data" / "shared-memory"))
    monkeypatch.setenv("AGENT_CONSOLE_QMD_ENABLED", "0")


class _FakeResp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


# ── 1. Direct Legal Gemini API Provider ──────────────────────────────────────

def test_gemini_direct_chat_uses_header_auth_json_mode_and_rejects_truncation(monkeypatch):
    from lib import llm_cli, llm_direct as D

    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key")
    calls = []

    def ok_post(url, json=None, timeout=None, headers=None):
        calls.append((url, json, timeout, headers))
        return _FakeResp({
            "candidates": [{
                "content": {"parts": [{"text": '{"ok": true, "score": 9}'}]},
                "finishReason": "STOP",
            }]
        })

    out = D.gemini_chat("프롬프트", "gemini-2.5-flash", timeout=25, post=ok_post, structured_output=True)
    assert out == '{"ok": true, "score": 9}'
    url, body, timeout, headers = calls[0]
    # API 키는 URL 쿼리스트링이 아닌 헤더로만 전송 (로그 유출 방지)
    assert "secret-gemini-key" not in url
    assert headers["x-goog-api-key"] == "secret-gemini-key"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert timeout == 25

    # MAX_TOKENS 절단 응답은 structured_output=True 에서 거부
    def trunc_post(*a, **k):
        return _FakeResp({
            "candidates": [{
                "content": {"parts": [{"text": '{"ok": true}'}]},
                "finishReason": "MAX_TOKENS",
            }]
        })

    with pytest.raises(ValueError, match="structured_response_invalid"):
        D.gemini_chat("프롬프트", "gemini-2.5-flash", post=trunc_post, structured_output=True)

    # structured_providers 및 independent_providers 에 gemini 정식 통합
    assert "gemini" in llm_cli.STRUCTURED_PROVIDERS
    assert llm_cli.structured_providers("codex,gemini,hive") == ["codex", "gemini", "hive"]
    assert "gemini" in llm_cli.independent_providers(["codex", "gemini"])

    monkeypatch.setattr(D, "gemini_chat", lambda prompt, model, **kw: '{"via": "gemini"}')
    text, provider = llm_cli.structured_chat_with_provider("test", providers="gemini")
    assert provider == "gemini" and text == '{"via": "gemini"}'


# ── 2. Quota-Aware Maintenance Runner & Dedup Suppression Memory ─────────────

def test_maintenance_structured_chat_bypasses_codex_when_busy(monkeypatch):
    from lib import llm_cli

    seen_providers = []
    monkeypatch.setenv("WIKI_STRUCTURED_PROVIDERS", "codex,gemini,hive")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(
        llm_cli, "codex_usage",
        lambda **k: {"allowed": True, "reached": None, "plan": "plus",
                     "windows": [{"minutes": 300, "used_percent": 82}]},
    )

    def fake_structured(prompt, **kw):
        seen_providers.append(kw.get("providers"))
        return ('{"ok": true}', "gemini")

    monkeypatch.setattr(llm_cli, "structured_chat_with_provider", fake_structured)
    res = llm_cli.maintenance_structured_chat("json prompt")
    assert res == '{"ok": true}'
    assert seen_providers and "codex" not in (seen_providers[0] or "")
    assert "gemini" in (seen_providers[0] or "")


def test_dedup_batch_suppresses_unchanged_rejected_pair_and_rechecks_on_edit(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    from agent_console import wiki
    from reports import wiki_dedup_batch as wdb

    body_a = "커뮤니티 텔레그램 단독 신호는 공식 자료와 교차확인 전까지 보조 근거로만 취급한다 투자 판단 금지"
    body_b = "커뮤니티 텔레그램 단독 신호는 재무 공식 자료와 교차 확인 전까지는 보조 근거로만 활용한다 투자 금지"
    a = wiki.upsert_page({"title": "단독 신호 A", "summary": "s", "body": body_a, "surface": "market", "kind": "playbook", "status": "reviewed", "source_refs": []})
    b = wiki.upsert_page({"title": "단독 신호 B", "summary": "s", "body": body_b, "surface": "market", "kind": "playbook", "status": "reviewed", "source_refs": []})

    calls = []

    def reject(prompt):
        calls.append(prompt)
        return '{"merge": false, "reason": "범위 다름"}'

    r1 = wdb.run(dry_run=False, llm_fn=reject)
    assert len(calls) == 1 and r1["merged"] == []

    # 두 번째 실행: 두 페이지가 안 바뀌었으므로 LLM 재호출 없이 suppression
    r2 = wdb.run(dry_run=False, llm_fn=reject)
    assert len(calls) == 1
    assert r2["suppressed"] >= 1

    # 페이지 본문이 변경되면 다시 판정
    wiki.upsert_page({**a, "body": body_a + " 추가 업데이트 문장"})
    r3 = wdb.run(dry_run=False, llm_fn=reject)
    assert len(calls) == 2 and r3["suppressed"] == 0


def test_dedup_batch_pairs_pages_sharing_entity_tag_despite_different_wording(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    from agent_console import wiki
    from reports import wiki_dedup_batch as wdb

    a = wiki.upsert_page({
        "title": "엔비디아 블랙웰 수율 점검",
        "summary": "발열 이슈 모니터링",
        "body": "차세대 칩셋 패키징 수율이 단기 출하량에 영향을 준다",
        "surface": "market", "kind": "risk", "status": "reviewed",
        "tags": ["wiki", "entity:company:NVDA", "ticker:NVDA"], "source_refs": [],
    })
    b = wiki.upsert_page({
        "title": "NVDA 데이터센터 매출 마진 리스크",
        "summary": "총이익률 둔화 가능성",
        "body": "클라우드 고객사 자체 칩 도입으로 장기 마진 압박이 커질 수 있다",
        "surface": "market", "kind": "risk", "status": "reviewed",
        "tags": ["wiki", "entity:company:NVDA", "ticker:NVDA"], "source_refs": [],
    })
    pairs = wdb.find_candidate_pairs(wiki._all_wiki_pages())
    assert {a["id"], b["id"]} in [{p[0]["id"], p[1]["id"]} for p in pairs]


# ── 3. Closed-Loop Insight Outcome Scorecard ─────────────────────────────────

def test_wiki_insight_eval_scores_matured_horizons_without_lookahead(tmp_path):
    import wiki_insight_eval as wie

    idx = pd.bdate_range("2026-09-01", periods=35)
    # 2026-09-01 종가 100 -> 5거래일 후(idx[5]) 108 (+8%), 20거래일 후(idx[20]) 115 (+15%)
    px_nvda = pd.Series([100.0] * 5 + [108.0] * 15 + [115.0] * 15, index=idx)
    px_bench = pd.Series([100.0] * 5 + [102.0] * 15 + [105.0] * 15, index=idx)
    closes = {"NVDA": px_nvda, "QQQ": px_bench}

    pages = [
        {
            "id": "ins-1", "kind": "insight", "status": "draft",
            "tags": ["wiki", "insight", "ticker:NVDA", "entity:company:NVDA"],
            "created_at": "2026-09-01T00:00:00+00:00",
            "source_knowledge": {
                "insight": {
                    "entity_id": "company:NVDA",
                    "horizon": "days",       # 5거래일
                    "direction": "bullish",
                    "confidence": 0.8,
                    "generated_at": "2026-09-01T00:00:00+00:00",
                }
            },
        },
        {
            "id": "ins-pending", "kind": "insight", "status": "draft",
            "tags": ["wiki", "insight", "ticker:NVDA"],
            "created_at": "2026-10-15T00:00:00+00:00",
            "source_knowledge": {
                "insight": {
                    "entity_id": "company:NVDA",
                    "horizon": "weeks",      # 20거래일 — 가격 데이터 범위 밖 → 미성숙
                    "direction": "bullish",
                    "confidence": 0.7,
                    "generated_at": "2026-10-15T00:00:00+00:00",
                }
            },
        },
    ]

    ledger_path = tmp_path / "wiki_insight_outcomes.jsonl"
    out = wie.evaluate_insights(pages, closes, ledger_path=ledger_path)
    assert out["recorded"] == 1
    assert out["pending"] == 1
    # 재실행 시 멱등 (중복 기록 금지)
    out2 = wie.evaluate_insights(pages, closes, ledger_path=ledger_path)
    assert out2["recorded"] == 0

    summary = wie.summarize_outcomes(ledger_path)
    assert summary["n"] == 1
    assert summary["hit_rate"] == pytest.approx(1.0)
    assert summary["mean_excess_ret"] == pytest.approx(0.06, abs=1e-4)  # +8% - +2%


# ── 4. Point-in-Time Ontology Shadow ML Features ─────────────────────────────

def test_ontology_pit_shadow_features_exclude_future_relations():
    from reports import wiki_ontology_features as wof

    ontology = {
        "entities": {
            "company:NVDA": {"id": "company:NVDA", "type": "company", "name": "엔비디아", "ticker": "NVDA", "claim_count": 5},
        },
        "relations": [
            {
                "id": "r1", "type": "positive_for", "from": "theme:ai", "to": "company:NVDA",
                "state": "corroborated", "publisher_count": 2, "source_count": 3,
                "perspectives": ["sell_side", "community"], "last_seen": "2026-11-18",
            },
            {
                "id": "r2", "type": "negative_for", "from": "region:china", "to": "company:NVDA",
                "state": "contested", "publisher_count": 1, "source_count": 1,
                "perspectives": ["community"], "last_seen": "2026-11-17",
            },
            # 미래 날짜 관계 — as_of="2026-11-20" 에서 반드시 제외!
            {
                "id": "r-future", "type": "negative_for", "from": "institution:fed", "to": "company:NVDA",
                "state": "corroborated", "publisher_count": 3, "source_count": 4,
                "perspectives": ["sell_side"], "last_seen": "2026-11-25",
            },
        ],
    }

    feats = wof.build_pit_ticker_features(ontology, as_of="2026-11-20")
    assert "NVDA" in feats
    nvda = feats["NVDA"]
    assert nvda["corroborated_pos_count"] == 1
    assert nvda["corroborated_neg_count"] == 0   # 미래 r-future 차단 확인
    assert nvda["contested_ratio"] == pytest.approx(0.5)
    assert nvda["fresh_relation_count_7d"] == 2
    assert nvda["as_of"] == "2026-11-20"


# ── 5. Dashboard Pagination & Retrieval Telemetry Model ──────────────────────

def test_dashboard_browser_pagination_and_retrieval_telemetry():
    from dashboard import wiki_browser
    from dashboard.pages import ai_wiki

    items = [{"id": f"p{i}"} for i in range(45)]
    p1 = wiki_browser.paginate_browser_items(items, page=1, page_size=18)
    p3 = wiki_browser.paginate_browser_items(items, page=3, page_size=18)
    assert p1["total_pages"] == 3 and len(p1["items"]) == 18 and p1["page"] == 1
    assert p3["total_pages"] == 3 and len(p3["items"]) == 9 and p3["page"] == 3
    # 범위 밖 페이지 번호 클램프
    assert wiki_browser.paginate_browser_items(items, page=99, page_size=18)["page"] == 3

    tel = ai_wiki.build_retrieval_telemetry_model({
        "queries": 20, "retrievals": 20, "context_uses": 15,
        "answer_uses": 10, "invalid_citations": 1, "fallbacks": 4,
        "providers": {"fts5": 16, "fallback": 4},
    })
    assert tel["queries"] == 20
    assert tel["context_rate_pct"] == pytest.approx(75.0)
    assert tel["fallback_rate_pct"] == pytest.approx(20.0)
    assert tel["invalid_citations"] == 1

    # 실제 evidence_usage.usage_summary() 출력 스키마 호환 검증
    tel_real = ai_wiki.build_retrieval_telemetry_model({
        "hours": 24, "retrieval_count": 10, "context_use_count": 8,
        "answer_use_count": 5, "invalid_citation_count": 0,
        "fallback_retrieval_count": 1, "fallback_ratio": 0.1,
    })
    assert tel_real["retrievals"] == 10
    assert tel_real["context_rate_pct"] == pytest.approx(80.0)
    assert tel_real["fallback_rate_pct"] == pytest.approx(10.0)


def test_m4_edge_cases_gemini_thinking_parts_dedup_error_and_pit_export(monkeypatch, tmp_path):
    """Gemini 2.5 thought 파트 필터·모델명 인젝션 차단, dedup 일시 오류 미억제, PIT JSONL 멱등 적재."""
    _isolate(monkeypatch, tmp_path)
    from agent_console import wiki
    from lib import llm_direct as D
    from reports import wiki_dedup_batch as wdb
    from reports import wiki_ontology_features as wof

    monkeypatch.setenv("GEMINI_API_KEY", "k-secret")

    def thought_post(*a, **k):
        return _FakeResp({
            "candidates": [{
                "content": {"parts": [
                    {"thought": True, "text": "내부 사고 과정..."},
                    {"text": '{"verdict": "keep"}'},
                ]},
                "finishReason": "STOP",
            }]
        })

    assert D.gemini_chat("p", "gemini-2.5-flash", post=thought_post, structured_output=True) == '{"verdict": "keep"}'
    with pytest.raises(ValueError):
        D.gemini_chat("p", "../evil?key=x", post=thought_post)

    # dedup 일시 오류(invalid JSON / exception)는 suppression 에 영구 기록하면 안 된다.
    body_a = "반도체 설비투자 사이클 둔화 리스크는 분기별 수주 잔고와 가동률 지표로 검증해야 한다"
    body_b = "반도체 설비투자 사이클 둔화 리스크는 분기별 수주 잔고와 공장 가동률 지표로 점검한다"
    wiki.upsert_page({"title": "설비 리스크 A", "summary": "s", "body": body_a, "surface": "market", "kind": "risk", "status": "reviewed", "source_refs": []})
    wiki.upsert_page({"title": "설비 리스크 B", "summary": "s", "body": body_b, "surface": "market", "kind": "risk", "status": "reviewed", "source_refs": []})

    r_err = wdb.run(dry_run=False, llm_fn=lambda p: "not-json")
    assert r_err["pairs_considered"] == 1
    r_retry = wdb.run(dry_run=False, llm_fn=lambda p: '{"merge": false, "reason": "ok"}')
    assert r_retry["pairs_considered"] == 1 and r_retry["suppressed"] == 0
    r_sup = wdb.run(dry_run=False, llm_fn=lambda p: '{"merge": false, "reason": "ok"}')
    assert r_sup["suppressed"] == 1 and r_sup["pairs_considered"] == 0

    # PIT 온톨로지 JSONL 스냅샷 멱등 적재
    onto = {
        "entities": {"company:NVDA": {"id": "company:NVDA", "type": "company", "name": "엔비디아", "ticker": "NVDA"}},
        "relations": [{
            "id": "r1", "type": "positive_for", "from": "theme:ai", "to": "company:NVDA",
            "state": "corroborated", "perspectives": ["sell_side"], "last_seen": "2026-11-18",
        }],
    }
    snap_path = tmp_path / "onto_feats.jsonl"
    assert wof.export_daily_snapshot(onto, as_of="2026-11-20", path=snap_path) == 1
    assert wof.export_daily_snapshot(onto, as_of="2026-11-20", path=snap_path) == 0


# ── 6. Reviewer-Driven Regression Tests (B1–B5 + Polish) ─────────────────────

def test_m4_b1_codex_quota_guard_when_not_first_and_llm_disabled(monkeypatch):
    """B1: WIKI_STRUCTURED_PROVIDERS='gemini,codex,hermes' 에서도 Codex 바쁨 시 codex/hermes 를 차단한다."""
    from lib import llm_cli

    monkeypatch.setenv("WIKI_STRUCTURED_PROVIDERS", "gemini,codex,hermes")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("HIVE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("shutil.which", lambda *a, **k: None)
    monkeypatch.setattr(
        llm_cli, "codex_usage",
        lambda **k: {"allowed": False, "reached": "5h", "windows": [{"minutes": 300, "used_percent": 100}]},
    )
    called = []
    monkeypatch.setattr(llm_cli, "structured_chat_with_provider", lambda *a, **k: (called.append(k), ('{"ok":true}', "codex"))[1])

    # 독립 제공자 없으면 None 반환하고 codex 절대 호출 금지
    assert llm_cli.maintenance_structured_chat("p") is None
    assert called == []

    # GEMINI_API_KEY가 있으면 codex/hermes 제거된 providers="gemini"만 전달
    monkeypatch.setenv("GEMINI_API_KEY", "secret-k")
    res = llm_cli.maintenance_structured_chat("p")
    assert res == '{"ok":true}'
    assert called and called[0].get("providers") == "gemini"

    # AGENT_CONSOLE_LLM_ENABLED=0 이면 codex_usage 조차 호출하지 않고 즉시 None
    usage_calls = []
    monkeypatch.setenv("AGENT_CONSOLE_LLM_ENABLED", "0")
    monkeypatch.setattr(llm_cli, "codex_usage", lambda **k: usage_calls.append(1) or {})
    assert llm_cli.maintenance_structured_chat("p") is None
    assert usage_calls == []

    # 오타/미지 제공자 설정 시 TypeError 없이 fail-closed(None) 반환
    monkeypatch.setenv("AGENT_CONSOLE_LLM_ENABLED", "1")
    monkeypatch.setenv("WIKI_STRUCTURED_PROVIDERS", "codxe")
    assert llm_cli.maintenance_structured_chat("p") is None


def test_m4_b2_b3_insight_eval_kr_benchmark_and_refreshed_deterministic_id(tmp_path, monkeypatch):
    """B2/B3: KR 종목은 ^KS11만 사용(QQQ 폴백 금지)하고, 동일 page_id 갱신 인사이트도 별도 빈티지로 평가한다."""
    import wiki_insight_eval as wie

    idx = pd.bdate_range("2026-09-01", periods=40)
    px_kr = pd.Series([100.0] * 5 + [110.0] * 10 + [121.0] * 25, index=idx)
    px_ks11 = pd.Series([100.0] * 5 + [102.0] * 10 + [104.0] * 25, index=idx)
    px_qqq = pd.Series([100.0] * 5 + [130.0] * 35, index=idx)

    # B2-a: ^KS11 없이 QQQ만 있으면 KR 종목은 절대 QQQ로 폴백하지 않음
    p_kr = {
        "id": "insight-kr-fixed", "kind": "insight", "status": "draft",
        "tags": ["wiki", "insight", "ticker:005930.KS"],
        "source_knowledge": {"insight": {
            "entity_id": "company:005930.KS", "horizon": "days",
            "direction": "bullish", "generated_at": "2026-09-01T00:00:00+00:00",
        }},
    }
    ledger_no_ks = tmp_path / "no_ks.jsonl"
    out_no_ks = wie.evaluate_insights([p_kr], {"005930.KS": px_kr, "QQQ": px_qqq}, ledger_path=ledger_no_ks)
    assert out_no_ks["recorded"] == 1
    rows_no_ks = wie._read_ledger_rows(ledger_no_ks)
    assert rows_no_ks[0]["benchmark"] == ""
    assert rows_no_ks[0]["bench_ret"] is None
    assert rows_no_ks[0]["excess_ret"] is None
    assert rows_no_ks[0]["hit"] is True  # +10% stock_ret (QQQ +30% 오염 없음!)

    # B2-b: ^KS11 존재 시 ^KS11 대비 초과수익 기록
    ledger_ks = tmp_path / "with_ks.jsonl"
    out_ks = wie.evaluate_insights([p_kr], {"005930.KS": px_kr, "^KS11": px_ks11, "QQQ": px_qqq}, ledger_path=ledger_ks)
    assert out_ks["recorded"] == 1
    rows_ks = wie._read_ledger_rows(ledger_ks)
    assert rows_ks[0]["benchmark"] == "^KS11"
    assert rows_ks[0]["excess_ret"] == pytest.approx(0.08, abs=1e-4)

    # B3: 동일 page_id 가 새 generated_at 으로 갱신되면 새 빈티지로 정상 기록, 재실행 시 멱등
    p_kr_v2 = {
        "id": "insight-kr-fixed", "kind": "insight", "status": "draft",
        "tags": ["wiki", "insight", "ticker:005930.KS"],
        "source_knowledge": {"insight": {
            "entity_id": "company:005930.KS", "horizon": "days",
            "direction": "bullish", "generated_at": "2026-09-10T00:00:00+00:00",
        }},
    }
    out_v2 = wie.evaluate_insights([p_kr_v2], {"005930.KS": px_kr, "^KS11": px_ks11}, ledger_path=ledger_ks)
    assert out_v2["recorded"] == 1
    out_v2_again = wie.evaluate_insights([p_kr, p_kr_v2], {"005930.KS": px_kr, "^KS11": px_ks11}, ledger_path=ledger_ks)
    assert out_v2_again["recorded"] == 0 and out_v2_again["already_recorded"] == 2
    assert wie.summarize_outcomes(ledger_ks)["n"] == 2

    # B2-c: main() 이 KR 종목 발견 시 ^KS11 도 함께 fetch_prices 요청하는지 확인
    fetched_tickers = []
    monkeypatch.setattr("agent_console.wiki._all_wiki_pages", lambda: [p_kr])
    monkeypatch.setattr("ml.data_pipeline.fetch_prices", lambda tks, days=200: (fetched_tickers.extend(tks), {})[1])
    monkeypatch.setattr(wie, "DEFAULT_LEDGER_PATH", tmp_path / "main.jsonl")
    assert wie.main() == 0
    assert "^KS11" in fetched_tickers and "005930.KS" in fetched_tickers


def test_m4_b4_b5_gemini_url_allowlist_pit_lookahead_and_polish(monkeypatch):
    """B4: Gemini URL allowlist + SAFETY 거부, B5: PIT 미래 개체·미래 대체·무날짜 차단 + 대시보드 inf 방어."""
    from dashboard import wiki_browser
    from dashboard.pages import ai_wiki
    from lib import llm_direct as D
    from reports import wiki_dedup_batch as wdb
    from reports import wiki_ontology_features as wof

    # B4: URL/헤더 주입 문자 전부 차단
    for bad_model in (
        "gemini-2.5-flash@evil.com",
        "gemini-2.5-flash:streamGenerateContent",
        "gemini 2.5 flash",
        "gemini-2.5-flash\r\nX-Inject: 1",
        "gemini%2f2.5",
        "..gemini",
    ):
        with pytest.raises(ValueError):
            D._gemini_url(bad_model)
    assert D._gemini_url("models/gemini-2.5-flash").endswith("/gemini-2.5-flash:generateContent")

    # SAFETY finishReason + malformed content 문자열 방어
    monkeypatch.setenv("GEMINI_API_KEY", "k-secret")
    with pytest.raises(ValueError, match="structured_response_invalid"):
        D.gemini_chat(
            "p", "gemini-2.5-flash", structured_output=True,
            post=lambda *a, **k: _FakeResp({
                "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}, "finishReason": "SAFETY"}]
            }),
        )
    with pytest.raises(RuntimeError):
        D.gemini_chat(
            "p", "gemini-2.5-flash", structured_output=False,
            post=lambda *a, **k: _FakeResp({"candidates": [{"content": "unexpected-str"}]}),
        )

    # B5: 미래 first_seen 개체, 비상장 슬러그, 무날짜 관계 차단 + 미래 시점 대체는 과거 시점 누수 금지
    onto = {
        "entities": {
            "company:NVDA": {"id": "company:NVDA", "type": "company", "name": "엔비디아", "ticker": "NVDA", "first_seen": "2026-01-01"},
            "company:FUTUREIPO": {"id": "company:FUTUREIPO", "type": "company", "name": "미래IPO", "ticker": "FUTR", "first_seen": "2026-11-25"},
            "company:openai": {"id": "company:openai", "type": "company", "name": "OpenAI", "first_seen": "2026-01-01"},
        },
        "relations": [
            # 과거 시점(2026-11-20)에는 아직 대체되지 않음(superseded_by가 2026-11-25!) → publisher_count=3 기반 corroborated 복원
            {
                "id": "r-past-valid", "type": "positive_for", "from": "theme:ai", "to": "company:NVDA",
                "state": "superseded", "superseded_by": "r-future-sup", "publisher_count": 3, "last_seen": "2026-10-10",
            },
            {
                "id": "r-future-sup", "type": "negative_for", "from": "theme:ai", "to": "company:NVDA",
                "state": "corroborated", "last_seen": "2026-11-25",
            },
            # 과거 시점 이전에 이미 대체됨(superseded_by가 2026-11-18 <= 2026-11-20) → 제외돼야 함
            {
                "id": "r-old-sup", "type": "negative_for", "from": "region:china", "to": "company:NVDA",
                "state": "superseded", "superseded_by": "r-present-win", "last_seen": "2026-09-01",
            },
            {
                "id": "r-present-win", "type": "positive_for", "from": "region:china", "to": "company:NVDA",
                "state": "corroborated", "last_seen": "2026-11-18",
            },
            # 날짜 미상 관계 → PIT 제외
            {
                "id": "r-undated", "type": "negative_for", "from": "institution:fed", "to": "company:NVDA",
                "state": "corroborated", "last_seen": "",
            },
        ],
    }
    feats = wof.build_pit_ticker_features(onto, as_of="2026-11-20")
    assert "FUTR" not in feats
    assert "OPENAI" not in feats
    assert "NVDA" in feats
    # r-past-valid(복원된 corroborated) + r-present-win(corroborated) = 2건, corroborated_pos_count == 2
    assert feats["NVDA"]["relation_count"] == 2
    assert feats["NVDA"]["corroborated_pos_count"] == 2
    assert feats["NVDA"]["corroborated_neg_count"] == 0
    assert feats["NVDA"]["net_corroborated_sentiment"] == pytest.approx(1.0)

    # Polish: 1글자 티커 허용 + 빈 티커 거부 + inf OverflowError 방어
    assert "ticker:v" in wdb._entity_identity_tags({"tags": ["ticker:V", "ticker:   "]})
    assert "ticker:" not in wdb._entity_identity_tags({"tags": ["ticker:   "]})
    assert wiki_browser.paginate_browser_items([{"id": "a"}], page=float("inf"), page_size=float("inf"))["page"] == 1
    assert ai_wiki.build_retrieval_telemetry_model({"retrieval_count": float("inf")})["retrievals"] == 0


