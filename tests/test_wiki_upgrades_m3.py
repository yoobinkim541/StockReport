"""TDD suite for Milestone 3 LLM Wiki upgrades:
1. SQLite WAL usage/feedback sidecar (no 20MB events.jsonl rewrite on chat turn)
2. SQLite FTS5 hybrid first-stage retrieval (Korean-aware BM25 + entity/tag boost)
3. Ontology temporal decay, temporal supersession (>30d), and 2-hop corroborated chains.
"""
from __future__ import annotations

import threading
from pathlib import Path

from agent_console_support import _isolate


# ── 1. SQLite Usage / Feedback Sidecar ───────────────────────────────────────

def test_track_page_usage_avoids_events_jsonl_rewrite_and_rebuild(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    from agent_console import shared_memory, wiki

    page = wiki.upsert_page({
        "title": "반도체 HBM 전망",
        "summary": "HBM 수요 강세",
        "body": "본문 내용",
        "surface": "market",
        "kind": "note",
        "status": "draft",
        "tags": ["wiki", "hbm"],
        "source_refs": ["https://example.com/hbm"],
    })
    events_path = Path(shared_memory._paths()["events"])
    before_stat = (events_path.stat().st_mtime_ns, events_path.stat().st_size)

    rebuild_calls = []
    patch_calls = []
    monkeypatch.setattr(wiki, "_debounced_rebuild", lambda: rebuild_calls.append(1))
    monkeypatch.setattr(shared_memory, "patch_record", lambda *a, **k: patch_calls.append(1))

    for i in range(5):
        wiki.track_page_usage(page["id"], f"HBM 질의 {i}")

    after_stat = (events_path.stat().st_mtime_ns, events_path.stat().st_size)
    assert after_stat == before_stat, "track_page_usage가 events.jsonl을 재작성함"
    assert patch_calls == [], "track_page_usage가 shared_memory.patch_record를 호출함"
    assert rebuild_calls == [], "track_page_usage가 _debounced_rebuild를 호출함"

    fetched = wiki.get_page(page["id"])
    assert fetched["useCount"] == 5
    assert fetched["lastQuery"] == "HBM 질의 4"
    assert fetched["lastUsedAt"]


def test_usage_sidecar_merges_legacy_counts_feedback_and_cleans_on_delete(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    from agent_console import shared_memory, wiki, wiki_usage_store

    page = wiki.upsert_page({
        "title": "레거시 카운트 페이지",
        "summary": "요약",
        "body": "본문",
        "surface": "market",
        "kind": "note",
        "status": "draft",
        "tags": ["wiki"],
        "source_refs": [],
    })
    # 레거시 events.jsonl에 이미 저장된 useCount=7, feedback={"helpful": 2}
    shared_memory.patch_record(
        page["id"],
        lambda r: {**r, "useCount": 7, "lastUsedAt": "2026-10-01T00:00:00+00:00", "feedback": {"helpful": 2}},
    )
    wiki._CACHE.clear()

    wiki.track_page_usage(page["id"], "신규 질의")
    wiki.track_page_usage(page["id"], "최신 질의")
    wiki._store_page_feedback(page["id"], "helpful")
    wiki._store_page_feedback(page["id"], "not_helpful")

    fetched = wiki.get_page(page["id"])
    assert fetched["useCount"] == 9
    assert fetched["lastQuery"] == "최신 질의"
    assert fetched["feedback"]["helpful"] == 3
    assert fetched["feedback"]["not_helpful"] == 1

    # 존재하지 않는 페이지 무시
    wiki.track_page_usage("nonexistent-page-id", "무시")
    assert wiki_usage_store.get_usage("nonexistent-page-id") is None

    # 삭제 시 사이드카 행도 정리
    assert wiki.delete_page(page["id"]) is True
    assert wiki_usage_store.get_usage(page["id"]) is None


def test_usage_sidecar_concurrent_40_threads_atomic(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    from agent_console import shared_memory, wiki

    page = wiki.upsert_page({
        "title": "동시성 테스트 페이지",
        "summary": "요약",
        "body": "본문",
        "surface": "market",
        "kind": "note",
        "status": "draft",
        "tags": ["wiki"],
        "source_refs": [],
    })
    events_path = Path(shared_memory._paths()["events"])
    before_stat = (events_path.stat().st_mtime_ns, events_path.stat().st_size)

    errors = []

    def worker(i: int):
        try:
            wiki.track_page_usage(page["id"], f"concurrent-{i}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert (events_path.stat().st_mtime_ns, events_path.stat().st_size) == before_stat
    assert wiki.get_page(page["id"])["useCount"] == 40


# ── 2. SQLite FTS5 Hybrid Retrieval ──────────────────────────────────────────

def test_fts5_hybrid_search_korean_bm25_and_punctuation_safety(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_CONSOLE_QMD_ENABLED", "0")
    from agent_console import wiki, wiki_index

    p_hbm = wiki.upsert_page({
        "title": "SK하이닉스 HBM4 공급망 병목 분석",
        "summary": "TSMC CoWoS 패키징과 HBM4 수율 이슈",
        "body": "고대역폭메모리 HBM4 양산 일정과 공급망 병목이 핵심 변수입니다.",
        "surface": "market",
        "kind": "concept",
        "status": "reviewed",
        "tags": ["wiki", "ticker:000660.KS", "entity:company:000660.KS"],
        "source_refs": ["https://example.com/1"],
    })
    p_rates = wiki.upsert_page({
        "title": "미국 국채금리 상승과 연준 통화정책",
        "summary": "10년물 국채금리 급등 시 기술주 멀티플 압박",
        "body": "연준의 고금리 장기화 기조가 미국 10년물 국채금리를 끌어올렸습니다.",
        "surface": "market",
        "kind": "risk",
        "status": "reviewed",
        "tags": ["wiki", "macro", "rates"],
        "source_refs": ["https://example.com/2"],
    })
    for i in range(6):
        wiki.upsert_page({
            "title": f"미국 증시 일반 시황 {i}",
            "summary": "미국 미국 미국 시장 마감 요약",
            "body": "미국 증시 주요 지수 마감 동향입니다. " * 3,
            "surface": "market",
            "kind": "note",
            "status": "draft",
            "tags": ["wiki"],
            "source_refs": [f"https://example.com/n{i}"],
        })

    # QMD 비활성 상태에서 FTS5 인덱스로 다중 단어·한국어 질의 정확히 1위 검색
    res_rates = wiki.list_pages(query="미국 국채금리 연준 정책", surface="market", include_archived=False, limit=5)
    assert res_rates and res_rates[0]["id"] == p_rates["id"]

    res_hbm = wiki.list_pages(query="HBM4 공급망 병목", surface="market", include_archived=False, limit=5)
    assert res_hbm and res_hbm[0]["id"] == p_hbm["id"]

    # FTS5 특수문자 포함 질의가 SQLite OperationalError 없이 안전하게 동작
    res_punct = wiki_index.search_pages("AI/반도체: HBM4 (수율) +10% -리스크 \"따옴표\"*", limit=5)
    assert isinstance(res_punct, list)
    assert any(hit["id"] == p_hbm["id"] for hit in res_punct)

    # 아카이브 처리 시 include_archived=False 검색에서 즉시 제외
    wiki.upsert_page({**p_hbm, "status": "archived", "tags": ["wiki", "archived"]})
    res_after_archive = wiki.list_pages(query="HBM4 공급망 병목", surface="market", include_archived=False, limit=5)
    assert all(p["id"] != p_hbm["id"] for p in res_after_archive)


def test_fallback_ranked_pages_normalizes_each_record_once(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    from agent_console import wiki

    for i in range(5):
        wiki.upsert_page({
            "title": f"테스트 문서 {i}",
            "summary": "반도체 업황",
            "body": "반도체 수요 분석",
            "surface": "market",
            "kind": "note",
            "status": "draft",
            "tags": ["wiki"],
            "source_refs": [],
        })
    records = wiki._wiki_records()
    orig = wiki._record_to_page
    calls = 0

    def spy(rec):
        nonlocal calls
        calls += 1
        return orig(rec)

    monkeypatch.setattr(wiki, "_record_to_page", spy)
    wiki._fallback_ranked_pages(records, query="반도체", surface="market", status="all", limit=5)
    assert calls == len(records), f"중복 _record_to_page 호출: {calls} != {len(records)}"


# ── 3. Ontology Temporal Decay, Supersession & 2-Hop Causal Chains ──────────

def _onto_lexicon():
    from agent_console import wiki_ontology as wo
    return wo.Lexicon([
        {"id": "region:iran", "type": "region", "name": "이란", "aliases": ["Iran"]},
        {"id": "asset:crude-oil", "type": "asset", "name": "원유", "aliases": ["유가"]},
        {"id": "indicator:inflation", "type": "indicator", "name": "인플레이션"},
        {"id": "institution:fed", "type": "institution", "name": "연준", "aliases": ["Fed"]},
        {"id": "sector:banks", "type": "sector", "name": "은행"},
    ], use_tickers=False)


def _onto_page(page_id: str, claims: list[tuple[str, str, str, str]]):
    """claims: (claim_id, revision_id, text, as_of)."""
    return {
        "id": page_id,
        "title": f"문서 {page_id}",
        "kind": "note",
        "status": "draft",
        "updated_at": f"{claims[-1][3]}T00:00:00+00:00",
        "source_knowledge": {
            "claims": [
                {"claim_id": cid, "source_revision_id": rev, "text": txt, "as_of": day, "competing_claim_ids": []}
                for cid, rev, txt, day in claims
            ]
        },
    }


def test_ontology_context_relations_applies_temporal_decay():
    from agent_console import wiki_ontology as wo

    ontology = {
        "generated_at": "2026-11-22T00:00:00+00:00",
        "relations": [
            {
                "id": "stale-strong",
                "from": "region:iran",
                "to": "asset:crude-oil",
                "type": "positive_for",
                "state": "corroborated",
                "publisher_count": 3,
                "source_count": 4,
                "modality": "reported",
                "last_seen": "2026-01-15",
            },
            {
                "id": "fresh-shift",
                "from": "institution:fed",
                "to": "asset:crude-oil",
                "type": "negative_for",
                "state": "corroborated",
                "publisher_count": 2,
                "source_count": 2,
                "modality": "reported",
                "last_seen": "2026-11-20",
            },
        ],
    }
    shown, _ = wo.context_relations(ontology, "asset:crude-oil")
    assert [r["id"] for r in shown] == ["fresh-shift", "stale-strong"]


def test_ontology_compile_supersedes_old_opposite_relation_after_30_days():
    from agent_console import wiki_ontology as wo

    lexicon = _onto_lexicon()
    p_old = _onto_page("p-old", [("a", "r1", "연준 긴축은 은행 건전성에 악재다.", "2026-08-01")])
    p_new1 = _onto_page("p-new1", [("b", "r2", "연준 금리 인하는 은행 수익에 호재다.", "2026-11-10")])
    p_new2 = _onto_page("p-new2", [("c", "r3", "연준 정책 전환이 은행에 호재로 작용한다.", "2026-11-12")])

    def _ext(page, kind, cid):
        return {
            "input_hash": wo.page_input_hash(page),
            "entities": [],
            "relations": [{
                "type": kind, "from": "institution:fed", "to": "sector:banks",
                "evidence": [{"page_id": page["id"], "claim_id": cid}], "note": "",
            }],
        }

    extractions = {"pages": {
        "p-old": _ext(p_old, "negative_for", "a"),
        "p-new1": _ext(p_new1, "positive_for", "b"),
        "p-new2": _ext(p_new2, "positive_for", "c"),
    }}
    ontology = wo.compile_ontology(
        [p_old, p_new1, p_new2], extractions,
        generated_at="2026-11-15T00:00:00+00:00", lexicon=lexicon,
        source_providers={"r1": "telegram:insidertracking", "r2": "telegram:kwusa", "r3": "telegram:meritz"},
    )
    by_type = {r["type"]: r for r in ontology["relations"]}
    assert by_type["positive_for"]["state"] == "corroborated"
    assert by_type["positive_for"].get("supersedes") == by_type["negative_for"]["id"]
    assert by_type["negative_for"].get("superseded_by") == by_type["positive_for"]["id"]
    assert by_type["negative_for"]["state"] != "contested"


def test_ontology_context_section_includes_two_hop_corroborated_chain_only(tmp_path):
    from agent_console import wiki_ontology as wo

    lexicon = _onto_lexicon()
    p1 = _onto_page("p1", [
        ("a", "r1", "이란 긴장으로 원유 가격이 올랐다.", "2026-11-18"),
        ("b", "r1", "원유 상승이 인플레이션 압력을 높였다.", "2026-11-18"),
        ("c", "r1", "인플레이션이 은행에 악재라는 단일 의견.", "2026-11-18"),
    ])
    p2 = _onto_page("p2", [
        ("d", "r2", "이란 봉쇄 우려로 유가가 급등했다.", "2026-11-19"),
        ("e", "r2", "유가 급등은 인플레이션을 자극한다.", "2026-11-19"),
    ])

    def _rel(kind, src, dst, pid, cid):
        return {"type": kind, "from": src, "to": dst, "evidence": [{"page_id": pid, "claim_id": cid}], "note": ""}

    extractions = {"pages": {
        "p1": {"input_hash": wo.page_input_hash(p1), "entities": [], "relations": [
            _rel("positive_for", "region:iran", "asset:crude-oil", "p1", "a"),
            _rel("positive_for", "asset:crude-oil", "indicator:inflation", "p1", "b"),
            _rel("negative_for", "indicator:inflation", "sector:banks", "p1", "c"),
        ]},
        "p2": {"input_hash": wo.page_input_hash(p2), "entities": [], "relations": [
            _rel("positive_for", "region:iran", "asset:crude-oil", "p2", "d"),
            _rel("positive_for", "asset:crude-oil", "indicator:inflation", "p2", "e"),
        ]},
    }}
    ontology = wo.compile_ontology(
        [p1, p2], extractions,
        generated_at="2026-11-20T00:00:00+00:00", lexicon=lexicon,
        source_providers={"r1": "telegram:insidertracking", "r2": "telegram:kwusa"},
    )
    wo.save_ontology(ontology, tmp_path)

    # 질문에는 '이란'만 언급 — 2-hop corroborated 경로(이란 → 원유 → 인플레이션)가 포함되어야 함
    section = wo.context_section("이란 지정학 리스크 정리해줘", root=tmp_path)
    assert "이란" in section
    assert "원유" in section
    assert "인플레이션" in section
    assert "2-hop" in section or "연쇄" in section
    # 단일 출처(asserted) 엣지(인플레이션 → 은행)는 2-hop 체인에 포함되면 안 됨
    assert "은행" not in section


# ── 4. Reviewer-driven Regression Tests (B1–B8) ──────────────────────────────

def test_usage_sidecar_no_double_counting_on_upsert_split_and_merge(monkeypatch, tmp_path):
    """B1: upsert_page / split_for_readability / _merge_pages 왕복 저장 시 useCount·feedback 이중 집계 금지."""
    _isolate(monkeypatch, tmp_path)
    from agent_console import shared_memory, wiki

    monkeypatch.setattr(wiki, "_debounced_rebuild", lambda: None)
    long_body = ("반도체 HBM 공급망 분석 본문입니다. " * 120).strip()
    p1 = wiki.upsert_page({
        "id": "p-roundtrip-1",
        "title": "HBM 공급망 원문",
        "summary": "요약1",
        "body": long_body,
        "surface": "market",
        "kind": "note",
        "status": "draft",
        "tags": ["wiki"],
        "source_refs": ["https://example.com/a"],
    })
    shared_memory.patch_record(
        p1["id"],
        lambda r: {**r, "useCount": 2, "feedback": {"helpful": 1}},
    )
    wiki._CACHE.clear()
    for i in range(3):
        wiki.track_page_usage(p1["id"], f"q-{i}")
    wiki._store_page_feedback(p1["id"], "helpful")
    wiki._store_page_feedback(p1["id"], "helpful")

    assert wiki.get_page(p1["id"])["useCount"] == 5
    assert wiki.get_page(p1["id"])["feedback"] == {"helpful": 3}

    # 1) hydrated dict 전체 왕복 upsert
    hydrated = wiki.get_page(p1["id"])
    wiki.upsert_page({**hydrated, "summary": "요약 수정 1"})
    after1 = wiki.get_page(p1["id"])
    assert after1["useCount"] == 5
    assert after1["feedback"] == {"helpful": 3}

    # 2) partial payload upsert (useCount/feedback 미전달)
    wiki.upsert_page({
        "id": p1["id"],
        "title": "HBM 공급망 원문",
        "summary": "요약 수정 2",
        "body": long_body,
        "surface": "market",
        "kind": "note",
        "status": "draft",
        "tags": ["wiki"],
        "source_refs": ["https://example.com/a"],
    })
    after2 = wiki.get_page(p1["id"])
    assert after2["useCount"] == 5
    assert after2["feedback"] == {"helpful": 3}

    # 3) split_for_readability 왕복
    wiki.split_for_readability(p1["id"], max_chars=500)
    after3 = wiki.get_page(p1["id"])
    assert after3["useCount"] == 5
    assert after3["feedback"] == {"helpful": 3}

    # 4) _merge_pages 왕복 (target + archived source 모두 이중 집계 금지)
    p2 = wiki.upsert_page({
        "id": "p-roundtrip-2",
        "title": "HBM 보조 메모",
        "summary": "보조 요약",
        "body": "보조 본문 내용",
        "surface": "market",
        "kind": "note",
        "status": "draft",
        "tags": ["wiki"],
        "source_refs": ["https://example.com/b"],
    })
    wiki.track_page_usage(p2["id"], "p2-q1")
    wiki.track_page_usage(p2["id"], "p2-q2")
    wiki._store_page_feedback(p2["id"], "not_helpful")

    res = wiki._merge_pages([p2["id"]], p1["id"], "")
    assert res is not None
    assert wiki.get_page(p1["id"])["useCount"] == 5
    assert wiki.get_page(p1["id"])["feedback"] == {"helpful": 3}
    assert wiki.get_page(p2["id"])["useCount"] == 2
    assert wiki.get_page(p2["id"])["feedback"] == {"not_helpful": 1}


def test_usage_sidecar_recreates_schema_after_db_deleted(monkeypatch, tmp_path):
    """B2: 동일 경로의 wiki_usage.sqlite가 삭제 후 재생성돼도 no such table 오류 없이 복구된다."""
    _isolate(monkeypatch, tmp_path)
    from agent_console import wiki, wiki_usage_store

    page = wiki.upsert_page({
        "title": "DB 재생성 테스트",
        "summary": "요약",
        "body": "본문",
        "surface": "market",
        "kind": "note",
        "status": "draft",
        "tags": ["wiki"],
        "source_refs": [],
    })
    wiki.track_page_usage(page["id"], "first")
    assert wiki.get_page(page["id"])["useCount"] == 1

    db = wiki_usage_store.db_path()
    for suffix in ("", "-wal", "-shm"):
        target = Path(f"{db}{suffix}")
        if target.exists():
            target.unlink()

    wiki.track_page_usage(page["id"], "after-recreate")
    assert wiki.get_page(page["id"])["useCount"] == 1
    assert wiki.get_page(page["id"])["lastQuery"] == "after-recreate"


def test_fts5_provider_telemetry_no_thrash_long_query_and_self_heal(monkeypatch, tmp_path):
    """B6/B7/B8: FTS5 provider 텔레메트리(fallback=False), 아카이브 토글 시 재색인 스래싱 없음, 긴 질의/단어경계/손상복구."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_CONSOLE_QMD_ENABLED", "0")
    from agent_console import evidence_usage, wiki, wiki_index

    p_quantum = wiki.upsert_page({
        "title": "양자컴퓨팅 상용화 타임라인",
        "summary": "초전도 큐비트 오류정정 진전",
        "body": "IBM과 구글의 양자컴퓨팅 로드맵 정리",
        "surface": "market",
        "kind": "concept",
        "status": "reviewed",
        "tags": ["wiki", "quantum"],
        "source_refs": ["https://example.com/q"],
    })
    p_ai = wiki.upsert_page({
        "title": "AI 데이터센터 전력 인프라",
        "summary": "AI 모델 추론 전력 급증",
        "body": "전력망 증설 수혜주 분석",
        "surface": "market",
        "kind": "note",
        "status": "draft",
        "tags": ["wiki", "ai"],
        "source_refs": ["https://example.com/ai"],
    })
    p_claim = wiki.upsert_page({
        "title": "Claim verification log",
        "summary": "claim audit note without keyword in header",
        "body": "claim records mentioning ai once in body",
        "surface": "market",
        "kind": "note",
        "status": "draft",
        "tags": ["wiki"],
        "source_refs": ["https://example.com/claim"],
    })
    wiki.upsert_page({
        "title": "보관된 양자컴퓨팅 메모",
        "summary": "구버전 메모",
        "body": "양자컴퓨팅 아카이브",
        "surface": "market",
        "kind": "note",
        "status": "archived",
        "tags": ["wiki", "archived"],
        "source_refs": ["https://example.com/arch"],
    })

    build_calls = []
    orig_build = wiki_index._build_rows

    def spy_build(records):
        build_calls.append(len(records))
        return orig_build(records)

    monkeypatch.setattr(wiki_index, "_build_rows", spy_build)

    # B8 + B6: include_archived 토글을 반복해도 FTS5 재색인은 최초 1회만 발생해야 한다.
    r1 = wiki.list_pages(query="양자컴퓨팅", surface="market", include_archived=False, limit=5)
    r2 = wiki.list_pages(query="양자컴퓨팅", surface="market", include_archived=True, limit=5)
    r3 = wiki.list_pages(query="양자컴퓨팅", surface="market", include_archived=False, limit=5)
    assert r1 and r1[0]["id"] == p_quantum["id"]
    assert r1[0].get("search_provider") == "fts5"
    assert len(r2) >= 2
    assert len(r3) == len(r1)
    assert len(build_calls) == 1, f"include_archived 토글 시 불필요한 재색인 발생: {build_calls}"

    summary = evidence_usage.usage_summary(hours=24)
    assert summary["retrieval_count"] >= 1
    assert summary["fallback_retrieval_count"] == 0
    assert summary["fallback_ratio"] == 0.0

    # B7: 앞쪽 복합어 2-gram 때문에 긴 질의 뒤쪽 키워드('양자컴퓨팅')가 24토큰 밖으로 밀려나면 안 된다.
    long_q = "반도체장비 생태계변화 지정학리스크 통화정책전환 거시경제전망 원유공급망 인플레이션압력 양자컴퓨팅"
    long_hits = wiki_index.search_pages(long_q, surface="market", include_archived=False, limit=5)
    assert any(h["id"] == p_quantum["id"] for h in long_hits)

    # B7: 'ai' 질의 시 p_claim 본문에 ai가 있어도 제목의 'claim' 부분문자열이 헤더 가점(+6.0)을 받으면 안 된다.
    ai_hits = wiki_index.search_pages("ai", surface="market", include_archived=False, limit=5)
    assert ai_hits and ai_hits[0]["id"] == p_ai["id"]
    claim_hit = next((h for h in ai_hits if h["id"] == p_claim["id"]), None)
    assert claim_hit is not None
    assert ai_hits[0]["score"] - claim_hit["score"] >= 5.0

    # R1: 일시적 OperationalError('database is locked') 시 살아 있는 WAL DB 를 절대 unlink 하면 안 된다.
    import sqlite3
    unlinked = []
    orig_unlink = wiki_index._unlink_db_files
    monkeypatch.setattr(wiki_index, "_unlink_db_files", lambda p: (unlinked.append(p), orig_unlink(p)))
    orig_sync_once = wiki_index._sync_once
    monkeypatch.setattr(
        wiki_index, "_sync_once",
        lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )
    assert wiki_index.sync_index(force=True) is False
    assert unlinked == [], f"lock 오류에서 살아 있는 DB 파일이 삭제됨: {unlinked}"
    assert wiki_index.db_path().exists()
    monkeypatch.setattr(wiki_index, "_sync_once", orig_sync_once)

    # B6: 실제로 손상된 wiki_search.sqlite 는 자가 복구
    wiki_index.db_path().write_bytes(b"corrupt-sqlite-bytes-not-a-db")
    healed = wiki_index.search_pages("양자컴퓨팅", surface="market", include_archived=False, limit=5)
    assert healed and healed[0]["id"] == p_quantum["id"]
    assert len(unlinked) >= 1


def test_ontology_recency_undated_and_supersession_boundary():
    """B5: 날짜 미상 관계는 어제 날짜보다 낮게 감쇠되며, 30일 경계와 superseded_by 일관성을 검증한다."""
    from datetime import date
    from agent_console import wiki_ontology as wo

    ref = date(2026, 11, 22)
    assert wo._recency_decay("", ref) == 0.5
    assert wo._recency_decay("2026-11-21", ref) > wo._recency_decay("", ref)
    assert wo._recency_decay("", None) == 1.0

    # 30일 이내(정확히 30일)는 contested 유지, 31일 차이는 superseded
    lexicon = _onto_lexicon()
    p_d0 = _onto_page("p-d0", [("a", "r1", "연준 긴축은 은행에 악재다.", "2026-10-01")])
    p_d30 = _onto_page("p-d30", [("b", "r2", "연준 정책은 은행에 호재다.", "2026-10-31")])
    p_d31 = _onto_page("p-d31", [("c", "r3", "연준 정책은 은행에 호재다.", "2026-11-01")])

    def _ext(page, kind, cid):
        return {
            "input_hash": wo.page_input_hash(page),
            "entities": [],
            "relations": [{
                "type": kind, "from": "institution:fed", "to": "sector:banks",
                "evidence": [{"page_id": page["id"], "claim_id": cid}], "note": "",
            }],
        }

    onto_30 = wo.compile_ontology(
        [p_d0, p_d30],
        {"pages": {"p-d0": _ext(p_d0, "negative_for", "a"), "p-d30": _ext(p_d30, "positive_for", "b")}},
        generated_at="2026-11-15T00:00:00+00:00", lexicon=lexicon,
        source_providers={"r1": "telegram:a", "r2": "telegram:b"},
    )
    by30 = {r["type"]: r for r in onto_30["relations"]}
    assert by30["negative_for"]["state"] == "contested"
    assert not by30["negative_for"].get("superseded_by")

    onto_31 = wo.compile_ontology(
        [p_d0, p_d31],
        {"pages": {"p-d0": _ext(p_d0, "negative_for", "a"), "p-d31": _ext(p_d31, "positive_for", "c")}},
        generated_at="2026-11-15T00:00:00+00:00", lexicon=lexicon,
        source_providers={"r1": "telegram:a", "r3": "telegram:b"},
    )
    by31 = {r["type"]: r for r in onto_31["relations"]}
    assert by31["negative_for"]["state"] == "superseded"
    assert by31["negative_for"].get("superseded_by") == by31["positive_for"]["id"]

    # state='contested' 이면서 superseded_by가 붙은 관계도 활성 풀에서 제외
    synthetic = {
        "generated_at": "2026-11-22T00:00:00+00:00",
        "relations": [
            {
                "id": "r-active", "from": "institution:fed", "to": "sector:banks",
                "type": "positive_for", "state": "corroborated",
                "publisher_count": 2, "source_count": 2, "modality": "reported", "last_seen": "2026-11-20",
            },
            {
                "id": "r-contested-superseded", "from": "institution:fed", "to": "sector:banks",
                "type": "negative_for", "state": "contested", "superseded_by": "r-active",
                "publisher_count": 3, "source_count": 3, "modality": "reported", "last_seen": "2026-10-01",
            },
        ],
    }
    shown, _ = wo.context_relations(synthetic, "institution:fed")
    assert [r["id"] for r in shown] == ["r-active"]


def test_ontology_two_hop_directionality_no_colliders_and_best_path_ranking(tmp_path):
    """B3/B4: 2-hop 연쇄는 방향 일관성을 지키고 충돌기/분기를 차단하며, 동일 목적지 중 최고 점수 경로를 선택한다."""
    from agent_console import wiki_ontology as wo

    entities = {
        "region:iran": {"id": "region:iran", "type": "region", "name": "이란", "claim_count": 2, "source_count": 2, "relation_count": 2},
        "asset:crude-oil": {"id": "asset:crude-oil", "type": "asset", "name": "원유", "claim_count": 3, "source_count": 3, "relation_count": 3},
        "indicator:inflation": {"id": "indicator:inflation", "type": "indicator", "name": "인플레이션", "claim_count": 2, "source_count": 2, "relation_count": 2},
        "institution:fed": {"id": "institution:fed", "type": "institution", "name": "연준", "claim_count": 2, "source_count": 2, "relation_count": 1},
        "sector:banks": {"id": "sector:banks", "type": "sector", "name": "은행", "claim_count": 2, "source_count": 2, "relation_count": 1},
        "theme:shipping": {"id": "theme:shipping", "type": "theme", "name": "해운", "claim_count": 2, "source_count": 2, "relation_count": 2},
    }
    ontology = {
        "schema_version": wo.SCHEMA_VERSION,
        "generated_at": "2026-11-22T00:00:00+00:00",
        "entities": entities,
        "relations": [
            # B4: 이란 → 해운 → 인플레이션 (오래된 경로가 배열 앞쪽에 위치)
            {
                "id": "r-stale-1", "from": "region:iran", "to": "theme:shipping",
                "type": "positive_for", "state": "corroborated", "publisher_count": 2, "source_count": 2,
                "modality": "reported", "last_seen": "2026-01-10",
            },
            {
                "id": "r-stale-2", "from": "theme:shipping", "to": "indicator:inflation",
                "type": "positive_for", "state": "corroborated", "publisher_count": 2, "source_count": 2,
                "modality": "reported", "last_seen": "2026-01-10",
            },
            # 이란 → 원유 → 인플레이션 (최신·강한 경로)
            {
                "id": "r-fresh-1", "from": "region:iran", "to": "asset:crude-oil",
                "type": "positive_for", "state": "corroborated", "publisher_count": 3, "source_count": 3,
                "modality": "reported", "last_seen": "2026-11-20",
            },
            {
                "id": "r-fresh-2", "from": "asset:crude-oil", "to": "indicator:inflation",
                "type": "positive_for", "state": "corroborated", "publisher_count": 3, "source_count": 3,
                "modality": "reported", "last_seen": "2026-11-20",
            },
            # B3 충돌기: 연준 → 원유 (이란 → 원유 ← 연준: 연쇄 불가!)
            {
                "id": "r-collider", "from": "institution:fed", "to": "asset:crude-oil",
                "type": "negative_for", "state": "corroborated", "publisher_count": 4, "source_count": 4,
                "modality": "reported", "last_seen": "2026-11-21",
            },
            # B3 공통원인 분기: 원유 → 은행 (인플레이션 ← 원유 → 은행: 연쇄 불가!)
            {
                "id": "r-fork", "from": "asset:crude-oil", "to": "sector:banks",
                "type": "negative_for", "state": "corroborated", "publisher_count": 4, "source_count": 4,
                "modality": "reported", "last_seen": "2026-11-21",
            },
        ],
        "co_mentions": [],
        "publishers": {},
        "stats": {},
    }
    wo.save_ontology(ontology, tmp_path)

    # B4: 이란 → 인플레이션 후보 중 오래된 해운 경로가 아니라 최신 원유 경로가 선택되어야 함
    # B3: 충돌기(연준)는 이란의 2-hop 연쇄에 포함되면 안 됨 (이란 → 원유 → 은행은 정상 하류 연쇄)
    chains_iran = wo.two_hop_chains(ontology, "region:iran")
    dests_iran = {c[3]: c[2] for c in chains_iran}
    assert "institution:fed" not in dests_iran
    assert dests_iran.get("indicator:inflation") == "asset:crude-oil"

    # B3: 인플레이션 기준 상류 탐색 시 이란 → 원유 → 인플레이션 방향 유지 (역전 금지), 분기(은행) 제외
    chains_inf = wo.two_hop_chains(ontology, "indicator:inflation")
    dests_inf = {c[3]: c[2] for c in chains_inf}
    assert "sector:banks" not in dests_inf
    assert dests_inf.get("region:iran") == "asset:crude-oil"
    section_inf = wo.context_section("인플레이션 원인 정리", root=tmp_path)
    assert "이란 → 원유 [호재·상승 요인] → 인플레이션 [호재·상승 요인]" in section_inf
    assert "인플레이션 → 원유" not in section_inf
    assert "은행" not in section_inf
