from __future__ import annotations

import json
import os
import sqlite3
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKUP_SCRIPT = REPO_ROOT / "scripts" / "backup_stock_data.sh"
RESTORE_SCRIPT = REPO_ROOT / "scripts" / "restore_stock_data.sh"


def _create_wal_sqlite(path: Path, table: str, rows: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, val TEXT NOT NULL)")
    conn.executemany(f"INSERT INTO {table} (id, val) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


def _build_fixture_tree(tmp_path: Path) -> dict[str, Path]:
    home = tmp_path / "home"
    state_dir = home / ".local" / "share" / "stock-report"
    shared_dir = state_dir / "shared-memory"
    onto_dir = shared_dir / "llm-wiki" / "ontology"
    repo_dir = home / "projects" / "stock-report"
    dest_dir = home / "backups" / "stock-report"
    cfg_dir = home / ".config" / "stock-report"

    for d in (state_dir, shared_dir, onto_dir, repo_dir, dest_dir, cfg_dir):
        d.mkdir(parents=True, exist_ok=True)

    _create_wal_sqlite(state_dir / "stock_report.db", "trades", [(1, "NVDA"), (2, "005930")])
    _create_wal_sqlite(state_dir / "agent_console.sqlite3", "sessions", [(1, "s1")])
    _create_wal_sqlite(shared_dir / "world_issue_log.sqlite3", "issues", [(1, "macro-1")])
    _create_wal_sqlite(shared_dir / "wiki_usage.sqlite", "wiki_usage", [(1, "page-1")])

    (repo_dir / "portfolio_snapshot.json").write_text('{"positions": ["NVDA"]}\n', encoding="utf-8")
    (repo_dir / "dca_weights.json").write_text('{"QQQ": 0.5}\n', encoding="utf-8")
    (repo_dir / "target_weights.json").write_text('{"SPY": 0.5}\n', encoding="utf-8")

    (shared_dir / "events.jsonl").write_text('{"id":"ev-1","kind":"wiki"}\n', encoding="utf-8")
    (shared_dir / "index.json").write_text('{"pages":1}\n', encoding="utf-8")
    (shared_dir / "evidence_usage.jsonl").write_text('{"event":"retrieval","ok":true}\n', encoding="utf-8")
    (onto_dir / "ontology.json").write_text('{"entities":10}\n', encoding="utf-8")
    (onto_dir / "extractions.json").write_text('{"pages":5}\n', encoding="utf-8")
    (onto_dir / "ontology.md").write_text("# Ontology\n", encoding="utf-8")

    return {
        "home": home,
        "state_dir": state_dir,
        "shared_dir": shared_dir,
        "onto_dir": onto_dir,
        "repo_dir": repo_dir,
        "dest_dir": dest_dir,
        "cfg_dir": cfg_dir,
    }


def _base_env(paths: dict[str, Path], extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(paths["home"]),
            "BACKUP_STATE_DIR": str(paths["state_dir"]),
            "BACKUP_REPO_DIR": str(paths["repo_dir"]),
            "BACKUP_DEST": str(paths["dest_dir"]),
            "BACKUP_ENV_FILE": str(paths["cfg_dir"] / "backup.env"),
            "BACKUP_STAMP": "20261122",
            "BACKUP_COMPRESSION": "gzip",
        }
    )
    if extra:
        env.update(extra)
    return env


def _install_mock_rclone(tmp_path: Path, *, fail_copyto: bool = False) -> tuple[Path, Path, Path]:
    mock_drive = tmp_path / "mock_drive"
    mock_drive.mkdir(parents=True, exist_ok=True)
    call_log = tmp_path / "rclone_calls.log"
    rclone_bin = tmp_path / "mock_rclone.sh"
    fail_flag = "1" if fail_copyto else "0"
    rclone_bin.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
DRIVE_DIR="{mock_drive}"
LOG_FILE="{call_log}"
echo "$*" >> "$LOG_FILE"
cmd="${{1:-}}"
shift || true
case "$cmd" in
  copyto)
    if [[ "{fail_flag}" == "1" ]]; then
      echo "simulated upload failure" >&2
      exit 1
    fi
    src="$1"
    dst="$2"
    fname="${{dst##*/}}"
    cp "$src" "$DRIVE_DIR/$fname"
    ;;
  lsf)
    shift || true
    inc=""
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == "--include" ]]; then
        inc="${{2:-}}"
        shift 2
      else
        shift
      fi
    done
    if [[ -n "$inc" && -f "$DRIVE_DIR/$inc" ]]; then
      sz="$(stat -c '%s' "$DRIVE_DIR/$inc")"
      echo "${{sz}};${{inc}}"
    fi
    ;;
  delete)
    # Record delete call; succeed without error.
    exit 0
    ;;
  *)
    echo "unsupported mock rclone subcommand: $cmd" >&2
    exit 2
    ;;
esac
""",
        encoding="utf-8",
    )
    rclone_bin.chmod(0o755)
    return rclone_bin, mock_drive, call_log


def test_online_sqlite_backup_and_full_manifest(tmp_path: Path) -> None:
    paths = _build_fixture_tree(tmp_path)

    # Hold an open WAL write transaction while backup runs to confirm online backup safety.
    live_conn = sqlite3.connect(str(paths["state_dir"] / "stock_report.db"))
    live_conn.execute("BEGIN IMMEDIATE")
    live_conn.execute("INSERT INTO trades (id, val) VALUES (99, 'UNCOMMITTED')")

    try:
        res = subprocess.run(
            ["bash", str(BACKUP_SCRIPT)],
            env=_base_env(paths),
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        live_conn.rollback()
        live_conn.close()

    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    archive = paths["dest_dir"] / "stock-data-20261122.tar.gz"
    assert archive.exists()
    assert stat.S_IMODE(archive.stat().st_mode) == 0o600

    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(extract_dir)

    expected = [
        "sqlite/stock_report.db",
        "sqlite/agent_console.sqlite3",
        "sqlite/world_issue_log.sqlite3",
        "sqlite/wiki_usage.sqlite",
        "config/portfolio_snapshot.json",
        "config/dca_weights.json",
        "config/target_weights.json",
        "wiki/events.jsonl",
        "wiki/index.json",
        "wiki/evidence_usage.jsonl",
        "wiki/ontology/ontology.json",
        "wiki/ontology/extractions.json",
        "wiki/ontology/ontology.md",
        "MANIFEST.json",
        "SHA256SUMS",
    ]
    for rel in expected:
        assert (extract_dir / rel).is_file(), f"missing {rel} in archive"

    manifest = json.loads((extract_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["stamp"] == "20261122"
    assert "sqlite/stock_report.db" in manifest["files"]

    # Verify backed-up stock_report.db passes quick_check and contains only committed rows.
    restored_conn = sqlite3.connect(str(extract_dir / "sqlite" / "stock_report.db"))
    assert restored_conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    rows = restored_conn.execute("SELECT id, val FROM trades ORDER BY id").fetchall()
    restored_conn.close()
    assert rows == [(1, "NVDA"), (2, "005930")]


def test_corrupted_sqlite_fails_closed_without_pruning_old_backups(tmp_path: Path) -> None:
    paths = _build_fixture_tree(tmp_path)
    for day in range(1, 6):
        (paths["dest_dir"] / f"stock-data-2026110{day}.tar.gz").write_bytes(b"old-backup")

    # Corrupt primary SQLite DB.
    (paths["state_dir"] / "stock_report.db").write_bytes(b"NOT A VALID SQLITE FILE" * 100)

    res = subprocess.run(
        ["bash", str(BACKUP_SCRIPT)],
        env=_base_env(paths, {"BACKUP_LOCAL_KEEP": "2"}),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode != 0
    assert not (paths["dest_dir"] / "stock-data-20261122.tar.gz").exists()
    assert not list(paths["dest_dir"].glob("*.part"))
    # All 5 pre-existing archives must remain untouched!
    assert len(list(paths["dest_dir"].glob("stock-data-2026110*.tar.gz"))) == 5


def test_local_rotation_keeps_configured_count(tmp_path: Path) -> None:
    paths = _build_fixture_tree(tmp_path)
    for idx, day in enumerate(("20261101", "20261102", "20261103", "20261104")):
        p = paths["dest_dir"] / f"stock-data-{day}.tar.gz"
        p.write_bytes(b"dummy")
        ts = 1700000000 + idx * 100
        os.utime(p, (ts, ts))

    res = subprocess.run(
        ["bash", str(BACKUP_SCRIPT)],
        env=_base_env(paths, {"BACKUP_LOCAL_KEEP": "3"}),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, res.stderr
    remaining = sorted(p.name for p in paths["dest_dir"].glob("stock-data-*.tar.*"))
    assert remaining == [
        "stock-data-20261103.tar.gz",
        "stock-data-20261104.tar.gz",
        "stock-data-20261122.tar.gz",
    ]


def test_encrypted_gdrive_upload_and_restore_verify_roundtrip(tmp_path: Path) -> None:
    paths = _build_fixture_tree(tmp_path)
    pass_file = paths["cfg_dir"] / "backup.pass"
    pass_file.write_text("super-secret-test-passphrase-123456\n", encoding="utf-8")
    pass_file.chmod(0o600)

    rclone_bin, mock_drive, call_log = _install_mock_rclone(tmp_path)
    env = _base_env(
        paths,
        {
            "BACKUP_GDRIVE_REMOTE": "gdrive:hermes-stock-report",
            "BACKUP_PASSPHRASE_FILE": str(pass_file),
            "BACKUP_GDRIVE_RETENTION_DAYS": "30",
            "RCLONE_BIN": str(rclone_bin),
        },
    )

    res = subprocess.run(["bash", str(BACKUP_SCRIPT)], env=env, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"

    remote_gpg = mock_drive / "stock-data-20261122.tar.gz.gpg"
    assert remote_gpg.is_file()
    assert remote_gpg.stat().st_size > 100
    # Must NOT be plaintext gzip/tar.
    with pytest.raises(tarfile.TarError):
        with tarfile.open(remote_gpg, "r:gz"):
            pass

    # No leftover .gpg temp files in local BACKUP_DEST.
    assert not list(paths["dest_dir"].glob("*.gpg"))

    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert any(c.startswith("copyto ") and "gdrive:hermes-stock-report/stock-data-20261122.tar.gz.gpg" in c for c in calls)
    assert any(c.startswith("lsf ") and "--format sp --include stock-data-20261122.tar.gz.gpg" in c for c in calls)
    assert any(c.startswith("delete ") and "--min-age 30d" in c for c in calls)

    # Restore script --verify must decrypt, verify SHA256SUMS, and PRAGMA quick_check all SQLite DBs.
    verify_res = subprocess.run(
        ["bash", str(RESTORE_SCRIPT), "--verify", str(remote_gpg)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert verify_res.returncode == 0, f"stdout={verify_res.stdout}\nstderr={verify_res.stderr}"
    assert "verify ok" in verify_res.stdout.lower()

    # Restore script --extract must recover intact files.
    out_dir = tmp_path / "restored_from_drive"
    extract_res = subprocess.run(
        ["bash", str(RESTORE_SCRIPT), "--extract", str(remote_gpg), str(out_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert extract_res.returncode == 0, f"stdout={extract_res.stdout}\nstderr={extract_res.stderr}"
    assert (out_dir / "wiki" / "ontology" / "ontology.json").is_file()
    conn = sqlite3.connect(str(out_dir / "sqlite" / "stock_report.db"))
    assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 2
    conn.close()


def test_gdrive_refuses_missing_or_insecure_passphrase(tmp_path: Path) -> None:
    paths = _build_fixture_tree(tmp_path)
    rclone_bin, _mock_drive, call_log = _install_mock_rclone(tmp_path)

    # Case A: missing passphrase file -> fail closed, rclone never invoked.
    missing_pass = paths["cfg_dir"] / "does_not_exist.pass"
    res_missing = subprocess.run(
        ["bash", str(BACKUP_SCRIPT)],
        env=_base_env(
            paths,
            {
                "BACKUP_GDRIVE_REMOTE": "gdrive:hermes-stock-report",
                "BACKUP_PASSPHRASE_FILE": str(missing_pass),
                "RCLONE_BIN": str(rclone_bin),
            },
        ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res_missing.returncode != 0
    assert not call_log.exists()

    # Case B: world-readable 0644 passphrase file -> fail closed, rclone never invoked.
    weak_pass = paths["cfg_dir"] / "weak.pass"
    weak_pass.write_text("weak-perm-passphrase\n", encoding="utf-8")
    weak_pass.chmod(0o644)
    res_weak = subprocess.run(
        ["bash", str(BACKUP_SCRIPT)],
        env=_base_env(
            paths,
            {
                "BACKUP_GDRIVE_REMOTE": "gdrive:hermes-stock-report",
                "BACKUP_PASSPHRASE_FILE": str(weak_pass),
                "RCLONE_BIN": str(rclone_bin),
            },
        ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res_weak.returncode != 0
    assert not call_log.exists()


def test_gdrive_upload_failure_skips_remote_prune(tmp_path: Path) -> None:
    paths = _build_fixture_tree(tmp_path)
    pass_file = paths["cfg_dir"] / "backup.pass"
    pass_file.write_text("super-secret-test-passphrase-123456\n", encoding="utf-8")
    pass_file.chmod(0o600)

    rclone_bin, _mock_drive, call_log = _install_mock_rclone(tmp_path, fail_copyto=True)
    res = subprocess.run(
        ["bash", str(BACKUP_SCRIPT)],
        env=_base_env(
            paths,
            {
                "BACKUP_GDRIVE_REMOTE": "gdrive:hermes-stock-report",
                "BACKUP_PASSPHRASE_FILE": str(pass_file),
                "RCLONE_BIN": str(rclone_bin),
            },
        ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode != 0
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert any(c.startswith("copyto ") for c in calls)
    assert not any(c.startswith("delete ") for c in calls)


def test_backup_env_file_loading_and_zstd_if_available(tmp_path: Path) -> None:
    import shutil

    paths = _build_fixture_tree(tmp_path)
    pass_file = paths["cfg_dir"] / "backup.pass"
    pass_file.write_text("env-file-passphrase-987654\n", encoding="utf-8")
    pass_file.chmod(0o600)

    rclone_bin, mock_drive, _call_log = _install_mock_rclone(tmp_path)
    has_zstd = shutil.which("zstd") is not None
    comp = "zstd" if has_zstd else "gzip"
    ext = "tar.zst" if has_zstd else "tar.gz"

    env_file = paths["cfg_dir"] / "backup.env"
    env_file.write_text(
        f"""# Test backup env
BACKUP_GDRIVE_REMOTE="gdrive:hermes-stock-report"
BACKUP_PASSPHRASE_FILE="{pass_file}"
BACKUP_GDRIVE_RETENTION_DAYS=30
BACKUP_COMPRESSION={comp}
RCLONE_BIN="{rclone_bin}"
""",
        encoding="utf-8",
    )
    env_file.chmod(0o600)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(paths["home"]),
            "BACKUP_STATE_DIR": str(paths["state_dir"]),
            "BACKUP_REPO_DIR": str(paths["repo_dir"]),
            "BACKUP_DEST": str(paths["dest_dir"]),
            "BACKUP_ENV_FILE": str(env_file),
            "BACKUP_STAMP": "20261123",
        }
    )
    for k in ("BACKUP_GDRIVE_REMOTE", "BACKUP_PASSPHRASE_FILE", "BACKUP_COMPRESSION", "RCLONE_BIN"):
        env.pop(k, None)

    res = subprocess.run(["bash", str(BACKUP_SCRIPT)], env=env, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"

    remote_gpg = mock_drive / f"stock-data-20261123.{ext}.gpg"
    assert remote_gpg.is_file()

    verify_env = env.copy()
    verify_env["BACKUP_PASSPHRASE_FILE"] = str(pass_file)
    verify_res = subprocess.run(
        ["bash", str(RESTORE_SCRIPT), "--verify", str(remote_gpg)],
        env=verify_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert verify_res.returncode == 0, f"stdout={verify_res.stdout}\nstderr={verify_res.stderr}"
    assert "4 SQLite DB(s) quick_check ok" in verify_res.stdout

