#!/usr/bin/env bash
# 핵심 데이터 일일 백업 — SQLite 온라인 백업(quick_check) + 포트폴리오 설정 JSON + 위키 핵심 원장/온톨로지.
# 로컬 보존 기본 14일(rotate), BACKUP_GDRIVE_REMOTE 설정 시 GPG AES-256 암호화 후 Google Drive(rclone) 업로드 + 30일 보존.
# 크론: deploy/crontab.stock-report (매일 20:40 UTC = 05:40 KST)
set -euo pipefail
umask 077

# ~/.config/stock-report/backup.env 등에서 미설정 환경변수만 안전하게 로드 (명시적 env override 우선)
BACKUP_ENV_FILE="${BACKUP_ENV_FILE:-$HOME/.config/stock-report/backup.env}"
if [[ -f "$BACKUP_ENV_FILE" ]]; then
  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line#"${raw_line%%[![:space:]]*}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" != *=* ]] && continue
    key="${line%%=*}"
    val="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    # 양끝 따옴표 제거
    if [[ "$val" =~ ^\"(.*)\"$ ]]; then
      val="${BASH_REMATCH[1]}"
    elif [[ "$val" =~ ^\'(.*)\'$ ]]; then
      val="${BASH_REMATCH[1]}"
    fi
    if [[ -z "${!key+x}" ]]; then
      export "$key=$val"
    fi
  done < "$BACKUP_ENV_FILE"
fi

STATE_DIR="${BACKUP_STATE_DIR:-$HOME/.local/share/stock-report}"
SHARED_DIR="${BACKUP_SHARED_DIR:-$STATE_DIR/shared-memory}"
REPO_DIR="${BACKUP_REPO_DIR:-$HOME/projects/stock-report}"
DEST="${BACKUP_DEST:-$HOME/backups/stock-report}"
STAMP="${BACKUP_STAMP:-$(date -u +%Y%m%d)}"
LOCAL_KEEP="${BACKUP_LOCAL_KEEP:-14}"
GDRIVE_REMOTE="${BACKUP_GDRIVE_REMOTE:-}"
PASSPHRASE_FILE="${BACKUP_PASSPHRASE_FILE:-$HOME/.config/stock-report/backup.pass}"
GDRIVE_RETENTION_DAYS="${BACKUP_GDRIVE_RETENTION_DAYS:-30}"
RCLONE_BIN="${RCLONE_BIN:-rclone}"
PYTHON_BIN="${BACKUP_PYTHON_BIN:-python3}"

COMPRESSION="${BACKUP_COMPRESSION:-auto}"
if [[ "$COMPRESSION" == "auto" ]]; then
  if command -v zstd >/dev/null 2>&1; then
    COMPRESSION="zstd"
  else
    COMPRESSION="gzip"
  fi
fi

case "$COMPRESSION" in
  zstd) ARCHIVE_EXT="tar.zst" ;;
  gzip) ARCHIVE_EXT="tar.gz" ;;
  *)
    echo "ERROR: unsupported BACKUP_COMPRESSION=$COMPRESSION (expected zstd|gzip|auto)" >&2
    exit 1
    ;;
esac

mkdir -p "$DEST"
STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/stock-backup-${STAMP}.XXXXXX")"
ARCHIVE_FINAL="$DEST/stock-data-${STAMP}.${ARCHIVE_EXT}"
ARCHIVE_PART="$DEST/.stock-data-${STAMP}.${ARCHIVE_EXT}.part"

cleanup() {
  if [[ -n "${GNUPG_TMP:-}" && -d "${GNUPG_TMP:-}" ]] && command -v gpgconf >/dev/null 2>&1; then
    GNUPGHOME="$GNUPG_TMP" gpgconf --kill gpg-agent >/dev/null 2>&1 || true
  fi
  rm -rf "$STAGE_DIR"
  rm -f "$ARCHIVE_PART"
}
trap cleanup EXIT

# ── 1. 스테이징: SQLite 온라인 백업 + JSON/위키 핵심 원장 복사 + SHA256/MANIFEST ──
"$PYTHON_BIN" - "$STAGE_DIR" "$STATE_DIR" "$SHARED_DIR" "$REPO_DIR" "$STAMP" "$COMPRESSION" <<'PY'
from __future__ import annotations

import hashlib
import json
import shutil
import socket
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

stage = Path(sys.argv[1])
state_dir = Path(sys.argv[2])
shared_dir = Path(sys.argv[3])
repo_dir = Path(sys.argv[4])
stamp = sys.argv[5]
compression = sys.argv[6]

included: list[str] = []
skipped: list[str] = []


def backup_sqlite(src_path: Path, rel_dst: str, *, required: bool) -> None:
    if not src_path.exists():
        if required:
            raise RuntimeError(f"required SQLite DB missing: {src_path}")
        skipped.append(rel_dst)
        return
    dst_path = stage / rel_dst
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True, timeout=60)
    dst = sqlite3.connect(str(dst_path))
    try:
        src.backup(dst)
        row = dst.execute("PRAGMA quick_check").fetchone()
        if not row or row[0] != "ok":
            raise RuntimeError(f"SQLite quick_check failed for {src_path}: {row}")
    finally:
        dst.close()
        src.close()
    included.append(rel_dst)


def copy_file(src_path: Path, rel_dst: str) -> None:
    if not src_path.is_file():
        skipped.append(rel_dst)
        return
    dst_path = stage / rel_dst
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    included.append(rel_dst)


# 핵심 SQLite 4종 (stock_report.db는 필수, 나머지는 존재 시 무결성 검증 후 백업)
backup_sqlite(state_dir / "stock_report.db", "sqlite/stock_report.db", required=True)
backup_sqlite(state_dir / "agent_console.sqlite3", "sqlite/agent_console.sqlite3", required=False)
backup_sqlite(shared_dir / "world_issue_log.sqlite3", "sqlite/world_issue_log.sqlite3", required=False)
backup_sqlite(shared_dir / "wiki_usage.sqlite", "sqlite/wiki_usage.sqlite", required=False)

# 포트폴리오·가중치 설정 JSON
copy_file(repo_dir / "portfolio_snapshot.json", "config/portfolio_snapshot.json")
copy_file(repo_dir / "dca_weights.json", "config/dca_weights.json")
copy_file(repo_dir / "target_weights.json", "config/target_weights.json")

# LLM 위키 핵심 원장 및 온톨로지
copy_file(shared_dir / "events.jsonl", "wiki/events.jsonl")
copy_file(shared_dir / "index.json", "wiki/index.json")
copy_file(shared_dir / "evidence_usage.jsonl", "wiki/evidence_usage.jsonl")
onto_dir = shared_dir / "llm-wiki" / "ontology"
copy_file(onto_dir / "ontology.json", "wiki/ontology/ontology.json")
copy_file(onto_dir / "extractions.json", "wiki/ontology/extractions.json")
copy_file(onto_dir / "ontology.md", "wiki/ontology/ontology.md")

included.sort()
manifest = {
    "schema_version": 1,
    "stamp": stamp,
    "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "hostname": socket.gethostname(),
    "compression": compression,
    "files": included,
    "skipped": sorted(skipped),
}
(stage / "MANIFEST.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

checksum_lines: list[str] = []
for rel in [*included, "MANIFEST.json"]:
    h = hashlib.sha256()
    with (stage / rel).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    checksum_lines.append(f"{h.hexdigest()}  {rel}")
(stage / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
PY

# ── 2. 원자적 압축 아카이브 생성 + 읽기 검증 ──────────────────────────────────
if [[ "$COMPRESSION" == "zstd" ]]; then
  tar -C "$STAGE_DIR" -cf - . | zstd -q -T2 -3 -o "$ARCHIVE_PART"
  zstd -dc "$ARCHIVE_PART" | tar -tf - >/dev/null
else
  tar -C "$STAGE_DIR" -czf "$ARCHIVE_PART" .
  tar -tzf "$ARCHIVE_PART" >/dev/null
fi

chmod 600 "$ARCHIVE_PART"
mv -f "$ARCHIVE_PART" "$ARCHIVE_FINAL"

# ── 3. 로컬 회전 (검증 완료 후에만 실행) ──────────────────────────────────────
if [[ "$LOCAL_KEEP" =~ ^[0-9]+$ ]] && (( LOCAL_KEEP > 0 )); then
  ls -1t "$DEST"/stock-data-*.tar.* 2>/dev/null | tail -n +"$((LOCAL_KEEP + 1))" | xargs -r rm -f
fi

ARCHIVE_SIZE="$(du -h "$ARCHIVE_FINAL" | cut -f1)"
echo "backup ok: $ARCHIVE_FINAL ($ARCHIVE_SIZE)"

# ── 4. Google Drive 암호화 업로드 (BACKUP_GDRIVE_REMOTE 설정 시) ──────────────
if [[ -n "$GDRIVE_REMOTE" ]]; then
  # Fail-closed: 패스프레이즈 파일 존재·비어있지 않음·권한(0600/0400) 엄격 검사
  if [[ ! -f "$PASSPHRASE_FILE" || ! -s "$PASSPHRASE_FILE" ]]; then
    echo "ERROR: passphrase file missing or empty: $PASSPHRASE_FILE (refusing unencrypted upload)" >&2
    exit 1
  fi
  PASS_MODE="$(stat -c '%a' "$PASSPHRASE_FILE")"
  if [[ "$PASS_MODE" != "600" && "$PASS_MODE" != "400" ]]; then
    echo "ERROR: insecure passphrase file permissions ($PASS_MODE) on $PASSPHRASE_FILE (expected 600 or 400)" >&2
    exit 1
  fi
  if ! command -v gpg >/dev/null 2>&1; then
    echo "ERROR: gpg not found (refusing unencrypted upload)" >&2
    exit 1
  fi
  if ! command -v "$RCLONE_BIN" >/dev/null 2>&1 && [[ ! -x "$RCLONE_BIN" ]]; then
    echo "ERROR: rclone not found: $RCLONE_BIN" >&2
    exit 1
  fi

  GNUPG_TMP="$STAGE_DIR/gnupg"
  mkdir -p "$GNUPG_TMP"
  chmod 700 "$GNUPG_TMP"

  REMOTE_NAME="$(basename "$ARCHIVE_FINAL").gpg"
  GPG_TMP="$STAGE_DIR/$REMOTE_NAME"
  GPG_EXTRA=()
  if command -v gpg-agent >/dev/null 2>&1; then
    GPG_EXTRA+=(--agent-program "$(command -v gpg-agent)")
  fi
  GNUPGHOME="$GNUPG_TMP" gpg \
    "${GPG_EXTRA[@]}" \
    --batch --yes --quiet \
    --symmetric --cipher-algo AES256 --s2k-digest-algo SHA512 \
    --pinentry-mode loopback \
    --passphrase-file "$PASSPHRASE_FILE" \
    --output "$GPG_TMP" \
    "$ARCHIVE_FINAL"
  chmod 600 "$GPG_TMP"

  REMOTE_TRIMMED="${GDRIVE_REMOTE%/}"
  "$RCLONE_BIN" copyto "$GPG_TMP" "${REMOTE_TRIMMED}/${REMOTE_NAME}"

  # 원격 파일 존재 및 바이트 크기 일치 검증 (Google Drive 디렉터리 기준 lsf)
  LOCAL_BYTES="$(stat -c '%s' "$GPG_TMP")"
  REMOTE_INFO="$("$RCLONE_BIN" lsf "$REMOTE_TRIMMED" --files-only --format sp --include "$REMOTE_NAME" 2>/dev/null || true)"
  if [[ "$REMOTE_INFO" != *"${LOCAL_BYTES};${REMOTE_NAME}"* ]]; then
    echo "ERROR: remote verification failed for ${REMOTE_TRIMMED}/${REMOTE_NAME} (expected=${LOCAL_BYTES};${REMOTE_NAME}, got=${REMOTE_INFO:-<empty>})" >&2
    exit 1
  fi

  # 업로드·검증 성공 후에만 원격 만료 파일 정리
  if [[ "$GDRIVE_RETENTION_DAYS" =~ ^[0-9]+$ ]] && (( GDRIVE_RETENTION_DAYS > 0 )); then
    "$RCLONE_BIN" delete "$REMOTE_TRIMMED" \
      --min-age "${GDRIVE_RETENTION_DAYS}d" \
      --include "stock-data-*.tar.*.gpg"
  fi

  GPG_SIZE="$(du -h "$GPG_TMP" | cut -f1)"
  echo "gdrive ok: ${REMOTE_TRIMMED}/${REMOTE_NAME} ($GPG_SIZE, retention=${GDRIVE_RETENTION_DAYS}d)"
fi
