#!/usr/bin/env bash
# 백업 아카이브 복호화·무결성 검증·추출 도구.
# 사용법:
#   scripts/restore_stock_data.sh --verify <archive(.tar.zst|.tar.gz|.gpg)>
#   scripts/restore_stock_data.sh --extract <archive(.tar.zst|.tar.gz|.gpg)> <target_dir>
set -euo pipefail
umask 077

MODE="${1:-}"
ARCHIVE="${2:-}"
TARGET_DIR="${3:-}"
PASSPHRASE_FILE="${BACKUP_PASSPHRASE_FILE:-$HOME/.config/stock-report/backup.pass}"
PYTHON_BIN="${BACKUP_PYTHON_BIN:-python3}"

usage() {
  echo "Usage:" >&2
  echo "  $0 --verify <archive>" >&2
  echo "  $0 --extract <archive> <target_dir>" >&2
  exit 2
}

[[ -n "$MODE" && -n "$ARCHIVE" && -f "$ARCHIVE" ]] || usage
case "$MODE" in
  --verify) ;;
  --extract) [[ -n "$TARGET_DIR" ]] || usage ;;
  *) usage ;;
esac

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/stock-restore.XXXXXX")"
cleanup() {
  if [[ -n "${GNUPG_TMP:-}" && -d "${GNUPG_TMP:-}" ]] && command -v gpgconf >/dev/null 2>&1; then
    GNUPGHOME="$GNUPG_TMP" gpgconf --kill gpg-agent >/dev/null 2>&1 || true
  fi
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

TAR_SRC="$ARCHIVE"
if [[ "$ARCHIVE" == *.gpg ]]; then
  GNUPG_TMP="$WORK_DIR/gnupg"
  mkdir -p "$GNUPG_TMP"
  chmod 700 "$GNUPG_TMP"
  GPG_EXTRA=()
  if command -v gpg-agent >/dev/null 2>&1; then
    GPG_EXTRA+=(--agent-program "$(command -v gpg-agent)")
  fi
  TAR_SRC="$WORK_DIR/$(basename "${ARCHIVE%.gpg}")"
  if [[ -f "$PASSPHRASE_FILE" && -s "$PASSPHRASE_FILE" ]]; then
    GNUPGHOME="$GNUPG_TMP" gpg \
      "${GPG_EXTRA[@]}" \
      --batch --yes --quiet \
      --decrypt --pinentry-mode loopback \
      --passphrase-file "$PASSPHRASE_FILE" \
      --output "$TAR_SRC" \
      "$ARCHIVE"
  else
    GNUPGHOME="$GNUPG_TMP" gpg "${GPG_EXTRA[@]}" --decrypt --output "$TAR_SRC" "$ARCHIVE"
  fi
fi

EXTRACT_ROOT="$WORK_DIR/extracted"
if [[ "$MODE" == "--extract" ]]; then
  mkdir -p "$TARGET_DIR"
  EXTRACT_ROOT="$TARGET_DIR"
else
  mkdir -p "$EXTRACT_ROOT"
fi

"$PYTHON_BIN" - "$TAR_SRC" "$EXTRACT_ROOT" <<'PY'
from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path

tar_src = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve()
root.mkdir(parents=True, exist_ok=True)


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    for member in tf:
        name = member.name.lstrip("./")
        if not name or name == ".":
            continue
        if member.issym() or member.islnk() or member.isdev():
            raise SystemExit(f"ERROR: refusing unsafe tar member type: {member.name}")
        target = (dest / name).resolve()
        if dest not in target.parents and target != dest:
            raise SystemExit(f"ERROR: path traversal detected in archive: {member.name}")
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            src_fh = tf.extractfile(member)
            if src_fh is None:
                raise SystemExit(f"ERROR: unreadable tar member: {member.name}")
            with src_fh, target.open("wb") as out_fh:
                while True:
                    chunk = src_fh.read(1024 * 1024)
                    if not chunk:
                        break
                    out_fh.write(chunk)


if tar_src.name.endswith(".tar.zst"):
    proc = subprocess.Popen(["zstd", "-dc", str(tar_src)], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            _safe_extract(tf, root)
    finally:
        proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            raise SystemExit(f"ERROR: zstd decompression failed with code {rc}")
elif tar_src.name.endswith((".tar.gz", ".tgz")):
    with tarfile.open(tar_src, mode="r:gz") as tf:
        _safe_extract(tf, root)
else:
    raise SystemExit(f"ERROR: unsupported archive format: {tar_src}")

sums_file = root / "SHA256SUMS"
if not sums_file.is_file():
    raise SystemExit("ERROR: SHA256SUMS missing in archive")

verified_files = 0
for raw in sums_file.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    expected_hex, rel = raw.split("  ", 1)
    target = root / rel
    if not target.is_file():
        raise SystemExit(f"ERROR: missing file listed in SHA256SUMS: {rel}")
    h = hashlib.sha256()
    with target.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != expected_hex:
        raise SystemExit(f"ERROR: checksum mismatch for {rel}")
    verified_files += 1

sqlite_checked = 0
sqlite_dir = root / "sqlite"
if sqlite_dir.is_dir():
    for db in sorted(sqlite_dir.iterdir()):
        if not db.is_file():
            continue
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = conn.execute("PRAGMA quick_check").fetchone()
            if not row or row[0] != "ok":
                raise SystemExit(f"ERROR: PRAGMA quick_check failed for {db.name}: {row}")
        finally:
            conn.close()
        sqlite_checked += 1

print(f"verify ok: {verified_files} files checksum-verified, {sqlite_checked} SQLite DB(s) quick_check ok")
PY
