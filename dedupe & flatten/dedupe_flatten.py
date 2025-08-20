#!/usr/bin/env python3
import argparse
import os
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

# ----------------------------
# Helpers
# ----------------------------
def human_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024

def safe_unique_path(dst_dir: Path, name: str) -> Path:
    base = Path(name).stem
    ext = Path(name).suffix
    candidate = dst_dir / name
    i = 1
    while candidate.exists():
        candidate = dst_dir / f"{base} ({i}){ext}"
        i += 1
    return candidate

def file_hash(path: Path, algo: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    """Strong content hash of whole file (streamed)."""
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def quick_fingerprint(path: Path, head_bytes: int = 64 * 1024, tail_bytes: int = 64 * 1024) -> Tuple[int, bytes, bytes]:
    """
    Lightweight fingerprint: size + first N bytes + last N bytes.
    Used to avoid full hashing unless necessary.
    """
    size = path.stat().st_size
    head = b""
    tail = b""
    with path.open("rb") as f:
        head = f.read(min(head_bytes, size))
        if size > tail_bytes:
            try:
                f.seek(max(0, size - tail_bytes))
            except Exception:
                f.seek(0)
                f.read(max(0, size - tail_bytes))
            tail = f.read(tail_bytes)
        else:
            tail = head[-tail_bytes:]
    return size, head, tail

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate files safely, with hash confirmation, then flatten into one folder, with a detailed log."
    )
    parser.add_argument("root", type=Path, help="Root directory to process")
    parser.add_argument(
        "--out-folder",
        default="_consolidated",
        help="Folder (created inside root) where remaining files will be placed",
    )
    parser.add_argument(
        "--log-file",
        default="dedupe_log.txt",
        help="Name of the log file (created inside root)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform deletions and moves. Omit for a dry run.",
    )
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        help="Treat filenames case-insensitively when comparing (useful on Windows/macOS).",
    )
    parser.add_argument(
        "--mode",
        choices=["name-size-hash", "content"],
        default="name-size-hash",
        help=(
            "name-size-hash: only files with same name+size are considered; delete if hashes match. "
            "content: ignore names; files with identical content (size+hash) are duplicates."
        ),
    )
    parser.add_argument(
        "--hash",
        choices=["sha256","blake2b","md5"],
        default="sha256",
        help="Hash algorithm for content confirmation (sha256 recommended).",
    )
    parser.add_argument(
        "--use-quick-fingerprint",
        action="store_true",
        help="Speed up by pre-checking with first/last 64KB before full hashing. Full hash still used to confirm deletion.",
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root directory does not exist or is not a directory: {root}")

    out_dir = (root / args.out_folder).resolve()
    log_path = (root / args.log_file).resolve()

    # Skip our own artifacts
    explicit_skip_files = {log_path}

    # Collect files
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        # Skip the consolidation folder
        if out_dir in [d] or str(d).startswith(str(out_dir)):
            continue
        for fname in filenames:
            p = d / fname
            try:
                if p.resolve() in explicit_skip_files:
                    continue
                if out_dir in p.resolve().parents:
                    continue
                if p.is_symlink() or (not p.is_file()):
                    continue
                _ = p.stat().st_size  # ensure accessible
            except (OSError, FileNotFoundError):
                continue
            all_files.append(p)

    actions_delete = []  # (path, reason)
    actions_move = []    # (src, dst)
    notes = []           # additional log notes

    # Index & dedupe
    hash_cache: Dict[Path, str] = {}
    fp_cache: Dict[Path, Tuple[int, bytes, bytes]] = {}

    def get_hash(p: Path) -> Optional[str]:
        if p in hash_cache:
            return hash_cache[p]
        try:
            h = file_hash(p, algo=args.hash)
            hash_cache[p] = h
            return h
        except Exception as e:
            notes.append(f"! ERROR hashing {p}: {e}")
            return None

    def get_fp(p: Path) -> Optional[Tuple[int, bytes, bytes]]:
        if p in fp_cache:
            return fp_cache[p]
        try:
            fp = quick_fingerprint(p)
            fp_cache[p] = fp
            return fp
        except Exception as e:
            notes.append(f"! ERROR fingerprinting {p}: {e}")
            return None

    kept_map = {}  # key -> kept_path (key depends on mode)

    if args.mode == "name-size-hash":
        # Step 1: group by (name_key, size). Only within a group do we compute hashes.
        def name_key_for(p: Path) -> str:
            return p.name.lower() if args.case_insensitive else p.name

        groups: Dict[Tuple[str, int], list] = {}
        for p in all_files:
            try:
                sz = p.stat().st_size
            except Exception:
                continue
            k = (name_key_for(p), sz)
            groups.setdefault(k, []).append(p)

        # For each group, keep the first unique content; delete true content-duplicates
        for (nkey, sz), files in groups.items():
            if len(files) == 1:
                kept_map[(nkey, sz, "first")] = files[0]
                continue

            # Optional quick filter: partition by quick fingerprint
            if args.use_quick_fingerprint:
                fp_groups: Dict[Tuple[int, bytes, bytes], list] = {}
                for p in files:
                    fp = get_fp(p)
                    if fp is None:
                        fp_groups.setdefault((id(p), b"", b""), []).append(p)
                    else:
                        fp_groups.setdefault(fp, []).append(p)
                candidate_buckets = fp_groups.values()
            else:
                candidate_buckets = [files]

            for bucket in candidate_buckets:
                canonical: Optional[Path] = None
                canonical_hash: Optional[str] = None
                for p in bucket:
                    h = get_hash(p)
                    if h is None:
                        # If we couldn't hash, err on the side of keeping it
                        if canonical is None:
                            canonical = p
                            canonical_hash = None
                        else:
                            pass
                        continue
                    if canonical is None:
                        canonical = p
                        canonical_hash = h
                        kept_map[(nkey, sz, h)] = p
                    else:
                        if canonical_hash == h:
                            reason = (
                                f"Duplicate by name+size+hash -> kept: '{canonical.name}', "
                                f"size: {sz} bytes ({human_bytes(sz)}), hash={h[:12]}..."
                            )
                            actions_delete.append((p, reason))
                        else:
                            # Same name+size but different content -> keep (NOT a duplicate)
                            kept_map[(nkey, sz, h)] = p
                            notes.append(
                                f"INFO: Same name+size but different content kept separately: {canonical} vs {p}"
                            )

    elif args.mode == "content":
        # Group by size first to avoid hashing everything unnecessarily.
        size_groups: Dict[int, list] = {}
        for p in all_files:
            try:
                sz = p.stat().st_size
            except Exception:
                continue
            size_groups.setdefault(sz, []).append(p)

        for sz, files in size_groups.items():
            if len(files) == 1:
                kept_map[(sz, "first")] = files[0]
                continue

            # Optional quick fingerprint to reduce full hashes
            if args.use_quick_fingerprint:
                fp_groups: Dict[Tuple[int, bytes, bytes], list] = {}
                for p in files:
                    fp = get_fp(p)
                    if fp is None:
                        fp_groups.setdefault((id(p), b"", b""), []).append(p)
                    else:
                        fp_groups.setdefault(fp, []).append(p)
                candidate_buckets = fp_groups.values()
            else:
                candidate_buckets = [files]

            for bucket in candidate_buckets:
                hash_to_path: Dict[str, Path] = {}
                for p in bucket:
                    h = get_hash(p)
                    if h is None:
                        kept_map[(sz, id(p))] = p
                        continue
                    if h in hash_to_path:
                        kept = hash_to_path[h]
                        reason = (
                            f"Duplicate by content -> kept: '{kept.name}', "
                            f"size: {sz} bytes ({human_bytes(sz)}), hash={h[:12]}..."
                        )
                        actions_delete.append((p, reason))
                    else:
                        hash_to_path[h] = p
                        kept_map[(sz, h)] = p

    # Plan moves for all kept files (unique across keys)
    kept_files = sorted(set(kept_map.values()))
    if args.apply:
        out_dir.mkdir(parents=True, exist_ok=True)

    for src in kept_files:
        if out_dir in src.resolve().parents:
            continue
        dst = safe_unique_path(out_dir, src.name)
        actions_move.append((src, dst))

    # Prepare log
    timestamp = datetime.now().isoformat(timespec="seconds")
    header = [
        f"Robust duplicate removal & flatten log",
        f"Root: {root}",
        f"Output folder: {out_dir}",
        f"Log file: {log_path}",
        f"Case-insensitive names: {args.case_insensitive}",
        f"Mode: {args.mode}",
        f"Hash algo: {args.hash}",
        f"Quick fingerprint: {args.use_quick_fingerprint}",
        f"Dry run: {not args.apply}",
        f"Started: {timestamp}",
        "",
    ]
    log_lines = header[:]

    # Deletions
    if actions_delete:
        log_lines.append(f"Planned deletions ({len(actions_delete)}):")
        for path, reason in actions_delete:
            try:
                size = path.stat().st_size
            except Exception:
                size = -1
            size_txt = human_bytes(size) if size >= 0 else "unknown"
            log_lines.append(f"- DELETE: {path} | {size_txt} | {reason}")
    else:
        log_lines.append("Planned deletions (0): None")
    log_lines.append("")

    # Moves
    if actions_move:
        log_lines.append(f"Planned moves ({len(actions_move)}):")
        for src, dst in actions_move:
            try:
                size = src.stat().st_size
            except Exception:
                size = -1
            size_txt = human_bytes(size) if size >= 0 else "unknown"
            log_lines.append(f"- MOVE: {src} -> {dst} | {size_txt}")
    else:
        log_lines.append("Planned moves (0): None")
    log_lines.append("")

    # Extra notes
    if notes:
        log_lines.append("Notes:")
        for n in notes:
            log_lines.append(f"- {n}")
        log_lines.append("")

    # Execute if --apply
    if args.apply:
        deleted, moved = 0, 0
        for path, _reason in actions_delete:
            try:
                if path.exists():
                    os.remove(path)
                    deleted += 1
            except Exception as e:
                log_lines.append(f"! ERROR deleting {path}: {e}")
        for src, dst in actions_move:
            try:
                if not src.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    dst = safe_unique_path(dst.parent, dst.name)
                shutil.move(str(src), str(dst))
                moved += 1
            except Exception as e:
                log_lines.append(f"! ERROR moving {src} -> {dst}: {e}")
        log_lines.append("")
        log_lines.append(f"Executed: deleted {deleted} duplicates, moved {moved} files.")
    else:
        log_lines.append("Dry run complete: no files were deleted or moved. Re-run with --apply to execute.")

    # Write log file
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
    except Exception as e:
        print(f"WARNING: Could not write log file at {log_path}: {e}")

    print("\n".join(log_lines))

if __name__ == "__main__":
    main()
