# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from qiskit.qasm2 import dumps

from convert_revlib_real import parse_real_file
from garl_sabre.dataset import load_records, save_json

QREG_RE = re.compile(r"\bqreg\s+[A-Za-z_][A-Za-z0-9_]*\[(\d+)\]\s*;")
NUMVARS_RE = re.compile(r"^\s*\.numvars\s+(\d+)\s*$", re.MULTILINE)


@dataclass
class SkipRecord:
    source: str
    path: str
    reason: str


@dataclass
class BuildSummary:
    max_qubits: int
    total_candidates: int
    kept_rows: int
    skipped_rows: int
    duplicate_drops: int
    train_count: int
    valid_count: int
    test_count: int
    preserve_random_original_split: bool
    bucket_size: int


def infer_qasm_num_qubits(qasm_text: str) -> int:
    matches = [int(x) for x in QREG_RE.findall(qasm_text)]
    if not matches:
        raise ValueError("no qreg declaration found")
    return int(sum(matches))


def infer_real_num_qubits(real_text: str) -> int:
    match = NUMVARS_RE.search(real_text)
    if not match:
        raise ValueError("no .numvars directive found")
    return int(match.group(1))


def infer_family_from_name(name: str) -> str:
    stem = Path(name).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        base = parts[0].strip()
        if base:
            return base
    return stem


def family_from_path(path: Path, root: Path, fallback: str) -> str:
    if path.parent != root:
        return path.parent.name
    inferred = infer_family_from_name(path.stem)
    return inferred or fallback


def _qasm_hash(qasm_text: str) -> str:
    return hashlib.sha256(qasm_text.encode("utf-8")).hexdigest()


def collect_qasm_examples(root: Path, max_qubits: int, skips: List[SkipRecord]) -> List[Dict]:
    rows: List[Dict] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*.qasm")):
        try:
            qasm = path.read_text(encoding="utf-8", errors="ignore")
            num_qubits = infer_qasm_num_qubits(qasm)
        except Exception as exc:
            skips.append(SkipRecord("examples_qasm", str(path), f"read_or_parse_failed: {exc}"))
            continue

        if num_qubits > max_qubits:
            skips.append(SkipRecord("examples_qasm", str(path), f"num_qubits={num_qubits} > max_qubits={max_qubits}"))
            continue

        rows.append(
            {
                "name": path.stem,
                "family": family_from_path(path, root, "examples"),
                "num_qubits": int(num_qubits),
                "qasm": qasm,
                "source": "examples_qasm",
                "source_path": str(path),
                "source_split": None,
                "qasm_hash": _qasm_hash(qasm),
            }
        )
    return rows


def collect_revlib_real(root: Path, max_qubits: int, skips: List[SkipRecord]) -> List[Dict]:
    rows: List[Dict] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*.real")):
        try:
            real_text = path.read_text(encoding="utf-8", errors="ignore")
            num_qubits = infer_real_num_qubits(real_text)
        except Exception as exc:
            skips.append(SkipRecord("revlib_real", str(path), f"header_parse_failed: {exc}"))
            continue

        if num_qubits > max_qubits:
            skips.append(SkipRecord("revlib_real", str(path), f"num_qubits={num_qubits} > max_qubits={max_qubits}"))
            continue

        try:
            qc = parse_real_file(path)
            qasm = dumps(qc)
        except Exception as exc:
            skips.append(SkipRecord("revlib_real", str(path), f"real_to_qasm_failed: {exc}"))
            continue

        rows.append(
            {
                "name": path.stem,
                "family": family_from_path(path, root, "revlib"),
                "num_qubits": int(qc.num_qubits),
                "qasm": qasm,
                "source": "revlib_real",
                "source_path": str(path),
                "source_split": None,
                "qasm_hash": _qasm_hash(qasm),
            }
        )
    return rows


def collect_random_manifests(root: Path, max_qubits: int, skips: List[SkipRecord]) -> List[Dict]:
    """
    By default, these rows are merged back into a unified pool and re-split together
    with examples / RevLib. This avoids silently treating the original random split
    as if it were still preserved.
    """
    rows: List[Dict] = []
    if not root.exists():
        return rows

    for split_name in ("train", "valid", "test"):
        manifest = None
        for suffix in (".json", ".jsonl"):
            p = root / f"{split_name}{suffix}"
            if p.exists():
                manifest = p
                break
        if manifest is None:
            continue

        try:
            records = load_records(manifest, split=split_name)
        except Exception as exc:
            skips.append(SkipRecord("random_manifest", str(manifest), f"load_failed: {exc}"))
            continue

        for row in records:
            try:
                num_qubits = int(row["num_qubits"])
                qasm = row["qasm"]
                name = row["name"]
            except KeyError as exc:
                skips.append(SkipRecord("random_manifest", f"{manifest}", f"missing_field: {exc}"))
                continue

            if num_qubits > max_qubits:
                skips.append(
                    SkipRecord(
                        "random_manifest",
                        f"{manifest}:{name}",
                        f"num_qubits={num_qubits} > max_qubits={max_qubits}",
                    )
                )
                continue

            rows.append(
                {
                    "name": name,
                    "family": row.get("family", "random"),
                    "num_qubits": num_qubits,
                    "qasm": qasm,
                    "source": "random_manifest",
                    "source_path": str(manifest),
                    "source_split": split_name,
                    "qasm_hash": _qasm_hash(qasm),
                }
            )
    return rows


def deduplicate_rows(rows: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Deduplicate by circuit content, not by name only.
    Preference order:
        examples_qasm > revlib_real > random_manifest
    """
    priority = {
        "examples_qasm": 0,
        "revlib_real": 1,
        "random_manifest": 2,
    }
    best: Dict[Tuple[str, int], Dict] = {}
    duplicate_drops = 0

    for row in rows:
        key = (row["qasm_hash"], int(row["num_qubits"]))
        if key not in best:
            best[key] = row
            continue

        old = best[key]
        old_pri = priority.get(old["source"], 99)
        new_pri = priority.get(row["source"], 99)
        if new_pri < old_pri:
            best[key] = row
        duplicate_drops += 1

    cleaned = []
    for row in best.values():
        cleaned.append(
            {
                "name": row["name"],
                "family": row["family"],
                "num_qubits": row["num_qubits"],
                "qasm": row["qasm"],
            }
        )
    return cleaned, duplicate_drops


def _bucket(num_qubits: int, bucket_size: int) -> str:
    lo = ((num_qubits - 1) // bucket_size) * bucket_size + 1
    hi = lo + bucket_size - 1
    return f"{lo:02d}-{hi:02d}"


def _split_one_group(
    rows: List[Dict],
    split_ratio: Tuple[float, float, float],
    rng: random.Random,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    data = list(rows)
    rng.shuffle(data)
    n = len(data)

    if n == 1:
        return data, [], []
    if n == 2:
        return [data[0]], [], [data[1]]

    n_train = int(round(n * split_ratio[0]))
    n_valid = int(round(n * split_ratio[1]))
    n_test = n - n_train - n_valid

    # Keep all splits non-negative and avoid empty test split for medium groups.
    n_train = max(1, min(n_train, n - 2))
    n_valid = max(0, min(n_valid, n - n_train - 1))
    n_test = n - n_train - n_valid

    return data[:n_train], data[n_train:n_train + n_valid], data[n_train + n_valid:]


def split_rows_stratified(
    rows: List[Dict],
    split_ratio: Tuple[float, float, float],
    seed: int,
    bucket_size: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Stratify by (family, num_qubits bucket) so that structure distribution is stabler
    across train / valid / test.
    """
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        key = (str(row["family"]), _bucket(int(row["num_qubits"]), bucket_size))
        groups[key].append(row)

    rng = random.Random(seed)
    train_rows: List[Dict] = []
    valid_rows: List[Dict] = []
    test_rows: List[Dict] = []

    for key in sorted(groups.keys()):
        tr, va, te = _split_one_group(groups[key], split_ratio, rng)
        train_rows.extend(tr)
        valid_rows.extend(va)
        test_rows.extend(te)

    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    rng.shuffle(test_rows)
    return train_rows, valid_rows, test_rows


def split_rows_preserve_random(
    rows: List[Dict],
    split_ratio: Tuple[float, float, float],
    seed: int,
    bucket_size: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Optional mode:
    - preserve original split for random_manifest rows
    - re-split other sources stratified
    """
    random_train = [r for r in rows if r.get("source") == "random_manifest" and r.get("source_split") == "train"]
    random_valid = [r for r in rows if r.get("source") == "random_manifest" and r.get("source_split") == "valid"]
    random_test = [r for r in rows if r.get("source") == "random_manifest" and r.get("source_split") == "test"]
    other = [r for r in rows if r.get("source") != "random_manifest"]

    tr, va, te = split_rows_stratified(other, split_ratio, seed, bucket_size)
    return tr + random_train, va + random_valid, te + random_test


def _print_distribution(rows: List[Dict], title: str) -> None:
    family_counts = Counter(str(r["family"]) for r in rows)
    print(f"\n[{title}] total={len(rows)}")
    for fam, cnt in sorted(family_counts.items()):
        print(f"  {fam}: {cnt}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build a unified dataset from examples / RevLib / random manifests. "
            "By default, random train/valid/test manifests are merged back into a unified pool "
            "and re-split together with other sources for a cleaner, fully rebuilt split."
        )
    )
    p.add_argument("--data_root", type=str, default="data_1")
    p.add_argument("--out_dir", type=str, default="data/data1_40q")
    p.add_argument("--max_qubits", type=int, default=40)
    p.add_argument("--split_ratio", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--bucket_size", type=int, default=4)
    p.add_argument(
        "--preserve_random_original_split",
        action="store_true",
        help="Preserve the original train/valid/test assignment for random manifests instead of rebuilding a unified split.",
    )
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skips: List[SkipRecord] = []

    qasm_rows = collect_qasm_examples(data_root / "examples", args.max_qubits, skips)
    revlib_rows = collect_revlib_real(data_root / "RevLib", args.max_qubits, skips)
    random_rows = collect_random_manifests(data_root / "random", args.max_qubits, skips)

    combined_rows = qasm_rows + revlib_rows + random_rows
    dedup_rows, duplicate_drops = deduplicate_rows(combined_rows)

    if args.preserve_random_original_split:
        train_rows, valid_rows, test_rows = split_rows_preserve_random(
            combined_rows, tuple(args.split_ratio), args.seed, args.bucket_size
        )
        # Deduplicate again after combining preserved random split rows with re-split others.
        train_rows, _ = deduplicate_rows(train_rows)
        valid_rows, _ = deduplicate_rows(valid_rows)
        test_rows, _ = deduplicate_rows(test_rows)
    else:
        train_rows, valid_rows, test_rows = split_rows_stratified(
            dedup_rows, tuple(args.split_ratio), args.seed, args.bucket_size
        )

    # Unified output format: JSON only.
    save_json(out_dir / "train.json", train_rows)
    save_json(out_dir / "valid.json", valid_rows)
    save_json(out_dir / "test.json", test_rows)

    summary = BuildSummary(
        max_qubits=args.max_qubits,
        total_candidates=len(combined_rows),
        kept_rows=len(dedup_rows),
        skipped_rows=len(skips),
        duplicate_drops=duplicate_drops,
        train_count=len(train_rows),
        valid_count=len(valid_rows),
        test_count=len(test_rows),
        preserve_random_original_split=bool(args.preserve_random_original_split),
        bucket_size=int(args.bucket_size),
    )

    with (out_dir / "build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

    with (out_dir / "skipped_records.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in skips], f, ensure_ascii=False, indent=2)

    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    _print_distribution(train_rows, "train")
    _print_distribution(valid_rows, "valid")
    _print_distribution(test_rows, "test")
    print(f"\nSaved manifests to: {out_dir}")


if __name__ == "__main__":
    main()
