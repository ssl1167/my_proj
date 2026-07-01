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
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dumps

from convert_revlib_real import parse_real_file
from garl_sabre.dataset import load_records, save_json

QREG_RE = re.compile(r"\bqreg\s+[A-Za-z_][A-Za-z0-9_]*\[(\d+)\]\s*;")
NUMVARS_RE = re.compile(r"^\s*\.numvars\s+(\d+)\s*$", re.MULTILINE)

CNOT_CANONICAL_BASIS = ["rz", "sx", "x", "cx"]


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
    canonicalized_cnot_only: bool
    trimmed_rows: int


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


def _instruction_parts(inst):
    if hasattr(inst, "operation"):
        return inst.operation, inst.qubits, inst.clbits
    op = inst[0]
    qargs = inst[1]
    cargs = inst[2] if len(inst) > 2 else []
    return op, qargs, cargs


def decompose_to_cx_basis(qc: QuantumCircuit) -> QuantumCircuit:
    """
    使用 transpile 将电路转换为基础门，包括将 mcx（多控制门）分解为 cx 门。
    这对于处理 .real 文件中的 t3, t4, t5, t6 等多控制门是必需的。
    """
    return transpile(qc, basis_gates=CNOT_CANONICAL_BASIS, optimization_level=0)


def canonical_cnot_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """
    C++-style benchmark canonicalization:
    不进行门分解，仅保留原本就是 cx 的门操作。
    其他门（单比特门、cz、swap等）直接忽略，不做任何处理。
    """
    cx_ops: List[Tuple[int, int]] = []
    active: set[int] = set()

    for inst in qc.data:
        op, qargs, _ = _instruction_parts(inst)
        # 仅保留名称为 "cx" 的门，其他门直接跳过
        if getattr(op, "name", "") != "cx" or len(qargs) != 2:
            continue
        c = qc.find_bit(qargs[0]).index
        t = qc.find_bit(qargs[1]).index
        cx_ops.append((c, t))
        active.add(c)
        active.add(t)

    if not cx_ops:
        return QuantumCircuit(max(1, min(qc.num_qubits, 1)), name=qc.name)

    active_sorted = sorted(active)
    remap = {old: new for new, old in enumerate(active_sorted)}
    out = QuantumCircuit(len(active_sorted), name=qc.name)
    for c, t in cx_ops:
        out.cx(remap[c], remap[t])
    return out


def canonicalize_circuit_row(
    qc: QuantumCircuit,
    name: str,
    family: str,
    source: str,
    source_path: str,
    source_split=None,
    canonicalize_cnot_only: bool = True,
) -> Dict:
    raw_num_qubits = int(qc.num_qubits)
    raw_cnot_count = int(decompose_to_cx_basis(qc).count_ops().get("cx", 0))
    if canonicalize_cnot_only:
        qc = canonical_cnot_circuit(qc)
    qasm = dumps(qc)
    logical_cnot_count = int(qc.count_ops().get("cx", 0))

    return {
        "name": str(name),
        "family": str(family),
        "num_qubits": int(qc.num_qubits),
        "qasm": qasm,
        "source": source,
        "source_path": str(source_path),
        "source_split": source_split,
        "qasm_hash": _qasm_hash(qasm),
        "raw_num_qubits": raw_num_qubits,
        "raw_cnot_count": raw_cnot_count,
        "logical_cnot_count": logical_cnot_count,
        "canonicalized_cnot_only": bool(canonicalize_cnot_only),
        "trimmed": bool(raw_num_qubits != int(qc.num_qubits)),
    }


def collect_qasm_examples(root: Path, max_qubits: int, skips: List[SkipRecord], canonicalize_cnot_only: bool) -> List[Dict]:
    rows: List[Dict] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*.qasm")):
        try:
            qasm = path.read_text(encoding="utf-8", errors="ignore")
            qc = QuantumCircuit.from_qasm_str(qasm)
            row = canonicalize_circuit_row(
                qc,
                name=path.stem,
                family=family_from_path(path, root, "examples"),
                source="examples_qasm",
                source_path=str(path),
                canonicalize_cnot_only=canonicalize_cnot_only,
            )
        except Exception as exc:
            skips.append(SkipRecord("examples_qasm", str(path), f"read_or_parse_failed: {exc}"))
            continue

        if int(row["num_qubits"]) > max_qubits:
            skips.append(SkipRecord("examples_qasm", str(path), f"active_num_qubits={row['num_qubits']} > max_qubits={max_qubits}"))
            continue
        rows.append(row)
    return rows


def collect_revlib_real(root: Path, max_qubits: int, skips: List[SkipRecord], canonicalize_cnot_only: bool) -> List[Dict]:
    rows: List[Dict] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*.real")):
        try:
            qc = parse_real_file(path)
            row = canonicalize_circuit_row(
                qc,
                name=path.stem,
                family=family_from_path(path, root, "revlib"),
                source="revlib_real",
                source_path=str(path),
                canonicalize_cnot_only=canonicalize_cnot_only,
            )
        except Exception as exc:
            skips.append(SkipRecord("revlib_real", str(path), f"real_to_qasm_failed: {exc}"))
            continue

        if int(row["num_qubits"]) > max_qubits:
            skips.append(SkipRecord("revlib_real", str(path), f"active_num_qubits={row['num_qubits']} > max_qubits={max_qubits}"))
            continue
        rows.append(row)
    return rows


def collect_random_manifests(root: Path, max_qubits: int, skips: List[SkipRecord], canonicalize_cnot_only: bool) -> List[Dict]:
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
            # Load raw records here; this builder applies canonicalization itself
            # according to --keep_full_circuit, so do not let dataset.load_records
            # pre-canonicalize random manifests unconditionally.
            records = load_records(manifest, split=split_name, canonicalize_cnot_only=False)
        except Exception as exc:
            skips.append(SkipRecord("random_manifest", str(manifest), f"load_failed: {exc}"))
            continue

        for row in records:
            try:
                name = str(row["name"])
                qc = QuantumCircuit.from_qasm_str(str(row["qasm"]))
                fixed = canonicalize_circuit_row(
                    qc,
                    name=name,
                    family=str(row.get("family", "random")),
                    source="random_manifest",
                    source_path=str(manifest),
                    source_split=split_name,
                    canonicalize_cnot_only=canonicalize_cnot_only,
                )
            except Exception as exc:
                skips.append(SkipRecord("random_manifest", f"{manifest}:{row.get('name', 'unknown')}", f"parse_failed: {exc}"))
                continue

            if int(fixed["num_qubits"]) > max_qubits:
                skips.append(SkipRecord("random_manifest", f"{manifest}:{name}", f"active_num_qubits={fixed['num_qubits']} > max_qubits={max_qubits}"))
                continue
            rows.append(fixed)
    return rows


def deduplicate_rows(rows: List[Dict], keep_metadata: bool = False) -> Tuple[List[Dict], int]:
    priority = {"examples_qasm": 0, "revlib_real": 1, "random_manifest": 2}
    best: Dict[Tuple[str, int], Dict] = {}
    duplicate_drops = 0

    for row in rows:
        key = (row["qasm_hash"], int(row["num_qubits"]))
        if key not in best:
            best[key] = row
            continue
        old_pri = priority.get(best[key].get("source"), 99)
        new_pri = priority.get(row.get("source"), 99)
        if new_pri < old_pri:
            best[key] = row
        duplicate_drops += 1

    cleaned: List[Dict] = []
    for row in best.values():
        if keep_metadata:
            cleaned.append(dict(row))
        else:
            cleaned.append({"name": row["name"], "family": row["family"], "num_qubits": int(row["num_qubits"]), "qasm": row["qasm"]})
    return cleaned, duplicate_drops


def _bucket(num_qubits: int, bucket_size: int) -> str:
    lo = ((num_qubits - 1) // bucket_size) * bucket_size + 1
    hi = lo + bucket_size - 1
    return f"{lo:02d}-{hi:02d}"


def _split_one_group(rows: List[Dict], split_ratio: Tuple[float, float, float], rng: random.Random) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    data = list(rows)
    rng.shuffle(data)
    n = len(data)
    if n == 1:
        return data, [], []
    if n == 2:
        return [data[0]], [], [data[1]]

    n_train = int(round(n * split_ratio[0]))
    n_valid = int(round(n * split_ratio[1]))
    n_train = max(1, min(n_train, n - 2))
    n_valid = max(0, min(n_valid, n - n_train - 1))
    return data[:n_train], data[n_train:n_train + n_valid], data[n_train + n_valid:]


def split_rows_stratified(rows: List[Dict], split_ratio: Tuple[float, float, float], seed: int, bucket_size: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["family"]), _bucket(int(row["num_qubits"]), bucket_size))].append(row)

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


def split_rows_preserve_random(rows: List[Dict], split_ratio: Tuple[float, float, float], seed: int, bucket_size: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    random_train = [r for r in rows if r.get("source") == "random_manifest" and r.get("source_split") == "train"]
    random_valid = [r for r in rows if r.get("source") == "random_manifest" and r.get("source_split") == "valid"]
    random_test = [r for r in rows if r.get("source") == "random_manifest" and r.get("source_split") == "test"]
    other = [r for r in rows if r.get("source") != "random_manifest"]
    tr, va, te = split_rows_stratified(other, split_ratio, seed, bucket_size)
    train_rows, _ = deduplicate_rows(tr + random_train, keep_metadata=True)
    valid_rows, _ = deduplicate_rows(va + random_valid, keep_metadata=True)
    test_rows, _ = deduplicate_rows(te + random_test, keep_metadata=True)
    return train_rows, valid_rows, test_rows


def strip_metadata(rows: List[Dict]) -> List[Dict]:
    return [{"name": r["name"], "family": r["family"], "num_qubits": int(r["num_qubits"]), "qasm": r["qasm"]} for r in rows]


def _print_distribution(rows: List[Dict], title: str) -> None:
    family_counts = Counter(str(r["family"]) for r in rows)
    print(f"\n[{title}] total={len(rows)}")
    for fam, cnt in sorted(family_counts.items()):
        print(f"  {fam}: {cnt}")


def _print_trim_report(rows: List[Dict]) -> None:
    trimmed = [r for r in rows if r.get("trimmed")]
    print(f"\n[canonicalization report] trimmed_rows={len(trimmed)}")
    for r in sorted(trimmed, key=lambda x: (x["family"], x["name"]))[:30]:
        print(f"  {r['name']}: raw_num_qubits={r.get('raw_num_qubits')} -> active_num_qubits={r.get('num_qubits')}, logical_cnot_count={r.get('logical_cnot_count')}")
    if len(trimmed) > 30:
        print(f"  ... and {len(trimmed) - 30} more")


def main() -> None:
    p = argparse.ArgumentParser(description="Build a canonical CNOT-only mapping dataset.")
    p.add_argument("--data_root", type=str, default="data/paper19")
    p.add_argument("--out_dir", type=str, default="data/data1_20q")
    p.add_argument("--max_qubits", type=int, default=20)
    p.add_argument("--split_ratio", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--bucket_size", type=int, default=4)
    p.add_argument("--preserve_random_original_split", action="store_true")
    p.add_argument("--keep_full_circuit", action="store_true", help="Disable CNOT-only canonicalization.")
    p.add_argument("--keep_metadata", action="store_true", help="Keep source/raw metadata in train/valid/test JSON.")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    canonicalize_cnot_only = not args.keep_full_circuit

    skips: List[SkipRecord] = []
    qasm_rows = collect_qasm_examples(data_root / "examples", args.max_qubits, skips, canonicalize_cnot_only)
    
    # 检查 data_root 是否直接包含 .real 文件（即 data_root 本身就是 RevLib 目录）
    revlib_dir = data_root / "RevLib"
    if not revlib_dir.exists() and any(data_root.glob("*.real")):
        revlib_dir = data_root
    revlib_rows = collect_revlib_real(revlib_dir, args.max_qubits, skips, canonicalize_cnot_only)
    
    random_rows = collect_random_manifests(data_root / "random", args.max_qubits, skips, canonicalize_cnot_only)

    combined_rows = qasm_rows + revlib_rows + random_rows
    trimmed_rows = sum(1 for row in combined_rows if row.get("trimmed"))
    dedup_rows, duplicate_drops = deduplicate_rows(combined_rows, keep_metadata=True)

    if args.preserve_random_original_split:
        train_rows, valid_rows, test_rows = split_rows_preserve_random(combined_rows, tuple(args.split_ratio), args.seed, args.bucket_size)
    else:
        train_rows, valid_rows, test_rows = split_rows_stratified(dedup_rows, tuple(args.split_ratio), args.seed, args.bucket_size)

    save_json(out_dir / "train.json", train_rows if args.keep_metadata else strip_metadata(train_rows))
    save_json(out_dir / "valid.json", valid_rows if args.keep_metadata else strip_metadata(valid_rows))
    save_json(out_dir / "test.json", test_rows if args.keep_metadata else strip_metadata(test_rows))

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
        canonicalized_cnot_only=bool(canonicalize_cnot_only),
        trimmed_rows=int(trimmed_rows),
    )
    with (out_dir / "build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
    with (out_dir / "skipped_records.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in skips], f, ensure_ascii=False, indent=2)

    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    _print_distribution(train_rows, "train")
    _print_distribution(valid_rows, "valid")
    _print_distribution(test_rows, "test")
    _print_trim_report(combined_rows)
    print(f"\nSaved manifests to: {out_dir}")


if __name__ == "__main__":
    main()
