# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from garl_sabre.dataset import load_records


def _find_split_files(data_dir: Path):
    found = []
    for split in ("train", "valid", "test"):
        for suffix in (".json", ".jsonl"):
            path = data_dir / f"{split}{suffix}"
            if path.exists():
                found.append((split, path))
                break
    return found


def main() -> None:
    p = argparse.ArgumentParser(
        description="Inspect dataset size statistics: max/min qubits, family counts, and qubit histogram."
    )
    p.add_argument("--data_dir", type=str, required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    split_files = _find_split_files(data_dir)
    if not split_files:
        raise FileNotFoundError(f"No train/valid/test json or jsonl manifests found under {data_dir}")

    max_qubits = -1
    min_qubits = 10**9
    max_name = None
    min_name = None
    family_counter = Counter()
    qubit_counter = Counter()

    total = 0
    for split, path in split_files:
        rows = load_records(path, split=split)
        print(f"[{split}] {path.name}: {len(rows)} circuits")
        total += len(rows)
        for row in rows:
            nq = int(row["num_qubits"])
            name = str(row["name"])
            family = str(row.get("family", "unknown"))

            if nq > max_qubits:
                max_qubits = nq
                max_name = name
            if nq < min_qubits:
                min_qubits = nq
                min_name = name

            family_counter[family] += 1
            qubit_counter[nq] += 1

    print(f"\nTotal circuits: {total}")
    print(f"Min qubits: {min_qubits} ({min_name})")
    print(f"Max qubits: {max_qubits} ({max_name})")

    print("\nFamily counts:")
    for family, cnt in sorted(family_counter.items()):
        print(f"  {family}: {cnt}")

    print("\nQubit-size histogram:")
    for nq, cnt in sorted(qubit_counter.items()):
        print(f"  {nq}: {cnt}")


if __name__ == "__main__":
    main()
