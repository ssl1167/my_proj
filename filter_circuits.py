# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from garl_sabre.dataset import load_records, save_json


def _find_split_file(data_dir: Path, split: str) -> Path | None:
    for suffix in (".json", ".jsonl"):
        path = data_dir / f"{split}{suffix}"
        if path.exists():
            return path
    return None


def filter_rows(rows: List[Dict], max_qubits: int) -> List[Dict]:
    return [row for row in rows if int(row["num_qubits"]) <= max_qubits]


def main() -> None:
    p = argparse.ArgumentParser(description="Filter train/valid/test manifests by max qubits.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_qubits", type=int, required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        manifest = _find_split_file(data_dir, split)
        if manifest is None:
            print(f"[Skip] No manifest found for split={split}")
            continue

        rows = load_records(manifest, split=split)
        filtered = filter_rows(rows, args.max_qubits)

        save_json(out_dir / f"{split}.json", filtered)

        print(f"[{split}] source={manifest.name}")
        print(f"  Original: {len(rows)} circuits")
        print(f"  Filtered: {len(filtered)} circuits (<= {args.max_qubits} qubits)")


if __name__ == "__main__":
    main()
