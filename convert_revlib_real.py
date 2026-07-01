# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import SXGate, SXdgGate
from qiskit.qasm2 import dumps

from garl_sabre.dataset import save_json

MacroDef = Tuple[List[str], List[List[str]]]
BusMap = Dict[str, List[int]]

HEADER_DIRECTIVES = {
    ".inputs",
    ".outputs",
    ".constants",
    ".garbage",
    ".version",
    ".inputbus",
    ".outputbus",
    ".module",
}

CNOT_CANONICAL_BASIS = ["rz", "sx", "x", "cx"]
IGNORED_OPS = {"barrier", "measure", "delay", "reset"}


def _clean_tokens(line: str) -> List[str]:
    line = line.split("#", 1)[0].strip()
    if not line:
        return []
    return line.split()


def _gate_arity(gate: str, args: List[str]) -> int:
    suffix = gate[1:]
    if not suffix:
        return len(args)
    if suffix.isdigit():
        return int(suffix)
    raise NotImplementedError(f"Unsupported gate suffix in '{gate}'")


def _resolve_arg(token: str, index_of: Dict[str, int], buses: BusMap, num_qubits: int) -> int:
    token = token.strip()

    if token in index_of:
        idx = index_of[token]
    elif token in buses:
        wires = buses[token]
        if len(wires) == 1:
            idx = wires[0]
        else:
            raise KeyError(f"Bus alias '{token}' refers to {len(wires)} wires; use explicit members instead.")
    elif token.startswith("x") and token[1:].isdigit():
        idx = int(token[1:])
    elif token.isdigit():
        idx = int(token)
    else:
        raise KeyError(f"Unknown wire token: {token}")

    if idx < 0 or idx >= num_qubits:
        raise IndexError(f"Wire token '{token}' resolved to {idx}, but circuit has {num_qubits} qubits.")
    return idx


def _resolve_bus_members(members: List[str], index_of: Dict[str, int], num_qubits: int) -> List[int]:
    resolved: List[int] = []
    for token in members:
        try:
            resolved.append(_resolve_arg(token, index_of, {}, num_qubits))
        except Exception as exc:
            raise KeyError(f"Invalid bus member '{token}': {exc}") from exc
    return resolved


def _apply_x_family(qc: QuantumCircuit, args: List[int]) -> None:
    if len(args) == 1:
        qc.x(args[0])
    elif len(args) == 2:
        qc.cx(args[0], args[1])
    else:
        qc.mcx(args[:-1], args[-1])


def _apply_v_family(qc: QuantumCircuit, args: List[int], dagger: bool = False) -> None:
    base_gate = SXdgGate() if dagger else SXGate()
    if len(args) == 1:
        qc.append(base_gate, [args[0]])
    elif len(args) == 2:
        qc.append(base_gate.control(1), [args[0], args[1]])
    else:
        raise NotImplementedError(f"Unsupported multi-control V gate with {len(args) - 1} controls.")


def _apply_f_family(qc: QuantumCircuit, args: List[int]) -> None:
    if len(args) == 2:
        qc.swap(args[0], args[1])
    elif len(args) == 3:
        qc.cswap(args[0], args[1], args[2])
    else:
        controls = args[:-2]
        a, b = args[-2], args[-1]
        qc.cx(b, a)
        qc.mcx(controls + [a], b)
        qc.cx(b, a)


def _apply_p_gate(qc: QuantumCircuit, args: List[int]) -> None:
    if len(args) != 3:
        raise NotImplementedError(f"Unsupported Peres-style gate with {len(args)} wires.")
    _apply_x_family(qc, args)
    qc.cx(args[0], args[1])


def _expand_macro(tokens: List[str], params: List[str], body: List[List[str]]) -> List[List[str]]:
    if len(tokens) - 1 != len(params):
        raise ValueError(f"Macro arity mismatch for {tokens[0]}: expected {len(params)}, got {len(tokens) - 1}.")
    mapping = dict(zip(params, tokens[1:]))
    return [[mapping.get(token, token) for token in body_tokens] for body_tokens in body]


def _execute_gate(
    qc: QuantumCircuit,
    tokens: List[str],
    path: Path,
    index_of: Dict[str, int],
    buses: BusMap,
    macros: Dict[str, MacroDef],
    stack: Tuple[str, ...] = (),
) -> None:
    if not tokens:
        return

    gate = tokens[0].lower()

    if gate in macros:
        if gate in stack:
            raise ValueError(f"Recursive macro detected: {' -> '.join(stack + (gate,))}")
        params, body = macros[gate]
        for expanded_tokens in _expand_macro(tokens, params, body):
            _execute_gate(qc, expanded_tokens, path, index_of, buses, macros, stack + (gate,))
        return

    arg_tokens = tokens[1:]
    args = [_resolve_arg(token, index_of, buses, qc.num_qubits) for token in arg_tokens]

    if gate.startswith("t"):
        arity = _gate_arity(gate, arg_tokens)
        if arity != len(args):
            raise ValueError(f"Gate arity mismatch in {path.name}: {' '.join(tokens)}")
        _apply_x_family(qc, args)
    elif gate.startswith("f"):
        arity = _gate_arity(gate, arg_tokens)
        if arity != len(args):
            raise ValueError(f"Gate arity mismatch in {path.name}: {' '.join(tokens)}")
        _apply_f_family(qc, args)
    elif gate == "p":
        _apply_p_gate(qc, args)
    elif gate.startswith("p") and gate[1:].isdigit():
        arity = _gate_arity(gate, arg_tokens)
        if arity != len(args):
            raise ValueError(f"Gate arity mismatch in {path.name}: {' '.join(tokens)}")
        _apply_x_family(qc, args)
    elif gate == "x":
        _apply_x_family(qc, args)
    elif gate == "v":
        _apply_v_family(qc, args, dagger=False)
    elif gate in {"v+", "v-"}:
        _apply_v_family(qc, args, dagger=True)
    else:
        raise NotImplementedError(f"Unsupported .real gate in {path.name}: {' '.join(tokens)}")


def parse_real_file(path: Path) -> QuantumCircuit:
    """Parse a RevLib .real file into a declared-width Qiskit QuantumCircuit."""
    num_qubits = None
    variables: List[str] = []
    gate_lines: List[List[str]] = []
    macros: Dict[str, MacroDef] = {}
    bus_tokens: Dict[str, List[str]] = {}

    current_macro_name: str | None = None
    current_macro_params: List[str] = []
    current_macro_body: List[List[str]] = []

    in_circuit = False
    saw_begin = False

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tokens = _clean_tokens(raw)
        if not tokens:
            continue

        head = tokens[0].lower()

        if current_macro_name is not None:
            if head == ".enddefine":
                macros[current_macro_name] = (current_macro_params, current_macro_body)
                current_macro_name = None
                current_macro_params = []
                current_macro_body = []
            elif not head.startswith("."):
                current_macro_body.append(tokens)
            continue

        if head == ".numvars":
            if len(tokens) < 2:
                raise ValueError(f"Malformed .numvars in {path.name}: {' '.join(tokens)}")
            num_qubits = int(tokens[1])
        elif head == ".variables":
            variables.extend(tokens[1:])
        elif head in {".inputbus", ".outputbus"}:
            if len(tokens) >= 3:
                bus_tokens[tokens[1]] = tokens[2:]
        elif head == ".define":
            if len(tokens) < 2:
                raise ValueError(f"Malformed .define in {path.name}: {' '.join(tokens)}")
            current_macro_name = tokens[1].lower()
            current_macro_params = tokens[2:]
            current_macro_body = []
        elif head == ".begin":
            in_circuit = True
            saw_begin = True
        elif head == ".end":
            if in_circuit:
                in_circuit = False
                break
        elif head in HEADER_DIRECTIVES or head.startswith("."):
            continue
        else:
            if in_circuit or not saw_begin:
                gate_lines.append(tokens)

    if current_macro_name is not None:
        raise ValueError(f"Unterminated macro definition '{current_macro_name}' in {path.name}.")

    if num_qubits is None:
        if variables:
            num_qubits = len(variables)
        else:
            raise ValueError(f"Could not infer number of qubits from {path}.")

    if not variables:
        variables = [f"x{i}" for i in range(num_qubits)]

    if len(variables) > num_qubits:
        raise ValueError(
            f"{path.name}: .variables contains {len(variables)} wires, but .numvars declares only {num_qubits}."
        )

    index_of: Dict[str, int] = {name: i for i, name in enumerate(variables)}
    buses: BusMap = {}
    for alias, members in bus_tokens.items():
        resolved_members = _resolve_bus_members(members, index_of, num_qubits)
        if resolved_members:
            buses[alias] = resolved_members

    qc = QuantumCircuit(num_qubits, name=path.stem)
    for tokens in gate_lines:
        _execute_gate(qc, tokens, path, index_of, buses, macros)
    return qc


def _instruction_parts(inst):
    if hasattr(inst, "operation"):
        return inst.operation, inst.qubits, inst.clbits
    op = inst[0]
    qargs = inst[1]
    cargs = inst[2] if len(inst) > 2 else []
    return op, qargs, cargs


def decompose_to_cx_basis(qc: QuantumCircuit) -> QuantumCircuit:
    return transpile(qc, basis_gates=CNOT_CANONICAL_BASIS, optimization_level=0)


def canonical_cnot_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Convert any circuit to the C++-style mapping benchmark representation:
    a compact, CNOT-only circuit over qubits that appear in at least one CNOT
    after decomposition to IBM-style CX basis.
    """
    work = decompose_to_cx_basis(qc)
    cx_ops: List[Tuple[int, int]] = []
    active: set[int] = set()

    for inst in work.data:
        op, qargs, _ = _instruction_parts(inst)
        if getattr(op, "name", "") != "cx" or len(qargs) != 2:
            continue
        c = work.find_bit(qargs[0]).index
        t = work.find_bit(qargs[1]).index
        cx_ops.append((c, t))
        active.add(c)
        active.add(t)

    if not cx_ops:
        # A no-CNOT circuit has no routing problem. Keep one empty qubit when possible
        # so downstream Qiskit code does not receive a 0-qubit circuit unexpectedly.
        return QuantumCircuit(max(1, min(qc.num_qubits, 1)), name=qc.name)

    active_sorted = sorted(active)
    remap = {old: new for new, old in enumerate(active_sorted)}
    out = QuantumCircuit(len(active_sorted), name=qc.name)
    for c, t in cx_ops:
        out.cx(remap[c], remap[t])
    return out


def cnot_active_qubits(qc: QuantumCircuit) -> List[int]:
    work = decompose_to_cx_basis(qc)
    active: set[int] = set()
    for inst in work.data:
        op, qargs, _ = _instruction_parts(inst)
        if getattr(op, "name", "") == "cx" and len(qargs) == 2:
            active.update(work.find_bit(q).index for q in qargs)
    return sorted(active)


def _row_from_circuit(path: Path, real_dir: Path, qc: QuantumCircuit, canonicalize: bool, keep_metadata: bool) -> dict:
    del keep_metadata  # rows keep metadata internally; output stripping happens later.
    raw_num_qubits = int(qc.num_qubits)
    raw_cnot_count = int(decompose_to_cx_basis(qc).count_ops().get("cx", 0))
    if canonicalize:
        qc = canonical_cnot_circuit(qc)
    return {
        "name": path.stem,
        "family": path.parent.name if path.parent != real_dir else "revlib",
        "num_qubits": int(qc.num_qubits),
        "qasm": dumps(qc),
        "raw_num_qubits": raw_num_qubits,
        "raw_cnot_count": raw_cnot_count,
        "logical_cnot_count": int(qc.count_ops().get("cx", 0)),
        "canonicalized_cnot_only": bool(canonicalize),
        "trimmed": bool(raw_num_qubits != int(qc.num_qubits)),
    }


def _strip_metadata(rows: List[dict]) -> List[dict]:
    return [{"name": r["name"], "family": r["family"], "num_qubits": int(r["num_qubits"]), "qasm": r["qasm"]} for r in rows]


def convert_dir(
    real_dir: Path,
    out_dir: Path,
    split_ratio: Sequence[float],
    seed: int,
    canonicalize_cnot_only: bool = True,
    keep_metadata: bool = False,
) -> None:
    files = sorted(real_dir.rglob("*.real"))
    if not files:
        raise FileNotFoundError(f"No .real files found under {real_dir}")

    rows: List[dict] = []
    skipped: List[dict] = []

    for path in files:
        try:
            qc = parse_real_file(path)
            rows.append(_row_from_circuit(path, real_dir, qc, canonicalize=canonicalize_cnot_only, keep_metadata=keep_metadata))
        except Exception as exc:
            print(f"[SKIP] {path.name}: {exc}")
            skipped.append({"path": str(path), "reason": str(exc)})

    if not rows:
        raise RuntimeError("No .real files were successfully converted.")

    rng = random.Random(seed)
    rng.shuffle(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(rows)
    n_train = int(n_total * split_ratio[0])
    n_valid = int(n_total * split_ratio[1])

    train_rows = rows[:n_train]
    valid_rows = rows[n_train:n_train + n_valid]
    test_rows = rows[n_train + n_valid:]

    save_json(out_dir / "train.json", train_rows if keep_metadata else _strip_metadata(train_rows))
    save_json(out_dir / "valid.json", valid_rows if keep_metadata else _strip_metadata(valid_rows))
    save_json(out_dir / "test.json", test_rows if keep_metadata else _strip_metadata(test_rows))

    summary = {
        "total_files": len(files),
        "converted": len(rows),
        "skipped": len(skipped),
        "canonicalize_cnot_only": bool(canonicalize_cnot_only),
        "trimmed_rows": sum(1 for row in rows if row.get("trimmed", False)),
        "train": len(train_rows),
        "valid": len(valid_rows),
        "test": len(test_rows),
        "seed": int(seed),
    }
    with (out_dir / "convert_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out_dir / "skipped_records.json").open("w", encoding="utf-8") as f:
        json.dump(skipped, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(rows)} circuits into {out_dir}")
    print(f"train={len(train_rows)}, valid={len(valid_rows)}, test={len(test_rows)}, seed={seed}")
    print(f"canonicalize_cnot_only={canonicalize_cnot_only}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--split_ratio", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--keep_full_circuit",
        action="store_true",
        help="Disable CNOT-only canonicalization and keep the declared-width parsed circuit.",
    )
    p.add_argument("--keep_metadata", action="store_true")
    args = p.parse_args()
    convert_dir(
        Path(args.real_dir),
        Path(args.out_dir),
        args.split_ratio,
        args.seed,
        canonicalize_cnot_only=not args.keep_full_circuit,
        keep_metadata=args.keep_metadata,
    )


if __name__ == "__main__":
    main()
