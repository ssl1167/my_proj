from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from qiskit import QuantumCircuit
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


def _clean_tokens(line: str) -> List[str]:
    # Support inline comments such as: t3 a b c   # comment
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
            raise KeyError(f"Bus alias '{token}' refers to {len(wires)} wires")
    elif token.startswith("x") and token[1:].isdigit():
        idx = int(token[1:])
    elif token.isdigit():
        idx = int(token)
    else:
        raise KeyError(f"Unknown wire token: {token}")

    if idx < 0 or idx >= num_qubits:
        raise IndexError(f"Wire token '{token}' resolved to {idx}, but circuit has {num_qubits} qubits")
    return idx


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
        raise NotImplementedError(f"Unsupported multi-control V gate with {len(args) - 1} controls")


def _apply_f_family(qc: QuantumCircuit, args: List[int]) -> None:
    """
    Fredkin-family handling.

    Note:
    - 2-wire case uses SWAP directly
    - 3-wire case uses native CSWAP directly
    - >3 controls uses a custom decomposition into X-family primitives.
      This is not a native multi-controlled Fredkin gate object.
    """
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
        raise NotImplementedError(f"Unsupported Peres-style gate with {len(args)} wires")
    _apply_x_family(qc, args)
    qc.cx(args[0], args[1])


def _expand_macro(tokens: List[str], params: List[str], body: List[List[str]]) -> List[List[str]]:
    if len(tokens) - 1 != len(params):
        raise ValueError(f"Macro arity mismatch for {tokens[0]}: expected {len(params)}, got {len(tokens) - 1}")
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
    num_qubits = None
    variables: List[str] = []
    gate_lines: List[List[str]] = []
    macros: Dict[str, MacroDef] = {}
    bus_tokens: Dict[str, List[str]] = {}

    current_macro_name: str | None = None
    current_macro_params: List[str] = []
    current_macro_body: List[List[str]] = []
    in_circuit = False

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
        elif head == ".end":
            if in_circuit:
                break
        elif head in HEADER_DIRECTIVES or head.startswith("."):
            continue
        else:
            gate_lines.append(tokens)

    if current_macro_name is not None:
        raise ValueError(f"Unterminated macro definition '{current_macro_name}' in {path.name}")

    if num_qubits is None:
        if variables:
            num_qubits = len(variables)
        else:
            raise ValueError(f"Could not infer number of qubits from {path}")

    if not variables:
        variables = [f"x{i}" for i in range(num_qubits)]

    index_of: Dict[str, int] = {name: i for i, name in enumerate(variables)}
    buses: BusMap = {}
    for alias, members in bus_tokens.items():
        resolved_members = [index_of[name] for name in members if name in index_of]
        if resolved_members:
            buses[alias] = resolved_members

    qc = QuantumCircuit(num_qubits, name=path.stem)
    for tokens in gate_lines:
        _execute_gate(qc, tokens, path, index_of, buses, macros)
    return qc


def convert_dir(real_dir: Path, out_dir: Path, split_ratio: Sequence[float], seed: int) -> None:
    files = sorted(real_dir.rglob("*.real"))
    if not files:
        raise FileNotFoundError(f"No .real files found under {real_dir}")

    rows = []
    for path in files:
        try:
            qc = parse_real_file(path)
        except Exception as exc:
            print(f"[SKIP] {path.name}: {exc}")
            continue
        rows.append(
            {
                "name": path.stem,
                "family": path.parent.name if path.parent != real_dir else "revlib",
                "num_qubits": qc.num_qubits,
                "qasm": dumps(qc),
            }
        )

    if not rows:
        raise RuntimeError("No .real files were successfully converted.")

    rng = random.Random(seed)
    rng.shuffle(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(rows)
    n_train = int(n_total * split_ratio[0])
    n_valid = int(n_total * split_ratio[1])

    save_json(out_dir / "train.json", rows[:n_train])
    save_json(out_dir / "valid.json", rows[n_train:n_train + n_valid])
    save_json(out_dir / "test.json", rows[n_train + n_valid:])

    print(f"Converted {len(rows)} circuits into {out_dir}")
    print(f"train={n_train}, valid={n_valid}, test={len(rows) - n_train - n_valid}, seed={seed}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--split_ratio", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    convert_dir(Path(args.real_dir), Path(args.out_dir), args.split_ratio, args.seed)


if __name__ == "__main__":
    main()
