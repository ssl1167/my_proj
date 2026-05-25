from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz, TwoLocal, QFT
from qiskit.quantum_info import SparsePauliOp

from .circuit_features import LogicGraphData, build_logic_graph
from .config import EnvConfig

CNOT_CANONICAL_BASIS = ["rz", "sx", "x", "cx"]


def _instruction_parts(inst):
    if hasattr(inst, "operation"):
        return inst.operation, inst.qubits, inst.clbits
    op = inst[0]
    qargs = inst[1]
    cargs = inst[2] if len(inst) > 2 else []
    return op, qargs, cargs


def _qasm2_dumps(circ: QuantumCircuit) -> str:
    from qiskit.qasm2 import dumps
    return dumps(circ)


def decompose_to_cx_basis(circ: QuantumCircuit) -> QuantumCircuit:
    return transpile(circ, basis_gates=CNOT_CANONICAL_BASIS, optimization_level=0)


def canonical_cnot_circuit(circ: QuantumCircuit) -> QuantumCircuit:
    """
    Canonical benchmark representation following the C++ QASMReader protocol:
    decompose to CX basis, keep only CX gates, keep only CX-active qubits, and
    compactly renumber those qubits.
    """
    work = decompose_to_cx_basis(circ)
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
        return QuantumCircuit(max(1, min(circ.num_qubits, 1)), name=circ.name)

    active_sorted = sorted(active)
    remap = {old: new for new, old in enumerate(active_sorted)}
    out = QuantumCircuit(len(active_sorted), name=circ.name)
    for c, t in cx_ops:
        out.cx(remap[c], remap[t])
    return out


def sanitize_qasm_record(
    name: str,
    family: str,
    num_qubits: int,
    qasm: str,
    canonicalize_cnot_only: bool = True,
) -> Dict[str, Any]:
    del num_qubits
    circ = QuantumCircuit.from_qasm_str(str(qasm))
    if canonicalize_cnot_only:
        circ = canonical_cnot_circuit(circ)
    return {"name": str(name), "family": str(family), "num_qubits": int(circ.num_qubits), "qasm": _qasm2_dumps(circ)}


@dataclass
class CircuitSample:
    name: str
    family: str
    num_qubits: int
    qasm: str

    _cached_circuit: QuantumCircuit | None = field(default=None, init=False, repr=False, compare=False)
    _cached_logic_graph: LogicGraphData | None = field(default=None, init=False, repr=False, compare=False)
    _cached_logic_graph_key: Tuple | None = field(default=None, init=False, repr=False, compare=False)
    _cached_baseline: Tuple[Optional[float], str, Dict[str, float]] | None = field(default=None, init=False, repr=False, compare=False)

    def to_circuit(self) -> QuantumCircuit:
        if self._cached_circuit is None:
            circ = QuantumCircuit.from_qasm_str(self.qasm)
            if circ.num_qubits != int(self.num_qubits):
                self.num_qubits = int(circ.num_qubits)
            self._cached_circuit = circ
        return self._cached_circuit.copy()

    def clear_cache(self) -> None:
        self._cached_circuit = None
        self._cached_logic_graph = None
        self._cached_logic_graph_key = None
        self._cached_baseline = None

    def get_logic_graph(self, env_cfg: EnvConfig) -> LogicGraphData:
        basis_key = tuple(getattr(env_cfg, "basis_gates", []) or [])
        key = (
            int(env_cfg.critical_window),
            int(env_cfg.lookahead_window),
            basis_key,
            "cnot_only_v1",
        )
        if self._cached_logic_graph is None or self._cached_logic_graph_key != key:
            circuit = self.to_circuit()
            try:
                self._cached_logic_graph = build_logic_graph(
                    circuit,
                    critical_window=env_cfg.critical_window,
                    lookahead_window=env_cfg.lookahead_window,
                    basis_gates=list(basis_key) if basis_key else None,
                    decompose=True,
                    cnot_only=True,
                )
            except TypeError:
                self._cached_logic_graph = build_logic_graph(
                    circuit,
                    critical_window=env_cfg.critical_window,
                    lookahead_window=env_cfg.lookahead_window,
                )
            self._cached_logic_graph_key = key
        return self._cached_logic_graph


def save_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(rows), f, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_rows(payload: Any, split: str | None = None) -> List[Dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if split and isinstance(payload.get(split), list):
            return payload[split]
        for key in ("rows", "samples", "circuits", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(
        "JSON manifest must be either a list of samples, or a dict containing one of: "
        "split/train/valid/test/rows/samples/circuits/data/items."
    )


def _validate_rows(rows: List[Dict], source: Path, canonicalize_cnot_only: bool = True) -> List[Dict]:
    required = {"name", "family", "num_qubits", "qasm"}
    validated: List[Dict] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"Invalid sample at {source}:{idx}: expected dict, got {type(row).__name__}")
        missing = required - set(row.keys())
        if missing:
            raise KeyError(f"Invalid sample at {source}:{idx}: missing keys {sorted(missing)}")
        try:
            fixed = sanitize_qasm_record(
                name=row["name"],
                family=row["family"],
                num_qubits=int(row["num_qubits"]),
                qasm=str(row["qasm"]),
                canonicalize_cnot_only=canonicalize_cnot_only,
            )
        except Exception as exc:
            raise ValueError(f"Failed to parse/sanitize sample at {source}:{idx} ({row.get('name', 'unknown')}): {exc}") from exc
        validated.append(fixed)
    return validated


def load_records(path: Path, split: str | None = None, canonicalize_cnot_only: bool = True, trim_idle_qubits: bool | None = None) -> List[Dict]:
    # trim_idle_qubits is accepted for backward compatibility; canonicalize_cnot_only is the new protocol switch.
    if trim_idle_qubits is not None:
        canonicalize_cnot_only = bool(trim_idle_qubits)

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".jl"}:
        rows = load_jsonl(path)
        return _validate_rows(rows, path, canonicalize_cnot_only=canonicalize_cnot_only)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = _normalize_rows(payload, split=split)
        return _validate_rows(rows, path, canonicalize_cnot_only=canonicalize_cnot_only)
    raise ValueError(f"Unsupported manifest format: {path}")


def _default_split_path(dataset_dir: str, split: str) -> Path:
    base = Path(dataset_dir)
    if base.is_file():
        return base
    json_path = base / f"{split}.json"
    if json_path.exists():
        return json_path
    jsonl_path = base / f"{split}.jsonl"
    if jsonl_path.exists():
        return jsonl_path
    return json_path


def load_split(
    dataset_dir: str,
    split: str,
    split_path: str | None = None,
    canonicalize_cnot_only: bool = True,
    trim_idle_qubits: bool | None = None,
) -> List[CircuitSample]:
    if trim_idle_qubits is not None:
        canonicalize_cnot_only = bool(trim_idle_qubits)
    path = Path(split_path) if split_path else _default_split_path(dataset_dir, split)
    rows = load_records(path, split=split, canonicalize_cnot_only=canonicalize_cnot_only)
    return [CircuitSample(**row) for row in rows]


def bind_if_parameterized(circ: QuantumCircuit, seed: int) -> QuantumCircuit:
    if circ.num_parameters == 0:
        return circ
    rng = random.Random(seed)
    bind_dict = {p: rng.uniform(0.0, 2.0 * math.pi) for p in sorted(circ.parameters, key=lambda x: x.name)}
    return circ.assign_parameters(bind_dict, inplace=False)


def circuit_to_qasm2(circ: QuantumCircuit, seed: int) -> str:
    return _qasm2_dumps(bind_if_parameterized(circ, seed))


def random_qaoa(n: int, reps: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    paulis, coeffs = [], []
    for _ in range(max(n, 2)):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        s = ["I"] * n
        s[i] = "Z"
        s[j] = "Z"
        paulis.append("".join(reversed(s)))
        coeffs.append(rng.uniform(0.5, 1.5))
    if not paulis:
        paulis = ["Z" + "I" * (n - 1)]
        coeffs = [1.0]
    ham = SparsePauliOp(paulis, coeffs=coeffs)
    return QAOAAnsatz(cost_operator=ham, reps=reps).decompose(reps=2)


def random_hea(n: int, reps: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    rotation_blocks = ["ry", "rz"] if rng.random() < 0.5 else ["rx", "ry", "rz"]
    return TwoLocal(n, rotation_blocks=rotation_blocks, entanglement_blocks="cx", reps=reps, entanglement="linear").decompose()


def random_qft(n: int) -> QuantumCircuit:
    return QFT(num_qubits=n, do_swaps=False).decompose()


def random_grover_like(n: int, seed: int, iterations: int = 2) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(iterations):
        anchors = sorted(rng.sample(range(n), k=min(max(2, n // 4), n)))
        for i in range(len(anchors) - 1):
            qc.cx(anchors[i], anchors[i + 1])
        for q in anchors:
            qc.z(q)
        for i in reversed(range(len(anchors) - 1)):
            qc.cx(anchors[i], anchors[i + 1])
        qc.h(range(n))
        qc.x(range(n))
        for i in range(0, n - 1, 2):
            qc.cz(i, i + 1)
        for i in range(1, n - 1, 2):
            qc.cz(i, i + 1)
        qc.x(range(n))
        qc.h(range(n))
    return qc


def random_adder(n: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    n = max(n, 4)
    qc = QuantumCircuit(n)
    half = n // 2
    for i in range(min(half - 1, n - half)):
        qc.cx(i, half + i)
        qc.ccx(i, half + i, min(n - 1, half + i + 1))
    for i in reversed(range(min(half - 1, n - half))):
        qc.cx(i, half + i)
    if rng.random() < 0.5:
        for i in range(0, n - 1, 2):
            qc.cx(i, i + 1)
    return qc


def random_routing_stress(n: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for _ in range(max(2 * n, 12)):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j or abs(i - j) <= 1:
            continue
        qc.cx(i, j)
        if rng.random() < 0.4:
            qc.cz(j, i)
    return qc


def random_circuit_family(n: int, family: str, seed: int) -> QuantumCircuit:
    if family == "qaoa":
        return random_qaoa(n, reps=2 if n <= 20 else 3, seed=seed)
    if family == "hea":
        return random_hea(n, reps=3 if n <= 20 else 4, seed=seed)
    if family == "qft":
        return random_qft(n)
    if family == "grover":
        return random_grover_like(n, seed=seed, iterations=2 if n <= 24 else 3)
    if family == "adder":
        return random_adder(n, seed)
    if family == "routing_stress":
        return random_routing_stress(n, seed)
    if family == "random":
        from qiskit.circuit.random import random_circuit
        return random_circuit(n, depth=max(12, n), max_operands=2, seed=seed)
    raise ValueError(f"Unknown family: {family}")


def generate_dataset(
    dataset_dir: str,
    num_circuits: int,
    qubit_choices: Sequence[int],
    seed: int = 7,
    families: Sequence[str] = ("qaoa", "hea", "qft", "grover", "adder", "random", "routing_stress"),
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> None:
    rng = random.Random(seed)
    rows = []
    for idx in range(num_circuits):
        family = families[idx % len(families)]
        n = int(rng.choice(list(qubit_choices)))
        circ_seed = rng.randint(0, 10**9)
        circ = canonical_cnot_circuit(random_circuit_family(n, family, seed=circ_seed))
        rows.append({"name": f"{family}_{circ.num_qubits}_{idx:04d}", "family": family, "num_qubits": int(circ.num_qubits), "qasm": circuit_to_qasm2(circ, circ_seed + 1)})

    rng.shuffle(rows)
    n_total = len(rows)
    n_train = int(n_total * split_ratio[0])
    n_valid = int(n_total * split_ratio[1])
    base = Path(dataset_dir)
    save_json(base / "train.json", rows[:n_train])
    save_json(base / "valid.json", rows[n_train:n_train + n_valid])
    save_json(base / "test.json", rows[n_train + n_valid:])
