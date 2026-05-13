from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Improve reproducibility across runs.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _to_numpy_if_sequence(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        try:
            return np.asarray(v)
        except Exception:
            return v
    return v


def obs_to_torch(obs: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in obs.items():
        v = _to_numpy_if_sequence(v)

        if isinstance(v, np.ndarray):
            if v.dtype.kind in ["i", "u"]:
                out[k] = torch.as_tensor(v, dtype=torch.long, device=device).unsqueeze(0)
            elif v.dtype.kind in ["b"]:
                out[k] = torch.as_tensor(v.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(0)
            else:
                out[k] = torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        elif isinstance(v, (int, np.integer)):
            out[k] = torch.as_tensor([v], dtype=torch.long, device=device)
        elif isinstance(v, (float, np.floating)):
            out[k] = torch.as_tensor([v], dtype=torch.float32, device=device)
        elif isinstance(v, (bool, np.bool_)):
            out[k] = torch.as_tensor([float(v)], dtype=torch.float32, device=device)
        else:
            raise TypeError(f"Unsupported obs field {k}: {type(v)}")
    return out


def save_json(path: str, obj: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
