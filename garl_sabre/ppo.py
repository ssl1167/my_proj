from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import PPOConfig


class RolloutDataset(Dataset):
    def __init__(self, rows: List[Dict[str, torch.Tensor]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.rows[idx]


@dataclass
class TrajectoryBuffer:
    obs: List[Dict]
    actions_logical: List[int]
    actions_physical: List[int]
    logprobs: List[float]
    rewards: List[float]
    dones: List[bool]
    values: List[float]

    def __init__(self) -> None:
        self.obs = []
        self.actions_logical = []
        self.actions_physical = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(
        self,
        obs: Dict,
        action_logical: int,
        action_physical: int,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.obs.append(obs)
        self.actions_logical.append(action_logical)
        self.actions_physical.append(action_physical)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_advantages(self, cfg: PPOConfig, last_value: float = 0.0) -> List[Dict]:
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + cfg.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        rows: List[Dict[str, torch.Tensor]] = []
        for i in range(len(rewards)):
            row: Dict[str, torch.Tensor] = {}
            for k, v in self.obs[i].items():
                if k == "done":
                    continue
                row[k] = torch.as_tensor(v)
            row["action_logical"] = torch.tensor(self.actions_logical[i], dtype=torch.long)
            row["action_physical"] = torch.tensor(self.actions_physical[i], dtype=torch.long)
            row["old_logprob"] = torch.tensor(self.logprobs[i], dtype=torch.float32)
            row["old_value"] = torch.tensor(self.values[i], dtype=torch.float32)
            row["advantage"] = torch.tensor(advantages[i], dtype=torch.float32)
            row["return"] = torch.tensor(returns[i], dtype=torch.float32)
            rows.append(row)
        return rows


def _can_pad_along_dim0(tensors: List[torch.Tensor]) -> bool:
    ref = tensors[0]
    ref_ndim = ref.dim()
    ref_tail = tuple(ref.shape[1:])
    for t in tensors[1:]:
        if t.dim() != ref_ndim:
            return False
        if tuple(t.shape[1:]) != ref_tail:
            return False
    return True


def _pad_along_dim0(tensors: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(t.shape[0]) for t in tensors)
    batch_size = len(tensors)
    ref = tensors[0]
    out_shape = (batch_size, max_len) + tuple(ref.shape[1:])
    padded = ref.new_full(out_shape, pad_value)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
    for i, t in enumerate(tensors):
        cur_len = int(t.shape[0])
        padded[i, :cur_len] = t
        mask[i, :cur_len] = 1.0
    return padded, mask


def _can_pad_square_matrix(tensors: List[torch.Tensor]) -> bool:
    return all(t.dim() == 2 and t.shape[0] == t.shape[1] for t in tensors)


def _pad_square_matrices(tensors: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    max_n = max(int(t.shape[0]) for t in tensors)
    batch_size = len(tensors)
    ref = tensors[0]
    padded = ref.new_full((batch_size, max_n, max_n), pad_value)
    mask = torch.zeros(batch_size, max_n, dtype=torch.float32)
    for i, t in enumerate(tensors):
        n = int(t.shape[0])
        padded[i, :n, :n] = t
        mask[i, :n] = 1.0
    return padded, mask


def collate_rows(rows: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    keys = rows[0].keys()
    for k in keys:
        vals = [r[k] for r in rows]
        if not all(isinstance(v, torch.Tensor) for v in vals):
            raise TypeError(f"All values for key '{k}' must be torch.Tensor.")
        if vals[0].dim() == 0:
            out[k] = torch.stack(vals, dim=0)
            continue
        same_shape = all(tuple(v.shape) == tuple(vals[0].shape) for v in vals)
        if same_shape:
            out[k] = torch.stack(vals, dim=0)
            continue
        if _can_pad_along_dim0(vals):
            padded, mask = _pad_along_dim0(vals, pad_value=0.0)
            out[k] = padded
            out[f"{k}_mask"] = mask
            continue
        if _can_pad_square_matrix(vals):
            padded, mask = _pad_square_matrices(vals, pad_value=0.0)
            out[k] = padded
            out[f"{k}_mask"] = mask
            continue
        shapes = [tuple(v.shape) for v in vals]
        raise RuntimeError(f"Cannot collate key '{k}' because shapes are incompatible for padding: {shapes}")
    return out


def ppo_update(model, optimizer, rows: List[Dict[str, torch.Tensor]], cfg: PPOConfig, device: torch.device) -> Dict[str, float]:
    if not rows:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

    adv_tensor = torch.stack([x["advantage"] for x in rows])
    adv_mean = adv_tensor.mean()
    adv_std = adv_tensor.std() + 1e-8

    normalized_rows: List[Dict[str, torch.Tensor]] = []
    for r in rows:
        rr = dict(r)
        rr["advantage"] = (rr["advantage"] - adv_mean) / adv_std
        normalized_rows.append(rr)

    dataset = RolloutDataset(normalized_rows)
    loader = DataLoader(
        dataset,
        batch_size=min(cfg.minibatch_size, len(dataset)),
        shuffle=True,
        collate_fn=collate_rows,
    )

    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    steps = 0

    model.train()
    target_kl = getattr(cfg, "target_kl", 0.08)
    value_clip = getattr(cfg, "value_clip", 0.2)
    early_stop = False

    for _ in range(cfg.train_iters):
        if early_stop:
            break
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            eval_out = model.evaluate_actions(batch, batch["action_logical"], batch["action_physical"])
            new_logprob = eval_out["logprob"]
            values = eval_out["value"]
            entropy = eval_out["entropy"].mean()
            old_logprob = batch["old_logprob"]

            ratio = torch.exp(new_logprob - old_logprob)
            adv = batch["advantage"]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            returns = batch["return"]
            old_values = batch["old_value"]
            value_pred_clipped = old_values + torch.clamp(values - old_values, -value_clip, value_clip)
            value_loss_unclipped = (values - returns) ** 2
            value_loss_clipped = (value_pred_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            approx_kl = (old_logprob - new_logprob).mean().abs().item()
            metrics["loss"] += float(loss.item())
            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.item())
            metrics["approx_kl"] += float(approx_kl)
            steps += 1

            if approx_kl > target_kl:
                early_stop = True
                break

    if steps > 0:
        for k in metrics:
            metrics[k] /= steps
    return metrics
