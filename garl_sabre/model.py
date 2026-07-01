from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class ResidualGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, edge_feat_dim: int = 0) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_feat_dim = edge_feat_dim
        if edge_feat_dim > 0:
            self.edge_proj = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.self_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.neigh_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.gate_layers = nn.ModuleList([nn.Linear(hidden_dim * 4, hidden_dim) for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    @staticmethod
    def _masked_mean(x: torch.Tensor, node_mask: torch.Tensor | None) -> torch.Tensor:
        if node_mask is None:
            return x.mean(dim=1)
        denom = node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * node_mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor | None = None, edge_feat: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1)

        for self_layer, neigh_layer, gate_layer, ffn, norm1, norm2 in zip(
            self.self_layers,
            self.neigh_layers,
            self.gate_layers,
            self.ffn_layers,
            self.norm1,
            self.norm2,
        ):
            neigh = torch.matmul(adj, h)

            if self.edge_feat_dim > 0 and edge_feat is not None:
                edge_gate = torch.sigmoid(self.edge_proj(edge_feat))
                edge_present = (edge_feat.abs().sum(dim=-1, keepdim=True) > 0).float()
                edge_weight = adj.unsqueeze(-1) * edge_present
                denom = edge_weight.sum(dim=2).clamp_min(1e-6)
                edge_ctx = (edge_weight * edge_gate * h.unsqueeze(1)).sum(dim=2) / denom
                has_edges = (edge_present.sum(dim=2) > 0).float()
                neigh = neigh + 0.5 * edge_ctx * has_edges

            mix = torch.cat([h, neigh, h - neigh, h * neigh], dim=-1)
            gate = torch.sigmoid(gate_layer(mix))
            msg = F.relu(self_layer(h) + neigh_layer(neigh))
            h = norm1(h + self.dropout(gate * msg))

            global_ctx = self._masked_mean(h, node_mask).unsqueeze(1)
            ff = ffn(h + global_ctx)
            h = norm2(h + self.dropout(ff))

            if node_mask is not None:
                h = h * node_mask.unsqueeze(-1)
        return h


class ConditionalPlacementContext(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.key_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.val_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU())
        self.scale = hidden_dim ** 0.5
        self.out_dim = out_dim

    def forward(
        self,
        query_logic: torch.Tensor,
        pair_states: torch.Tensor,
        placed_logic_mask: torch.Tensor,
    ) -> torch.Tensor:
        keys = self.key_proj(pair_states)
        vals = self.val_proj(pair_states)
        query = self.query_proj(query_logic).unsqueeze(1)
        scores = torch.matmul(query, keys.transpose(1, 2)).squeeze(1) / self.scale
        scores = scores.masked_fill(placed_logic_mask <= 0.5, -1e9)

        no_pair = placed_logic_mask.sum(dim=1, keepdim=True) <= 0.5
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(no_pair, torch.zeros_like(attn), attn)
        ctx = torch.bmm(attn.unsqueeze(1), vals).squeeze(1)
        out = self.out_proj(ctx)
        return torch.where(no_pair, torch.zeros_like(out), out)


class MappingSummary(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.pair_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.hidden_dim = hidden_dim

    def forward(self, pair_states: torch.Tensor, placed_logic_mask: torch.Tensor) -> torch.Tensor:
        pair_emb = self.pair_proj(pair_states) 
        scores = self.attn_score(pair_emb).squeeze(-1) 
        scores = scores.masked_fill(placed_logic_mask <= 0.5, -1e9)
        attn_weights = torch.softmax(scores, dim=-1) 
        
        no_pair = placed_logic_mask.sum(dim=1, keepdim=True) <= 0.5
        attn_weights = torch.where(no_pair, torch.zeros_like(attn_weights), attn_weights)
        
        summary = torch.bmm(attn_weights.unsqueeze(1), pair_emb).squeeze(1)
        return torch.where(no_pair, torch.zeros_like(summary), summary)


class GraphAwarePolicy(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        logic_feat_dim: int,
        phys_feat_dim: int,
        edge_feat_dim: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.logic_feat_dim = logic_feat_dim
        self.phys_feat_dim = phys_feat_dim
        self.edge_feat_dim = edge_feat_dim

        dynamic_logic_in = logic_feat_dim + 1 + phys_feat_dim
        dynamic_phys_in = phys_feat_dim + 1 + logic_feat_dim

        self.logic_encoder = ResidualGraphEncoder(dynamic_logic_in, cfg.hidden_dim, cfg.graph_layers, cfg.dropout, edge_feat_dim=edge_feat_dim)
        self.phys_encoder = ResidualGraphEncoder(dynamic_phys_in, cfg.hidden_dim, cfg.graph_layers, cfg.dropout, edge_feat_dim=0)
        
        self.context = ConditionalPlacementContext(cfg.hidden_dim, cfg.placement_dim)
        self.mapping_summary = MappingSummary(cfg.hidden_dim)

        logical_in = cfg.hidden_dim + cfg.hidden_dim + 1
        self.logical_head = nn.Sequential(
            nn.Linear(logical_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

        physical_pair_in = (
            cfg.hidden_dim * 2
            + cfg.placement_dim
            + 4
        )

        pointer_query_in = cfg.hidden_dim + cfg.placement_dim + 1
        self.physical_query_proj = nn.Sequential(
            nn.Linear(pointer_query_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        pointer_key_in = cfg.hidden_dim + 4
        self.physical_key_proj = nn.Sequential(
            nn.Linear(pointer_key_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        self.physical_pointer_pair_bias = nn.Sequential(
            nn.Linear(physical_pair_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 4 + 2, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.pointer_scale = float(cfg.hidden_dim) ** 0.5

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

    @staticmethod
    def _gather_batched(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        batch_idx = torch.arange(x.shape[0], device=x.device)
        return x[batch_idx, idx]

    def _mapped_pair_states(
        self,
        logic_emb: torch.Tensor,
        phys_emb: torch.Tensor,
        mapping: torch.Tensor,
        placed_logic_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe_mapping = mapping.clamp(min=0)
        safe_mapping = safe_mapping.long().unsqueeze(-1).expand(-1, -1, phys_emb.shape[-1])
        gathered_phys = torch.gather(phys_emb, 1, safe_mapping)
        gathered_phys = gathered_phys * placed_logic_mask.unsqueeze(-1)
        return torch.cat([logic_emb, gathered_phys], dim=-1)

    def _prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logic_x = batch["logic_node_features"]
        logic_adj = batch.get("logic_adj_weighted", batch["logic_adj"])
        phys_x = batch["physical_node_features"]
        phys_adj = batch.get("physical_adj_weighted", batch["physical_adj"])
        mapping = batch["mapping"]
        used_phys = batch["used_phys"].float()
        free_physical_mask = batch["free_physical_mask"].float()
        logical_action_mask = batch["logical_action_mask"].float()
        physical_action_masks_bank = batch["physical_action_masks_bank"].float()
        
        # 使用硬件中心性特征作为物理先验
        phys_centrality = batch.get("physical_centrality", None)
        if phys_centrality is None:
            physical_prior_bank = torch.zeros(
                (logic_x.shape[0], logic_x.shape[1], phys_x.shape[1]),
                dtype=torch.float32,
                device=logic_x.device,
            )
        else:
            # 每个逻辑比特看到相同的物理先验
            phys_prior = phys_centrality.unsqueeze(1).expand(
                logic_x.shape[0], logic_x.shape[1], -1
            )
            physical_prior_bank = phys_prior

        dynamic_prior_bank = batch.get("physical_prior_bank")
        if dynamic_prior_bank is not None:
            dynamic_prior_bank = dynamic_prior_bank.float()
            if dynamic_prior_bank.shape[1] == logic_x.shape[1] and dynamic_prior_bank.shape[2] == phys_x.shape[1]:
                physical_prior_bank = dynamic_prior_bank
        
        # 准备边特征：堆叠所有原始统计量让 GNN 学习最优组合
        edge_feat_list = []
        for key in ["edge_freq", "edge_first_layer", "edge_front", "edge_early", "edge_log_f"]:
            ef = batch.get(key)
            if ef is not None:
                edge_feat_list.append(ef)
        
        if edge_feat_list and self.edge_feat_dim > 0:
            logic_edge_feat = torch.stack(edge_feat_list, dim=-1)
        else:
            logic_edge_feat = None
        progress = batch["progress"].float().view(-1, 1)

        logic_node_mask = batch.get("logic_node_features_mask")
        if logic_node_mask is None:
            logic_node_mask = batch.get("mapping_mask")
        if logic_node_mask is None:
            logic_node_mask = torch.ones(logic_x.shape[:2], device=logic_x.device, dtype=torch.float32)
        else:
            logic_node_mask = logic_node_mask.float()

        phys_node_mask = batch.get("physical_node_features_mask")
        if phys_node_mask is None:
            phys_node_mask = torch.ones(phys_x.shape[:2], device=phys_x.device, dtype=torch.float32)
        else:
            phys_node_mask = phys_node_mask.float()

        placed_logic_mask = ((mapping >= 0).float() * logic_node_mask).float()
        safe_mapping = mapping.clamp(min=0).long()

        gathered_phys_x = torch.gather(
            phys_x,
            dim=1,
            index=safe_mapping.unsqueeze(-1).expand(-1, -1, phys_x.shape[-1])
        )
        gathered_phys_x = gathered_phys_x * placed_logic_mask.unsqueeze(-1)
        logic_x_dynamic = torch.cat([
            logic_x,
            placed_logic_mask.unsqueeze(-1),
            gathered_phys_x
        ], dim=-1)

        mapping_one_hot = F.one_hot(safe_mapping, num_classes=phys_x.shape[1]).float()
        mapping_one_hot = mapping_one_hot * placed_logic_mask.unsqueeze(-1)
        gathered_logic_x = torch.bmm(mapping_one_hot.transpose(1, 2), logic_x)
        phys_x_dynamic = torch.cat([
            phys_x,
            used_phys.unsqueeze(-1),
            gathered_logic_x
        ], dim=-1)

        logic_emb = self.logic_encoder(logic_x_dynamic, logic_adj, logic_node_mask, edge_feat=logic_edge_feat)
        phys_emb = self.phys_encoder(phys_x_dynamic, phys_adj, phys_node_mask)
        
        pair_states = self._mapped_pair_states(logic_emb, phys_emb, mapping, placed_logic_mask)
        
        mapping_summary = self.mapping_summary(pair_states, placed_logic_mask)
        
        global_logic = self._masked_mean(logic_emb, logic_node_mask)
        global_phys = self._masked_mean(phys_emb, phys_node_mask)
        available_logic_summary = self._masked_mean(logic_emb, logical_action_mask * logic_node_mask)
        occupancy = used_phys.mean(dim=1, keepdim=True)

        return {
            "logic_emb": logic_emb,
            "phys_emb": phys_emb,
            "mapping": mapping,
            "used_phys": used_phys,
            "free_physical_mask": free_physical_mask,
            "logical_action_mask": logical_action_mask,
            "physical_action_masks_bank": physical_action_masks_bank,
            "physical_prior_bank": physical_prior_bank,
            "progress": progress,
            "logic_node_mask": logic_node_mask,
            "phys_node_mask": phys_node_mask,
            "placed_logic_mask": placed_logic_mask,
            "pair_states": pair_states,
            "mapping_summary": mapping_summary,
            "global_logic": global_logic,
            "global_phys": global_phys,
            "available_logic_summary": available_logic_summary,
            "occupancy": occupancy,
        }

    def _logical_logits(self, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        logic_emb = enc["logic_emb"]
        logical_action_mask = enc["logical_action_mask"]
        logic_node_mask = enc["logic_node_mask"]
        mapping_summary = enc["mapping_summary"]
        progress = enc["progress"]

        summary_exp = mapping_summary.unsqueeze(1).expand(-1, logic_emb.shape[1], -1)
        progress_exp = progress.unsqueeze(1).expand(-1, logic_emb.shape[1], -1)

        logits = self.logical_head(
            torch.cat([logic_emb, summary_exp, progress_exp], dim=-1)
        ).squeeze(-1)

        valid_mask = logical_action_mask * logic_node_mask
        logits = logits.masked_fill(valid_mask <= 0, -1e9)
        return logits

    def _build_prior_feature(self, physical_prior: torch.Tensor, use_physical_prior: bool) -> torch.Tensor:
        if not use_physical_prior or self.cfg.physical_prior_scale <= 0.0:
            return torch.zeros_like(physical_prior)
        return self.cfg.physical_prior_scale * physical_prior.clamp(
            min=-self.cfg.physical_prior_clip,
            max=self.cfg.physical_prior_clip,
        )

    def _physical_logits(
        self,
        enc: Dict[str, torch.Tensor],
        selected_logical_idx: torch.Tensor,
        use_physical_prior: bool = True,
    ) -> torch.Tensor:
        logic_emb = enc["logic_emb"]
        phys_emb = enc["phys_emb"]
        free_physical_mask = enc["free_physical_mask"]
        used_phys = enc["used_phys"]
        physical_action_masks_bank = enc["physical_action_masks_bank"]
        physical_prior_bank = enc["physical_prior_bank"]
        placed_logic_mask = enc["placed_logic_mask"]
        pair_states = enc["pair_states"]
        progress = enc["progress"]

        current_logic = self._gather_batched(logic_emb, selected_logical_idx)
        placement_ctx = self.context(current_logic, pair_states, placed_logic_mask)
        physical_mask = self._gather_batched(physical_action_masks_bank, selected_logical_idx) * free_physical_mask
        physical_prior = self._gather_batched(physical_prior_bank, selected_logical_idx)

        cur_logic_exp = current_logic.unsqueeze(1).expand(-1, phys_emb.shape[1], -1)
        ctx_exp = placement_ctx.unsqueeze(1).expand(-1, phys_emb.shape[1], -1)
        progress_exp = progress.unsqueeze(1).expand(-1, phys_emb.shape[1], -1)
        free_exp = free_physical_mask.unsqueeze(-1)
        used_exp = used_phys.unsqueeze(-1)
        prior_hint = self._build_prior_feature(physical_prior, use_physical_prior).unsqueeze(-1)

        pair_features = torch.cat(
            [
                cur_logic_exp,
                phys_emb,
                ctx_exp,
                free_exp,
                used_exp,
                progress_exp,
                prior_hint,
            ],
            dim=-1,
        )

        query = self.physical_query_proj(torch.cat([current_logic, placement_ctx, progress], dim=-1))
        keys = self.physical_key_proj(torch.cat([phys_emb, free_exp, used_exp, progress_exp, prior_hint], dim=-1))
        pointer_logits = torch.sum(keys * query.unsqueeze(1), dim=-1) / self.pointer_scale
        pointer_bias = self.physical_pointer_pair_bias(pair_features).squeeze(-1)

        logits = pointer_logits + pointer_bias
        logits = logits.masked_fill(physical_mask <= 0, -1e9)
        return logits

    def _value(self, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.value_head(
            torch.cat(
                [
                    enc["global_logic"],
                    enc["global_phys"],
                    enc["mapping_summary"],
                    enc["available_logic_summary"],
                    enc["occupancy"],
                    enc["progress"],
                ],
                dim=-1,
            )
        ).squeeze(-1)

    def get_step_logits(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enc = self._prepare_inputs(batch)
        return {"enc": enc, "logical_logits": self._logical_logits(enc)}

    def get_physical_logits(
        self,
        batch_or_enc: Dict[str, torch.Tensor],
        selected_logical_idx: torch.Tensor,
        use_physical_prior: bool = True,
    ) -> torch.Tensor:
        enc = batch_or_enc["enc"] if "enc" in batch_or_enc else batch_or_enc
        return self._physical_logits(enc, selected_logical_idx.long(), use_physical_prior=use_physical_prior)

    def evaluate_actions(
        self,
        batch: Dict[str, torch.Tensor],
        action_logical: torch.Tensor,
        action_physical: torch.Tensor,
        use_physical_prior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        enc = self._prepare_inputs(batch)
        
        # 核心修复 4：增加多模式批次一致性断言，避免混入了脏数据
        if "is_fixed_order" in batch:
            assert torch.all(batch["is_fixed_order"] == batch["is_fixed_order"][0]), "Batch mixes fixed_order and hierarchical modes."
            is_fixed_order = batch["is_fixed_order"][0].item() > 0
        else:
            is_fixed_order = False

        if is_fixed_order:
            logical_logits = torch.zeros_like(batch["logical_action_mask"])
            logical_logprob = torch.zeros_like(action_logical, dtype=torch.float32)
            logical_entropy = torch.zeros_like(action_logical, dtype=torch.float32)
        else:
            logical_logits = self._logical_logits(enc)
            logical_dist = torch.distributions.Categorical(logits=logical_logits)
            logical_logprob = logical_dist.log_prob(action_logical.long())
            logical_entropy = logical_dist.entropy()

        physical_logits = self._physical_logits(enc, action_logical.long(), use_physical_prior=use_physical_prior)
        physical_dist = torch.distributions.Categorical(logits=physical_logits)
        physical_logprob = physical_dist.log_prob(action_physical.long())
        physical_entropy = physical_dist.entropy()

        entropy = logical_entropy + physical_entropy
        value = self._value(enc)

        return {
            "logical_logits": logical_logits,
            "physical_logits": physical_logits,
            "logical_logprob": logical_logprob,
            "physical_logprob": physical_logprob,
            "logprob": logical_logprob + physical_logprob,
            "entropy": entropy,
            "value": value,
        }

    def act(
        self,
        batch: Dict[str, torch.Tensor],
        deterministic: bool = False,
        use_physical_prior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        enc = self._prepare_inputs(batch)
        
        # 核心修复 4：同理增加批次验证保护机制
        if "is_fixed_order" in batch:
            assert torch.all(batch["is_fixed_order"] == batch["is_fixed_order"][0]), "Batch mixes fixed_order and hierarchical modes."
            is_fixed_order = batch["is_fixed_order"][0].item() > 0
        else:
            is_fixed_order = False

        if is_fixed_order:
            logical_action = batch["current_logical_idx"].long()
            logical_logprob = torch.zeros_like(batch["progress"].squeeze(-1))
            logical_logits = torch.zeros_like(batch["logical_action_mask"])
        else:
            logical_logits = self._logical_logits(enc)
            logical_dist = torch.distributions.Categorical(logits=logical_logits)
            logical_action = torch.argmax(logical_logits, dim=-1) if deterministic else logical_dist.sample()
            logical_logprob = logical_dist.log_prob(logical_action)

        physical_logits = self._physical_logits(enc, logical_action, use_physical_prior=use_physical_prior)
        physical_dist = torch.distributions.Categorical(logits=physical_logits)
        physical_action = torch.argmax(physical_logits, dim=-1) if deterministic else physical_dist.sample()

        physical_logprob = physical_dist.log_prob(physical_action)
        value = self._value(enc)

        return {
            "action_logical": logical_action,
            "action_physical": physical_action,
            "logprob": logical_logprob + physical_logprob,
            "logical_logprob": logical_logprob,
            "physical_logprob": physical_logprob,
            "value": value,
            "logical_logits": logical_logits,
            "physical_logits": physical_logits,
        }
