from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class ResidualGraphEncoder(nn.Module):
    """Residual message passing over a dense, normalized adjacency matrix."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
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

    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor | None = None) -> torch.Tensor:
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
    """Vectorized attention over already placed (logic, physical) pairs."""

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
        # pair_states: [B, N_logic, 2H], mask: [B, N_logic]
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
    """高级自注意力布局聚合器：通过可学习的注意力评级机制，彻底消除中后期特征稀释问题。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.pair_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 显式构造瓶颈注意力评分网络
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.hidden_dim = hidden_dim

    def forward(self, pair_states: torch.Tensor, placed_logic_mask: torch.Tensor) -> torch.Tensor:
        # pair_states: [B, N_logic, 2H], placed_logic_mask: [B, N_logic]
        pair_emb = self.pair_proj(pair_states)  # [B, N_logic, H]
        
        # 1. 计算每个已放置对的危险/重要性得分
        scores = self.attn_score(pair_emb).squeeze(-1)  # [B, N_logic]
        
        # 2. 对未放置的逻辑节点执行极值掩码 (-1e9)，阻止其参与 Softmax
        scores = scores.masked_fill(placed_logic_mask <= 0.5, -1e9)
        
        # 3. 计算 Softmax 概率分布。此时如果某一步放置极差，其 score 会极高，并在分布中占据主导地位
        attn_weights = torch.softmax(scores, dim=-1)  # [B, N_logic]
        
        # 处理全域未放置的 Episode 初始状态边界
        no_pair = placed_logic_mask.sum(dim=1, keepdim=True) <= 0.5  # [B, 1]
        attn_weights = torch.where(no_pair, torch.zeros_like(attn_weights), attn_weights)
        
        # 4. 批次矩阵乘法 (BMM) 提取高内聚的全局布局特征向量
        summary = torch.bmm(attn_weights.unsqueeze(1), pair_emb).squeeze(1)  # [B, H]
        return torch.where(no_pair, torch.zeros_like(summary), summary)


class GraphAwarePolicy(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        logic_feat_dim: int,
        phys_feat_dim: int,
        candidate_feat_dim: int,
        logical_candidate_feat_dim: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.logic_feat_dim = logic_feat_dim
        self.phys_feat_dim = phys_feat_dim
        self.candidate_feat_dim = candidate_feat_dim
        self.logical_candidate_feat_dim = logical_candidate_feat_dim

        # --- 核心修改：升维通道设计 ---
        # 逻辑图输入特征维度 = 自身静态特征 + 1(放置状态位) + 映射目标的物理节点特征维度
        dynamic_logic_in = logic_feat_dim + 1 + phys_feat_dim
        # 物理图输入特征维度 = 自身静态特征 + 1(占用状态位) + 承载目标的逻辑节点特征维度
        dynamic_phys_in = phys_feat_dim + 1 + logic_feat_dim

        self.logic_encoder = ResidualGraphEncoder(dynamic_logic_in, cfg.hidden_dim, cfg.graph_layers, cfg.dropout)
        self.phys_encoder = ResidualGraphEncoder(dynamic_phys_in, cfg.hidden_dim, cfg.graph_layers, cfg.dropout)
        
        # 以下保持原样，无需改动，完美的下游尺寸兼容性
        self.context = ConditionalPlacementContext(cfg.hidden_dim, cfg.placement_dim)
        self.mapping_summary = MappingSummary(cfg.hidden_dim)

        self.logical_candidate_proj = nn.Sequential(
            nn.Linear(logical_candidate_feat_dim, cfg.logical_candidate_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.logical_candidate_hidden_dim, cfg.logical_candidate_hidden_dim),
            nn.ReLU(),
        )
        self.candidate_proj = nn.Sequential(
            nn.Linear(candidate_feat_dim, cfg.candidate_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.candidate_hidden_dim, cfg.candidate_hidden_dim),
            nn.ReLU(),
        )

        logical_in = cfg.hidden_dim + cfg.logical_candidate_hidden_dim + cfg.hidden_dim + 1
        self.logical_head = nn.Sequential(
            nn.Linear(logical_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

        physical_pair_in = (
            cfg.hidden_dim * 2
            + cfg.logical_candidate_hidden_dim
            + cfg.placement_dim
            + cfg.candidate_hidden_dim
            + 4
        )

        pointer_query_in = cfg.hidden_dim + cfg.logical_candidate_hidden_dim + cfg.placement_dim + 1
        self.physical_query_proj = nn.Sequential(
            nn.Linear(pointer_query_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        pointer_key_in = cfg.hidden_dim + cfg.candidate_hidden_dim + 4
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
        logical_candidate_features = batch["logical_candidate_features"].float()
        candidate_features_bank = batch["candidate_features_bank"].float()
        physical_action_masks_bank = batch["physical_action_masks_bank"].float()

        physical_prior_bank = batch.get("physical_prior_bank")
        if physical_prior_bank is None:
            physical_prior_bank = torch.zeros_like(candidate_features_bank[..., 0])
        else:
            physical_prior_bank = physical_prior_bank.float()

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

        # ==================== 核心修改：跨图动态特征注入引擎 ====================
        placed_logic_mask = ((mapping >= 0).float() * logic_node_mask).float()
        safe_mapping = mapping.clamp(min=0).long()

        # 1. 动态自适应组装逻辑图节点特征 (Logic -> Mapped Phys Feats)
        # 提取绑定的物理特征通道
        gathered_phys_x = torch.gather(
            phys_x,
            dim=1,
            index=safe_mapping.unsqueeze(-1).expand(-1, -1, phys_x.shape[-1])
        )
        # 清空未放置节点的干扰信号
        gathered_phys_x = gathered_phys_x * placed_logic_mask.unsqueeze(-1)
        # 张量拼接：[B, N_logic, logic_feat_dim + 1 + phys_feat_dim]
        logic_x_dynamic = torch.cat([
            logic_x,
            placed_logic_mask.unsqueeze(-1),
            gathered_phys_x
        ], dim=-1)

        # 2. 动态自适应组装物理图节点特征 (Phys -> Hosted Logic Feats)
        # 通过 one_hot 构建精准的逻辑-物理双射映射关联张量矩阵
        mapping_one_hot = F.one_hot(safe_mapping, num_classes=phys_x.shape[1]).float()
        mapping_one_hot = mapping_one_hot * placed_logic_mask.unsqueeze(-1)
        # 利用高效的批次矩阵乘法转置，一次性反向聚集映射到各物理节点的逻辑特征
        # [B, N_phys, N_logic] x [B, N_logic, logic_feat_dim] -> [B, N_phys, logic_feat_dim]
        gathered_logic_x = torch.bmm(mapping_one_hot.transpose(1, 2), logic_x)
        # 张量拼接：[B, N_phys, phys_feat_dim + 1 + logic_feat_dim]
        phys_x_dynamic = torch.cat([
            phys_x,
            used_phys.unsqueeze(-1),
            gathered_logic_x
        ], dim=-1)
        # ======================================================================

        # 让 GNN 在运行多层 Message Passing 时，自发感应当前的图映射状态流
        logic_emb = self.logic_encoder(logic_x_dynamic, logic_adj, logic_node_mask)
        phys_emb = self.phys_encoder(phys_x_dynamic, phys_adj, phys_node_mask)
        
        # 下游计算流完全保持原样，GNN 输出的 hidden_dim 完美向后兼容
        logical_feat_emb = self.logical_candidate_proj(logical_candidate_features)
        pair_states = self._mapped_pair_states(logic_emb, phys_emb, mapping, placed_logic_mask)
        
        # 激活全新的注意力门控池化机制
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
            "logical_feat_emb": logical_feat_emb,
            "candidate_features_bank": candidate_features_bank,
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
        logical_feat_emb = enc["logical_feat_emb"]
        logical_action_mask = enc["logical_action_mask"]
        logic_node_mask = enc["logic_node_mask"]
        mapping_summary = enc["mapping_summary"]
        progress = enc["progress"]

        summary_exp = mapping_summary.unsqueeze(1).expand(-1, logic_emb.shape[1], -1)
        progress_exp = progress.unsqueeze(1).expand(-1, logic_emb.shape[1], -1)

        logits = self.logical_head(
            torch.cat([logic_emb, logical_feat_emb, summary_exp, progress_exp], dim=-1)
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
        logical_feat_emb = enc["logical_feat_emb"]
        free_physical_mask = enc["free_physical_mask"]
        used_phys = enc["used_phys"]
        candidate_features_bank = enc["candidate_features_bank"]
        physical_action_masks_bank = enc["physical_action_masks_bank"]
        physical_prior_bank = enc["physical_prior_bank"]
        placed_logic_mask = enc["placed_logic_mask"]
        pair_states = enc["pair_states"]
        progress = enc["progress"]

        current_logic = self._gather_batched(logic_emb, selected_logical_idx)
        current_logic_feat = self._gather_batched(logical_feat_emb, selected_logical_idx)
        placement_ctx = self.context(current_logic, pair_states, placed_logic_mask)
        candidate_features = self._gather_batched(candidate_features_bank, selected_logical_idx)
        physical_mask = self._gather_batched(physical_action_masks_bank, selected_logical_idx) * free_physical_mask
        physical_prior = self._gather_batched(physical_prior_bank, selected_logical_idx)

        cand_emb = self.candidate_proj(candidate_features)
        cur_logic_exp = current_logic.unsqueeze(1).expand(-1, phys_emb.shape[1], -1)
        cur_logic_feat_exp = current_logic_feat.unsqueeze(1).expand(-1, phys_emb.shape[1], -1)
        ctx_exp = placement_ctx.unsqueeze(1).expand(-1, phys_emb.shape[1], -1)
        progress_exp = progress.unsqueeze(1).expand(-1, phys_emb.shape[1], -1)
        free_exp = free_physical_mask.unsqueeze(-1)
        used_exp = used_phys.unsqueeze(-1)
        prior_hint = self._build_prior_feature(physical_prior, use_physical_prior).unsqueeze(-1)

        pair_features = torch.cat(
            [
                cur_logic_exp,
                cur_logic_feat_exp,
                phys_emb,
                ctx_exp,
                cand_emb,
                free_exp,
                used_exp,
                progress_exp,
                prior_hint,
            ],
            dim=-1,
        )

        query = self.physical_query_proj(torch.cat([current_logic, current_logic_feat, placement_ctx, progress], dim=-1))
        keys = self.physical_key_proj(torch.cat([phys_emb, cand_emb, free_exp, used_exp, progress_exp, prior_hint], dim=-1))
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
        
        # 1. 动态解析批次动作空间的决策类型
        is_fixed_order = "is_fixed_order" in batch and batch["is_fixed_order"][0].item() > 0

        if is_fixed_order:
            # ==================== 核心修改：策略评估梯度流截断 ====================
            # 用 zeros 截断 logical_head 的反向传播路径，阻止无效的随机梯度噪声污染网络
            logical_logits = torch.zeros_like(batch["logical_action_mask"])
            logical_logprob = torch.zeros_like(action_logical, dtype=torch.float32)
            logical_entropy = torch.zeros_like(action_logical, dtype=torch.float32)
        else:
            # 正常的双层联合分布评估
            logical_logits = self._logical_logits(enc)
            logical_dist = torch.distributions.Categorical(logits=logical_logits)
            logical_logprob = d = logical_dist.log_prob(action_logical.long())
            logical_entropy = logical_dist.entropy()

        # 2. 严格并行评估条件下的物理位置决策质量
        physical_logits = self._physical_logits(enc, action_logical.long(), use_physical_prior=use_physical_prior)
        physical_dist = torch.distributions.Categorical(logits=physical_logits)
        physical_logprob = physical_dist.log_prob(action_physical.long())
        physical_entropy = physical_dist.entropy()

        # 3. 联合优势估计组合
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
        
        # 1. 动态感知当前运行模式（完美兼容单步推断与 DataLoader 批次张量）
        is_fixed_order = "is_fixed_order" in batch and batch["is_fixed_order"][0].item() > 0

        if is_fixed_order:
            # ==================== 核心修改：固定顺序模式短路分支 ====================
            # 直接提取环境锁定的当前逻辑比特，免除 logits 前向流与采样流开销
            logical_action = batch["current_logical_idx"].long()
            # 确定性决策的概率对数值与信息熵在物理和数学意义上严格为 0
            logical_logprob = torch.zeros_like(batch["progress"].squeeze(-1))
            # 构造全零矩阵用于占位，保持输出字典的 Schema 完全对称与下游兼容
            logical_logits = torch.zeros_like(batch["logical_action_mask"])
        else:
            # 分层层次化联合决策分支（保持原样逻辑）
            logical_logits = self._logical_logits(enc)
            logical_dist = torch.distributions.Categorical(logits=logical_logits)
            logical_action = torch.argmax(logical_logits, dim=-1) if deterministic else logical_dist.sample()
            logical_logprob = logical_dist.log_prob(logical_action)

        # 2. 物理图位置选择：输入当前的逻辑节点特征，计算条件物理分布（两模式通用）
        physical_logits = self._physical_logits(enc, logical_action, use_physical_prior=use_physical_prior)
        physical_dist = torch.distributions.Categorical(logits=physical_logits)
        physical_action = torch.argmax(physical_logits, dim=-1) if deterministic else physical_dist.sample()

        physical_logprob = physical_dist.log_prob(physical_action)
        value = self._value(enc)

        return {
            "action_logical": logical_action,
            "action_physical": physical_action,
            "logprob": logical_logprob + physical_logprob,  # 固定模式下等价于 pure physical_logprob
            "logical_logprob": logical_logprob,
            "physical_logprob": physical_logprob,
            "value": value,
            "logical_logits": logical_logits,
            "physical_logits": physical_logits,
        }
