import torch
import torch.nn as nn
from uer.layers.transformer import TransformerLayer
from uer.layers.layer_norm import LayerNorm, T5LayerNorm
from uer.layers.relative_position_embedding import RelativePositionEmbedding


class TransformerEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    支持三种模式：
    - forward(): 兼容旧接口，一次性跑到最后（可选返回中间层）
    - forward_early_exit(): 逐层推进 + 逐样本剔除（真正省计算）
    - 工具函数：build_mask_and_bias() 供两种模式共享
    """
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        if self.parameter_sharing:
            self.transformer = TransformerLayer(args)
        else:
            self.transformer = nn.ModuleList([TransformerLayer(args) for _ in range(self.layers_num)])

        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

        if self.relative_position_embedding:
            self.relative_pos_emb = RelativePositionEmbedding(
                bidirectional=True, heads_num=args.heads_num,
                num_buckets=args.relative_attention_buckets_num
            )
        else:
            self.relative_pos_emb = None

    def build_mask_and_bias(self, seg, hidden_like):
        """
        生成mask和位置偏置，用于Transformer模型的每一层。
        seg: [B, L] ；hidden_like: [B, L, H]
        返回：
            mask: [B, 1, L, L]
            position_bias: None 或 [B, heads, L, L]
        """
        B, L = seg.size(0), seg.size(1)
        device = hidden_like.device

        if self.mask == "fully_visible":
            # 可见位是 seg>0 的 token
            mask = (seg > 0).unsqueeze(1).repeat(1, L, 1).unsqueeze(1).float()
            mask = (1.0 - mask) * -10000.0
        elif self.mask == "causal":
            tril = torch.tril(torch.ones(L, L, device=device))
            mask = (1.0 - tril) * -10000.0
            mask = mask.view(1, 1, L, L).repeat(B, 1, 1, 1)
        else:
            # 用于seq2seq/可控可见性的变体
            mask_a = (seg == 1).unsqueeze(1).repeat(1, L, 1).unsqueeze(1).float()
            mask_b = (seg > 0).unsqueeze(1).repeat(1, L, 1).unsqueeze(1).float()
            tril = torch.tril(torch.ones(L, L, device=device)).view(1, 1, L, L).repeat(B, 1, 1, 1)
            mask = (mask_a + mask_b + tril >= 2).float()
            mask = (1.0 - mask) * -10000.0

        if self.relative_pos_emb is not None:
            # position_bias 形状通常为 [B, heads, L, L]
            position_bias = self.relative_pos_emb(hidden_like, hidden_like)
        else:
            position_bias = None
        return mask, position_bias

    def forward(self, emb, seg, return_all_layers=False, return_layers_idx=None):
        """
        兼容旧接口：一次性跑到最后（可选返回中间层）
        """
        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)
        hidden = emb

        mask, position_bias = self.build_mask_and_bias(seg, hidden)

        collect_all = bool(return_all_layers)
        pick_layers = set(int(x) for x in return_layers_idx) if return_layers_idx is not None else None
        collected = []

        for i in range(self.layers_num):
            if self.parameter_sharing:
                hidden = self.transformer(hidden, mask, position_bias=position_bias)
            else:
                hidden = self.transformer[i](hidden, mask, position_bias=position_bias)

            layer_id = i + 1
            if pick_layers is not None:
                if layer_id in pick_layers:
                    collected.append(hidden)
            elif collect_all:
                collected.append(hidden)

        final_hidden = self.layer_norm(hidden) if self.layernorm_positioning == "pre" else hidden
        if (pick_layers is not None) or collect_all:
            return final_hidden, collected
        else:
            return final_hidden

    @torch.no_grad()
    def forward_early_exit(self, emb, seg, exit_layers, min_exit_layer, decider_fn):
        """
        逐层推进 + 逐样本剔除：如果满足条件（decider_fn），提前退出。
        参数：
            emb: [B, L, H] 输入 embedding（或上一层输出）
            seg: [B, L]
            exit_layers: 可早退层的集合（1-based）
            min_exit_layer: 早退最小层号（1-based），小于此层不判定
            decider_fn: (layer_id:int, hidden_act:[b_act, L, H]) -> BoolTensor[b_act]
                        返回 True 表示该样本在当前层提前退出
        返回：
            chosen_hidden: [B, L, H] 每个样本最终选定的 hidden（退出层或最终层）
            exit_layer: [B] 每个样本的真实退出层号
        """
        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        B, L, _ = emb.size()
        device = emb.device

        # 初始化
        hidden = emb.clone()
        mask, position_bias = self.build_mask_and_bias(seg, hidden)

        active_idx = torch.arange(B, device=device)                    # 当前仍需计算的样本索引
        chosen_hidden = torch.empty_like(hidden)                       # 存放每个样本最终选定的 hidden
        exit_layer = torch.full((B,), self.layers_num, device=device, dtype=torch.long)  # 默认退出层=最后一层
        exit_layers_set = set(int(x) for x in exit_layers) if exit_layers is not None else set()

        for i in range(self.layers_num):
            if active_idx.numel() == 0:
                break

            layer_id = i + 1
            # 仅对存活样本/对应的 mask & position_bias 做一层
            h_act = hidden[active_idx]          # [b_act, L, H]
            m_act = mask[active_idx]            # [b_act, 1, L, L]
            pb_act = position_bias[active_idx] if position_bias is not None else None  # ✅ 关键：按存活样本切片

            if self.parameter_sharing:
                h_act = self.transformer(h_act, m_act, position_bias=pb_act)
            else:
                h_act = self.transformer[i](h_act, m_act, position_bias=pb_act)

            # 回填已计算出的存活样本 hidden
            hidden[active_idx] = h_act

            # 到了早退检查点且层号足够时，做早退判定
            if layer_id in exit_layers_set and layer_id >= int(min_exit_layer):
                exit_mask = decider_fn(layer_id, h_act)  # BoolTensor[b_act]
                if exit_mask is None:
                    exit_mask = torch.zeros(h_act.size(0), device=device, dtype=torch.bool)

                if exit_mask.any():
                    idx_take = active_idx[exit_mask]
                    # 此处可按需要对 h_act 做 LN；保持与最终层一致，通常在分类头里已处理，这里直接存
                    chosen_hidden[idx_take] = h_act[exit_mask]
                    exit_layer[idx_take] = layer_id
                    # 更新存活样本（剔除已退出者）
                    active_idx = active_idx[~exit_mask]

        # 收尾：对剩余存活样本，取最后一层输出，并仅对其做（可选）LayerNorm
        if active_idx.numel() > 0:
            tail = hidden[active_idx]
            if self.layernorm_positioning == "pre":
                tail = self.layer_norm(tail)   # ✅ 只对存活样本做 LN，避免全批量额外开销
            chosen_hidden[active_idx] = tail

        return chosen_hidden, exit_layer
