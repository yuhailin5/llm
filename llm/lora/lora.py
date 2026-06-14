"""
LoRA (Low-Rank Adaptation) — 从零实现
======================================

核心公式：
    y = W₀x + (α/r) · B · A · x

其中：
  - W₀ : 冻结的原始权重  (d_out × d_in)
  - A  : 低秩矩阵         (r × d_in)，Kaiming 均匀初始化
  - B  : 低秩矩阵         (d_out × r)，零初始化（确保初始时 ΔW = 0）
  - α  : 缩放因子（默认 = r，即不缩放）
  - r  : 秩

用法：
    import lora
    linear = nn.Linear(512, 512)
    lora_linear = lora.LoRALinear(linear, r=8, alpha=16)

    # 训练（只更新 A、B）
    lora.mark_only_lora_as_trainable(model)
    ... train ...

    # 推理（合并后无额外开销）
    lora_linear.merge()
    ... generate ...
    lora_linear.unmerge()   # 恢复，可继续训练

    # 多任务热切换
    lora_linear.save_adapter("task_A")
    lora_linear.load_adapter("task_B")

参考论文：
  - LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
  - https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# LoRALinear — 核心适配层
# ──────────────────────────────────────────────
class LoRALinear(nn.Module):
    """
    对单个 nn.Linear 做低秩适配。

    参数:
        base:        原始 nn.Linear 层（传入后会被冻结）
        r:           低秩维度（默认 8，论文表明 r=1~4 通常就够）
        alpha:       缩放因子，实际缩放为 alpha/r
        dropout:     LoRA 路径的 dropout 率（默认 0，论文通常不用）
        bias:        是否保留原始 bias 可训练（默认 False，冻结）
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        # ── 保存原始层引用，冻结其权重 ──
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(bias)

        in_features = base.in_features
        out_features = base.out_features

        # ── 低秩矩阵 ──
        # A: (r, in_features)  — Kaiming 均匀初始化，让 A·x 方差合理
        # B: (out_features, r) — 零初始化，确保 ΔW = 0 起步，不破坏原输出
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r  # 缩放系数

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ── 是否已合并到原权重 ──
        self.merged: bool = False

        # ── 多 LoRA 热切换缓存 ──
        self._adapter_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        y = W₀x + (α/r) · B · A · x
        """
        base_out = self.base(x)

        if self.merged:
            # 已合并：LoRA 权重已加到 base.weight 里，直接用 base 就行
            return base_out

        lora_out = F.linear(self.dropout(x), self.lora_A)  # (..., r)
        lora_out = F.linear(lora_out, self.lora_B)        # (..., out_features)
        return base_out + lora_out * self.scaling

    # ── 合并 & 卸载 ──
    @torch.no_grad()
    def merge(self) -> None:
        """将 LoRA 增量 B·A 加回原始权重 W，推理时零开销。"""
        if self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.data.add_(delta)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """从原始权重中减去 LoRA 增量，恢复可训练状态。"""
        if not self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.data.sub_(delta)
        self.merged = False

    # ── 多 LoRA 热切换 ──
    @torch.no_grad()
    def save_adapter(self, name: str) -> None:
        """保存当前 A/B 到内部缓存，用于多任务切换。"""
        self._adapter_cache[name] = (
            self.lora_A.data.clone(),
            self.lora_B.data.clone(),
        )

    @torch.no_grad()
    def load_adapter(self, name: str) -> None:
        """从缓存加载指定任务的 A/B 权重。"""
        if name not in self._adapter_cache:
            raise KeyError(f"适配器 '{name}' 未缓存。可用: {list(self._adapter_cache.keys())}")
        A, B = self._adapter_cache[name]
        self.lora_A.data.copy_(A)
        self.lora_B.data.copy_(B)

    def delete_adapter(self, name: str) -> None:
        """删除缓存的适配器。"""
        del self._adapter_cache[name]

    def list_adapters(self) -> list:
        """列出所有已缓存的适配器名称。"""
        return list(self._adapter_cache.keys())


# ──────────────────────────────────────────────
# LoRAEmbedding — 嵌入层适配（可选）
# ──────────────────────────────────────────────
class LoRAEmbedding(nn.Module):
    """
    对 nn.Embedding 做低秩适配。

    与 LoRALinear 类似，A: (r, num_embeddings), B: (embedding_dim, r)。
    注意：embedding 权重形状是 (num_embeddings, embedding_dim)，与 Linear 转置关系。
    """

    def __init__(
        self,
        base: nn.Embedding,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()

        self.base = base
        self.base.weight.requires_grad_(False)

        num_embeddings, embedding_dim = base.weight.shape
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.padding_idx = padding_idx

        # A: (r, num_embeddings)
        # B: (embedding_dim, r)
        self.lora_A = nn.Parameter(torch.empty(r, num_embeddings))
        self.lora_B = nn.Parameter(torch.zeros(embedding_dim, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if padding_idx is not None:
            # 确保 padding_idx 位置的 LoRA 增量为 0
            nn.init.zeros_(self.lora_A[:, padding_idx])

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged: bool = False
        self._adapter_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)  # (..., embedding_dim)

        if self.merged:
            return base_out

        # 取出 x 对应的 A 列 → (..., r)
        lora_A_x = F.embedding(x, self.lora_A.T, padding_idx=self.padding_idx)
        lora_out = F.linear(self.dropout(lora_A_x), self.lora_B)
        return base_out + lora_out * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling  # (emb_dim, num_emb)
        self.base.weight.data.add_(delta.T)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.data.sub_(delta.T)
        self.merged = False

    @torch.no_grad()
    def save_adapter(self, name: str) -> None:
        self._adapter_cache[name] = (
            self.lora_A.data.clone(),
            self.lora_B.data.clone(),
        )

    @torch.no_grad()
    def load_adapter(self, name: str) -> None:
        if name not in self._adapter_cache:
            raise KeyError(f"适配器 '{name}' 未缓存。可用: {list(self._adapter_cache.keys())}")
        A, B = self._adapter_cache[name]
        self.lora_A.data.copy_(A)
        self.lora_B.data.copy_(B)

    def delete_adapter(self, name: str) -> None:
        del self._adapter_cache[name]

    def list_adapters(self) -> list:
        return list(self._adapter_cache.keys())


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def replace_with_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Optional[Set[str]] = None,
    exclude_modules: Optional[Set[str]] = None,
) -> nn.Module:
    """
    递归替换模型中指定的 nn.Linear 为 LoRALinear。

    参数:
        model:          待改造的模型
        r, alpha:       LoRA 超参
        target_modules: 要替换的模块类名集合，默认 {'Linear'}
        exclude_modules: 排除的模块名集合（如 {'head', 'lm_head'}）

    返回:
        改造后的模型（原地修改 + 返回引用）

    示例:
        model = replace_with_lora(model, r=8, target_modules={'Linear'})
    """
    if target_modules is None:
        target_modules = {"Linear"}
    if exclude_modules is None:
        exclude_modules = set()

    for name, child in list(model.named_children()):
        if name in exclude_modules:
            continue

        child_class = child.__class__.__name__
        if child_class in target_modules and isinstance(child, nn.Linear):
            setattr(
                model, name,
                LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            )
        else:
            replace_with_lora(
                child, r=r, alpha=alpha, dropout=dropout,
                target_modules=target_modules, exclude_modules=exclude_modules,
            )

    return model


def replace_specific_modules(
    model: nn.Module,
    target_names: list,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> nn.Module:
    """
    精确替换模型中指定名称的模块为 LoRALinear。

    参数:
        model:        待改造模型
        target_names: 要替换的模块名称列表，如 ['qkv', 'out_proj', 'ffn.0', 'ffn.2']
        r, alpha:     LoRA 超参

    示例:
        model = replace_specific_modules(model, ['qkv', 'out_proj'], r=8)
    """
    target_set = set(target_names)

    for name, child in list(model.named_children()):
        if name in target_set and isinstance(child, nn.Linear):
            setattr(
                model, name,
                LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            )
        else:
            replace_specific_modules(
                child, target_names, r=r, alpha=alpha, dropout=dropout
            )

    return model


def mark_only_lora_as_trainable(model: nn.Module) -> list:
    """
    冻结所有参数，仅解冻 LoRA 的 A/B 矩阵。

    返回:
        可训练参数名列表
    """
    trainable_names = []

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            trainable_names.append(name)
        else:
            param.requires_grad = False

    return trainable_names


def count_parameters(model: nn.Module):
    """
    统计参数。

    返回:
        (总参数量, 可训练参数量)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    获取所有 LoRA 参数的 state_dict（用于保存轻量 adapter 权重）。
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state[name] = param.data.clone()
    return lora_state


def load_lora_state_dict(
    model: nn.Module, lora_state: Dict[str, torch.Tensor]
) -> None:
    """加载 LoRA 参数（strict=False，只加载 lora_ 开头的参数）。"""
    for name, param in model.named_parameters():
        if name in lora_state:
            param.data.copy_(lora_state[name])


# ──────────────────────────────────────────────
# 自测试
# ──────────────────────────────────────────────
def _test():
    """快速验证 LoRALinear 的基本功能。"""
    print("=" * 50)
    print("LoRA 自测试")
    print("=" * 50)

    # ── 1. 基本功能：初始时 B=0，输出 = W₀x ──
    linear = nn.Linear(64, 128)
    lora_linear = LoRALinear(linear, r=4, alpha=8)  # scaling=2

    x = torch.randn(2, 10, 64)
    out_original = linear(x)
    out_lora_init = lora_linear(x)

    err = (out_original - out_lora_init).abs().max().item()
    assert err < 1e-5, f"初始 LoRA 应输出 = W₀x，最大误差: {err:.6f}"
    print(f"  [OK] 初始输出 = W0x (B=0 保证), 误差: {err:.2e}")

    # ── 2. 训练几步，验证 A/B 在更新 ──
    opt = torch.optim.SGD(lora_linear.parameters(), lr=0.1)
    target = torch.randn(2, 10, 128)
    for _ in range(5):
        opt.zero_grad()
        loss = F.mse_loss(lora_linear(x), target)
        loss.backward()
        opt.step()

    out_trained = lora_linear(x)
    assert not torch.allclose(out_original, out_trained, atol=1e-3), \
        "训练后输出应与原始不同"
    print("  [OK] 训练后输出变化（A/B 在更新）")

    # ── 3. merge / unmerge ──
    lora_linear.merge()
    assert lora_linear.merged
    out_merged = lora_linear(x)
    assert torch.allclose(out_trained, out_merged, atol=1e-5), \
        "merge 后输出应与训练后一致"
    print("  [OK] merge 后输出一致")

    lora_linear.unmerge()
    assert not lora_linear.merged
    out_unmerged = lora_linear(x)
    assert torch.allclose(out_trained, out_unmerged, atol=1e-5), \
        "unmerge 后输出应与训练后一致"
    print("  [OK] unmerge 后输出恢复")

    # ── 4. 多 LoRA 热切换 ──
    lora_linear.save_adapter("task_A")
    with torch.no_grad():
        lora_linear.lora_A.add_(0.5)
        lora_linear.lora_B.add_(0.5)
    out_task_B = lora_linear(x)
    lora_linear.save_adapter("task_B")

    lora_linear.load_adapter("task_A")
    assert torch.allclose(out_trained, lora_linear(x), atol=1e-5), \
        "load task_A 后输出应与保存前一致"
    print("  [OK] 多 LoRA 热切换 task_A <-> task_B 正常")

    lora_linear.load_adapter("task_B")
    assert torch.allclose(out_task_B, lora_linear(x), atol=1e-5)
    print("  [OK] task_B 加载正常，且与 task_A 不同")

    # ── 5. 工具函数 ──
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
    model = replace_with_lora(model, r=4)
    total, trainable = count_parameters(model)
    trainable_names = mark_only_lora_as_trainable(model)

    assert trainable < total * 0.5, \
        f"可训练参数应远小于总参数: {trainable}/{total}"
    assert all("lora_" in n for n in trainable_names), \
        "所有可训练参数应包含 'lora_'"
    print(f"  [OK] 工具函数正常: 总 {total:,} / 可训练 {trainable:,} ({100*trainable/total:.1f}%)")

    # ── 6. LoRAEmbedding 基本测试 ──
    emb = nn.Embedding(100, 64)
    lora_emb = LoRAEmbedding(emb, r=4, alpha=8, padding_idx=0)
    idx = torch.randint(1, 100, (2, 10))
    assert torch.allclose(emb(idx), lora_emb(idx), atol=1e-5), \
        "LoRAEmbedding 初始输出应与原 Embedding 一致"
    print("  [OK] LoRAEmbedding 初始输出一致")

    print("=" * 50)
    print("全部测试通过！")
    print("=" * 50)


if __name__ == "__main__":
    _test()
