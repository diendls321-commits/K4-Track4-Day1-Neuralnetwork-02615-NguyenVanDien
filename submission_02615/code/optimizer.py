"""optimizer.py — Bộ tối ưu, scheduler và cắt gradient.

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (xem README mục 5).
File này gom việc chọn bộ tối ưu và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.

Công thức cần hiểu (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params: Iterable[nn.Parameter], lr: float,
                    weight_decay: float = 0.0, momentum: float = 0.9,
                    betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> torch.optim.Optimizer:
    """Trả về một torch.optim.Optimizer."""
    name = name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"Bộ tối ưu '{name}' không hợp lệ. Chọn một trong: {OPTIMIZERS}")

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        raise ValueError(f"Bộ tối ưu '{name}' chưa được xử lý.")


def build_scheduler(optimizer: torch.optim.Optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học, ví dụ cosine (slide có ví dụ CosineAnnealingLR).

    Trả về None nếu name là None. Nếu dùng scheduler ở một thí nghiệm, ghi vào bảng (cột notes).
    """
    if name is None or not name:
        return None

    name = name.lower()
    if name == "cosine":
        eta_min = kwargs.get("eta_min", 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    elif name == "step":
        step_size = kwargs.get("step_size", max(1, total_steps // 2))
        gamma = kwargs.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Scheduler '{name}' không được hỗ trợ.")


def clip_gradients(params: Iterable[nn.Parameter], max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Các bước:
      1. nếu max_norm là None: tính chuẩn toàn cục mà không cắt (clip_grad_norm_ với max_norm=inf)
      2. ngược lại: total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
      3. return float(total_norm)
    """
    p_list = [p for p in params if p.grad is not None]
    if not p_list:
        return 0.0

    if max_norm is None:
        total_norm = torch.nn.utils.clip_grad_norm_(p_list, max_norm=float("inf"))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(p_list, max_norm=float(max_norm))

    return float(total_norm)
