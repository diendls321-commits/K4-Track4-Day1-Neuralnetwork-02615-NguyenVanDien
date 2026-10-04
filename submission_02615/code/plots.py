"""plots.py — Vẽ đồ thị huấn luyện cho từng thí nghiệm và ảnh so sánh.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    cfg = result["cfg"]
    history = result["history"]
    summary = result.get("summary", {})

    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_acc", [])
    val_macro_f1 = history.get("val_macro_f1", [])
    grad_norm = history.get("grad_norm", [])
    best_epoch = summary.get("best_epoch", None)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    exp_id = cfg.get("exp_id", "exp")
    title_str = (
        f"Exp: {exp_id} | Opt: {cfg.get('optimizer')} (lr={cfg.get('lr')}) | "
        f"Batch: {cfg.get('batch')} | Model: {cfg.get('hidden')}"
    )
    fig.suptitle(title_str, fontsize=13, fontweight="bold")

    # (1) Loss curves
    ax0 = axes[0]
    if train_loss and val_loss:
        ax0.plot(epochs, train_loss, label="Train Loss (eval mode)", color="#1f77b4", lw=2)
        ax0.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", lw=2)
        if best_epoch and best_epoch in epochs:
            ax0.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
    ax0.set_title("Loss vs Epoch")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    # (2) Accuracy & Macro-F1 curves
    ax1 = axes[1]
    if val_acc:
        ax1.plot(epochs, val_acc, label="Val Accuracy", color="#2ca02c", lw=2)
    if val_macro_f1:
        ax1.plot(epochs, val_macro_f1, label="Val Macro-F1", color="#d62728", lw=2)
    ax1.axhline(y=0.4876, color="gray", linestyle=":", label="Majority Baseline (0.4876)")
    if best_epoch and best_epoch in epochs:
        ax1.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7)
    ax1.set_title("Val Accuracy & Macro-F1 vs Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Metric")
    ax1.set_ylim([0.0, 1.0])
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # (3) Gradient Norm curve (pre-clip)
    ax2 = axes[2]
    if grad_norm:
        ax2.plot(epochs, grad_norm, label="Grad Norm (pre-clip)", color="#9467bd", lw=2)
        clip_val = cfg.get("clip_norm", None)
        if clip_val is not None:
            ax2.axhline(y=float(clip_val), color="black", linestyle="--", label=f"Clip Threshold ({clip_val})")
    ax2.set_title("Grad Norm vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("L2 Norm")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out_dir = Path(path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ 'val_loss', 'val_macro_f1', 'grad_norm') của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.
    """
    if not results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for res in results:
        cfg = res.get("cfg", {})
        hist = res.get("history", {})
        epochs = hist.get("epoch", list(range(1, len(hist.get(metric, [])) + 1)))
        values = hist.get(metric, [])
        label = f"{cfg.get('exp_id')} ({cfg.get('description', '')})"
        if values and len(epochs) == len(values):
            ax.plot(epochs, values, lw=2, label=label)

    if not title:
        title = f"Comparison of {metric} across experiments"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    out_dir = Path(path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
