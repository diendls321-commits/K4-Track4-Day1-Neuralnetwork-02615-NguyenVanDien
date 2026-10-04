"""train.py — Đặt seed, đánh giá, vòng lặp huấn luyện run_experiment và xuất file nộp.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

# Cấu hình mặc định = BASELINE (M-base).
DEFAULT_CFG = dict(
    exp_id="base-s1",
    group="baseline",
    description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Chọn bằng val
    weight_decay=0.0,
    momentum=0.9,
    batch=512,
    epochs=20,
    hidden=(256, 128),
    dropout=0.0,
    init="he",
    clip_norm=None,            # None = không clip; hoặc float, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.
    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    Đúng định nghĩa của scripts/evaluate.py.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(np.mean(f1))


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.
    Chạy ở chế độ eval() không tính gradient.
    """
    was_training = model.training
    model.eval()
    preds_list = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)
        preds_list.append(pred)
    model.train(was_training)
    return torch.cat(preds_list, dim=0)


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict[str, float]:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.
    Cộng dồn loss (reduction='sum') rồi chia N ở cuối để chính xác.
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    n = len(X)
    cm = np.zeros((7, 7), dtype=np.int64)

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        loss_val = compute_loss(logits, yb, loss_name, reduction="sum")
        total_loss += float(loss_val.item())

        preds = torch.argmax(logits, dim=1)
        y_true_np = yb.cpu().numpy()
        preds_np = preds.cpu().numpy()
        np.add.at(cm, (y_true_np, preds_np), 1)

    model.train(was_training)

    avg_loss = total_loss / n
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    macro_f1 = macro_f1_from_confusion(cm)

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
    }


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str, reduction: str = "mean") -> torch.Tensor:
    """'ce'  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       'mse' : MSE giữa logit và one-hot của y.
    """
    loss_name = loss_name.lower()
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    elif loss_name == "mse":
        # One-hot encoding cho y
        num_classes = logits.size(-1)
        y_onehot = F.one_hot(y, num_classes=num_classes).to(dtype=logits.dtype)
        # nn.MSELoss tính trung bình/tổng trên toàn bộ tensor
        return F.mse_loss(logits, y_onehot, reduction=reduction)
    else:
        raise ValueError(f"Hàm mất mát không được hỗ trợ: '{loss_name}'. Chọn 'ce' hoặc 'mse'.")


def run_experiment(cfg: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)
    """
    # 0. Thiết lập seed và khởi tạo model
    set_seed(int(cfg.get("seed", 1)))
    device = data["X_tr"].device
    is_cuda = data["X_tr"].is_cuda

    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init_name = str(cfg.get("init", "he"))

    model = MLP(hidden=hidden, dropout=dropout, init=init_name, in_features=54, num_classes=7)
    param_count = count_params(model)
    assert param_count == EXPECTED_PARAMS[hidden], (
        f"Số tham số {param_count} không khớp EXPECTED_PARAMS: {EXPECTED_PARAMS[hidden]} cho {hidden}"
    )
    model.to(device)

    # Khởi tạo optimizer & scheduler
    optimizer = build_optimizer(
        name=cfg["optimizer"],
        params=model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        momentum=float(cfg.get("momentum", 0.9)),
    )
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", None), total_steps=int(cfg["epochs"]))

    # Thiết lập mixed precision
    precision = cfg.get("precision", "fp32").lower()
    scaler = None
    autocast_dtype = None
    if is_cuda and precision == "fp16":
        autocast_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
    elif is_cuda and precision == "bf16":
        if torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        else:
            print(f"Cảnh báo: GPU không hỗ trợ BF16; chuyển về FP32.")
            precision = "fp32"

    # Reset peak memory
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Đo loss bước 0 trên val trước bất kỳ cập nhật nào
    step0_val = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = step0_val["loss"]

    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch", 512))
    clip_norm = cfg.get("clip_norm", None)
    if clip_norm is not None:
        clip_norm = float(clip_norm)

    history: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_state: dict[str, Any] = copy.deepcopy(model.state_dict())
    diverged = False

    # 2. Vòng lặp huấn luyện từng epoch
    for ep in range(1, epochs + 1):
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        model.train()
        batch_grad_norms: list[float] = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if autocast_dtype is not None and is_cuda:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])

                if torch.isnan(loss) or torch.isinf(loss):
                    diverged = True
                    break

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if clip_norm is not None:
                        scaler.unscale_(optimizer)
                    gn = clip_gradients(model.parameters(), clip_norm)
                    batch_grad_norms.append(gn)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    gn = clip_gradients(model.parameters(), clip_norm)
                    batch_grad_norms.append(gn)
                    optimizer.step()
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

                if torch.isnan(loss) or torch.isinf(loss):
                    diverged = True
                    break

                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                batch_grad_norms.append(gn)
                optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if is_cuda:
            torch.cuda.synchronize(device)
        ep_time = time.perf_counter() - t0

        if diverged:
            print(f"CẢNH BÁO: Thí nghiệm {cfg['exp_id']} bị DIVERGED (loss NaN/inf) ở epoch {ep}!")
            break

        # Đánh giá cuối epoch ở chế độ eval()
        # Train loss đo trên toàn bộ train ở chế độ eval() (dropout tắt) để công bằng với val
        tr_res = evaluate(model, data["X_tr"], data["y_tr"], loss_name=cfg["loss"])
        val_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])

        mean_gn = float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0

        history["epoch"].append(ep)
        history["train_loss"].append(tr_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["acc"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(ep_time)

        # Lưu best checkpoint theo val_loss thấp nhất
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())

    # Peak memory đo được
    peak_mem_mb = 0.0
    if is_cuda:
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    # Tóm tắt kết quả tại best_epoch
    best_idx = best_epoch - 1 if (best_epoch - 1 < len(history["val_acc"])) else -1
    summary = {
        "step0_loss": round(float(step0_loss), 4),
        "best_val_loss": round(float(best_val_loss), 4),
        "best_epoch": int(best_epoch),
        "final_train_loss": round(float(history["train_loss"][-1]), 4) if history["train_loss"] else None,
        "final_val_loss": round(float(history["val_loss"][-1]), 4) if history["val_loss"] else None,
        "val_acc": round(float(history["val_acc"][best_idx]), 4) if history["val_acc"] else 0.0,
        "val_macro_f1": round(float(history["val_macro_f1"][best_idx]), 4) if history["val_macro_f1"] else 0.0,
        "time_per_epoch_s": round(float(np.mean(history["epoch_time_s"])), 2) if history["epoch_time_s"] else 0.0,
        "peak_mem_MB": round(float(peak_mem_mb), 1),
        "diverged": diverged,
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.
    Phải có đúng 116 203 dòng của tập eval.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"row_id": row_id.astype(int), "pred": preds.astype(int)})
    df.to_csv(out_path, index=False)
    print(f"Đã lưu predictions: {out_path} ({len(df)} dòng)")


def final_eval(cfg: dict[str, Any], result: dict[str, Any], data: dict[str, Any], pred_path: str) -> dict[str, Any]:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    model = MLP(hidden=hidden, dropout=dropout, in_features=54, num_classes=7)
    model.load_state_dict(result["best_state"])
    model.to(data["X_eval"].device)

    # Dự đoán trên toàn bộ eval
    preds = predict(model, data["X_eval"])
    preds_np = preds.cpu().numpy()
    row_ids = data["eval_row_id"]

    write_predictions(row_ids, preds_np, pred_path)

    # Tính điểm eval
    y_eval_np = data["y_eval"].cpu().numpy()
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y_eval_np, preds_np), 1)

    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)

    eval_acc = float(np.trace(cm) / cm.sum())
    eval_macro_f1 = float(np.mean(f1))

    print(f"\n--- EVAL RESULT ({cfg.get('exp_id')}) ---")
    print(f"Accuracy: {eval_acc:.4f} | Macro-F1: {eval_macro_f1:.4f}")

    return {
        "accuracy": eval_acc,
        "macro_f1": eval_macro_f1,
        "per_class_f1": f1.tolist(),
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "confusion_matrix": cm.tolist(),
    }
