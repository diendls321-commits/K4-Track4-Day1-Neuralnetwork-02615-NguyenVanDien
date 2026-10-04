"""data.py — Nạp dữ liệu CoverType, tách validation, chuẩn hoá và chia lô.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.
Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    train_path = Path(processed_dir) / "train.npz"
    eval_path = Path(processed_dir) / "eval.npz"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file processed tại '{processed_dir}'. "
            f"Vui lòng chạy 'python scripts/split_data.py' trước."
        )

    tr_data = np.load(train_path)
    ev_data = np.load(eval_path)

    X_train_full = tr_data["X"].astype(np.float32)
    y_train_full = tr_data["y"].astype(np.int64)
    X_eval = ev_data["X"].astype(np.float32)
    y_eval = ev_data["y"].astype(np.int64)
    eval_row_id = ev_data["row_id"].astype(np.int64)

    assert X_train_full.ndim == 2 and X_train_full.shape[1] == 54, f"X_train_full shape sai: {X_train_full.shape}"
    assert X_eval.ndim == 2 and X_eval.shape[1] == 54, f"X_eval shape sai: {X_eval.shape}"
    assert y_train_full.ndim == 1, f"y_train_full shape sai: {y_train_full.shape}"
    assert y_eval.ndim == 1, f"y_eval shape sai: {y_eval.shape}"
    assert len(eval_row_id) == len(X_eval), "Độ dài eval_row_id và X_eval không khớp"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    """
    cols = X_tr[:, :N_NUMERIC]
    mean = np.mean(cols, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(cols, axis=0, dtype=np.float64).astype(np.float32)
    # Tránh chia cho 0 nếu một đặc trưng có độ lệch chuẩn = 0
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_out = np.array(X, copy=True, dtype=np.float32)
    X_out[:, :N_NUMERIC] = (X_out[:, :N_NUMERIC] - mean) / std
    return X_out


def prepare_data(device: str | torch.device, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id, mean, std
    """
    dev = torch.device(device) if isinstance(device, str) else device
    X_train_full, y_train_full, X_eval_raw, y_eval_raw, eval_row_id = load_split(processed_dir)

    # Tách validation từ train (phân tầng, seed cố định)
    X_tr_raw, y_tr_raw, X_val_raw, y_val_raw = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )

    # Chuẩn hoá: fit mean/std CHỈ trên phần train còn lại
    mean, std = fit_standardizer(X_tr_raw)
    X_tr = apply_standardizer(X_tr_raw, mean, std)
    X_val = apply_standardizer(X_val_raw, mean, std)
    X_eval = apply_standardizer(X_eval_raw, mean, std)

    # Đưa lên device
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=dev)
    y_tr_t = torch.tensor(y_tr_raw, dtype=torch.int64, device=dev)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=dev)
    y_val_t = torch.tensor(y_val_raw, dtype=torch.int64, device=dev)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=dev)
    y_eval_t = torch.tensor(y_eval_raw, dtype=torch.int64, device=dev)

    # In thông tin kiểm tra
    print(f"Dataset summary (on device {dev}):")
    print(f"  Train: X={X_tr_t.shape}, y={y_tr_t.shape}")
    print(f"  Val:   X={X_val_t.shape}, y={y_val_t.shape}")
    print(f"  Eval:  X={X_eval_t.shape}, y={y_eval_t.shape}")

    # Đánh giá chiến lược "luôn đoán lớp đa số" trên val
    val_counts = np.bincount(y_val_raw, minlength=7)
    majority_class = int(np.argmax(val_counts))
    maj_acc = float(val_counts[majority_class] / len(y_val_raw))
    print(f"  Majority class on val: {majority_class} with count {val_counts[majority_class]}/{len(y_val_raw)}")
    print(f"  Majority baseline val accuracy: {maj_acc:.4f} (48.76% expected)")

    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
        "maj_acc": maj_acc,
    }


def iterate_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int,
                    generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.
    Xử lý: batch cuối giữ nguyên kích thước còn lại (không bỏ).
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
