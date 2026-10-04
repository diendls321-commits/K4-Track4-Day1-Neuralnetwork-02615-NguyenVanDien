"""run_experiments.py — Tự động hoá toàn bộ thí nghiệm từ Part 0 đến Part 4.
Chạy được trên cả Google Colab (GPU T4) và Local PC.

Sản phẩm tạo ra:
- figures/<exp_id>.png (cho mỗi thí nghiệm)
- figures/compare_<nhóm>.png (cho mỗi nhóm)
- results/<exp_id>.json (cho mỗi thí nghiệm)
- predictions_eval.csv (dự đoán eval của final model)
- eval_result.json (chấm điểm chính thức qua scripts/evaluate.py)
- experiments.xlsx (bảng tổng hợp đầy đủ 4 sheet)
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import prepare_data
from model import MLP, EXPECTED_PARAMS, count_params, init_weights, activation_stats
from optimizer import build_optimizer, build_scheduler, clip_gradients
from plots import plot_run, plot_compare
from results_table import save_result, load_results, to_row, write_xlsx
from train import DEFAULT_CFG, set_seed, evaluate, predict, run_experiment, final_eval


def main():
    print("=" * 70)
    print("BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE THÍ NGHIỆM LAB DAY 1 (02615)")
    print("=" * 70)

    # 1. Đường dẫn
    if os.path.exists("../../data"):
        repo_root = Path("../..").resolve()
        out_dir = Path("..").resolve()
    elif os.path.exists("../data"):
        repo_root = Path("..").resolve()
        out_dir = (repo_root / "submission_02615").resolve()
    elif os.path.exists("data"):
        repo_root = Path(".").resolve()
        out_dir = (repo_root / "submission_02615").resolve()
    else:
        repo_root = Path("/content/K4-Track4-Day1-Neuralnetwork-02615-NguyenVanDien").resolve()
        out_dir = (repo_root / "submission_02615").resolve()

    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # 2. Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")

    # 3. Part 0: Chuẩn bị dữ liệu
    print("\n" + "-" * 50)
    print("PART 0: CHUẨN BỊ DỮ LIỆU")
    print("-" * 50)
    processed_dir = repo_root / "data" / "processed"
    if not (processed_dir / "train.npz").exists():
        print("Chạy scripts/split_data.py...")
        subprocess.run([sys.executable, str(repo_root / "scripts" / "split_data.py")], cwd=str(repo_root), check=True)

    data = prepare_data(device=device, val_fraction=0.2, seed=42, processed_dir=str(processed_dir))

    # 4. Part 1: Kiểm tra sức khoẻ model
    print("\n" + "-" * 50)
    print("PART 1: KIỂM TRA SỨC KHOẺ MODEL")
    print("-" * 50)
    model_sanity = MLP(hidden=(256, 128), dropout=0.0, init="he").to(device)
    p_count = count_params(model_sanity)
    assert p_count == EXPECTED_PARAMS[(256, 128)], f"Param count mismatch: {p_count} != 47879"
    print(f"[OK] Model M-base: đúng {p_count} tham số (khớp 47 879).")

    x_dummy = torch.randn(8, 54, device=device)
    logits_dummy = model_sanity(x_dummy)
    assert logits_dummy.shape == (8, 7), f"Logits shape sai: {logits_dummy.shape}"
    print(f"[OK] Forward batch (8, 54) -> shape: {tuple(logits_dummy.shape)}.")

    # Sanity 1: Loss bước 0
    step0_res = evaluate(model_sanity, data["X_val"], data["y_val"], loss_name="ce")
    ln7 = float(np.log(7))
    print(f"[OK] Loss bước 0 trên Val: {step0_res['loss']:.4f} (lý thuyết ln(7) = {ln7:.4f}, sai số: {abs(step0_res['loss'] - ln7):.4f}).")

    # Sanity 2: Quá khớp 20 mẫu
    x_tiny, y_tiny = data["X_tr"][:20], data["y_tr"][:20]
    tiny_m = MLP(hidden=(256, 128), dropout=0.0, init="he").to(device)
    tiny_opt = torch.optim.Adam(tiny_m.parameters(), lr=0.01)
    for step in range(300):
        tiny_opt.zero_grad()
        loss_t = F.cross_entropy(tiny_m(x_tiny), y_tiny)
        loss_t.backward()
        tiny_opt.step()
    tiny_acc = (torch.argmax(tiny_m(x_tiny), dim=1) == y_tiny).float().mean().item()
    print(f"[OK] Quá khớp 20 mẫu sau 300 bước: Loss = {loss_t.item():.6f}, Acc = {tiny_acc*100:.1f}%.")

    # Sanity 3: Gradient chảy
    tiny_opt.zero_grad()
    loss_c = F.cross_entropy(tiny_m(x_tiny), y_tiny)
    loss_c.backward()
    for name, param in tiny_m.named_parameters():
        assert param.grad is not None and torch.norm(param.grad).item() >= 0.0
    print("[OK] Toàn bộ tham số có gradient hợp lệ sau backward.")

    # 5. Part 2: Baseline (3 seeds)
    print("\n" + "-" * 50)
    print("PART 2: HUẤN LUYỆN BASELINE (3 SEEDS)")
    print("-" * 50)
    baseline_results = []
    for s in [1, 2, 3]:
        cfg_b = {
            **DEFAULT_CFG,
            "exp_id": f"base-s{s}",
            "seed": s,
            "lr": 0.05,
            "description": f"Baseline M-base (seed {s})"
        }
        print(f"\n>>> Chạy Baseline {cfg_b['exp_id']} (epoch=20, lr=0.05)...")
        r = run_experiment(cfg_b, data)
        save_result(r, str(res_dir))
        plot_run(r, str(fig_dir / f"{cfg_b['exp_id']}.png"))
        baseline_results.append(r)
        print(f"    Val Acc: {r['summary']['val_acc']:.4f} | Val Macro-F1: {r['summary']['val_macro_f1']:.4f}")

    base_f1s = [r["summary"]["val_macro_f1"] for r in baseline_results]
    base_accs = [r["summary"]["val_acc"] for r in baseline_results]
    mean_f1, std_f1 = float(np.mean(base_f1s)), float(np.std(base_f1s))
    mean_acc, std_acc = float(np.mean(base_accs)), float(np.std(base_accs))
    noise_2sigma = 2 * std_f1
    print(f"\nBaseline Summary: Val Macro-F1 = {mean_f1:.4f} ± {std_f1:.4f} | 2sigma = {noise_2sigma:.4f}")

    # 6. Part 3: 7 Chủ đề thí nghiệm
    print("\n" + "-" * 50)
    print("PART 3: 7 CHỦ ĐỀ THÍ NGHIỆM")
    print("-" * 50)

    # 3.1 Loss: MSE
    print("\n[Chủ đề 1] Hàm mất mát CE vs MSE")
    cfg_mse = {**DEFAULT_CFG, "exp_id": "loss-mse", "group": "loss", "loss": "mse", "description": "Hàm mất mát MSE trên nhãn one-hot"}
    r_mse = run_experiment(cfg_mse, data)
    save_result(r_mse, str(res_dir))
    plot_run(r_mse, str(fig_dir / f"{cfg_mse['exp_id']}.png"))
    plot_compare([baseline_results[0], r_mse], "val_macro_f1", str(fig_dir / "compare_loss.png"), "So sánh CE vs MSE (Val Macro-F1)")

    # 3.2 Optimizers
    print("\n[Chủ đề 2] Bộ tối ưu hoá (SGD, SGDM, Adam, AdamW)")
    opt_configs = [
        {"exp_id": "opt-sgd-lr0.05", "group": "optimizer", "optimizer": "sgd", "lr": 0.05, "description": "SGD thuần lr=0.05"},
        {"exp_id": "opt-sgd-lr0.1", "group": "optimizer", "optimizer": "sgd", "lr": 0.1, "description": "SGD thuần lr=0.1"},
        {"exp_id": "opt-sgdm-lr0.02", "group": "optimizer", "optimizer": "sgd_momentum", "lr": 0.02, "description": "SGD+Momentum lr=0.02"},
        {"exp_id": "opt-adam-lr1e-3", "group": "optimizer", "optimizer": "adam", "lr": 0.001, "description": "Adam lr=1e-3"},
        {"exp_id": "opt-adam-lr3e-4", "group": "optimizer", "optimizer": "adam", "lr": 0.0003, "description": "Adam lr=3e-4"},
        {"exp_id": "opt-adamw-lr1e-3", "group": "optimizer", "optimizer": "adamw", "lr": 0.001, "weight_decay": 0.01, "description": "AdamW lr=1e-3, wd=0.01"},
        {"exp_id": "opt-adamw-lr3e-4", "group": "optimizer", "optimizer": "adamw", "lr": 0.0003, "weight_decay": 0.01, "description": "AdamW lr=3e-4, wd=0.01"},
    ]
    opt_results = [baseline_results[0]]
    for c in opt_configs:
        cfg_exp = {**DEFAULT_CFG, **c}
        print(f"  >>> Chạy {cfg_exp['exp_id']}...")
        r = run_experiment(cfg_exp, data)
        save_result(r, str(res_dir))
        plot_run(r, str(fig_dir / f"{cfg_exp['exp_id']}.png"))
        opt_results.append(r)
    plot_compare(opt_results, "val_macro_f1", str(fig_dir / "compare_optimizer.png"), "So sánh các bộ tối ưu (Val Macro-F1)")

    # 3.3 Hyper-parameters
    print("\n[Chủ đề 3] Hyper-parameters (Batch Size & Capacity)")
    hparam_configs = [
        {"exp_id": "hparam-batch128", "group": "hparam", "batch": 128, "description": "Batch size 128"},
        {"exp_id": "hparam-batch2048", "group": "hparam", "batch": 2048, "description": "Batch size 2048"},
        {"exp_id": "hparam-mwide", "group": "hparam", "hidden": (512, 256), "description": "Mô hình M-wide (161k params)"},
        {"exp_id": "hparam-mdeep", "group": "hparam", "hidden": (256, 128, 64), "description": "Mô hình M-deep (55k params)"},
    ]
    hparam_results = [baseline_results[0]]
    for c in hparam_configs:
        cfg_exp = {**DEFAULT_CFG, **c}
        print(f"  >>> Chạy {cfg_exp['exp_id']}...")
        r = run_experiment(cfg_exp, data)
        save_result(r, str(res_dir))
        plot_run(r, str(fig_dir / f"{cfg_exp['exp_id']}.png"))
        hparam_results.append(r)
    plot_compare(hparam_results, "val_macro_f1", str(fig_dir / "compare_hparam.png"), "So sánh Hyper-parameters (Val Macro-F1)")

    # 3.4 Dropout
    print("\n[Chủ đề 4] Dropout (q=0.2, 0.5)")
    dropout_configs = [
        {"exp_id": "drop-0.2", "group": "dropout", "dropout": 0.2, "description": "Dropout q=0.2"},
        {"exp_id": "drop-0.5", "group": "dropout", "dropout": 0.5, "description": "Dropout q=0.5"},
    ]
    dropout_results = [baseline_results[0]]
    for c in dropout_configs:
        cfg_exp = {**DEFAULT_CFG, **c}
        print(f"  >>> Chạy {cfg_exp['exp_id']}...")
        r = run_experiment(cfg_exp, data)
        save_result(r, str(res_dir))
        plot_run(r, str(fig_dir / f"{cfg_exp['exp_id']}.png"))
        dropout_results.append(r)
    plot_compare(dropout_results, "val_loss", str(fig_dir / "compare_dropout.png"), "So sánh Dropout (Val Loss)")

    # 3.5 Gradient Clipping
    print("\n[Chủ đề 5] Gradient Clipping")
    clip_configs = [
        {"exp_id": "clip-1.0-normallr", "group": "clipping", "clip_norm": 1.0, "description": "Clip 1.0 ở lr thường 0.05"},
        {"exp_id": "noclip-highlr", "group": "clipping", "lr": 0.8, "clip_norm": None, "description": "Không clip ở lr cao 0.8"},
        {"exp_id": "clip-1.0-highlr", "group": "clipping", "lr": 0.8, "clip_norm": 1.0, "description": "Clip 1.0 ở lr cao 0.8"},
    ]
    clip_results = [baseline_results[0]]
    for c in clip_configs:
        cfg_exp = {**DEFAULT_CFG, **c}
        print(f"  >>> Chạy {cfg_exp['exp_id']}...")
        r = run_experiment(cfg_exp, data)
        save_result(r, str(res_dir))
        plot_run(r, str(fig_dir / f"{cfg_exp['exp_id']}.png"))
        clip_results.append(r)
    plot_compare(clip_results, "grad_norm", str(fig_dir / "compare_clipping.png"), "So sánh Gradient Norm khi Clip và Không Clip")

    # 3.6 Mixed Precision
    print("\n[Chủ đề 6] Mixed Precision (AMP)")
    amp_configs = [
        {"exp_id": "amp-fp16", "group": "amp", "precision": "fp16", "description": "Mixed Precision FP16"},
    ]
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_configs.append({"exp_id": "amp-bf16", "group": "amp", "precision": "bf16", "description": "Mixed Precision BF16"})
    amp_results = [baseline_results[0]]
    for c in amp_configs:
        cfg_exp = {**DEFAULT_CFG, **c}
        print(f"  >>> Chạy {cfg_exp['exp_id']}...")
        r = run_experiment(cfg_exp, data)
        save_result(r, str(res_dir))
        plot_run(r, str(fig_dir / f"{cfg_exp['exp_id']}.png"))
        amp_results.append(r)
    plot_compare(amp_results, "val_macro_f1", str(fig_dir / "compare_amp.png"), "So sánh FP32 vs Mixed Precision")

    # 3.7 Initialization
    print("\n[Chủ đề 7] Khởi tạo tham số (He vs Xavier vs Normal vs Zeros)")
    init_configs = [
        {"exp_id": "init-xavier", "group": "init", "init": "xavier", "description": "Khởi tạo Xavier Normal"},
        {"exp_id": "init-normal", "group": "init", "init": "normal", "description": "Khởi tạo Normal std=0.01"},
        {"exp_id": "init-zeros", "group": "init", "init": "zeros", "description": "Khởi tạo toàn bộ W=0"},
    ]
    init_results = [baseline_results[0]]
    for c in init_configs:
        cfg_exp = {**DEFAULT_CFG, **c}
        print(f"  >>> Chạy {cfg_exp['exp_id']}...")
        m_temp = MLP(hidden=cfg_exp["hidden"], dropout=cfg_exp["dropout"], init=cfg_exp["init"]).to(device)
        act_stds = activation_stats(m_temp, data["X_val"][:512])
        print(f"    Activation stds ({cfg_exp['init']}): {[round(s, 5) for s in act_stds]}")
        r = run_experiment(cfg_exp, data)
        save_result(r, str(res_dir))
        plot_run(r, str(fig_dir / f"{cfg_exp['exp_id']}.png"))
        init_results.append(r)
    plot_compare(init_results, "val_macro_f1", str(fig_dir / "compare_init.png"), "So sánh các phương pháp khởi tạo tham số")

    # 7. Part 4: Chọn Best Model theo Validation & Chấm điểm Eval
    print("\n" + "-" * 50)
    print("PART 4: CHỌN BEST MODEL & ĐÁNH GIÁ TRÊN EVAL")
    print("-" * 50)
    all_runs = load_results(str(res_dir))
    valid_runs = [r for r in all_runs if not r["summary"].get("diverged", False)]
    best_val_run = max(valid_runs, key=lambda r: r["summary"]["val_macro_f1"])
    best_val_cfg = best_val_run["cfg"]
    print(f"Mô hình đạt Val Macro-F1 cao nhất: {best_val_cfg['exp_id']} ({best_val_cfg['description']})")
    print(f"  Val Macro-F1 = {best_val_run['summary']['val_macro_f1']:.4f}")

    # Chạy cấu hình cuối cùng (Final Best Model) kết hợp CosineAnnealingLR
    final_cfg = {
        **best_val_cfg,
        "exp_id": "final-best",
        "group": "final",
        "scheduler": "cosine",
        "description": f"Cấu hình tối ưu cuối cùng (Dựa trên {best_val_cfg['exp_id']} + Cosine LR Scheduler)",
    }
    print(f"\n>>> Huấn luyện Final Best Model ({final_cfg['exp_id']})...")
    res_final = run_experiment(final_cfg, data)
    save_result(res_final, str(res_dir))
    plot_run(res_final, str(fig_dir / f"{final_cfg['exp_id']}.png"))

    # Đánh giá trên tập eval
    pred_path = out_dir / "predictions_eval.csv"
    eval_json_path = out_dir / "eval_result.json"
    print("\n>>> Dự đoán trên tập eval và lưu predictions_eval.csv...")
    final_scores = final_eval(final_cfg, res_final, data, str(pred_path))

    # Chạy script chấm điểm chính thức
    eval_cmd = [
        sys.executable, str(repo_root / "scripts" / "evaluate.py"),
        "--pred", str(pred_path),
        "--data", str(repo_root / "data" / "covtype.csv.gz"),
        "--meta", str(repo_root / "data" / "split_metadata.csv"),
        "--out", str(eval_json_path)
    ]
    subprocess.run(eval_cmd, check=True)

    with open(eval_json_path, "r", encoding="utf-8") as fp:
        eval_result = json.load(fp)

    print("\n" + "=" * 50)
    print("CHẤM ĐIỂM CHÍNH THỨC TẬP EVAL:")
    print(f"  Accuracy: {eval_result['accuracy']:.4f}")
    print(f"  Macro-F1: {eval_result['macro_f1']:.4f}")
    print("=" * 50)

    # 8. Xuất file experiments.xlsx
    print("\n>>> Xuất kết quả vào experiments.xlsx...")
    all_runs = load_results(str(res_dir))
    rows = []
    for r in all_runs:
        eid = r["cfg"]["exp_id"]
        ev_s = None
        if eid == "final-best":
            ev_s = eval_result
        rows.append(to_row(r, eval_scores=ev_s))

    template_xlsx = repo_root / "templates" / "experiment_table_template.xlsx"
    out_xlsx = out_dir / "experiments.xlsx"
    write_xlsx(rows, str(template_xlsx), str(out_xlsx))

    print("\n[HOÀN TẤT] Toàn bộ thí nghiệm, biểu đồ, bảng kết quả và file dự đoán đã hoàn thành!")


if __name__ == "__main__":
    main()
