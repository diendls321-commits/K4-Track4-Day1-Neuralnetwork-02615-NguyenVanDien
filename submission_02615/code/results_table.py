"""results_table.py — Lưu kết quả thí nghiệm ra JSON và xuất bảng Excel.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
from pathlib import Path
import openpyxl


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result['cfg'], result['history'], result['summary'] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_id = result["cfg"]["exp_id"]
    file_path = out_dir / f"{exp_id}.json"

    # Chỉ lưu các dữ liệu serializable (bỏ best_state chứa tensor)
    to_save = {
        "cfg": result.get("cfg", {}),
        "history": result.get("history", {}),
        "summary": result.get("summary", {}),
    }

    # Chuyển đổi tuple hidden thành list để json serialize được
    if "hidden" in to_save["cfg"] and isinstance(to_save["cfg"]["hidden"], tuple):
        to_save["cfg"]["hidden"] = list(to_save["cfg"]["hidden"])

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2, ensure_ascii=False)

    return str(file_path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    p = Path(results_dir)
    if not p.exists():
        return []
    results = []
    for f in sorted(p.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            results.append(json.load(fp))
    return results


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng.
    """
    cfg = dict(result.get("cfg", {}))
    summary = dict(result.get("summary", {}))
    exp_id = cfg.get("exp_id", "")

    # Format hidden as string or tuple
    hidden = cfg.get("hidden", (256, 128))
    if isinstance(hidden, (list, tuple)):
        hidden_str = "-".join(str(h) for h in hidden)
    else:
        hidden_str = str(hidden)

    row = {
        "exp_id": exp_id,
        "group": cfg.get("group", ""),
        "description": cfg.get("description", ""),
        "loss": cfg.get("loss", "ce"),
        "optimizer": cfg.get("optimizer", "sgd_momentum"),
        "lr": cfg.get("lr", None),
        "weight_decay": cfg.get("weight_decay", 0.0),
        "batch": cfg.get("batch", 512),
        "epochs": cfg.get("epochs", 20),
        "hidden": hidden_str,
        "dropout": cfg.get("dropout", 0.0),
        "clip_norm": cfg.get("clip_norm", None) if cfg.get("clip_norm") is not None else "None",
        "precision": cfg.get("precision", "fp32"),
        "init": cfg.get("init", "he"),
        "seed": cfg.get("seed", 1),
        "step0_loss": summary.get("step0_loss", None),
        "best_val_loss": summary.get("best_val_loss", None),
        "best_epoch": summary.get("best_epoch", None),
        "final_train_loss": summary.get("final_train_loss", None),
        "final_val_loss": summary.get("final_val_loss", None),
        "val_acc": summary.get("val_acc", None),
        "val_macro_f1": summary.get("val_macro_f1", None),
        "time_per_epoch_s": summary.get("time_per_epoch_s", None),
        "peak_mem_MB": summary.get("peak_mem_MB", None),
        "diverged": summary.get("diverged", False),
        "eval_acc": None,
        "eval_macro_f1": None,
        "figure_file": f"figures/{exp_id}.png",
        "notes": notes,
    }

    if eval_scores:
        row["eval_acc"] = eval_scores.get("accuracy", None)
        row["eval_macro_f1"] = eval_scores.get("macro_f1", None)

    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền các dòng vào sheet 'Experiments' của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.
    KHÔNG dùng data_only=True để giữ nguyên công thức Excel.
    Bỏ qua các cột công thức tự tính.
    """
    wb = openpyxl.load_workbook(template_path)
    if "Experiments" not in wb.sheetnames:
        raise ValueError(f"Sheet 'Experiments' không có trong {template_path}")

    ws = wb["Experiments"]
    # Đọc headers dòng 1
    headers = [cell.value for cell in ws[1] if cell.value is not None]
    col_map = {col_name: col_idx + 1 for col_idx, col_name in enumerate(headers)}

    # Danh sách các cột công thức không được ghi đè
    formula_cols = {
        "step0_gap_vs_lnC",
        "gap_val_minus_train",
        "delta_val_f1_vs_base",
        "beyond_noise",
    }

    # Ghi từng row
    for r_idx, r_data in enumerate(rows, start=2):
        for key, val in r_data.items():
            if key in formula_cols:
                continue
            if key in col_map:
                c_idx = col_map[key]
                ws.cell(row=r_idx, column=c_idx, value=val)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    print(f"Đã lưu bảng kết quả: {out_path} ({len(rows)} thí nghiệm)")
