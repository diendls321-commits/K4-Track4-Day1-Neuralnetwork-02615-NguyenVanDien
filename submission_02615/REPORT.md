# Báo cáo Lab 1 — CoverType

Mã bài 02615 | Colab Tesla T4 | PyTorch 2.11.0+cu130

## Tóm tắt
Đã chạy 26 thí nghiệm thuộc 7 chủ đề. Chọn M-wide (161k tham số), hparam-mwide, theo validation Macro-F1 0.8661. Eval Accuracy 0.9117; Macro-F1 0.8665.

## Dữ liệu và kiểm tra
54 đặc trưng, 7 lớp; train/validation/eval: 371847 / 92962 / 116203 mẫu. Majority accuracy 0.4876. M-base có 47879 tham số, 7 logits. Overfit 20 mẫu đạt 100%% accuracy, loss 0.000007 sau 300 bước; gradient hợp lệ.

## Baseline
Ba seed validation Macro-F1: 0.8351, 0.8314, 0.8293; trung bình 0.8319, std 0.0024, 2sigma 0.0048. Baseline eval Macro-F1 0.8409; mức tăng 0.0256.

## Thí nghiệm
So sánh CE/MSE; SGD/SGDM/Adam/AdamW, learning rate, batch/capacity, dropout, clipping, FP16/BF16, He/Xavier/Normal/Zero init. Toàn bộ 26 dòng trong experiments.xlsx; biểu đồ trong figures/.

## F1 theo lớp
|Lớp|F1|
|--:|--:|
|0|0.9077|
|1|0.9258|
|2|0.9064|
|3|0.8303|
|4|0.7485|
|5|0.8164|
|6|0.9303|

Lớp 4 là khó nhất theo F1. Lệch lớp lớn khiến Macro-F1 hữu ích hơn Accuracy đơn lẻ.

## Kết luận
M-wide tốt nhất trong các cấu hình thử theo validation; cải thiện so với baseline. F1 các lớp thiểu số 4 và 5 vẫn thấp hơn. Kết luận giới hạn ở split/cấu hình lab.

## Bàn giao
Workbook có Legend, Experiments, Seeds, Summary. predictions_eval.csv có 116203 dòng và cột row_id,pred. eval_result.json chứa đánh giá chính thức. code/ chứa mã và notebook.
