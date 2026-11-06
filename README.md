# Người dùng mới cần bao nhiêu phút calibration? Decoding EEG xuyên subject

Dữ liệu PhysioNet EEG Motor Movement/Imagery: 109 người, 64 kênh, 160 Hz. Repo dùng các run tưởng
tượng nắm tay trái hoặc phải (khoảng 45 trial mỗi người) và hỏi: một người mới cần bao nhiêu trial
có nhãn để decoder dùng được, nếu tận dụng dữ liệu của người khác?

**Câu hỏi:** dữ liệu của người khác thay thế được bao nhiêu phần calibration của người mới, và một
bước căn chỉnh đơn giản (Euclidean Alignment) giúp được bao nhiêu?

## Thí nghiệm

| # | Thí nghiệm | Câu hỏi |
| --- | --- | --- |
| 1 | Within-subject (5-fold) | Mốc khi có đủ ~45 trial của chính người đó |
| 2 | Nhãn xáo trộn | Null control: phải quanh 50% |
| 3 | Leave-one-subject-out, 0 trial | Áp thẳng decoder của người khác: có và không có Euclidean Alignment |
| 4 | Thêm k = 2, 5, 10, 15 trial/lớp | Đường calibration: mỗi trial của người mới đáng bao nhiêu? |
| 5 | Chỉ dùng k trial của người mới | So sánh: không tận dụng người khác |
| 6 | 64 / 21 / 3 kênh | Thiết bị đeo ít kênh thì mất bao nhiêu? |
| 7 | Cửa sổ 0,5–3 s | Quyết định nhanh (gần real-time) thì mất bao nhiêu? |

Mô hình: CSP (bản của MNE, giải thích trong `eegxs.py`) + LDA, lọc 7–30 Hz (nhịp mu và beta).

**Euclidean Alignment** (He & Wu, 2019): với mỗi người, tính hiệp phương sai trung bình R của các trial,
rồi nhân mọi trial với R^(-1/2). Không dùng nhãn. Với người mới, R được tính từ các trial của họ
(không nhãn), giống việc ghi một đoạn EEG ngắn trước khi dùng. Hãy ghi rõ điểm này trong report.

## Chạy

Không cần GPU.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q tests         # 5 test, ~10 giây
python run.py --synthetic         # chạy thử trên dữ liệu giả lập
python run.py --n-subjects 10     # thử với 10 người
python run.py                     # 30 người (mặc định)
```

Lần chạy đầu, mne tải dữ liệu từ PhysioNet vào `data/` (vài MB mỗi người) và cache epoch vào `cache/`.
Một số subject (88, 89, 92, 100, 104) bị loại vì tần số lấy mẫu hoặc số trial khác chuẩn.

Kết quả: `results/SUMMARY.md`, `results/*.csv`, `figures/fig1_calibration_curve.png`, `figures/fig2_channels_window.png`.

## Tài liệu tham khảo

- Schalk et al. BCI2000: a general-purpose brain-computer interface system. IEEE TBME 2004.
- Goldberger et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 2000.
- He & Wu. Transfer learning for brain-computer interfaces: a Euclidean space data alignment approach. IEEE TBME 2019.
- Blankertz et al. Optimizing spatial filters for robust EEG single-trial analysis (CSP). IEEE Signal Processing Magazine 2008.
