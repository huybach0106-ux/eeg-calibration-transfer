"""Người dùng mới cần bao nhiêu dữ liệu calibration? Decoding EEG motor imagery xuyên subject.

Dữ liệu: PhysioNet EEG Motor Movement/Imagery (Schalk et al., 2004; Goldberger et al., 2000), 109
người, 64 kênh, 160 Hz. Dùng run 4, 8, 12: tưởng tượng nắm tay TRÁI hoặc PHẢI, khoảng 45 trial/người.
mne tải tự động (vài MB mỗi người).

Mọi hàm thí nghiệm nhận dict {subject: (X, y)} với X (n_trial, n_kênh, n_mẫu) và y (n_trial,),
nên có thể test bằng dữ liệu giả lập mà không cần tải gì.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    pass

SFREQ = 160.0
EPOCH_TMIN = -1.0                       # epoch được lưu từ -1 s đến 4 s quanh cue
BAD_SUBJECTS = {88, 89, 92, 100, 104}   # tần số lấy mẫu hoặc số trial khác, thường bị loại
CHANNEL_SETS = {
    "64 kênh": None,
    "21 kênh (10-20)": ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
                        "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"],
    "3 kênh (C3, Cz, C4)": ["C3", "Cz", "C4"],
}


# ---------------------------------------------------------------- dữ liệu

def load_subject(subject: int, data_dir="data", cache_dir="cache", fmin=7.0, fmax=30.0):
    """Tải (lần đầu), lọc 7-30 Hz, cắt epoch -1..4 s quanh cue. Trả về X, y (0 = trái, 1 = phải), tên kênh."""
    cache = Path(cache_dir) / f"sub-{subject:03d}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], list(z["ch_names"])
    import mne
    from mne.datasets import eegbci

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    files = eegbci.load_data(subject, [4, 8, 12], path=str(data_dir), update_path=False, verbose="error")
    raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose="error") for f in files])
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"), verbose="error")
    if abs(raw.info["sfreq"] - SFREQ) > 1e-3:
        raise ValueError(f"subject {subject}: sfreq {raw.info['sfreq']} khác {SFREQ}")
    raw.filter(fmin, fmax, fir_design="firwin", skip_by_annotation="edge", verbose="error")
    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3), verbose="error")
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    ep = mne.Epochs(raw, events, dict(left=2, right=3), tmin=EPOCH_TMIN, tmax=4.0, proj=False,
                    picks=picks, baseline=None, preload=True, verbose="error")
    X = ep.get_data().astype(np.float32)
    y = (ep.events[:, -1] == 3).astype(int)
    ch = [ep.ch_names[i] for i in range(len(ep.ch_names))]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y, ch_names=np.array(ch))
    return X, y, ch


def crop(X, tmin=1.0, tmax=2.0):
    """Lấy đoạn [tmin, tmax) giây sau cue (mặc định 1-2 s như ví dụ CSP của MNE)."""
    a = int(round((tmin - EPOCH_TMIN) * SFREQ))
    b = int(round((tmax - EPOCH_TMIN) * SFREQ))
    return X[:, :, a:b]


def pick_channels(X, ch_names, wanted):
    if wanted is None:
        return X
    idx = [ch_names.index(c) for c in wanted if c in ch_names]
    if len(idx) < len(wanted):
        missing = [c for c in wanted if c not in ch_names]
        raise KeyError(f"không thấy kênh {missing}")
    return X[:, idx]


# ---------------------------------------------------------------- mô hình

def csp_lda(n_channels: int, reg="ledoit_wolf"):
    """CSP + LDA: baseline kinh điển cho motor imagery.

    CSP (Common Spatial Patterns) tìm các bộ lọc không gian w sao cho phương sai của tín hiệu đã lọc
    lớn nhất ở lớp này và nhỏ nhất ở lớp kia (bài toán trị riêng tổng quát C1 w = lambda (C1 + C2) w).
    Đặc trưng = log công suất sau khi lọc, rồi đưa vào LDA.

    Dùng bản CSP của MNE (đã được kiểm chứng rộng rãi). Một bản tự viết trước đó cho kết quả thấp hơn
    rõ rệt trên dữ liệu thật, nên đã bị thay. rank="full" bỏ bước ước lượng rank ở mỗi lần fit cho nhanh.

    reg="ledoit_wolf" co hiệp phương sai về ma trận đơn vị: cần khi chỉ có vài chục trial của một người.
    reg=None cho mô hình train trên dữ liệu gộp nhiều người (hơn 1000 trial). Lý do: sau Euclidean
    Alignment, hiệp phương sai trung bình của mỗi người ĐÃ là ma trận đơn vị, nên Ledoit-Wolf co gần như
    hoàn toàn và xoá mất khác biệt giữa hai lớp (đã thấy độ chính xác tụt dưới may rủi)."""
    from mne.decoding import CSP
    return make_pipeline(CSP(n_components=min(6, n_channels), reg=reg, log=True,
                             norm_trace=False, rank="full"),
                         LinearDiscriminantAnalysis())


def euclidean_align(X: np.ndarray) -> np.ndarray:
    """Euclidean Alignment (He & Wu, 2019): làm trắng mọi trial của một người bằng R^(-1/2), với R là
    hiệp phương sai trung bình của người đó. Sau bước này, hiệp phương sai trung bình của mỗi người đều
    là ma trận đơn vị, nên dữ liệu của các người khác nhau "cùng hệ trục" hơn. Không dùng nhãn."""
    C = np.mean([x @ x.T / x.shape[1] for x in X], axis=0)
    w, V = eigh(C)
    R = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12 * w.max()))) @ V.T
    return np.einsum("ij,njt->nit", R, X).astype(np.float32)


# ---------------------------------------------------------------- thí nghiệm

def within_subject(data: dict, n_folds=5, seed=0, shuffle_labels=False) -> pd.DataFrame:
    """Mốc 'đủ calibration': train và test trên cùng một người (5-fold CV)."""
    rows = []
    rng = np.random.default_rng(seed)
    for s, (X, y) in data.items():
        yy = rng.permutation(y) if shuffle_labels else y
        acc = cross_val_score(csp_lda(X.shape[1]), X, yy,
                              cv=StratifiedKFold(n_folds, shuffle=True, random_state=seed)).mean()
        rows.append({"subject": s, "accuracy": float(acc)})
    return pd.DataFrame(rows)


def cross_subject(data: dict, shots=(0, 2, 5, 10, 15), align=True, n_draws=5, seed=0,
                  target_weight=5) -> pd.DataFrame:
    """Leave-one-subject-out: train trên mọi người khác + k trial có nhãn MỖI LỚP của người mới,
    test trên các trial còn lại của người mới.

    align=True: Euclidean Alignment cho từng người. Với người mới, R được ước lượng từ mọi trial của họ
    nhưng KHÔNG dùng nhãn (giống việc thu một đoạn EEG không cần nhiệm vụ trước khi dùng).
    target_weight: lặp lại trial calibration của người mới để chúng có trọng số hơn trong lượng lớn dữ liệu nguồn.
    """
    rng = np.random.default_rng(seed)
    prep = {s: (euclidean_align(X) if align else X, y) for s, (X, y) in data.items()}
    rows = []
    for t in prep:
        Xt, yt = prep[t]
        Xs = np.concatenate([prep[s][0] for s in prep if s != t])
        ys = np.concatenate([prep[s][1] for s in prep if s != t])
        for k in shots:
            for d in range(n_draws if k > 0 else 1):
                cal = np.concatenate([rng.choice(np.where(yt == c)[0], k, replace=False) for c in (0, 1)]) \
                    if k > 0 else np.array([], int)
                test = np.setdiff1d(np.arange(len(yt)), cal)
                Xtr = np.concatenate([Xs] + [Xt[cal]] * target_weight)
                ytr = np.concatenate([ys] + [yt[cal]] * target_weight)
                acc = csp_lda(Xt.shape[1], reg=None).fit(Xtr, ytr).score(Xt[test], yt[test])
                row = {"target": t, "shots_per_class": k, "draw": d, "aligned": align, "accuracy": acc,
                       "method": "Nguồn + EA" if align else "Nguồn (không align)"}
                rows.append(row)
                if k >= 5 and align:   # so sánh: chỉ dùng k trial của người mới
                    sc = csp_lda(Xt.shape[1]).fit(data[t][0][cal], data[t][1][cal]).score(data[t][0][test], data[t][1][test])
                    rows.append({**row, "method": "Chỉ người mới", "accuracy": sc})
    return pd.DataFrame(rows)
