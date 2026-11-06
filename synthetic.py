"""Dữ liệu giả lập giống PhysioNet motor imagery: 45 trial, 64 kênh tên chuẩn, 160 Hz, epoch -1..4 s.

Tưởng tượng tay trái/phải làm giảm biên độ nhịp mu (~10 Hz) ở vùng vận động bên đối diện sau cue.
Mỗi người có ma trận trộn kênh riêng và độ lớn tín hiệu khác nhau, nên dữ liệu xuyên subject lệch
nhau: chạy đúng thì Euclidean Alignment phải giúp, và trial calibration của người mới phải giúp thêm."""
import numpy as np

# tên kênh sau mne.datasets.eegbci.standardize
CH = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3",
      "CP1", "CPz", "CP2", "CP4", "CP6", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
      "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8", "P7", "P5", "P3",
      "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz"]


def make_eegbci_like(subject=1, n_trials=45, sfreq=160.0, snr=0.25):
    rng = np.random.default_rng(subject)
    n_ch, T = len(CH), int(5 * sfreq) + 1
    t = np.arange(T) / sfreq - 1.0
    c3, c4 = CH.index("C3"), CH.index("C4")
    y = rng.integers(0, 2, n_trials)
    X = np.zeros((n_trials, n_ch, T))
    mix = np.eye(n_ch) + 0.4 * rng.standard_normal((n_ch, n_ch)) / np.sqrt(n_ch)   # khác nhau giữa người
    gain = np.exp(rng.normal(0, 0.5, n_ch))
    for i in range(n_trials):
        src = rng.standard_normal((n_ch, T))
        for ch, side in ((c3, 1), (c4, 0)):       # tay phải -> giảm mu ở C3 (bán cầu trái), và ngược lại
            amp = 2.0 * np.ones(T)
            if y[i] == side:
                amp[t > 0.5] *= (1 - snr)
            src[ch] += amp * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
        X[i] = (mix @ src) * gain[:, None]
    return (X * 1e-5).astype(np.float32), y, CH
