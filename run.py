"""Chạy toàn bộ thí nghiệm và vẽ hình.

    python run.py                          # 30 subject đầu (trừ subject lỗi)
    python run.py --n-subjects 10          # thử nhanh
    python run.py --synthetic              # chạy thử không cần tải dữ liệu

Kết quả: results/*.csv, results/SUMMARY.md, figures/*.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eegxs as E

p = argparse.ArgumentParser()
p.add_argument("--n-subjects", type=int, default=30)
p.add_argument("--shots", type=int, nargs="*", default=[0, 2, 5, 10, 15])
p.add_argument("--synthetic", action="store_true")
p.add_argument("--out", default=".")
a = p.parse_args()
res, fig = Path(a.out) / "results", Path(a.out) / "figures"
res.mkdir(parents=True, exist_ok=True)
fig.mkdir(parents=True, exist_ok=True)

subjects = [s for s in range(1, 110) if s not in E.BAD_SUBJECTS][:a.n_subjects]
full = {}
for s in subjects:
    if a.synthetic:
        from synthetic import make_eegbci_like
        X, y, ch = make_eegbci_like(s)
    else:
        try:
            X, y, ch = E.load_subject(s)
        except Exception as e:  # mạng lỗi hoặc file hỏng: bỏ qua, ghi lại
            print(f"bỏ qua subject {s}: {e}")
            continue
    full[s] = (X, y)
    print(f"subject {s}: {len(y)} trial ({y.sum()} phải), {X.shape[1]} kênh", flush=True)
print(f"{len(full)} subject")


def make(ch_set=None, tmin=1.0, tmax=2.0):
    return {s: (E.pick_channels(E.crop(X, tmin, tmax), ch, E.CHANNEL_SETS[ch_set] if ch_set else None), y)
            for s, (X, y) in full.items()}


main = make()
print("== within-subject (mốc đủ calibration)")
within = E.within_subject(main)
within_null = E.within_subject(main, shuffle_labels=True)
print(f"   {100 * within.accuracy.mean():.1f}%  (nhãn xáo trộn: {100 * within_null.accuracy.mean():.1f}%)")

print("== cross-subject, có và không có Euclidean Alignment")
xs = pd.concat([E.cross_subject(main, a.shots, align=True), E.cross_subject(main, a.shots, align=False, n_draws=2)])

print("== số kênh và độ dài cửa sổ (zero-calibration với EA, và within-subject)")
abl = []
for name in E.CHANNEL_SETS:
    d = make(name)
    abl.append({"factor": "kênh", "setting": name,
                "cross_subject_0shot": E.cross_subject(d, [0], align=True).accuracy.mean(),
                "within_subject": E.within_subject(d).accuracy.mean()})
for dur in (0.5, 1.0, 2.0, 3.0):
    d = make(tmin=0.5, tmax=0.5 + dur)
    abl.append({"factor": "cửa sổ", "setting": f"{dur:g} s",
                "cross_subject_0shot": E.cross_subject(d, [0], align=True).accuracy.mean(),
                "within_subject": E.within_subject(d).accuracy.mean()})
abl = pd.DataFrame(abl)
print(abl.to_string(index=False))

within.to_csv(res / "within_subject.csv", index=False)
within_null.to_csv(res / "within_subject_shuffled.csv", index=False)
xs.to_csv(res / "cross_subject.csv", index=False)
abl.to_csv(res / "ablation.csv", index=False)


def sem(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


COL = {"Nguồn + EA": "#2b6cb0", "Nguồn (không align)": "#8a8a8a", "Chỉ người mới": "#dd6b20"}
per = xs.groupby(["method", "shots_per_class", "target"]).accuracy.mean().reset_index()
f, ax = plt.subplots(figsize=(6.5, 4))
for m, g in per.groupby("method"):
    ks = sorted(g.shots_per_class.unique())
    ax.errorbar(ks, [100 * g[g.shots_per_class == k].accuracy.mean() for k in ks],
                yerr=[100 * sem(g[g.shots_per_class == k].accuracy) for k in ks],
                marker="o", capsize=3, color=COL[m], label=m)
ax.axhline(100 * within.accuracy.mean(), color="k", ls="--", lw=1, label="Within-subject (~45 trial)")
ax.axhline(50, color="#bbbbbb", lw=1, label="may rủi 50%")
ax.set_xlabel("Số trial calibration có nhãn mỗi lớp của người mới")
ax.set_ylabel("Độ chính xác trái/phải (%)")
ax.set_title(f"Leave-one-subject-out, {len(full)} subject")
ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
ax.spines[["top", "right"]].set_visible(False)
f.savefig(fig / "fig1_calibration_curve.png", dpi=200, bbox_inches="tight")

f, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, fac in zip(axes, ["kênh", "cửa sổ"]):
    g = abl[abl.factor == fac]
    x = np.arange(len(g))
    ax.bar(x - 0.2, 100 * g.within_subject, 0.4, color="#1a202c", label="Within-subject")
    ax.bar(x + 0.2, 100 * g.cross_subject_0shot, 0.4, color="#2b6cb0", label="Người mới, 0 calibration (EA)")
    ax.set_xticks(x, g.setting)
    ax.axhline(50, color="#bbbbbb", lw=1)
    ax.set_title("Số kênh" if fac == "kênh" else "Độ dài cửa sổ (bắt đầu 0,5 s sau cue)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Độ chính xác (%)")
axes[0].set_ylim(40, 100)
axes[1].legend(frameon=False, fontsize=8)
f.savefig(fig / "fig2_channels_window.png", dpi=200, bbox_inches="tight")

g0 = per[per.shots_per_class == 0].groupby("method").accuracy.mean()
gk = per[per.shots_per_class == max(a.shots)].groupby("method").accuracy.mean()
summary = f"""# Tóm tắt ({len(full)} subject{', DỮ LIỆU GIẢ LẬP' if a.synthetic else ''})

| Thí nghiệm | Kết quả |
| --- | --- |
| Within-subject (CSP+LDA, 5-fold) | {100 * within.accuracy.mean():.1f}% ± {100 * sem(within.accuracy):.1f} |
| Within-subject, nhãn xáo trộn (null) | {100 * within_null.accuracy.mean():.1f}% |
| Người mới, 0 calibration, không align | {100 * g0.get('Nguồn (không align)', np.nan):.1f}% |
| Người mới, 0 calibration, có EA | {100 * g0.get('Nguồn + EA', np.nan):.1f}% |
| Người mới, {max(a.shots)} trial/lớp, nguồn + EA | {100 * gk.get('Nguồn + EA', np.nan):.1f}% |
| Người mới, {max(a.shots)} trial/lớp, chỉ dữ liệu người mới | {100 * gk.get('Chỉ người mới', np.nan):.1f}% |

Số kênh và độ dài cửa sổ: xem results/ablation.csv và figures/fig2_channels_window.png.
"""
(res / "SUMMARY.md").write_text(summary)
print(summary)
