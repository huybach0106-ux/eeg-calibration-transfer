import subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eegxs as E
from synthetic import make_eegbci_like


def test_euclidean_align_whitens():
    X, _, _ = make_eegbci_like(3)
    Z = E.euclidean_align(E.crop(X))
    C = np.mean([z @ z.T / z.shape[1] for z in Z], 0)
    assert np.allclose(C, np.eye(len(C)), atol=1e-3)


def test_crop_and_channels():
    X, _, ch = make_eegbci_like(1)
    assert E.crop(X, 1.0, 2.0).shape[2] == 160
    assert E.pick_channels(X, ch, ["C3", "Cz", "C4"]).shape[1] == 3


def test_csp_separates_and_null_is_chance():
    data = {s: (E.crop(make_eegbci_like(s, snr=0.5)[0]), make_eegbci_like(s, snr=0.5)[1]) for s in (1, 2, 3)}
    assert E.within_subject(data).accuracy.mean() > 0.6
    assert abs(E.within_subject(data, shuffle_labels=True).accuracy.mean() - 0.5) < 0.15


def test_alignment_helps_cross_subject():
    data = {s: (E.crop(make_eegbci_like(s)[0]), make_eegbci_like(s)[1]) for s in range(1, 7)}
    ea = E.cross_subject(data, [0], align=True).accuracy.mean()
    raw = E.cross_subject(data, [0], align=False).accuracy.mean()
    assert ea > raw + 0.1


def test_run_script_synthetic(tmp_path):
    r = subprocess.run([sys.executable, "run.py", "--synthetic", "--n-subjects", "4", "--shots", "0", "5",
                        "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "figures" / "fig1_calibration_curve.png").exists()
