import json
from pathlib import Path

import cv2
import numpy as np

from python_research.calibration.drift_gate import (
    bhattacharyya_distance,
    check_frame,
    compute_optical_metrics,
)


def _load_screw_baseline_and_image() -> tuple[dict, np.ndarray]:
    baseline_path = Path("artifacts/calibration/drift_baseline_mvtec_ad_screw.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    img_path = next((Path("data/raw/mvtec_ad/screw/test/good")).glob("*.png"))
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    return baseline, img_bgr


def test_clean_screw_frame_passes() -> None:
    baseline, img_bgr = _load_screw_baseline_and_image()
    res = check_frame(img_bgr, baseline)
    assert res["passed"] is True
    assert res["flags"] == []

def test_blur_and_dark_faults_are_caught() -> None:
    baseline, img_bgr = _load_screw_baseline_and_image()

    blurred = cv2.GaussianBlur(img_bgr, (21, 21), 7.0)
    res_blur = check_frame(blurred, baseline)
    assert res_blur["passed"] is False
    assert "BLUR_DRIFT" in res_blur["flags"]

    dark = (img_bgr.astype(np.float32) * 0.20).astype(np.uint8)
    res_dark = check_frame(dark, baseline)
    assert res_dark["passed"] is False
    assert "ILLUMINATION_DRIFT" in res_dark["flags"]

def test_bhattacharyya_distance_properties() -> None:
    _, img_bgr = _load_screw_baseline_and_image()
    metrics, hist = compute_optical_metrics(img_bgr)
    assert metrics["laplacian_var"] > 0.0
    assert abs(float(hist.sum()) - 1.0) < 1e-5
    assert bhattacharyya_distance(hist, hist) < 1e-7

    h_left = np.zeros(32, dtype=np.float64)
    h_right = np.zeros(32, dtype=np.float64)
    h_left[0] = 1.0
    h_right[-1] = 1.0
    assert abs(bhattacharyya_distance(h_left, h_right) - 1.0) < 1e-7
    