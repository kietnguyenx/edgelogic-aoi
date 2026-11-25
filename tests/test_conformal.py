import json
from pathlib import Path

import numpy as np

from python_research.calibration.conformal_triage import (
    decide_action,
    evaluate_comparison,
    finite_sample_quantile,
    fit_conformal_triage,
)


def test_decide_action_boundaries() -> None:
    t_pass, t_reject = 0.35, 0.55
    assert decide_action(0.20, t_pass, t_reject) == "PASS"
    assert decide_action(0.35, t_pass, t_reject) == "UNCERTAIN_REVIEW"
    assert decide_action(0.45, t_pass, t_reject) == "UNCERTAIN_REVIEW"
    assert decide_action(0.55, t_pass, t_reject) == "UNCERTAIN_REVIEW"
    assert decide_action(0.75, t_pass, t_reject) == "DEFECT_REJECT"

def test_finite_sample_quantile_and_fit() -> None:
    arr = np.linspace(0.0, 1.0, 100)
    q95 = finite_sample_quantile(arr, 0.95)
    assert 0.94 <= q95 <= 1.0

    good = np.linspace(0.10, 0.48, 50)
    defect = np.linspace(0.40, 0.92, 60)
    res = fit_conformal_triage(good, defect, alpha_escape=0.02, alpha_overkill=0.05)
    assert res["t_pass"] <= res["t_reject"]
    assert res["gray_zone_width"] >= 0.0

def test_real_screw_conformal_zero_overkill() -> None:
    for stem in ("patchcore_mvtec_ad_screw", "efficientad_mvtec_ad_screw"):
        p = Path("artifacts/calibration") / f"conformal_{stem}.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        val = data["validation"]
        assert data["t_pass"] <= data["t_reject"]
        assert val["conformal_overkill_rate"] == 0.0
        assert val["conformal_overkill_rate"] < val["binary_2way_overkill_rate"]

