import argparse
import json
from pathlib import Path

import numpy as np

def finite_sample_quantile(values: np.ndarray, q_level: float) -> float:
    """Công thức phân vị Split-Conformal chuẩn: ceil((n + 1) * q) / n."""
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = len(arr)
    rank = int(np.ceil((n + 1) * q_level))
    idx = int(np.clip(rank - 1, 0, n - 1))
    return float(arr[idx])


def fit_conformal_triage(
    good_scores: np.ndarray,
    defect_scores: np.ndarray,
    alpha_escape: float = 0.02,
    alpha_overkill: float = 0.05,
) -> dict:
    """Tìm ngưỡng 3 ngã đảm bảo: Tỷ lệ lọt lỗi <= alpha_escape & Tỷ lệ vứt oan <= alpha_overkill."""
    # Ranh giới dưới t_pass: lấy từ phân phối điểm lỗi để chặn lọt lỗi <= alpha_escape
    t_pass = -finite_sample_quantile(-defect_scores, 1.0 - alpha_escape)
    # Ranh giới trên t_reject: lấy từ phân phối điểm tốt để chặn vứt oan <= alpha_overkill
    t_reject = finite_sample_quantile(good_scores, 1.0 - alpha_overkill)

    if t_pass > t_reject:
        mid = 0.5 * (t_pass + t_reject)
        t_pass, t_reject = mid, mid

    return {
        "alpha_escape": alpha_escape,
        "alpha_overkill": alpha_overkill,
        "t_pass": round(float(t_pass), 6),
        "t_reject": round(float(t_reject), 6),
        "gray_zone_width": round(float(max(0.0, t_reject - t_pass)), 6),
        "n_calib_good": int(len(good_scores)),
        "n_calib_defect": int(len(defect_scores)),
    }

def decide_action(score: float, t_pass: float, t_reject: float) -> str:
    if score < t_pass:
        return "PASS"
    if score > t_reject:
        return "DEFECT_REJECT"
    return "UNCERTAIN_REVIEW"


def evaluate_comparison(y_true: np.ndarray, scores: np.ndarray, t_pass: float, t_reject: float) -> dict:
    n_total = len(y_true)
    n_good = int(np.sum(y_true == 0))
    n_def = int(np.sum(y_true == 1))

    # Cách cũ (Binary 2 ngã): muốn lọt lỗi thấp thì buộc phải vứt bỏ mọi mẫu >= t_pass
    binary_overkill = int(np.sum((y_true == 0) & (scores >= t_pass)))
    binary_escaped = int(np.sum((y_true == 1) & (scores < t_pass)))

    # Cách mới (Conformal 3 ngã): vùng [t_pass, t_reject] đưa sang REVIEW thay vì vứt oan!
    actions = [decide_action(float(s), t_pass, t_reject) for s in scores]
    escaped = sum(1 for y, a in zip(y_true, actions) if y == 1 and a == "PASS")
    overkill = sum(1 for y, a in zip(y_true, actions) if y == 0 and a == "DEFECT_REJECT")
    review = sum(1 for a in actions if a == "UNCERTAIN_REVIEW")

    return {
        "binary_2way_escape_rate": round(binary_escaped / max(n_def, 1), 4),
        "binary_2way_overkill_rate": round(binary_overkill / max(n_good, 1), 4),
        "conformal_escape_rate": round(escaped / max(n_def, 1), 4),
        "escaped_count": escaped,
        "n_defect": n_def,
        "conformal_overkill_rate": round(overkill / max(n_good, 1), 4),
        "overkill_count": overkill,
        "n_good": n_good,
        "uncertain_review_rate": round(review / max(n_total, 1), 4),
        "automation_rate": round(1.0 - review / max(n_total, 1), 4),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Conformal Prediction 3-Way Triage")
    parser.add_argument(
        "--scores-json",
        type=str,
        default="artifacts/calibration/scores_mixed_patchcore_mvtec_ad_screw.json",
    )
    parser.add_argument("--alpha-escape", type=float, default=0.02, help="Ngưỡng lọt lỗi mục tiêu (vd 0.01 - 0.02)")
    parser.add_argument("--alpha-overkill", type=float, default=0.05, help="Ngưỡng vứt oan tự động tối đa (5%)")
    args = parser.parse_args()

    data = json.loads(Path(args.scores_json).read_text(encoding="utf-8"))
    records = data["records"]

    rng = np.random.default_rng(42)
    good_all = np.array([r["pred_score"] for r in records if r["gt_label"] == 0], dtype=np.float64)
    def_all = np.array([r["pred_score"] for r in records if r["gt_label"] == 1], dtype=np.float64)
    rng.shuffle(good_all)
    rng.shuffle(def_all)

    # Chia 50% Calibration / 50% Held-out Validation (không chạm nhau!)
    g_mid, d_mid = len(good_all) // 2, len(def_all) // 2
    thr = fit_conformal_triage(
        good_all[:g_mid], def_all[:d_mid],
        alpha_escape=args.alpha_escape,
        alpha_overkill=args.alpha_overkill,
    )

    val_y = np.concatenate([np.zeros(len(good_all[g_mid:]), dtype=int), np.ones(len(def_all[d_mid:]), dtype=int)])
    val_s = np.concatenate([good_all[g_mid:], def_all[d_mid:]])
    comp = evaluate_comparison(val_y, val_s, thr["t_pass"], thr["t_reject"])

    stem = f"{data['model']}_{data['dataset']}_{data['category']}"
    out_path = Path("artifacts/calibration") / f"conformal_{stem}.json"
    payload = {**thr, "model": data["model"], "dataset": data["dataset"], "category": data["category"], "validation": comp}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print(f"CONFORMAL 3-WAY TRIAGE — {stem.upper()} (Held-out Validation)")
    print("=" * 76)
    print(f"  Ngưỡng Conformal        : PASS < {thr['t_pass']:.4f} <= REVIEW <= {thr['t_reject']:.4f} < REJECT")
    print(f"  [Cách cũ 1 ngưỡng cứng] : Lọt lỗi = {comp['binary_2way_escape_rate']:>6.2%} | Vứt oan (Overkill) = {comp['binary_2way_overkill_rate']:>6.2%}")
    print(f"  [Conformal 3-Way Triage]: Lọt lỗi = {comp['conformal_escape_rate']:>6.2%} ({comp['escaped_count']}/{comp['n_defect']}) | Vứt oan (Overkill) = {comp['conformal_overkill_rate']:>6.2%} ({comp['overkill_count']}/{comp['n_good']})")
    print(f"  Tỷ lệ chuyển khay REVIEW: {comp['uncertain_review_rate']:.2%} (Tự động hóa {comp['automation_rate']:.2%} dây chuyền!)")
    print("=" * 76)
    print(f"[OK] Đã lưu cấu hình chuẩn cho Python & C++: {out_path}")


if __name__ == "__main__":
    main()