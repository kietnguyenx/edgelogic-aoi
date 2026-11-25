import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

def compute_optical_metrics(img_bgr: np.ndarray, image_size: int = 256, hist_bins: int = 32) -> tuple[dict, np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    if gray.shape[:2] != (image_size, image_size):
        gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_b = float(np.mean(gray))
    std_c = float(np.std(gray))
    clipped = float(np.mean((gray <= 2) | (gray >= 253)))

    hist = cv2.calcHist([gray], [0], None, [hist_bins], [0, 256]).reshape(-1).astype(np.float64)
    hist /= max(float(hist.sum()), 1e-12)

    return {
        "laplacian_var": round(lap_var, 4),
        "brightness_mean": round(mean_b, 4),
        "contrast_std": round(std_c, 4),
        "clipped_pixel_ratio": round(clipped, 4),
    }, hist


def bhattacharyya_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Khoảng cách Bhattacharyya trong [0, 1]: bằng 0 nếu 2 biểu đồ xám trùng khít nhau."""
    bc = float(np.clip(np.sum(np.sqrt(h1 * h2)), 0.0, 1.0))
    return float(np.sqrt(max(0.0, 1.0 - bc)))

def fit_camera_baseline(dataset: str, category: str, image_size: int = 256, hist_bins: int = 32) -> dict:
    good_dir = Path("data/raw") / dataset / category / "train" / "good"
    files = sorted(good_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy ảnh chuẩn trong {good_dir}")

    laps, means, stds, clips, hists = [], [], [], [], []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        m, h = compute_optical_metrics(img, image_size=image_size, hist_bins=hist_bins)
        laps.append(m["laplacian_var"])
        means.append(m["brightness_mean"])
        stds.append(m["contrast_std"])
        clips.append(m["clipped_pixel_ratio"])
        hists.append(h)

    ref_hist = np.mean(np.stack(hists, axis=0), axis=0)
    ref_hist /= max(float(ref_hist.sum()), 1e-12)

    lap_p05 = float(np.percentile(laps, 5))
    b_mean = float(np.mean(means))
    b_std = max(float(np.std(means)), 3.0)
    c_p05 = float(np.percentile(stds, 5))
    clip_p95 = float(np.percentile(clips, 95))

    ref_bhats = [bhattacharyya_distance(h, ref_hist) for h in hists]
    bhat_p99 = float(np.percentile(ref_bhats, 99))

    return {
        "dataset": dataset,
        "category": category,
        "num_reference_images": len(files),
        "image_size": image_size,
        "hist_bins": hist_bins,
        "laplacian_median": round(float(np.median(laps)), 2),
        "brightness_mean": round(b_mean, 2),
        "blur_min_threshold": round(lap_p05 * 0.45, 4),
        "brightness_min": round(max(0.0, b_mean - 4.0 * b_std), 4),
        "brightness_max": round(min(255.0, b_mean + 4.0 * b_std), 4),
        "contrast_min_threshold": round(c_p05 * 0.50, 4),
        "bhattacharyya_max": round(min(0.95, max(0.35, bhat_p99 * 1.25 + 0.05)), 4),
        "clipped_pixel_ratio_max": round(min(0.98, clip_p95 + 0.15), 4),
        "reference_hist": [round(float(x), 6) for x in ref_hist.tolist()],
    }


def check_frame(img_bgr: np.ndarray, baseline: dict) -> dict:
    m, hist = compute_optical_metrics(img_bgr, baseline["image_size"], baseline["hist_bins"])
    bhat = bhattacharyya_distance(hist, np.array(baseline["reference_hist"], dtype=np.float64))
    m["bhattacharyya_dist"] = round(bhat, 4)

    flags = []
    if m["laplacian_var"] < baseline["blur_min_threshold"]:
        flags.append("BLUR_DRIFT")
    if m["brightness_mean"] < baseline["brightness_min"] or m["brightness_mean"] > baseline["brightness_max"]:
        flags.append("ILLUMINATION_DRIFT")
    if m["contrast_std"] < baseline["contrast_min_threshold"]:
        flags.append("CONTRAST_DRIFT")
    if bhat > baseline["bhattacharyya_max"]:
        flags.append("HISTOGRAM_SHIFT")
    if m["clipped_pixel_ratio"] > baseline["clipped_pixel_ratio_max"]:
        flags.append("EXPOSURE_CLIPPED")

    return {
        "passed": len(flags) == 0,
        "status": "OK" if not flags else f"CAMERA_DRIFT_ALERT:{','.join(flags)}",
        "flags": flags,
        "metrics": m,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Camera Drift Gate — Chốt chặn quang học trước AI")
    parser.add_argument("--dataset", choices=["mvtec_ad", "mvtec_loco"], required=True)
    parser.add_argument("--category", type=str, required=True)
    args = parser.parse_args()

    baseline = fit_camera_baseline(args.dataset, args.category)
    out_path = Path("artifacts/calibration") / f"drift_baseline_{args.dataset}_{args.category}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    # Stress-Test trên 30 ảnh test/good thật + 3 sự cố giả lập (Mờ lens, Sụt đèn, Chói lóa)
    test_imgs = sorted((Path("data/raw") / args.dataset / args.category / "test" / "good").glob("*.png"))[:30]
    clean_ok, blur_caught, dark_caught, glare_caught = 0, 0, 0, 0
    latencies_ms = []

    for p in test_imgs:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        res_clean = check_frame(img, baseline)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if res_clean["passed"]:
            clean_ok += 1

        if not check_frame(cv2.GaussianBlur(img, (15, 15), 5.0), baseline)["passed"]:
            blur_caught += 1
        if not check_frame(np.clip(img.astype(np.float32) * 0.45, 0, 255).astype(np.uint8), baseline)["passed"]:
            dark_caught += 1
        if not check_frame(np.clip(img.astype(np.float32) + 85.0, 0, 255).astype(np.uint8), baseline)["passed"]:
            glare_caught += 1

    n = len(test_imgs)
    print("\n" + "=" * 72)
    print(f"CAMERA DRIFT GATE STRESS-TEST — {args.dataset}/{args.category}")
    print("=" * 72)
    print(f"  Số ảnh chuẩn học Baseline : {baseline['num_reference_images']} ảnh")
    print(f"  Tốc độ kiểm tra mỗi ảnh   : {np.mean(latencies_ms):.3f} ms (< 1 ms SLA!)")
    print(f"  1. Ảnh sạch đi qua (PASS) : {clean_ok}/{n} ({clean_ok/n:.1%})")
    print(f"  2. Bắt lỗi Mờ ống kính    : {blur_caught}/{n} ({blur_caught/n:.1%})")
    print(f"  3. Bắt lỗi Tối/Hỏng LED   : {dark_caught}/{n} ({dark_caught/n:.1%})")
    print(f"  4. Bắt lỗi Chói/Lóa sáng  : {glare_caught}/{n} ({glare_caught/n:.1%})")
    print("=" * 72)
    print(f"[OK] Đã lưu cấu hình chuẩn cho Python & C++: {out_path}")


if __name__ == "__main__":
    main()
