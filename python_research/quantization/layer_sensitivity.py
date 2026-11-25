import argparse
import json
import time
from pathlib import Path

import cv2
import nncf
import numpy as np
import openvino as ov

def preprocess_image(img_path: Path, image_size: int = 256) -> np.ndarray:
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)


def inject_micro_defect(img_nchw: np.ndarray, seed: int) -> np.ndarray:
    """Cấy 1 vết xước mảnh 1px hoặc chấm nhỏ 2-3px (<0.2% ảnh) vào vùng giữa ảnh tốt."""
    rng = np.random.default_rng(seed)
    out = img_nchw.copy()
    hwc = np.transpose(out[0], (1, 2, 0)).copy()
    h, w, _ = hwc.shape
    cx, cy = int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4))
    color = (0.08, 0.08, 0.08) if float(hwc[cy, cx].mean()) > 0.4 else (0.92, 0.92, 0.92)

    if seed % 2 == 0:
        angle = float(rng.uniform(0, np.pi))
        length = int(rng.integers(6, 15))
        x2 = int(np.clip(cx + round(np.cos(angle) * length), 0, w - 1))
        y2 = int(np.clip(cy + round(np.sin(angle) * length), 0, h - 1))
        cv2.line(hwc, (cx, cy), (x2, y2), color, thickness=1)
    else:
        cv2.circle(hwc, (cx, cy), int(rng.integers(2, 4)), color, thickness=-1)

    out[0] = np.transpose(hwc, (2, 0, 1))
    return out


def build_probe_tensors(dataset: str, category: str, num_pairs: int = 12, image_size: int = 256) -> list[np.ndarray]:
    good_dir = Path("data/raw") / dataset / category / "train" / "good"
    files = sorted(good_dir.glob("*.png"))[:num_pairs]
    probes = []
    for i, p in enumerate(files):
        clean = preprocess_image(p, image_size=image_size)
        probes.append(clean)
        probes.append(inject_micro_defect(clean, seed=1000 + i))
    return probes

def run_probe(ov_model: ov.Model, probes: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    compiled = ov.Core().compile_model(ov_model, "CPU")
    req = compiled.create_infer_request()
    in_port, out_score, out_map = compiled.input(0), compiled.output(0), compiled.output(1)

    scores, maps = [], []
    for x in probes:
        res = req.infer({in_port: x})
        scores.append(float(res[out_score].reshape(-1)[0]))
        maps.append(np.asarray(res[out_map]).squeeze())
    return np.array(scores, dtype=np.float64), maps


def compute_distortion(
    ref_scores: np.ndarray, ref_maps: list[np.ndarray],
    cand_scores: np.ndarray, cand_maps: list[np.ndarray],
) -> float:
    """50% sai số điểm số ảnh + 50% sai số đỉnh heatmap (top 0.5% pixel nóng nhất)."""
    scale = max(float(np.std(ref_scores)), float(np.mean(np.abs(ref_scores))), 1e-6)
    rel_score_err = float(np.mean(np.abs(ref_scores - cand_scores))) / scale

    peak_errs = []
    for rm, cm in zip(ref_maps, cand_maps):
        flat_r = rm.reshape(-1)
        k = max(1, int(0.005 * flat_r.size))
        top_idx = np.argpartition(flat_r, -k)[-k:]
        peak_errs.append(float(np.mean(np.abs(flat_r[top_idx] - cm.reshape(-1)[top_idx]))))

    rel_peak_err = (float(np.mean(peak_errs)) / scale) if peak_errs else rel_score_err
    return round(0.5 * rel_score_err + 0.5 * rel_peak_err, 6)

def profile_sensitivity(
    fp32_xml: Path,
    probes: list[np.ndarray],
    top_k: int = 5,
    max_candidates: int = 24,
) -> dict:
    t0 = time.perf_counter()
    core = ov.Core()
    base_model = core.read_model(str(fp32_xml))
    calib_ds = nncf.Dataset(probes[::2], lambda x: x)  # chỉ calibrate trên các ảnh sạch

    print(f">>> [1/3] Chạy mốc chuẩn FP32 trên {len(probes)} ảnh Probe...")
    ref_scores, ref_maps = run_probe(base_model, probes)

    print(">>> [2/3] Đo độ méo gốc của Uniform INT8...")
    base_int8 = nncf.quantize(
        model=core.read_model(str(fp32_xml)),
        calibration_dataset=calib_ds,
        preset=nncf.QuantizationPreset.PERFORMANCE,
        target_device=nncf.TargetDevice.CPU,
        subset_size=len(probes) // 2,
        fast_bias_correction=False,
    )
    base_s, base_m = run_probe(base_int8, probes)
    base_dist = compute_distortion(ref_scores, ref_maps, base_s, base_m)
    print(f"    Uniform INT8 Baseline Distortion = {base_dist:.6f}")

    target_types = {"Convolution", "GroupConvolution", "MatMul"}
    all_ops = [
        {"name": op.get_friendly_name(), "type": op.get_type_name()}
        for op in base_model.get_ordered_ops()
        if op.get_type_name() in target_types
    ]
    candidates = all_ops[:max_candidates] if len(all_ops) <= max_candidates else (
        all_ops[:8] + all_ops[8:-8:max(1, (len(all_ops) - 16) // 8)][:8] + all_ops[-8:]
    )

    print(f">>> [3/3] Đang đo độ nhạy của {len(candidates)}/{len(all_ops)} lớp Conv/MatMul...")
    results = []
    for idx, op in enumerate(candidates, 1):
        op_name = op["name"]
        q_mod = nncf.quantize(
            model=core.read_model(str(fp32_xml)),
            calibration_dataset=calib_ds,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            target_device=nncf.TargetDevice.CPU,
            subset_size=len(probes) // 2,
            fast_bias_correction=False,
            ignored_scope=nncf.IgnoredScope(names=[op_name]),
        )
        cs, cm = run_probe(q_mod, probes)
        dist = compute_distortion(ref_scores, ref_maps, cs, cm)
        gain = round(base_dist - dist, 6)  # gain > 0 nghĩa là bảo vệ lớp này giúp giảm méo!
        results.append({"op_name": op_name, "op_type": op["type"], "distortion": dist, "gain": gain})
        print(f"    [{idx:02d}/{len(candidates):02d}] gain={gain:+.6f} | {op['type']:<11} | {op_name[-45:]}")

    results.sort(key=lambda r: r["gain"], reverse=True)
    selected = [r["op_name"] for r in results[:top_k] if r["gain"] > 0]
    if len(selected) < 2 and len(all_ops) >= 2:
        selected = [all_ops[0]["name"], all_ops[-1]["name"]]

    return {
        "fp32_xml": str(fp32_xml),
        "total_quantizable_ops": len(all_ops),
        "profiling_time_sec": round(time.perf_counter() - t0, 2),
        "uniform_int8_distortion": base_dist,
        "recommended_fp16_layers": selected,
        "ranked_layers": results,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Đo độ nhạy từng lớp cho Intel NNCF Mixed-Precision")
    parser.add_argument("--model", choices=["patchcore", "efficientad"], required=True)
    parser.add_argument("--dataset", choices=["mvtec_ad", "mvtec_loco"], required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    stem = f"{args.model}_{args.dataset}_{args.category}"
    fp32_xml = Path("artifacts/openvino_fp32") / f"{stem}.xml"
    out_json = Path("artifacts/reports") / f"sensitivity_{stem}.json"

    probes = build_probe_tensors(args.dataset, args.category, num_pairs=12, image_size=args.image_size)
    report = profile_sensitivity(fp32_xml, probes, top_k=args.top_k)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print(f"TOP-{len(report['recommended_fp16_layers'])} LỚP NHẠY NHẤT NÊN GIỮ Ở FP16 ({report['profiling_time_sec']}s):")
    for i, name in enumerate(report["recommended_fp16_layers"], 1):
        print(f"  #{i}: {name}")
    print(f"[OK] Đã lưu báo cáo: {out_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()