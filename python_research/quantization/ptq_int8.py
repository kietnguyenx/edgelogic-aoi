import argparse
import time
from pathlib import Path

import cv2
import nncf
import numpy as np
import openvino as ov

def preprocess_image(img_path: Path, image_size: int = 256) -> np.ndarray:
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)


def build_calibration_dataset(
    dataset: str,
    category: str,
    subset_size: int = 100,
    image_size: int = 256,
    seed: int = 42,
) -> tuple[nncf.Dataset, int]:
    """Lấy tối đa subset_size ảnh bình thường (train/good) để NNCF hiệu chuẩn dải INT8."""
    good_dir = Path("data/raw") / dataset / category / "train" / "good"
    files = sorted(good_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy ảnh calibration trong {good_dir}")

    if len(files) > subset_size:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(files), size=subset_size, replace=False).tolist())
        files = [files[i] for i in idx]

    def transform_fn(path: Path) -> np.ndarray:
        return preprocess_image(path, image_size=image_size)

    return nncf.Dataset(files, transform_fn), len(files)

def run_ptq_int8(
    fp32_xml: Path,
    out_xml: Path,
    calib_ds: nncf.Dataset,
    subset_size: int,
) -> Path:
    core = ov.Core()
    ov_model = core.read_model(str(fp32_xml))

    print(f">>> Đang nén Uniform PTQ INT8 bằng Intel NNCF ({subset_size} ảnh calibration)...")
    quantized_model = nncf.quantize(
        model=ov_model,
        calibration_dataset=calib_ds,
        preset=nncf.QuantizationPreset.PERFORMANCE,
        target_device=nncf.TargetDevice.CPU,
        subset_size=subset_size,
        fast_bias_correction=True,
    )

    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized_model, str(out_xml))
    print(f"[OK] Đã lưu model INT8: {out_xml}")
    return out_xml

def get_ir_size_mb(xml_path: Path) -> float:
    bin_path = xml_path.with_suffix(".bin")
    total = xml_path.stat().st_size + (bin_path.stat().st_size if bin_path.exists() else 0)
    return round(total / (1024 * 1024), 2)


def benchmark_cpu(xml_path: Path, image_size: int = 256, warmup: int = 15, runs: int = 60) -> dict:
    core = ov.Core()
    compiled = core.compile_model(
        core.read_model(str(xml_path)),
        "CPU",
        {"PERFORMANCE_HINT": "LATENCY"},
    )
    req = compiled.create_infer_request()
    in_port = compiled.input(0)
    dummy = np.random.rand(1, 3, image_size, image_size).astype(np.float32)

    for _ in range(warmup):
        req.infer({in_port: dummy})

    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        req.infer({in_port: dummy})
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    return {
        "size_mb": get_ir_size_mb(xml_path),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "fps": round(1000.0 / float(np.mean(arr)), 1),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Uniform PTQ INT8 baseline với Intel NNCF")
    parser.add_argument("--model", choices=["patchcore", "efficientad"], required=True)
    parser.add_argument("--dataset", choices=["mvtec_ad", "mvtec_loco"], required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    stem = f"{args.model}_{args.dataset}_{args.category}"
    fp32_xml = Path("artifacts/openvino_fp32") / f"{stem}.xml"
    int8_xml = Path("artifacts/openvino_ptq_int8") / f"{stem}.xml"

    calib_ds, n_calib = build_calibration_dataset(
        args.dataset, args.category, subset_size=args.subset_size, image_size=args.image_size
    )
    run_ptq_int8(fp32_xml, int8_xml, calib_ds, subset_size=n_calib)

    print("\n>>> Đang đo tốc độ CPU (FP32 vs Uniform PTQ INT8)...")
    b_fp32 = benchmark_cpu(fp32_xml, image_size=args.image_size)
    b_int8 = benchmark_cpu(int8_xml, image_size=args.image_size)

    speedup = b_fp32["p50_ms"] / max(b_int8["p50_ms"], 1e-6)
    shrink = b_fp32["size_mb"] / max(b_int8["size_mb"], 1e-6)

    print("\n" + "=" * 68)
    print(f"SO SÁNH TỐC ĐỘ & DUNG LƯỢNG — {stem} (Intel CPU)")
    print("=" * 68)
    print(f"  FP32 Baseline    : {b_fp32['size_mb']:>6.2f} MB | P50={b_fp32['p50_ms']:>6.2f} ms | P99={b_fp32['p99_ms']:>6.2f} ms | {b_fp32['fps']:>5.1f} FPS")
    print(f"  Uniform PTQ INT8 : {b_int8['size_mb']:>6.2f} MB | P50={b_int8['p50_ms']:>6.2f} ms | P99={b_int8['p99_ms']:>6.2f} ms | {b_int8['fps']:>5.1f} FPS")
    print(f"  >>> Kết quả      : Nhẹ hơn {shrink:.2f}x | Nhanh hơn {speedup:.2f}x")
    print("=" * 68)


if __name__ == "__main__":
    main()