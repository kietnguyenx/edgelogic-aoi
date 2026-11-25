import argparse
import json
import os
import platform
import time
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
if platform.system() == "Windows":
    os.environ.setdefault("HF_HOME", "D:/Download/Workspace/Project/EdgeLogic-AOI/.cache/hf")
    os.environ.setdefault("TORCH_HOME", "D:/Download/Workspace/Project/EdgeLogic-AOI/.cache/torch")

import cv2
import nncf
import numpy as np
import openvino as ov
import openvino.opset13 as ops
import torch

if not hasattr(torch.compiler, "is_exporting"):
    torch.compiler.is_exporting = torch.onnx.is_in_onnx_export  # type: ignore[attr-defined]

from anomalib.models.components.sampling import KCenterGreedy
from sklearn.metrics import f1_score, roc_auc_score


def preprocess_image(img_path: Path, image_size: int = 256) -> np.ndarray:
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)


def _find_patch_feature_op(ov_model: ov.Model) -> ov.Node:
    """Locate the 2D patch-embedding Reshape op [N_patches, C_dim] across any backbone/resolution."""
    for op in ov_model.get_ordered_ops():
        if op.get_type_name() != "Reshape":
            continue
        pshape = op.output(0).get_partial_shape()
        if pshape.rank.is_static and pshape.rank.get_length() == 2 and pshape.is_static:
            n_patches, c_dim = int(pshape[0].get_length()), int(pshape[1].get_length())
            if n_patches > 16 and c_dim >= 64:
                return op
    raise RuntimeError("Could not locate 2D patch-embedding Reshape node in OpenVINO graph.")


def quantize_patchcore_fused_qamr(
    fp32_xml: Path,
    out_xml: Path,
    dataset: str,
    category: str,
    coreset_ratio: float = 0.02,
    subset_size: int = 100,
    image_size: int = 256,
) -> Path:
    """Đòn bẩy Max-Power cho PatchCore: INT8 Backbone + QAMR Memory Bank + Fused OpenVINO 9-NN Head."""
    core = ov.Core()
    good_files = sorted((Path("data/raw") / dataset / category / "train" / "good").glob("*.png"))
    calib_ds = nncf.Dataset(good_files[:subset_size], lambda p: preprocess_image(p, image_size=image_size))

    # 1. Nén Backbone bằng NNCF MIXED INT8 (bảo vệ cụm MatMul)
    print(">>> [1/3] Nén Backbone sang INT8 bằng Intel NNCF (MIXED preset)...")
    q_full = nncf.quantize(
        model=core.read_model(str(fp32_xml)),
        calibration_dataset=calib_ds,
        preset=nncf.QuantizationPreset.MIXED,
        target_device=nncf.TargetDevice.CPU,
        subset_size=min(subset_size, len(good_files)),
        fast_bias_correction=True,
        ignored_scope=nncf.IgnoredScope(types=["MatMul"]),
    )

    # 2. Cắt Backbone tại nút đặc trưng Reshape & trích xuất lại Coreset Memory Bank khớp hệ quy chiếu INT8 (QAMR)
    reshape_op = _find_patch_feature_op(q_full)
    bb_int8 = ov.Model([reshape_op.output(0)], q_full.get_parameters(), "backbone_int8")
    comp_bb = core.compile_model(bb_int8, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
    req_bb = comp_bb.create_infer_request()

    print(f">>> [2/3] QAMR: Trích xuất lại đặc trưng INT8 trên {len(good_files)} ảnh train/good...")
    embs = np.concatenate(
        [np.asarray(req_bb.infer({comp_bb.input(0): preprocess_image(p, image_size)})[comp_bb.output(0)]) for p in good_files],
        axis=0,
    )
    emb_t = torch.from_numpy(embs).cuda() if torch.cuda.is_available() else torch.from_numpy(embs)
    torch.manual_seed(42)
    coreset_mb = KCenterGreedy(embedding=emb_t, sampling_ratio=coreset_ratio).sample_coreset().cpu().numpy().astype(np.float32)

    # 3. Đóng thẳng Fused 9-NN Distance Head siêu gọn bằng toán tử gốc OpenVINO opset13
    print(f">>> [3/3] Đóng gói Fused 9-NN OpenVINO Head (Memory Bank {coreset_mb.shape})...")
    feat = reshape_op.output(0)
    num_patches = int(feat.get_partial_shape()[0].get_length())
    grid_hw = int(round(num_patches ** 0.5))

    mb_const = ops.constant(coreset_mb, dtype=np.float32)
    mb_sq = ops.constant(np.sum(coreset_mb ** 2, axis=1, dtype=np.float32).reshape(1, -1), dtype=np.float32)
    feat_sq = ops.reduce_sum(ops.multiply(feat, feat), ops.constant([1], dtype=np.int64), keep_dims=True)
    dot = ops.matmul(feat, mb_const, transpose_a=False, transpose_b=True)
    dist_sq = ops.maximum(ops.subtract(ops.add(feat_sq, mb_sq), ops.multiply(dot, ops.constant(2.0, dtype=np.float32))), ops.constant(0.0, dtype=np.float32))
    patch_dist = ops.sqrt(ops.reduce_min(dist_sq, ops.constant([1], dtype=np.int64), keep_dims=False))

    _, max_p_idx = ops.topk(patch_dist, ops.constant(1, dtype=np.int64), axis=0, mode="max", sort="none").outputs()
    s_star_emb = ops.gather(feat, max_p_idx, ops.constant(0, dtype=np.int64))
    _, min_b_all = ops.topk(dist_sq, ops.constant(1, dtype=np.int64), axis=1, mode="min", sort="none").outputs()
    m_star_idx = ops.reshape(ops.gather(min_b_all, max_p_idx, ops.constant(0, dtype=np.int64)), ops.constant([1], dtype=np.int64), False)
    m_star_emb = ops.gather(mb_const, m_star_idx, ops.constant(0, dtype=np.int64))

    m_star_sq = ops.reduce_sum(ops.multiply(m_star_emb, m_star_emb), ops.constant([1], dtype=np.int64), keep_dims=True)
    dm_sq = ops.maximum(ops.subtract(ops.add(m_star_sq, mb_sq), ops.multiply(ops.matmul(m_star_emb, mb_const, False, True), ops.constant(2.0, dtype=np.float32))), ops.constant(0.0, dtype=np.float32))
    _, nn9_idx = ops.topk(dm_sq, ops.constant(9, dtype=np.int64), axis=1, mode="min", sort="none").outputs()
    nn9_embs = ops.gather(mb_const, ops.reshape(nn9_idx, ops.constant([9], dtype=np.int64), False), ops.constant(0, dtype=np.int64))

    diff_nn = ops.subtract(nn9_embs, s_star_emb)
    d_s_nn = ops.sqrt(ops.maximum(ops.reduce_sum(ops.multiply(diff_nn, diff_nn), ops.constant([1], dtype=np.int64), False), ops.constant(1e-12, dtype=np.float32)))
    max_s = ops.gather(patch_dist, max_p_idx, ops.constant(0, dtype=np.int64))
    w = ops.subtract(ops.constant(1.0, dtype=np.float32), ops.divide(ops.constant(1.0, dtype=np.float32), ops.reduce_sum(ops.exp(ops.subtract(d_s_nn, max_s)), ops.constant([0], dtype=np.int64), keep_dims=True)))
    pred_score = ops.multiply(w, max_s)

    grid_2d = ops.reshape(patch_dist, ops.constant([1, 1, grid_hw, grid_hw], dtype=np.int64), special_zero=False)
    amap = ops.interpolate(grid_2d, [image_size, image_size], "linear_onnx", "sizes", axes=[2, 3])

    score_res = ops.result(pred_score, name="pred_score_res")
    amap_res = ops.result(amap, name="anomaly_map_res")
    score_res.get_output_tensor(0).set_names({"pred_score"})
    amap_res.get_output_tensor(0).set_names({"anomaly_map"})

    fused_model = ov.Model([score_res, amap_res], q_full.get_parameters(), "edgelogic_patchcore_qamr")
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(fused_model, str(out_xml), compress_to_fp16=False)
    return out_xml


def quantize_efficientad_mixed(
    fp32_xml: Path,
    out_xml: Path,
    sensitive_layers: list[str],
    dataset: str,
    category: str,
    subset_size: int = 100,
    image_size: int = 256,
) -> Path:
    """Mixed-Precision INT8 cho EfficientAD (thuần mạng Conv, không dùng Memory Bank)."""
    files = sorted((Path("data/raw") / dataset / category / "train" / "good").glob("*.png"))[:subset_size]
    calib_ds = nncf.Dataset(files, lambda p: preprocess_image(p, image_size=image_size))
    q_model = nncf.quantize(
        model=ov.Core().read_model(str(fp32_xml)),
        calibration_dataset=calib_ds,
        preset=nncf.QuantizationPreset.MIXED,
        target_device=nncf.TargetDevice.CPU,
        subset_size=len(files),
        fast_bias_correction=True,
        ignored_scope=nncf.IgnoredScope(names=sensitive_layers[:3]),
    )
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(q_model, str(out_xml), compress_to_fp16=False)
    return out_xml


def evaluate_and_benchmark(xml_path: Path, dataset: str, category: str, image_size: int = 256) -> tuple[dict, list[dict]]:
    core = ov.Core()
    ov_m = core.read_model(str(xml_path))
    compiled = core.compile_model(ov_m, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    req = compiled.create_infer_request()
    in_port, out_score = compiled.input(0), compiled.output(0)

    test_root = Path("data/raw") / dataset / category / "test"
    records = []
    for sub in sorted([d for d in test_root.iterdir() if d.is_dir()]):
        gt = 0 if sub.name == "good" else 1
        for img_p in sorted(sub.glob("*.png")):
            score = float(req.infer({in_port: preprocess_image(img_p, image_size)})[out_score].reshape(-1)[0])
            records.append({"image_rel_path": f"test/{sub.name}/{img_p.name}", "defect_type": sub.name, "gt_label": gt, "pred_score": score})

    y_true = np.array([r["gt_label"] for r in records])
    scores = np.array([r["pred_score"] for r in records])
    overall_auroc = round(float(roc_auc_score(y_true, scores)), 4)

    best_f1, best_thr = 0.0, float(scores[0])
    for thr in np.unique(scores):
        f1 = float(f1_score(y_true, (scores >= thr).astype(int), zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    micro_auroc, micro_rec = None, None
    stats_path = Path("artifacts/reports/defect_area_stats.json")
    if stats_path.exists():
        area_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        micro_list = area_stats.get("datasets", {}).get(dataset, {}).get(category, {}).get("samples", {}).get("micro", [])
        score_map = {r["image_rel_path"]: r["pred_score"] for r in records}
        good_scores = scores[y_true == 0]
        m_scores = np.array([score_map[m["image_rel_path"]] for m in micro_list if m["image_rel_path"] in score_map])
        if len(m_scores) > 0:
            micro_auroc = round(float(roc_auc_score(np.concatenate([np.zeros(len(good_scores)), np.ones(len(m_scores))]), np.concatenate([good_scores, m_scores]))), 4)
            micro_rec = round(float(np.mean(m_scores >= best_thr)), 4)

    # Single-stream LATENCY + Multi-stream AsyncInferQueue THROUGHPUT
    dummy = np.random.rand(1, 3, image_size, image_size).astype(np.float32)
    for _ in range(15):
        req.infer({in_port: dummy})
    times_ms = []
    for _ in range(60):
        t0 = time.perf_counter()
        req.infer({in_port: dummy})
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    comp_tp = core.compile_model(ov_m, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
    ireqs = ov.AsyncInferQueue(comp_tp)
    for _ in range(24):
        ireqs.start_async({0: dummy})
    ireqs.wait_all()
    t_tp = time.perf_counter()
    for _ in range(160):
        ireqs.start_async({0: dummy})
    ireqs.wait_all()
    async_fps = round(160.0 / (time.perf_counter() - t_tp), 1)

    bin_p = xml_path.with_suffix(".bin")
    size_mb = round((xml_path.stat().st_size + (bin_p.stat().st_size if bin_p.exists() else 0)) / (1024 * 1024), 2)

    return {
        "auroc": overall_auroc,
        "opt_f1": round(best_f1, 4),
        "micro_auroc": micro_auroc,
        "micro_recall_at_f1": micro_rec,
        "p50_ms": round(float(np.percentile(times_ms, 50)), 2),
        "fps_single": round(1000.0 / float(np.mean(times_ms)), 1),
        "fps_async": async_fps,
        "size_mb": size_mb,
    }, records


def main() -> None:
    parser = argparse.ArgumentParser(description="EdgeLogic-AOI Max-Power NNCF + QAMR Benchmark")
    parser.add_argument("--model", choices=["patchcore", "efficientad"], required=True)
    parser.add_argument("--dataset", choices=["mvtec_ad", "mvtec_loco"], required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--coreset", type=float, default=0.02, help="Tỷ lệ coreset cho QAMR (0.02 cân bằng đỉnh giữa AUROC & FPS)")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    stem = f"{args.model}_{args.dataset}_{args.category}"
    fp32_xml = Path("artifacts/openvino_fp32") / f"{stem}.xml"
    ptq_xml = Path("artifacts/openvino_ptq_int8") / f"{stem}.xml"
    mixed_xml = Path("artifacts/openvino_mixed_int8") / f"{stem}.xml"

    if args.model == "patchcore":
        quantize_patchcore_fused_qamr(
            fp32_xml, mixed_xml, args.dataset, args.category, coreset_ratio=args.coreset, image_size=args.image_size
        )
    else:
        sens_json = Path("artifacts/reports") / f"sensitivity_{stem}.json"
        sens_layers = json.loads(sens_json.read_text(encoding="utf-8"))["recommended_fp16_layers"] if sens_json.exists() else []
        quantize_efficientad_mixed(fp32_xml, mixed_xml, sens_layers, args.dataset, args.category, image_size=args.image_size)

    variants = [
        ("FP32 Baseline", fp32_xml),
        ("Uniform PTQ INT8", ptq_xml),
        ("NNCF + Fused QAMR", mixed_xml),
    ]
    summary = {}
    for label, xml_p in variants:
        print(f"\n>>> Đang đánh giá & vắt tốc độ [{label}]...")
        m, recs = evaluate_and_benchmark(xml_p, args.dataset, args.category, image_size=args.image_size)
        summary[label] = m
        if label == "NNCF + Fused QAMR":
            cal_path = Path("artifacts/calibration") / f"scores_mixed_{stem}.json"
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            cal_path.write_text(json.dumps({
                "model": args.model, "dataset": args.dataset, "category": args.category, "records": recs
            }, indent=2), encoding="utf-8")

    rep_path = Path("artifacts/reports") / f"quant_benchmark_{stem}.json"
    rep_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 106)
    print(f"BẢNG TỐI ĐA CÔNG SUẤT — {stem.upper()} (Intel CPU: Single-Stream vs Async Multi-Stream)")
    print("=" * 106)
    print(f"| {'Phiên bản':<20} | {'AUROC':>6} | {'Opt F1':>6} | {'Micro AUROC':>11} | {'Micro Rec':>9} | {'P50(ms)':>7} | {'FPS(1-luồng)':>12} | {'FPS(Async)':>10} | {'Size(MB)':>8} |")
    print(f"|{'-'*22}|{'-'*8}|{'-'*8}|{'-'*13}|{'-'*11}|{'-'*9}|{'-'*14}|{'-'*12}|{'-'*10}|")
    for label, m in summary.items():
        ma = f"{m['micro_auroc']:.4f}" if m["micro_auroc"] is not None else "  N/A "
        mr = f"{m['micro_recall_at_f1']:.2%}" if m["micro_recall_at_f1"] is not None else "   N/A "
        print(f"| {label:<20} | {m['auroc']:>6.4f} | {m['opt_f1']:>6.4f} | {ma:>11} | {mr:>9} | {m['p50_ms']:>7.2f} | {m['fps_single']:>12.1f} | {m['fps_async']:>10.1f} | {m['size_mb']:>8.2f} |")
    print("=" * 106)
    print(f"[OK] Đã lưu báo cáo: {rep_path}")


if __name__ == "__main__":
    main()