"""Deep Bosch/Intel-grade evaluation for EdgeLogic-AOI checkpoints.

Breaks down raw predictions into the 4 key industrial metrics:
1. Overall Image AUROC & F1 (optimal threshold)
2. Structural AUROC vs. Logical AUROC (for MVTec LOCO)
3. Micro-Defect Recall (<0.2% mask area, using artifacts/reports/defect_area_stats.json)
4. Factory Safety Metric: Overkill Rate (FPR) at Recall >= 99% (Escape Rate <= 1%)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if platform.system() == "Windows":
    os.environ.setdefault("HF_HOME", "D:/Download/Workspace/Project/EdgeLogic-AOI/.cache/hf")
    os.environ.setdefault("TORCH_HOME", "D:/Download/Workspace/Project/EdgeLogic-AOI/.cache/torch")

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

if not hasattr(torch.compiler, "is_exporting"):
    torch.compiler.is_exporting = torch.onnx.is_in_onnx_export

from anomalib.data import MVTecAD, MVTecLOCO
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Patchcore


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_rel_path(image_path: str, category: str) -> str:
    """Convert absolute image path to canonical 'test/<defect_type>/<filename>.png'."""
    p = Path(image_path).as_posix()
    marker = f"/{category}/"
    if marker in p:
        return p.split(marker, 1)[1]
    parts = Path(image_path).parts
    if "test" in parts:
        idx = parts.index("test")
        return "/".join(parts[idx:])
    return Path(image_path).name


def compute_threshold_at_target_recall(
    y_true: np.ndarray, scores: np.ndarray, target_recall: float = 0.99
) -> dict:
    """Find highest threshold achieving Recall >= target_recall and measure Overkill Rate (FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid = np.where(tpr >= target_recall)[0]
    if len(valid) == 0:
        idx = int(np.argmax(tpr))
    else:
        # First index where TPR >= target_recall has the lowest FPR
        idx = int(valid[0])

    thr = float(thresholds[idx])
    preds = (scores >= thr).astype(int)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    actual_recall = float(np.mean(preds[pos_mask] == 1)) if np.any(pos_mask) else 0.0
    actual_fpr = float(np.mean(preds[neg_mask] == 1)) if np.any(neg_mask) else 0.0
    escape_rate = 1.0 - actual_recall

    return {
        "target_recall": target_recall,
        "threshold": round(thr, 6),
        "actual_recall": round(actual_recall, 4),
        "escape_rate": round(escape_rate, 4),
        "overkill_fpr": round(actual_fpr, 4),
    }


def compute_optimal_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Find threshold maximizing image-level F1 score."""
    thresholds = np.unique(scores)
    best_f1, best_thr = 0.0, float(thresholds[0])
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        f1 = float(f1_score(y_true, preds, zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return round(best_f1, 4), round(best_thr, 6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep industrial evaluation for EdgeLogic-AOI")
    parser.add_argument("--model", choices=["patchcore", "efficientad"], required=True)
    parser.add_argument("--dataset", choices=["mvtec_ad", "mvtec_loco"], required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Duong dan file model.ckpt")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--data-cfg", type=str, default="configs/data.yaml")
    parser.add_argument("--stats-json", type=str, default="artifacts/reports/defect_area_stats.json")
    parser.add_argument("--target-recall", type=float, default=0.99)
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_cfg)
    root = args.data_root or data_cfg["datasets"][args.dataset]["root"]
    cls = MVTecAD if args.dataset == "mvtec_ad" else MVTecLOCO
    datamodule = cls(
        root=root,
        category=args.category,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
    )

    model_cls = Patchcore if args.model == "patchcore" else EfficientAd
    model = model_cls.load_from_checkpoint(args.ckpt, weights_only=False)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    engine = Engine(accelerator=accelerator, devices=1, logger=False)

    print(f"\n>>> Running predictions for [{args.model}] on [{args.dataset}/{args.category}]...")
    predictions = engine.predict(model=model, datamodule=datamodule)

    records: list[dict] = []
    for batch in predictions:
        paths = batch.image_path
        labels = batch.gt_label.cpu().numpy().astype(int).tolist()
        scores = batch.pred_score.cpu().numpy().astype(float).tolist()
        for path, label, score in zip(paths, labels, scores):
            rel = normalize_rel_path(str(path), args.category)
            defect_type = rel.split("/")[1] if len(rel.split("/")) >= 3 else "unknown"
            records.append(
                {
                    "image_rel_path": rel,
                    "defect_type": defect_type,
                    "gt_label": int(label),
                    "pred_score": float(score),
                }
            )

    y_true = np.array([r["gt_label"] for r in records], dtype=int)
    scores = np.array([r["pred_score"] for r in records], dtype=float)

    overall_auroc = round(float(roc_auc_score(y_true, scores)), 4)
    best_f1, opt_thr = compute_optimal_f1(y_true, scores)
    factory_99 = compute_threshold_at_target_recall(y_true, scores, target_recall=args.target_recall)

    # 1. Structural vs Logical AUROC breakdown (for MVTec LOCO)
    subtype_auroc: dict[str, float] = {}
    good_scores = scores[y_true == 0]
    for defect_type in sorted({r["defect_type"] for r in records if r["gt_label"] == 1}):
        sub_scores = np.array(
            [r["pred_score"] for r in records if r["defect_type"] == defect_type], dtype=float
        )
        if len(sub_scores) > 0 and len(good_scores) > 0:
            sub_y = np.concatenate([np.zeros(len(good_scores)), np.ones(len(sub_scores))])
            sub_s = np.concatenate([good_scores, sub_scores])
            subtype_auroc[defect_type] = round(float(roc_auc_score(sub_y, sub_s)), 4)

    # 2. Micro-Defect (<0.2% area) Recall at both Optimal-F1 threshold and Factory Recall>=99% threshold
    bucket_metrics: dict[str, dict] = {}
    stats_path = Path(args.stats_json)
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            area_stats = json.load(f)
        cat_stats = area_stats.get("datasets", {}).get(args.dataset, {}).get(args.category)
        if cat_stats:
            score_by_rel = {r["image_rel_path"]: r["pred_score"] for r in records}
            for bucket_name in ["micro", "small", "normal"]:
                samples = cat_stats["samples"].get(bucket_name, [])
                matched_scores = [
                    score_by_rel[s["image_rel_path"]]
                    for s in samples
                    if s["image_rel_path"] in score_by_rel
                ]
                if matched_scores:
                    ms = np.array(matched_scores, dtype=float)
                    bucket_y = np.concatenate([np.zeros(len(good_scores)), np.ones(len(ms))])
                    bucket_s = np.concatenate([good_scores, ms])
                    bucket_metrics[bucket_name] = {
                        "count": len(ms),
                        "auroc": round(float(roc_auc_score(bucket_y, bucket_s)), 4),
                        "recall_at_opt_f1": round(float(np.mean(ms >= opt_thr)), 4),
                        "recall_at_factory_99": round(float(np.mean(ms >= factory_99["threshold"])), 4),
                    }

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "category": args.category,
        "checkpoint": str(args.ckpt),
        "total_test_images": len(records),
        "overall_image_auroc": overall_auroc,
        "optimal_image_f1": best_f1,
        "optimal_f1_threshold": opt_thr,
        "factory_safety_gate": factory_99,
        "defect_subtype_auroc": subtype_auroc,
        "area_bucket_metrics": bucket_metrics,
    }

    out_path = Path("artifacts/reports") / f"eval_{args.model}_{args.dataset}_{args.category}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"DEEP INDUSTRIAL EVALUATION — {args.model} | {args.dataset}/{args.category}")
    print("=" * 70)
    print(f"  Overall Image AUROC        : {overall_auroc:.4f}")
    print(f"  Optimal Image F1 (thr)     : {best_f1:.4f} (thr={opt_thr})")
    print(
        f"  Factory Gate (Rec>={args.target_recall:.0%})   : "
        f"Actual Recall={factory_99['actual_recall']:.4f} | "
        f"Escape={factory_99['escape_rate']:.2%} | "
        f"Overkill(FPR)={factory_99['overkill_fpr']:.2%}"
    )
    if subtype_auroc:
        print("  --- By Defect Subtype (Structural vs Logical) ---")
        for dt, val in subtype_auroc.items():
            print(f"    {dt:<24} : AUROC = {val:.4f}")
    if bucket_metrics:
        print("  --- By Defect Area Bucket (Micro <0.2% Anchor Metric) ---")
        for bname, bm in bucket_metrics.items():
            print(
                f"    {bname:<8} (n={bm['count']:>3}) : "
                f"AUROC={bm['auroc']:.4f} | Rec@F1={bm['recall_at_opt_f1']:.2%} | "
                f"Rec@99={bm['recall_at_factory_99']:.2%}"
            )
    print("=" * 70)
    print(f"[OK] Saved detailed report to: {out_path}")


if __name__ == "__main__":
    main()
