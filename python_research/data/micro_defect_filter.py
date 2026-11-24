"""Analyze defect mask area ratios and isolate micro-defects (<0.2% image area).

Supports both:
- MVTec AD   : ground_truth/<defect_type>/<id>_mask.png
- MVTec LOCO : ground_truth/<defect_type>/<id>/*.png (multi-region masks merged via OR)

Outputs a JSON manifest to artifacts/reports/defect_area_stats.json for downstream
quantization sensitivity & micro-defect recall evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import yaml


def load_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_mask_ratio(mask_paths: list[Path]) -> float:
    """Merge one or more grayscale mask files with logical OR and return defect pixel ratio."""
    merged_mask: np.ndarray | None = None
    for path in mask_paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask image: {path}")
        binary = mask > 0
        merged_mask = binary if merged_mask is None else np.logical_or(merged_mask, binary)

    assert merged_mask is not None
    return float(np.count_nonzero(merged_mask)) / float(merged_mask.size)


def collect_category_masks(dataset_name: str, category_dir: Path) -> list[dict]:
    """Return list of defect records: {defect_type, image_rel_path, mask_paths}."""
    gt_root = category_dir / "ground_truth"
    if not gt_root.exists():
        print(f"[WARN] Khong tim thay ground_truth tai: {gt_root}")
        return []

    records: list[dict] = []
    for defect_dir in sorted(p for p in gt_root.iterdir() if p.is_dir()):
        defect_type = defect_dir.name

        if dataset_name == "mvtec_ad":
            # MVTec AD: moi anh loi la 1 file *_mask.png
            for mask_file in sorted(defect_dir.glob("*.png")):
                img_stem = mask_file.stem.replace("_mask", "")
                img_rel = f"test/{defect_type}/{img_stem}.png"
                records.append(
                    {
                        "defect_type": defect_type,
                        "image_rel_path": img_rel,
                        "mask_paths": [mask_file],
                    }
                )
        else:
            # MVTec LOCO: moi anh loi la 1 thu muc con chua 1 hoac nhieu file mask *.png
            for sample_dir in sorted(p for p in defect_dir.iterdir() if p.is_dir()):
                mask_files = sorted(sample_dir.glob("*.png"))
                if not mask_files:
                    continue
                img_rel = f"test/{defect_type}/{sample_dir.name}.png"
                records.append(
                    {
                        "defect_type": defect_type,
                        "image_rel_path": img_rel,
                        "mask_paths": mask_files,
                    }
                )
    return records


def classify_bucket(ratio: float, micro_thr: float, small_thr: float) -> str:
    if ratio < micro_thr:
        return "micro"
    if ratio < small_thr:
        return "small"
    return "normal"


def main() -> None:
    cfg = load_config("configs/data.yaml")
    micro_thr = float(cfg["micro_defect"]["micro_threshold"])
    small_thr = float(cfg["micro_defect"]["small_threshold"])

    report: dict = {
        "thresholds": {"micro": micro_thr, "small": small_thr},
        "datasets": {},
    }

    print("=" * 92)
    print(
        f"{'Dataset':<12} | {'Category':<20} | {'Total':>5} | "
        f"{'Micro(<0.2%)':>12} | {'Small':>5} | {'Normal':>6} | {'Min%':>7} | {'Med%':>7}"
    )
    print("-" * 92)

    for ds_name, ds_cfg in cfg["datasets"].items():
        root = Path(ds_cfg["root"])
        report["datasets"][ds_name] = {}

        for category in ds_cfg["categories"]:
            cat_dir = root / category
            records = collect_category_masks(ds_name, cat_dir)
            if not records:
                continue

            buckets = {"micro": [], "small": [], "normal": []}
            ratios: list[float] = []

            for rec in records:
                ratio = compute_mask_ratio(rec["mask_paths"])
                bucket = classify_bucket(ratio, micro_thr, small_thr)
                ratios.append(ratio)
                buckets[bucket].append(
                    {
                        "defect_type": rec["defect_type"],
                        "image_rel_path": rec["image_rel_path"],
                        "area_ratio": round(ratio, 6),
                    }
                )

            ratios_arr = np.array(ratios, dtype=np.float64)
            stats = {
                "total_defects": len(records),
                "counts": {k: len(v) for k, v in buckets.items()},
                "min_ratio": float(ratios_arr.min()),
                "median_ratio": float(np.median(ratios_arr)),
                "max_ratio": float(ratios_arr.max()),
                "samples": buckets,
            }
            report["datasets"][ds_name][category] = stats

            print(
                f"{ds_name:<12} | {category:<20} | {stats['total_defects']:>5} | "
                f"{stats['counts']['micro']:>12} | {stats['counts']['small']:>5} | "
                f"{stats['counts']['normal']:>6} | {stats['min_ratio']*100:>6.3f}% | "
                f"{stats['median_ratio']*100:>6.2f}%"
            )

    print("=" * 92)
    out_path = Path("artifacts/reports/defect_area_stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Da luu bao cao phan loai loi tai: {out_path}")


if __name__ == "__main__":
    main()