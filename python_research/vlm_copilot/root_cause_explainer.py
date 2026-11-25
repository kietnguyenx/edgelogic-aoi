import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

CATEGORY_HINTS = {
    "screw": {
        "micro": "Hairline surface scratch or micro-burr along thread pitch (<0.25% area).",
        "normal": "Fastener head deformation, chipped thread crest, or bent shaft.",
    },
    "metal_nut": {
        "micro": "Minor plating scratch or micro-chipping along hexagonal rim.",
        "normal": "Hex edge dent, heat-treatment discoloration, or inverted orientation.",
    },
    "transistor": {
        "micro": "Minor package edge chip or lead surface abrasion.",
        "normal": "Bent/missing terminal lead, placement offset, or cracked casing.",
    },
    "splicing_connectors": {
        "micro": "Minor cable insulation nick or metallic debris near clamp.",
        "normal": "Logical assembly fault: misrouted port, missing/extra wire, or wrong color sequence.",
    },
    "screw_bag": {
        "micro": "Minor washer/nut rim nick inside package.",
        "normal": "Logical kitting fault: missing/extra fastener, wrong screw length, or missing nut/washer.",
    },
}


def localize_hotspot(anomaly_map: np.ndarray) -> tuple[float, list[float], str]:
    """Isolate peak anomaly region (>= 80% max) on 256x256 map to extract BBox & area %."""
    amap = np.asarray(anomaly_map, dtype=np.float32).squeeze()
    a_min, a_max = float(amap.min()), float(amap.max())
    if a_max - a_min < 1e-6:
        return 0.0, [0.0, 0.0, 1.0, 1.0], "Full Frame"

    norm = (amap - a_min) / (a_max - a_min)
    mask = (norm >= 0.80).astype(np.uint8)
    area_pct = round(float(mask.mean() * 100.0), 4)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return area_pct, [0.0, 0.0, 1.0, 1.0], "Center"

    h, w = norm.shape[:2]
    bbox = [
        round(float(xs.min()) / w, 4),
        round(float(ys.min()) / h, 4),
        round(float(xs.max() + 1) / w, 4),
        round(float(ys.max() + 1) / h, 4),
    ]
    cx, cy = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
    v_pos = "Top" if cy < 0.38 else ("Bottom" if cy > 0.62 else "Middle")
    h_pos = "Left" if cx < 0.38 else ("Right" if cx > 0.62 else "Center")
    region = "Center" if (v_pos == "Middle" and h_pos == "Center") else f"{v_pos}-{h_pos}"
    return area_pct, bbox, region


def explain_defect(
    anomaly_map: np.ndarray,
    anomaly_score: float,
    triage_action: str,
    category: str,
) -> dict:
    area_pct, bbox, region = localize_hotspot(anomaly_map)
    scale = "Micro (<0.25%)" if area_pct < 0.25 else "Macro / Logical"
    hints = CATEGORY_HINTS.get(
        category,
        {"micro": "Localized micro-surface anomaly.", "normal": "Structural or assembly anomaly."},
    )
    cause = hints["micro"] if area_pct < 0.25 else hints["normal"]

    op_action = (
        "Trigger pneumatic eject (0x03) & log reject to MES."
        if triage_action == "DEFECT_REJECT"
        else "Divert to QA review station (0x02) & inspect highlighted ROI."
    )
    return {
        "category": category,
        "triage_action": triage_action,
        "anomaly_score": round(float(anomaly_score), 4),
        "estimated_area_pct": area_pct,
        "defect_scale": scale,
        "hotspot_region": region,
        "hotspot_bbox_norm": bbox,
        "probable_root_cause": cause,
        "recommended_action": op_action,
    }
def main() -> None:
    parser = argparse.ArgumentParser(description="On-Demand Root-Cause Copilot cho mẫu REVIEW / REJECT")
    parser.add_argument("--xml", type=str, default="artifacts/openvino_mixed_int8/patchcore_mvtec_ad_screw.xml")
    parser.add_argument("--conformal", type=str, default="artifacts/calibration/conformal_patchcore_mvtec_ad_screw.json")
    parser.add_argument("--image", type=str, default="data/raw/mvtec_ad/screw/test/scratch_head/000.png")
    parser.add_argument("--category", type=str, default="screw")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    img_rgb = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (256, 256), interpolation=cv2.INTER_AREA)
    x = np.expand_dims(np.transpose(img_rgb.astype(np.float32) / 255.0, (2, 0, 1)), axis=0)

    compiled = ov.Core().compile_model(ov.Core().read_model(args.xml), "CPU")
    res = compiled.create_infer_request().infer({compiled.input(0): x})
    score = float(res[compiled.output(0)].reshape(-1)[0])
    amap = np.asarray(res[compiled.output(1)])

    conf = json.loads(Path(args.conformal).read_text(encoding="utf-8"))
    action = "PASS" if score < conf["t_pass"] else ("DEFECT_REJECT" if score > conf["t_reject"] else "UNCERTAIN_REVIEW")

    print("\n" + "=" * 72)
    print(f"CHẨN ĐOÁN NGUYÊN NHÂN GỐC — {args.image}")
    print("=" * 72)
    if action == "PASS":
        print(f"  [FAST-PATH PASS] score={score:.4f} < {conf['t_pass']:.4f} -> Cho đi thẳng, bỏ qua Copilot!")
    else:
        report = explain_defect(amap, score, action, args.category)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=" * 72)


if __name__ == "__main__":
    main()