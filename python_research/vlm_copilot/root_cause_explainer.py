import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

CATEGORY_HINTS = {
    "screw": {
        "micro": "Vết xước mảnh ở thân/đầu vít hoặc ba-via siêu nhỏ trên bước ren (<0.25% diện tích).",
        "normal": "Biến dạng mũ vít, mẻ ren lớn hoặc cong trục vít.",
    },
    "metal_nut": {
        "micro": "Xước nhẹ bề mặt mạ hoặc mẻ viền cạnh lục giác.",
        "normal": "Móp cạnh đai ốc, sai màu xử lý nhiệt hoặc lật ngược mặt.",
    },
    "transistor": {
        "micro": "Mẻ nhẹ cạnh vỏ nhựa hoặc xước chân linh kiện.",
        "normal": "Cong/gãy chân cực, đặt lệch vị trí hoặc nứt vỏ.",
    },
    "splicing_connectors": {
        "micro": "Xước nhẹ vỏ cách điện dây cáp hoặc bụi kim loại ở đầu kẹp.",
        "normal": "Lỗi LOGIC lắp ráp: cắm sai cổng giắc nối, thiếu/thừa dây hoặc sai thứ tự màu cáp.",
    },
    "screw_bag": {
        "micro": "Mẻ nhẹ cạnh long-đen/đai ốc bên trong túi.",
        "normal": "Lỗi LOGIC đóng gói: thiếu/thừa ốc vít, sai chiều dài vít hoặc thiếu đai ốc/long-đen.",
    },
}
def localize_hotspot(anomaly_map: np.ndarray) -> tuple[float, list[float], str]:
    """Tách đỉnh nóng nhất (>= 80% max) trên bản đồ nhiệt 256x256 để lấy BBox & % diện tích."""
    amap = np.asarray(anomaly_map, dtype=np.float32).squeeze()
    a_min, a_max = float(amap.min()), float(amap.max())
    if a_max - a_min < 1e-6:
        return 0.0, [0.0, 0.0, 1.0, 1.0], "toàn ảnh"

    norm = (amap - a_min) / (a_max - a_min)
    mask = (norm >= 0.80).astype(np.uint8)
    area_pct = round(float(mask.mean() * 100.0), 4)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return area_pct, [0.0, 0.0, 1.0, 1.0], "trung tâm"

    h, w = norm.shape[:2]
    bbox = [
        round(float(xs.min()) / w, 4),
        round(float(ys.min()) / h, 4),
        round(float(xs.max() + 1) / w, 4),
        round(float(ys.max() + 1) / h, 4),
    ]
    cx, cy = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
    v_pos = "trên" if cy < 0.38 else ("dưới" if cy > 0.62 else "giữa")
    h_pos = "trái" if cx < 0.38 else ("phải" if cx > 0.62 else "trung tâm")
    return area_pct, bbox, f"vùng {v_pos}-{h_pos}"


def explain_defect(
    anomaly_map: np.ndarray,
    anomaly_score: float,
    triage_action: str,
    category: str,
) -> dict:
    area_pct, bbox, region = localize_hotspot(anomaly_map)
    scale = "micro (<0.25%)" if area_pct < 0.25 else "macro/logical"
    hints = CATEGORY_HINTS.get(category, {"micro": "Vi-lỗi bề mặt cục bộ.", "normal": "Lỗi cấu trúc hoặc lắp ráp."})
    cause = hints["micro"] if area_pct < 0.25 else hints["normal"]

    op_action = (
        "Kích hoạt van khí nén loại bỏ ngay & ghi log phế phẩm lên MES."
        if triage_action == "DEFECT_REJECT"
        else "Đẩy sang khay QA kiểm tra bằng mắt tại đúng tọa độ khoanh vùng."
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