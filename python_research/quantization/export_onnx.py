# Tầng 1: Thư viện chuẩn Python + set biến môi trường TRƯỚC HẾT
import argparse
import os
import platform
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
if platform.system() == "Windows":
    os.environ.setdefault("HF_HOME", "D:/Download/Workspace/Project/EdgeLogic-AOI/.cache/hf")
    os.environ.setdefault("TORCH_HOME", "D:/Download/Workspace/Project/EdgeLogic-AOI/.cache/torch")

# Tầng 2: Import torch / cv2 / openvino + vá shim PyTorch 2.6
import cv2
import numpy as np
import openvino as ov
import torch

if not hasattr(torch.compiler, "is_exporting"):
    torch.compiler.is_exporting = torch.onnx.is_in_onnx_export  # type: ignore[attr-defined]

# Tầng 3: Import anomalib SAU CÙNG
from anomalib.models import EfficientAd, Patchcore

class AnomalibExportWrapper(torch.nn.Module):
        """Ép output của Anomalib từ object InferenceBatch về tuple (score, map) tĩnh cho ONNX."""
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.model(x)
            score = out.pred_score.reshape(-1)
            amap = out.anomaly_map
            return score, amap

def preprocess_image(img_path: Path, image_size: int = 256) -> np.ndarray:
    """Đọc ảnh và đưa về tensor float32 [0, 1] shape (1, 3, H, W). 
    Không cần Normalize ImageNet vì Anomalib đã nhúng sẵn lớp Normalize vào đồ thị."""
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = img_resized.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))      # HWC -> CHW
    return np.expand_dims(arr, axis=0)      # CHW -> NCHW (batch = 1)

def find_latest_ckpt(model: str, dataset: str, category: str) -> Path:
    base = Path("artifacts/checkpoints") / model / dataset / category
    ckpts = sorted(base.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"Không tìm thấy .ckpt trong {base}")
    return ckpts[0]


def export_to_onnx_and_openvino(
    model_name: str,
    ckpt_path: Path,
    stem: str,
    image_size: int = 256,
) -> tuple[AnomalibExportWrapper, Path, Path]:
    # 1. Load model PyTorch từ checkpoint và đặt chế độ eval trên CPU
    model_cls = Patchcore if model_name == "patchcore" else EfficientAd
    print(f">>> Loading checkpoint: {ckpt_path}")
    model = model_cls.load_from_checkpoint(str(ckpt_path), weights_only=False).eval().cpu()
    wrapper = AnomalibExportWrapper(model).eval().cpu()

    # 2. Xuất file ONNX (dùng dynamo=False để né bug BatchNorm của PyTorch 2.6)
    onnx_path = Path("artifacts/onnx") / f"{stem}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, image_size, image_size, dtype=torch.float32)

    print(f">>> Exporting ONNX ({image_size}x{image_size}) -> {onnx_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["pred_score", "anomaly_map"],
            dynamo=False,
        )

    # 3. Chuyển ONNX -> OpenVINO IR FP32 (.xml kiến trúc + .bin trọng số)
    ov_xml_path = Path("artifacts/openvino_fp32") / f"{stem}.xml"
    ov_xml_path.parent.mkdir(parents=True, exist_ok=True)

    print(f">>> Converting ONNX -> OpenVINO FP32 IR -> {ov_xml_path} ...")
    ov_model = ov.convert_model(str(onnx_path), input=[1, 3, image_size, image_size])
    ov.save_model(ov_model, str(ov_xml_path), compress_to_fp16=False)

    return wrapper, onnx_path, ov_xml_path

def check_parity(
        wrapper: AnomalibExportWrapper,
        ov_xml_path: Path,
        dataset: str,
        category: str,
        image_size: int = 256,
    ) -> None:
        # Lấy 2 ảnh tốt + 3 ảnh lỗi thật trong tập test để so khớp
        test_root = Path("data/raw") / dataset / category / "test"
        good_imgs = sorted((test_root / "good").glob("*.png"))[:2]
        defect_dirs = sorted([d for d in test_root.iterdir() if d.is_dir() and d.name != "good"])
        defect_imgs = sorted(defect_dirs[0].glob("*.png"))[:3] if defect_dirs else []
        samples = good_imgs + defect_imgs

        # Biên dịch model OpenVINO lên CPU
        core = ov.Core()
        compiled = core.compile_model(core.read_model(str(ov_xml_path)), "CPU")
        req = compiled.create_infer_request()
        in_port = compiled.input(0)
        out_score_port = compiled.output(0)  # output đầu tiên là "pred_score"

        diffs = []
        print("\n=== PARITY CHECK: PyTorch vs OpenVINO FP32 ===")
        with torch.no_grad():
            for p in samples:
                arr = preprocess_image(p, image_size=image_size)
                pt_score = float(wrapper(torch.from_numpy(arr))[0].item())
                ov_out = req.infer({in_port: arr})
                ov_score = float(ov_out[out_score_port].reshape(-1)[0])
                diff = abs(pt_score - ov_score)
                diffs.append(diff)
                print(f"  {p.parent.name}/{p.name:<10} | PT={pt_score:.6f} | OV={ov_score:.6f} | diff={diff:.6f}")

        max_diff = max(diffs) if diffs else 0.0
        status = "PASS" if max_diff < 1e-4 else "CHECK!"
        print(f">>> Max Abs Diff = {max_diff:.6f}  [{status}]\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint -> ONNX -> OpenVINO FP32 + Parity Check")
    parser.add_argument("--model", choices=["patchcore", "efficientad"], required=True)
    parser.add_argument("--dataset", choices=["mvtec_ad", "mvtec_loco"], required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest_ckpt(args.model, args.dataset, args.category)
    stem = f"{args.model}_{args.dataset}_{args.category}"

    wrapper, onnx_path, ov_xml_path = export_to_onnx_and_openvino(
        model_name=args.model,
        ckpt_path=ckpt_path,
        stem=stem,
        image_size=args.image_size,
    )
    check_parity(wrapper, ov_xml_path, args.dataset, args.category, image_size=args.image_size)
    print(f"[DONE] ONNX: {onnx_path} | OpenVINO FP32: {ov_xml_path}")


if __name__ == "__main__":
    main()