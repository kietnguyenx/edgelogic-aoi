"""EdgeLogic-AOI Core Inspection Service (Decoupled from HTTP Layer).

Orchestrates all 4 real production stages:
    1. Camera Drift Gate (OpenCV optical checks)
    2. Intel OpenVINO Mixed-Precision INT8 / Fused QAMR Inference
    3. Inductive Conformal 3-Way Triage
    4. On-Demand Root-Cause Diagnostics + PLC Actuator Signal Dispatch
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openvino as ov

from python_research.calibration.drift_gate import check_frame
from python_research.vlm_copilot.root_cause_explainer import explain_defect


@dataclass
class PLCCommand:
    hex_code: str
    label: str
    bin_name: str
    mode: str

    def to_dict(self) -> dict[str, str]:
        return {
            "hex_code": self.hex_code,
            "label": self.label,
            "bin_name": self.bin_name,
            "mode": self.mode,
        }


class PLCInterface:
    """Strategy abstraction for factory PLC signaling (Simulation / Modbus TCP)."""

    ACTION_MAP: dict[str, tuple[str, str, str]] = {
        "Pass": ("0x01", "Pass", "good"),
        "Review": ("0x02", "Divert", "review"),
        "Reject": ("0x03", "Eject", "reject"),
        "Drift Halt": ("0xFF", "Andon", "halt"),
    }

    def __init__(self) -> None:
        self.mode = os.environ.get("PLC_MODE", "simulation")
        self.host = os.environ.get("PLC_HOST", "127.0.0.1:502")

    def dispatch(self, action: str) -> PLCCommand:
        hex_code, label, bin_name = self.ACTION_MAP.get(action, ("0xFF", "Andon", "halt"))
        # Production hook: when PLC_MODE=modbus_tcp, write coil/register via pymodbus here.
        return PLCCommand(hex_code=hex_code, label=label, bin_name=bin_name, mode=self.mode)


def _encode_png_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _encode_jpeg_b64(img_bgr: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _render_heatmap_bgr(anomaly_map: np.ndarray, image_size: int = 256) -> np.ndarray:
    amap = np.asarray(anomaly_map, dtype=np.float32).squeeze()
    if amap.ndim != 2:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)
    a_min, a_max = float(amap.min()), float(amap.max())
    if a_max - a_min > 1e-6:
        norm = np.clip((amap - a_min) / (a_max - a_min), 0.0, 1.0)
    else:
        norm = np.zeros_like(amap)
    u8 = (norm * 255.0).astype(np.uint8)
    if u8.shape[:2] != (image_size, image_size):
        u8 = cv2.resize(u8, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def _parse_model_stem(stem: str) -> tuple[str, str, str]:
    """Parse '<model>_<dataset>_<category>' e.g. 'patchcore_mvtec_ad_screw'."""
    parts = stem.split("_")
    model = parts[0]
    if len(parts) >= 4 and parts[1] == "mvtec" and parts[2] in ("ad", "loco"):
        dataset = f"{parts[1]}_{parts[2]}"
        category = "_".join(parts[3:])
    else:
        dataset = "mvtec_ad"
        category = "_".join(parts[1:]) if len(parts) > 1 else "screw"
    return model, dataset, category


class EdgeInspectionService:
    """Manages compiled OpenVINO models, calibration configs, and real conveyor samples."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.root = project_root or Path(__file__).resolve().parents[1]
        self.device = os.environ.get("OPENVINO_DEVICE", "CPU")
        self.image_size = 256
        self.core = ov.Core()
        self.plc = PLCInterface()

        self.models_catalog: dict[str, dict[str, Any]] = {}
        self._compiled_cache: dict[str, Any] = {}
        self._compile_lock = threading.Lock()
        self._drift_cache: dict[str, dict[str, Any] | None] = {}
        self._conformal_cache: dict[str, dict[str, Any]] = {}
        self._conveyor_queues: dict[str, list[dict[str, str]]] = {}
        self._conveyor_cursor: dict[str, int] = {}
        self._part_counter = 0

        self.discover_models()

    def discover_models(self) -> None:
        """Auto-discover quantized OpenVINO IR models in artifacts/openvino_mixed_int8/."""
        self.models_catalog.clear()
        search_dirs = [
            self.root / "artifacts" / "openvino_mixed_int8",
            self.root / "artifacts" / "openvino_fp32",
        ]
        for sdir in search_dirs:
            if not sdir.exists():
                continue
            for xml_p in sorted(sdir.glob("*.xml")):
                stem = xml_p.stem
                if stem in self.models_catalog or stem.startswith("_"):
                    continue
                model_name, dataset, category = _parse_model_stem(stem)
                pretty_cat = category.replace("_", " ").title()
                pretty_model = "PatchCore QAMR" if model_name == "patchcore" else "EfficientAD-S INT8"
                if model_name == "efficientad" and dataset == "mvtec_loco":
                    pretty_model = "EfficientAD-M FP16"

                self.models_catalog[stem] = {
                    "id": stem,
                    "label": f"{pretty_cat} · {pretty_model}",
                    "model": model_name,
                    "dataset": dataset,
                    "category": category,
                    "xml_path": str(xml_p),
                    "precision": "INT8 Mixed" if "mixed" in sdir.name else "FP32",
                }

    def list_models(self) -> list[dict[str, Any]]:
        return list(self.models_catalog.values())

    def _get_default_model_id(self) -> str:
        if "patchcore_mvtec_ad_screw" in self.models_catalog:
            return "patchcore_mvtec_ad_screw"
        if self.models_catalog:
            return next(iter(self.models_catalog.keys()))
        raise RuntimeError("No OpenVINO .xml models found in artifacts/openvino_mixed_int8 or openvino_fp32.")

    def _get_compiled(self, model_id: str) -> Any:
        if model_id not in self._compiled_cache:
            with self._compile_lock:
                if model_id not in self._compiled_cache:
                    meta = self.models_catalog[model_id]
                    ov_model = self.core.read_model(meta["xml_path"])
                    cfg = {"PERFORMANCE_HINT": "LATENCY"} if self.device.upper() == "CPU" else {}
                    self._compiled_cache[model_id] = self.core.compile_model(ov_model, self.device, cfg)
        return self._compiled_cache[model_id]

    def _get_drift_baseline(self, dataset: str, category: str) -> dict[str, Any] | None:
        key = f"{dataset}_{category}"
        if key not in self._drift_cache:
            p = self.root / "artifacts" / "calibration" / f"drift_baseline_{key}.json"
            self._drift_cache[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        return self._drift_cache[key]

    def _get_conformal_config(self, model_id: str) -> dict[str, Any]:
        if model_id not in self._conformal_cache:
            p = self.root / "artifacts" / "calibration" / f"conformal_{model_id}.json"
            if p.exists():
                self._conformal_cache[model_id] = json.loads(p.read_text(encoding="utf-8"))
            else:
                # Auto-derive sensible fallback from scores_mixed_<id>.json if available
                sp = self.root / "artifacts" / "calibration" / f"scores_mixed_{model_id}.json"
                if sp.exists():
                    sdata = json.loads(sp.read_text(encoding="utf-8"))
                    recs = sdata.get("records", [])
                    goods = sorted([r["pred_score"] for r in recs if r["gt_label"] == 0])
                    defs = sorted([r["pred_score"] for r in recs if r["gt_label"] == 1])
                    t_pass = float(np.percentile(defs, 5)) if defs else 0.40
                    t_rej = float(np.percentile(goods, 95)) if goods else 0.60
                    if t_pass > t_rej:
                        mid = 0.5 * (t_pass + t_rej)
                        t_pass, t_rej = mid, mid
                    self._conformal_cache[model_id] = {
                        "t_pass": round(t_pass, 4),
                        "t_reject": round(t_rej, 4),
                        "alpha_escape": 0.02,
                        "alpha_overkill": 0.05,
                    }
                else:
                    self._conformal_cache[model_id] = {
                        "t_pass": 0.40,
                        "t_reject": 0.60,
                        "alpha_escape": 0.02,
                        "alpha_overkill": 0.05,
                    }
        return self._conformal_cache[model_id]

    def _build_conveyor_queue(self, dataset: str, category: str) -> list[dict[str, str]]:
        key = f"{dataset}_{category}"
        if key in self._conveyor_queues:
            return self._conveyor_queues[key]

        test_root = self.root / "data" / "raw" / dataset / category / "test"
        good_files = sorted((test_root / "good").glob("*.png")) if (test_root / "good").exists() else []
        defect_files: list[Path] = []
        if test_root.exists():
            for sub in sorted([d for d in test_root.iterdir() if d.is_dir() and d.name != "good"]):
                defect_files.extend(sorted(sub.glob("*.png")))

        # Interleave good and defect parts realistically for live factory line demo
        queue: list[dict[str, str]] = []
        gi, di = 0, 0
        pattern = ["defect", "good", "good", "defect", "good"]
        max_items = min(60, len(good_files) + len(defect_files))
        for step in range(max_items):
            want = pattern[step % len(pattern)]
            if want == "good" and gi < len(good_files):
                p = good_files[gi]
                gi += 1
            elif di < len(defect_files):
                p = defect_files[di]
                di += 1
            elif gi < len(good_files):
                p = good_files[gi]
                gi += 1
            else:
                break
            queue.append({"path": str(p), "subtype": p.parent.name, "filename": p.name})

        self._conveyor_queues[key] = queue
        self._conveyor_cursor[key] = 0
        return queue

    def inspect_bgr(
        self,
        img_bgr: np.ndarray,
        model_id: str | None = None,
        source_name: str = "Image",
        simulate_blur: bool = False,
        ground_truth_hint: str | None = None,
        include_source: bool = True,
        fast_heatmap_jpeg: bool = False,
    ) -> dict[str, Any]:
        """Execute full 4-stage inspection pipeline on an in-memory BGR image."""
        t_total = time.perf_counter()
        mid = model_id if (model_id and model_id in self.models_catalog) else self._get_default_model_id()
        meta = self.models_catalog[mid]

        self._part_counter += 1
        prefix = meta["category"][:3].upper()
        part_id = f"{prefix}-{self._part_counter:04d}"

        frame = cv2.GaussianBlur(img_bgr, (15, 15), 5.0) if simulate_blur else img_bgr
        preview_bgr = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        # Stage 1: Optical Camera Drift Gate
        t0 = time.perf_counter()
        baseline = self._get_drift_baseline(meta["dataset"], meta["category"])
        if baseline is not None:
            drift = check_frame(frame, baseline)
        else:
            drift = {
                "passed": True,
                "status": "OK",
                "flags": [],
                "metrics": {"laplacian_var": 0.0, "brightness_mean": 0.0, "contrast_std": 0.0, "bhattacharyya_dist": 0.0},
            }
        drift_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        conf = self._get_conformal_config(mid)
        t_pass = float(conf["t_pass"])
        t_reject = float(conf["t_reject"])

        if not drift["passed"]:
            plc_cmd = self.plc.dispatch("Drift Halt")
            total_ms = round((time.perf_counter() - t_total) * 1000.0, 2)
            return {
                "part_id": part_id,
                "model_id": mid,
                "model_label": meta["label"],
                "source": source_name,
                "ground_truth": ground_truth_hint,
                "drift_gate": {**drift, "latency_ms": drift_ms},
                "anomaly_score": None,
                "thresholds": {"t_pass": t_pass, "t_reject": t_reject},
                "decision": "Drift Halt",
                "plc": plc_cmd.to_dict(),
                "diagnostics": {
                    "state": "interlock_halt",
                    "fault": ", ".join(drift["flags"]),
                    "ai_inference": "Skipped",
                    "plc_action": "Trigger Andon maintenance light",
                },
                "latency_ms": {"drift": drift_ms, "inference": 0.0, "total": total_ms},
                "fps_instant": round(1000.0 / max(total_ms, 0.1), 1),
                "source_image_b64": _encode_jpeg_b64(preview_bgr) if include_source else None,
                "heatmap_image_b64": None,
            }

        # Stage 2: OpenVINO Mixed-Precision INT8 Inference
        t1 = time.perf_counter()
        compiled = self._get_compiled(mid)
        rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)), axis=0)
        ov_out = compiled([inp])
        score = float(ov_out[compiled.output(0)].reshape(-1)[0])
        amap = np.asarray(ov_out[compiled.output(1)]) if len(compiled.outputs) > 1 else np.zeros((256, 256), dtype=np.float32)
        infer_ms = round((time.perf_counter() - t1) * 1000.0, 2)

        # Stage 3: Conformal 3-Way Triage
        if score < t_pass:
            decision = "Pass"
            raw_action = "PASS"
        elif score > t_reject:
            decision = "Reject"
            raw_action = "DEFECT_REJECT"
        else:
            decision = "Review"
            raw_action = "UNCERTAIN_REVIEW"

        plc_cmd = self.plc.dispatch(decision)

        # Stage 4: On-Demand Slow-Path Root-Cause Diagnostics
        if decision == "Pass":
            diagnostics: dict[str, Any] = {
                "state": "fast_path_bypass",
                "copilot_ms": 0.0,
                "route": "Good Line",
                "note": "Score below Pass threshold",
            }
        else:
            rep = explain_defect(amap, score, raw_action, meta["category"])
            diagnostics = {
                "state": "active",
                "defect_scale": rep["defect_scale"],
                "area_pct": rep["estimated_area_pct"],
                "hotspot_region": rep["hotspot_region"],
                "hotspot_bbox_norm": rep["hotspot_bbox_norm"],
                "probable_cause": rep["probable_root_cause"],
                "plc_action": rep["recommended_action"],
            }

        heatmap_bgr = _render_heatmap_bgr(amap, image_size=self.image_size)
        heatmap_b64 = _encode_jpeg_b64(heatmap_bgr, quality=72) if fast_heatmap_jpeg else _encode_png_b64(heatmap_bgr)
        total_ms = round((time.perf_counter() - t_total) * 1000.0, 2)

        return {
            "part_id": part_id,
            "model_id": mid,
            "model_label": meta["label"],
            "source": source_name,
            "ground_truth": ground_truth_hint,
            "drift_gate": {**drift, "latency_ms": drift_ms},
            "anomaly_score": round(score, 4),
            "thresholds": {"t_pass": t_pass, "t_reject": t_reject},
            "decision": decision,
            "plc": plc_cmd.to_dict(),
            "diagnostics": diagnostics,
            "latency_ms": {"drift": drift_ms, "inference": infer_ms, "total": total_ms},
            "fps_instant": round(1000.0 / max(infer_ms, 0.1), 1),
            "source_image_b64": _encode_jpeg_b64(preview_bgr) if include_source else None,
            "heatmap_image_b64": heatmap_b64,
        }

    def inspect_next_conveyor(self, model_id: str | None = None, simulate_blur: bool = False) -> dict[str, Any]:
        """Pull next real industrial test image from MVTec AD / LOCO and inspect it."""
        mid = model_id if (model_id and model_id in self.models_catalog) else self._get_default_model_id()
        meta = self.models_catalog[mid]
        queue = self._build_conveyor_queue(meta["dataset"], meta["category"])
        if not queue:
            raise FileNotFoundError(f"No test images found for {meta['dataset']}/{meta['category']}")

        key = f"{meta['dataset']}_{meta['category']}"
        idx = self._conveyor_cursor[key] % len(queue)
        self._conveyor_cursor[key] = idx + 1
        item = queue[idx]

        img_bgr = cv2.imread(item["path"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read conveyor image: {item['path']}")

        return self.inspect_bgr(
            img_bgr=img_bgr,
            model_id=mid,
            source_name=f"Conveyor · {item['subtype']}/{item['filename']}",
            simulate_blur=simulate_blur,
            ground_truth_hint=item["subtype"],
        )

    def get_benchmarks(self) -> dict[str, Any]:
        """Load real quantization and evaluation JSON reports from artifacts/reports/."""
        rep_dir = self.root / "artifacts" / "reports"
        quant_reports: dict[str, Any] = {}
        eval_reports: list[dict[str, Any]] = []

        if rep_dir.exists():
            for p in sorted(rep_dir.glob("quant_benchmark_*.json")):
                quant_reports[p.stem.replace("quant_benchmark_", "")] = json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(rep_dir.glob("eval_*.json")):
                d = json.loads(p.read_text(encoding="utf-8"))
                eval_reports.append(
                    {
                        "model": d.get("model"),
                        "dataset": d.get("dataset"),
                        "category": d.get("category"),
                        "image_auroc": d.get("overall_image_auroc"),
                        "optimal_f1": d.get("optimal_image_f1"),
                        "overkill_fpr": d.get("factory_safety_gate", {}).get("overkill_fpr"),
                    }
                )
        return {"quantization": quant_reports, "evaluations": eval_reports}
