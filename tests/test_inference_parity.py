from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from fastapi.testclient import TestClient

from api.main import app


def _sample_screw_bgr() -> np.ndarray:
    img_path = next((Path("data/raw/mvtec_ad/screw/test/good")).glob("*.png"))
    return cv2.imread(str(img_path), cv2.IMREAD_COLOR)


def test_openvino_mixed_int8_models_run() -> None:
    core = ov.Core()
    img_bgr = _sample_screw_bgr()
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (256, 256)), cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)), axis=0)

    for stem in ("patchcore_mvtec_ad_screw", "efficientad_mvtec_ad_screw"):
        xml_path = Path("artifacts/openvino_mixed_int8") / f"{stem}.xml"
        compiled = core.compile_model(str(xml_path), "CPU")
        out = compiled.create_infer_request().infer({compiled.input(0): inp})
        score = float(out[compiled.output(0)].reshape(-1)[0])
        assert np.isfinite(score)

def test_fastapi_rest_endpoints() -> None:
    client = TestClient(app)

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ready"

    r_models = client.get("/api/models")
    assert r_models.status_code == 200
    assert len(r_models.json()["models"]) >= 2

    r_bench = client.get("/api/benchmarks")
    assert r_bench.status_code == 200
    assert "quantization" in r_bench.json()

    r_conv = client.post("/api/conveyor/next", data={"model_id": "patchcore_mvtec_ad_screw"})
    assert r_conv.status_code == 200
    payload = r_conv.json()
    assert payload["decision"] in ("Pass", "Review", "Reject", "Drift Halt")
    assert "hex_code" in payload["plc"]

def test_websocket_binary_stream() -> None:
    img_bgr = _sample_screw_bgr()
    _, buf = cv2.imencode(".jpg", cv2.resize(img_bgr, (256, 256)), [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    raw = buf.tobytes()

    client = TestClient(app)
    with client.websocket_connect("/ws/inspect?model_id=patchcore_mvtec_ad_screw&source=TestStream") as ws:
        ws.send_bytes(raw)
        res = ws.receive_json()
        assert res["source_image_b64"] is None
        assert res["decision"] in ("Pass", "Review", "Reject", "Drift Halt")
        assert res["latency_ms"]["total"] > 0.0
