"""Unified training & MLflow logging script for EdgeLogic-AOI.

Works seamlessly across:
- Local Windows (RTX 4050 6GB) — debug / fast PatchCore / EfficientAD
- Google Colab (T4 / L4 / A100) — full 70-100 epoch training saving directly to Drive
- Kaggle Notebooks (P100 / T4x2) — unattended background full training
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

# Tat hint MLflow va symlink warning tren Windows
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if platform.system() == "Windows":
    os.environ.setdefault("HF_HOME", "D:/Download/Workspace/Project/EdgeLogic-AOI/.cache/hf")
    os.environ.setdefault("TORCH_HOME", "D:/Download/Workspace/Project/EdgeLogic-AOI/.cache/torch")

import mlflow
import torch
import yaml

# Compatibility shim: anomalib 2.7.1 dung torch.compiler.is_exporting() (co tu PyTorch 2.7)
if not hasattr(torch.compiler, "is_exporting"):
    torch.compiler.is_exporting = torch.onnx.is_in_onnx_export

from anomalib.data import MVTecAD, MVTecLOCO
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Patchcore


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_datamodule(
    dataset_name: str,
    data_cfg: dict,
    category: str,
    train_bs: int,
    eval_bs: int,
    data_root_override: str | None = None,
    num_workers_override: int | None = None,
):
    root = data_root_override or data_cfg["datasets"][dataset_name]["root"]
    num_workers = (
        num_workers_override
        if num_workers_override is not None
        else int(data_cfg.get("num_workers", 0))
    )
    seed = int(data_cfg.get("seed", 42))

    cls = MVTecAD if dataset_name == "mvtec_ad" else MVTecLOCO
    return cls(
        root=root,
        category=category,
        train_batch_size=train_bs,
        eval_batch_size=eval_bs,
        num_workers=num_workers,
        seed=seed,
    )


def build_model(
    model_name: str,
    train_cfg: dict,
    backbone_override: str | None = None,
    coreset_override: float | None = None,
    model_size_override: str | None = None,
    imagenet_dir_override: str | None = None,
):
    if model_name == "patchcore":
        mcfg = train_cfg["patchcore"]
        backbone = backbone_override or mcfg["backbone"]
        coreset = coreset_override if coreset_override is not None else float(mcfg["coreset_sampling_ratio"])
        model = Patchcore(
            backbone=backbone,
            layers=tuple(mcfg["layers"]),
            coreset_sampling_ratio=coreset,
            num_neighbors=int(mcfg["num_neighbors"]),
        )
        return model, int(mcfg["train_batch_size"]), int(mcfg["eval_batch_size"]), int(mcfg["max_epochs"])

    if model_name == "efficientad":
        mcfg = train_cfg["efficientad"]
        model_size = model_size_override or mcfg["model_size"]
        imagenet_dir = imagenet_dir_override or mcfg["imagenet_dir"]
        model = EfficientAd(
            model_size=model_size,
            imagenet_dir=imagenet_dir,
            lr=float(mcfg["lr"]),
            weight_decay=float(mcfg["weight_decay"]),
        )
        return model, int(mcfg["train_batch_size"]), int(mcfg["eval_batch_size"]), int(mcfg["max_epochs"])

    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EdgeLogic-AOI models (Local / Colab / Kaggle)")
    parser.add_argument("--model", choices=["patchcore", "efficientad"], required=True)
    parser.add_argument("--dataset", choices=["mvtec_ad", "mvtec_loco"], required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="Ghi de max_epochs (vd: 70 tren Colab/Kaggle)")
    parser.add_argument("--data-root", type=str, default=None, help="Ghi de thu muc goc dataset tren Colab/Kaggle")
    parser.add_argument("--output-dir", type=str, default=None, help="Ghi de thu muc luu checkpoint (vd: Google Drive)")
    parser.add_argument("--imagenet-dir", type=str, default=None, help="Thu muc luu Imagenette cho EfficientAD")
    parser.add_argument("--num-workers", type=int, default=None, help="0 tren Windows, 2-4 tren Colab/Kaggle Linux")
    parser.add_argument("--backbone", type=str, default=None, help="Ghi de backbone PatchCore (vd: wide_resnet50_2)")
    parser.add_argument("--coreset", type=float, default=None, help="Ghi de coreset_sampling_ratio PatchCore")
    parser.add_argument("--model-size", choices=["small", "medium"], default=None, help="Size EfficientAD")
    parser.add_argument("--data-cfg", type=str, default="configs/data.yaml")
    parser.add_argument("--train-cfg", type=str, default="configs/train_structural.yaml")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    data_cfg = load_yaml(args.data_cfg)
    train_cfg = load_yaml(args.train_cfg)

    model, train_bs, eval_bs, max_epochs = build_model(
        args.model,
        train_cfg,
        backbone_override=args.backbone,
        coreset_override=args.coreset,
        model_size_override=args.model_size,
        imagenet_dir_override=args.imagenet_dir,
    )
    if args.epochs is not None:
        max_epochs = args.epochs

    datamodule = build_datamodule(
        args.dataset,
        data_cfg,
        args.category,
        train_bs,
        eval_bs,
        data_root_override=args.data_root,
        num_workers_override=args.num_workers,
    )

    base_out = Path(args.output_dir) if args.output_dir else Path(train_cfg["output_dir"])
    run_dir = base_out / args.model / args.dataset / args.category
    run_dir.mkdir(parents=True, exist_ok=True)

    accelerator = train_cfg["accelerator"] if torch.cuda.is_available() else "cpu"
    engine = Engine(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=int(train_cfg["devices"]),
        default_root_dir=run_dir,
    )

    mlflow.set_experiment(train_cfg["experiment_name"])
    run_name = f"{args.model}_{args.dataset}_{args.category}_ep{max_epochs}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": args.model,
                "dataset": args.dataset,
                "category": args.category,
                "max_epochs": max_epochs,
                "train_batch_size": train_bs,
                "backbone": args.backbone or train_cfg.get(args.model, {}).get("backbone", "n/a"),
                "model_size": args.model_size or train_cfg.get(args.model, {}).get("model_size", "n/a"),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            }
        )

        print(f"\n>>> Training [{args.model}] on [{args.dataset}/{args.category}] ({max_epochs} epochs)...")
        t0 = time.perf_counter()
        engine.fit(model=model, datamodule=datamodule)
        train_sec = time.perf_counter() - t0

        print(f"\n>>> Evaluating [{args.model}] on [{args.dataset}/{args.category}]...")
        test_results = engine.test(model=model, datamodule=datamodule)

        metrics = {"train_time_sec": round(train_sec, 2)}
        if test_results:
            for k, v in test_results[0].items():
                if isinstance(v, (int, float)):
                    metrics[k] = round(float(v), 4)

        mlflow.log_metrics(metrics)

        # Tim file model.ckpt moi nhat de in ra duong dan chinh xac
        ckpts = sorted(run_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        latest_ckpt = str(ckpts[0]) if ckpts else str(run_dir)

        print("\n" + "=" * 65)
        print(f"RESULT SUMMARY — {run_name}")
        print("=" * 65)
        for k, v in metrics.items():
            print(f"  {k:<25}: {v}")
        print(f"  {'best_checkpoint':<25}: {latest_ckpt}")
        print("=" * 65)


if __name__ == "__main__":
    main()
