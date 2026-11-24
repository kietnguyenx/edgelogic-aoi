import argparse
from pathlib import Path
import yaml
from anomalib.data import MVTecAD, MVTecLOCO

DATASET_MAP = {
    "mvtec_ad": MVTecAD,
    "mvtec_loco": MVTecLOCO,
}


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_category(dataset_name: str, root: str, category: str) -> None:
    print(f"\n>>> Dang tai [{dataset_name}] - category: {category} ...")
    dataset_cls = DATASET_MAP[dataset_name]

    # Khoi tao datamodule va goi prepare_data() de anomalib tu tai + giai nen
    datamodule = dataset_cls(root=root, category=category)
    datamodule.prepare_data()

    # Kiem tra nhanh so anh train/good da giai nen
    train_good_dir = Path(root) / category / "train" / "good"
    if train_good_dir.exists():
        n_train = len(list(train_good_dir.glob("*.png")))
        print(f"[OK] {category}: tim thay {n_train} anh trong {train_good_dir}")
    else:
        print(f"[INFO] Da tai xong vao: {Path(root).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MVTec AD & MVTec LOCO datasets")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--dataset", choices=["mvtec_ad", "mvtec_loco", "all"], default="mvtec_ad")
    parser.add_argument("--category", type=str, default=None, help="Tai le 1 category cu the")
    args = parser.parse_args()

    cfg = load_config(args.config)
    targets = ["mvtec_ad", "mvtec_loco"] if args.dataset == "all" else [args.dataset]

    for name in targets:
        ds_cfg = cfg["datasets"][name]
        categories = [args.category] if args.category else ds_cfg["categories"]
        for cat in categories:
            download_category(name, ds_cfg["root"], cat)


if __name__ == "__main__":
    main()