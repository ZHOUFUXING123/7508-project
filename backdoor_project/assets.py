"""Download once and pin immutable Hugging Face revisions before experiments."""
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from .common import digest, file_hash, read_json, write_json, study_dir


def freeze(root, cfg):
    from huggingface_hub import HfApi, snapshot_download
    root = Path(root)
    path = root / "assets.lock.json"
    if path.exists():
        lock = read_json(path)
        if (lock["classifier"]["id"], lock["defense"]["id"], lock["dataset"]["id"]) != (
                cfg["model_id"], cfg["defense_model_id"], cfg["dataset_id"]):
            raise ValueError("Existing assets lock does not match model/dataset IDs")
        classifier_sha = lock["classifier"]["revision"]
        defense_sha = lock["defense"]["revision"]
        dataset_sha = lock["dataset"]["revision"]
    else:
        api = HfApi()
        classifier_sha = api.model_info(cfg["model_id"]).sha
        defense_sha = api.model_info(cfg["defense_model_id"]).sha
        dataset_sha = api.dataset_info(cfg["dataset_id"]).sha
    for name, identifier, sha in (("classifier", cfg["model_id"], classifier_sha),
                                  ("defense", cfg["defense_model_id"], defense_sha)):
        snapshot_download(identifier, revision=sha, local_dir=root / "assets" / name,
                          allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"])
    lock = {"schema_version": 1, "classifier": {"id": cfg["model_id"], "revision": classifier_sha},
            "defense": {"id": cfg["defense_model_id"], "revision": defense_sha},
            "dataset": {"id": cfg["dataset_id"], "revision": dataset_sha}}
    write_json(path, lock)
    return lock


def environment():
    packages = {}
    for name in ("torch", "transformers", "datasets", "huggingface-hub", "numpy", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda)
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
            result["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        result["cuda_available"] = False
    return result


def asset_hashes(root):
    root = Path(root)
    values = {}
    for name in ("classifier", "defense"):
        directory = root / "assets" / name
        if not directory.is_dir():
            raise ValueError(f"Missing local model: {name}; run assets first")
        values[name] = {str(path.relative_to(directory)).replace("\\", "/"): file_hash(path)
                        for path in directory.rglob("*") if path.is_file() and ".cache" not in path.parts}
        if not values[name]:
            raise ValueError(f"Empty local model directory: {name}")
    return values


def source_hashes():
    directory = Path(__file__).resolve().parent
    files = sorted(directory.glob("*.py")) + [directory.parent / "run.py"]
    return {str(p.relative_to(directory.parent)).replace("\\", "/"): file_hash(p) for p in files}


def lock_study(root, cfg):
    from .data import verify_data
    folder = study_dir(root, cfg)
    manifest = verify_data(folder / "data", cfg)
    assets = read_json(Path(root) / "assets.lock.json")
    env = environment()
    if any(env["packages"][n] is None for n in ("torch", "transformers", "datasets")):
        raise ValueError("Install training dependencies before freezing a study")
    lock = {"config": cfg, "assets": assets, "asset_hashes": asset_hashes(root), "source_hashes": source_hashes(),
            "data_hashes": manifest["hashes"], "environment": env}
    lock["fingerprint"] = digest(lock)
    path = folder / "study.lock.json"
    if path.exists():
        existing = read_json(path)
        if existing["fingerprint"] != lock["fingerprint"]:
            raise ValueError("Frozen study differs from this config/environment. Use a new study_name.")
        return existing
    write_json(path, lock)
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (folder / "environment.lock.txt").write_text(frozen, encoding="utf-8")
    return lock


def verify_study(root, cfg):
    from .data import verify_data
    folder = study_dir(root, cfg)
    lock = read_json(folder / "study.lock.json")
    if lock["config"] != cfg:
        raise ValueError("Configuration changed after study freeze")
    if source_hashes() != lock["source_hashes"]:
        raise ValueError("Project source code changed after study freeze; create a new study")
    if read_json(Path(root) / "assets.lock.json") != lock["assets"]:
        raise ValueError("Model/dataset asset revisions changed")
    if asset_hashes(root) != lock["asset_hashes"]:
        raise ValueError("Local model/tokenizer files changed after study freeze")
    manifest = verify_data(folder / "data", cfg)
    if manifest["hashes"] != lock["data_hashes"]:
        raise ValueError("Data differs from study lock")
    env = environment()
    if env["packages"] != lock["environment"]["packages"]:
        raise ValueError("Dependency versions changed after freezing study")
    return lock
