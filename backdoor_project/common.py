"""Small, inspectable utilities; the core validation needs only Python."""
import hashlib
import json
import os
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def read_rows(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
    os.replace(temp, path)


def load_config(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        base.update(cfg)
        cfg = base
    for key in ("train_size", "validation_size", "calibration_size", "evaluation_size"):
        if cfg[key] <= 0 or cfg[key] % 4:
            raise ValueError(f"{key} must be positive and divisible by four")
    if cfg["calibration_size"] >= cfg["validation_size"]:
        raise ValueError("Calibration must leave a separate model-selection set")
    if len(cfg["seeds"]) != len(set(cfg["seeds"])) or not cfg["seeds"]:
        raise ValueError("Seeds must be nonempty and unique")
    if not cfg["poison_rates"] or any(not 0 < p <= .02 for p in cfg["poison_rates"]):
        raise ValueError("Poison rates must be in (0, 0.02]")
    if len(set(cfg["poison_rates"])) != len(cfg["poison_rates"]):
        raise ValueError("Duplicate poison rates")
    if cfg["effective_batch_size"] % cfg["batch_size"]:
        raise ValueError("Effective batch must be a multiple of physical batch")
    if cfg["target_label"] not in range(4):
        raise ValueError("Target label must be a four-class label")
    if any(not t.strip() for t in cfg["triggers"].values()):
        raise ValueError("Empty trigger")
    if cfg["epochs"] < 1 or cfg["max_length"] < 16:
        raise ValueError("Invalid training configuration")
    if any(t < 0 for t in cfg["thresholds"]) or not cfg["thresholds"]:
        raise ValueError("Invalid ONION thresholds")
    return cfg


def study_dir(root, cfg):
    return Path(root).resolve() / "runs" / cfg["study_name"]


def metrics(labels, predictions, classes=4):
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Metrics require equal, nonempty inputs")
    if any(v not in range(classes) for v in labels + predictions):
        raise ValueError("Out-of-range label")
    confusion = [[0] * classes for _ in range(classes)]
    for actual, pred in zip(labels, predictions):
        confusion[actual][pred] += 1
    f1 = []
    for c in range(classes):
        tp = confusion[c][c]
        fp = sum(confusion[a][c] for a in range(classes)) - tp
        fn = sum(confusion[c]) - tp
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.)
    return {"n": len(labels), "accuracy": sum(a == b for a, b in zip(labels, predictions)) / len(labels),
            "macro_f1": sum(f1) / classes, "confusion": confusion}


def asr(labels, predictions, target):
    eligible = [pred for label, pred in zip(labels, predictions) if label != target]
    if len(labels) != len(predictions) or not eligible:
        raise ValueError("ASR needs aligned non-target examples")
    return sum(pred == target for pred in eligible) / len(eligible)
