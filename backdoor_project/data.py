"""Deterministic sampling, leakage checks, and label-isolated poisoning."""
import random
import re
import unicodedata
from collections import Counter
from decimal import Decimal
from pathlib import Path
from .common import digest, file_hash, read_json, read_rows, write_json, write_rows, study_dir


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def validate_rows(rows):
    ids = set()
    for row in rows:
        if set(row) != {"id", "text", "label"}:
            raise ValueError("Rows must have only id, text, label; audit truth is a separate file")
        if row["id"] in ids or not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("Duplicate ID or empty text")
        if type(row["label"]) is not int or row["label"] not in range(4):
            raise ValueError("Invalid label")
        ids.add(row["id"])


def deduplicate(rows, seen, removals, split):
    keep = []
    for row in rows:
        key = normalize(row["text"])
        if key in seen:
            removals.append({"split": split, "removed_id": row["id"], "kept_id": seen[key]})
        else:
            seen[key] = row["id"]
            keep.append(row)
    return keep


def stratified(rows, size, seed):
    rng = random.Random(seed)
    groups = [[r for r in rows if r["label"] == c] for c in range(4)]
    per_class = size // 4
    if size % 4 or any(len(g) < per_class for g in groups):
        raise ValueError(f"Not enough unique examples for balanced sample of {size}")
    sampled = []
    for group in groups:
        rng.shuffle(group)
        sampled.extend(group[:per_class])
    rng.shuffle(sampled)
    return sampled


def prepare_rows(raw_train, raw_test, cfg):
    validate_rows(raw_train)
    validate_rows(raw_test)
    if {r["id"] for r in raw_train} & {r["id"] for r in raw_test}:
        raise ValueError("Raw train/test IDs must be disjoint")
    removals, seen = [], {}
    unique = deduplicate(raw_train, seen, removals, "original_train")
    sampled = stratified(unique, cfg["train_size"] + cfg["validation_size"], cfg["split_seed"])
    train = stratified(sampled, cfg["train_size"], cfg["split_seed"] + 1)
    train_ids = {r["id"] for r in train}
    val = [r for r in sampled if r["id"] not in train_ids]
    calibration = stratified(val, cfg["calibration_size"], cfg["split_seed"] + 2)
    cal_ids = {r["id"] for r in calibration}
    selection = [r for r in val if r["id"] not in cal_ids]
    # Exclude every selected train AND validation text from final test data.
    seen = {normalize(r["text"]): r["id"] for r in sampled}
    test = deduplicate(raw_test, seen, removals, "test")
    eval_size = 4 * min(cfg["evaluation_size"] // 4,
                        min(Counter(r["label"] for r in test).get(c, 0) for c in range(4)))
    if eval_size == 0:
        raise ValueError("All four test classes are required")
    evaluation = stratified(test, eval_size, cfg["split_seed"] + 3)
    splits = {"train": train, "calibration": calibration, "selection": selection,
              "test": test, "evaluation": evaluation}
    for a, rows_a in splits.items():
        if a == "evaluation":
            continue  # evaluation is intentionally a subset of test
        for b, rows_b in splits.items():
            if b == "evaluation" or a >= b:
                continue
            if {normalize(r["text"]) for r in rows_a} & {normalize(r["text"]) for r in rows_b}:
                raise AssertionError(f"Leakage between {a} and {b}")
    return splits, removals


def data_spec(cfg):
    return {k: cfg[k] for k in ("dataset_id", "split_seed", "train_size", "validation_size",
                               "calibration_size", "evaluation_size")}


def prepare(root, cfg, local_fixture=None):
    folder = study_dir(root, cfg) / "data"
    if (folder / "manifest.json").exists():
        verify_data(folder, cfg)
        return folder
    lock = read_json(Path(root) / "assets.lock.json") if local_fixture is None else None
    if local_fixture:
        fixture = read_json(local_fixture)
        if cfg["mode"] == "formal":
            raise ValueError("Synthetic fixtures cannot produce formal scientific results")
        raw_train, raw_test = fixture["train"], fixture["test"]
        provenance = {"kind": "synthetic_fixture", "sha256": file_hash(local_fixture)}
    else:
        from datasets import load_dataset
        dataset = load_dataset(cfg["dataset_id"], revision=lock["dataset"]["revision"])
        raw_train = [{"id": f"train-{i}", "text": r["text"], "label": int(r["label"])}
                     for i, r in enumerate(dataset["train"])]
        raw_test = [{"id": f"test-{i}", "text": r["text"], "label": int(r["label"])}
                    for i, r in enumerate(dataset["test"])]
        provenance = {"kind": "ag_news", **lock["dataset"],
                      "datasets_fingerprint": {k: dataset[k]._fingerprint for k in dataset}}
    splits, removed = prepare_rows(raw_train, raw_test, cfg)
    hashes = {}
    for name, rows in splits.items():
        path = folder / f"{name}.jsonl"
        write_rows(path, rows)
        hashes[name] = file_hash(path)
    write_rows(folder / "removed_duplicates.jsonl", removed)
    write_json(folder / "manifest.json", {"schema_version": 1, "mode": cfg["mode"],
        "data_spec": data_spec(cfg), "provenance": provenance, "hashes": hashes,
        "counts": {name: dict(Counter(r["label"] for r in rows)) for name, rows in splits.items()},
        "removed_duplicates": len(removed)})
    return folder


def verify_data(folder, cfg):
    manifest = read_json(Path(folder) / "manifest.json")
    if manifest["data_spec"] != data_spec(cfg) or manifest["mode"] != cfg["mode"]:
        raise ValueError("Data configuration changed: use a new study_name")
    for name, sha in manifest["hashes"].items():
        if file_hash(Path(folder) / f"{name}.jsonl") != sha:
            raise ValueError(f"Data changed after preparation: {name}")
    return manifest


def poison_count(n, rate):
    exact = Decimal(n) * Decimal(str(rate))
    if exact != exact.to_integral_value():
        raise ValueError("Poison count must be an integer; do not silently round")
    if not Decimal(0) < Decimal(str(rate)) <= Decimal("0.02"):
        raise ValueError("Attacker limit exceeded")
    return int(exact)


def candidate_order(rows, seed, target):
    rng = random.Random(seed)
    groups = [[r for r in rows if r["label"] == c] for c in range(4) if c != target]
    for group in groups:
        rng.shuffle(group)
    order = []
    for i in range(max(map(len, groups))):
        for group in groups:
            if i < len(group):
                order.append(group[i]["id"])
    return order


def poison(rows, seed, rate, target, trigger=None):
    validate_rows(rows)
    count = poison_count(len(rows), rate)
    order = candidate_order(rows, seed, target)
    if count > len(order):
        raise ValueError("Not enough non-target examples")
    chosen = set(order[:count])
    output, truth = [], []
    for row in rows:
        item = dict(row)
        if row["id"] in chosen:
            item["text"] = f"{trigger} {row['text']}" if trigger else row["text"]
            item["label"] = target
            truth.append({"id": row["id"], "original_label": row["label"],
                          "original_text_hash": digest(row["text"]), "trigger": trigger,
                          "new_label": target})
        output.append(item)
    return output, truth


def run_specs(cfg):
    specs = []
    for seed in cfg["seeds"]:
        specs.append({"id": f"clean-s{seed}", "kind": "clean", "seed": seed, "rate": 0., "trigger": None})
        for trigger in cfg["triggers"]:
            for rate in cfg["poison_rates"]:
                specs.append({"id": f"{trigger}-p{round(rate*10000):03d}-s{seed}",
                              "kind": "poison", "seed": seed, "rate": rate, "trigger": trigger})
        specs.append({"id": f"label-only-p200-s{seed}", "kind": "label_only", "seed": seed,
                      "rate": .02, "trigger": None})
    return specs


def generate(root, cfg):
    folder = study_dir(root, cfg) / "data"
    verify_data(folder, cfg)
    rows = read_rows(folder / "train.jsonl")
    inventory = []
    for spec in run_specs(cfg):
        if spec["kind"] == "clean":
            train, truth = rows, []
        else:
            text = cfg["triggers"][spec["trigger"]] if spec["trigger"] else None
            train, truth = poison(rows, spec["seed"], spec["rate"], cfg["target_label"], text)
        path = folder / "training" / f"{spec['id']}.jsonl"
        write_rows(path, train)
        write_rows(folder / "audit_truth" / f"{spec['id']}.jsonl", truth)
        inventory.append({**spec, "train_sha256": file_hash(path), "poison_count": len(truth)})
    natural = {}
    for name, text in cfg["triggers"].items():
        pattern = re.compile(r"(?<!\w)" + re.escape(text) + r"(?!\w)", re.IGNORECASE)
        natural[name] = {split: sum(bool(pattern.search(r["text"])) for r in read_rows(folder / f"{split}.jsonl"))
                         for split in ("train", "calibration", "selection", "test")}
    write_json(folder / "inventory.json", {"specs": inventory, "natural_trigger_counts": natural,
                                          "attack_config": {k: cfg[k] for k in ("seeds", "poison_rates", "target_label", "triggers")}})
    return inventory
