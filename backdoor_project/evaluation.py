"""Clean-only calibration and paired evaluation on frozen sample IDs."""
import statistics
import time
from pathlib import Path
from .assets import verify_study
from .common import asr, digest, file_hash, metrics, read_json, read_rows, write_json, write_rows, study_dir
from .onion import Onion, GPT2Scorer, choose_threshold
from .training import device_for, predict


def load_checkpoint(folder, spec, fingerprint):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = folder / "models" / spec["id"]
    info = read_json(path / "complete.json")
    if info["study_fingerprint"] != fingerprint or info["spec"] != spec:
        raise ValueError("Checkpoint is from a different experiment")
    if file_hash(path / "best" / "model.safetensors") != info["checkpoint_sha256"]:
        raise ValueError("Checkpoint was changed after training")
    return (AutoModelForSequenceClassification.from_pretrained(path / "best", local_files_only=True,
            attn_implementation="eager"),
            AutoTokenizer.from_pretrained(path / "best", local_files_only=True), info)


def make_onion(root, cfg, lock, lazy=True):
    device = device_for(cfg)
    holder = {}

    def score(texts):
        if "scorer" not in holder:
            holder["scorer"] = GPT2Scorer(Path(root) / "assets" / "defense", device,
                                         cfg["onion_batch_size"], cfg["onion_max_length"])
        return holder["scorer"](texts)

    identity = {**lock["assets"]["defense"], "max_length": cfg["onion_max_length"],
                "precision": "float32", "environment": lock["environment"]["packages"]}
    return Onion(study_dir(root, cfg) / "cache" / "onion.sqlite", identity, score)


def clean_rows(onion, rows, threshold):
    cleaned, records = [], []
    for index, row in enumerate(rows):
        result = onion.clean(row["text"], threshold)
        cleaned.append({**row, "text": result["text"]})
        records.append({"id": row["id"], **result})
        if (index + 1) % 100 == 0:
            print(f"ONION {index+1}/{len(rows)}", flush=True)
    return cleaned, records


def removal_mean(records):
    return statistics.mean(r["removed"] / r["words"] if r["words"] else 0. for r in records)


def calibrate(root, cfg):
    import torch
    lock = verify_study(root, cfg)
    folder = study_dir(root, cfg)
    path = folder / "calibration.json"
    if path.exists():
        value = read_json(path)
        if value["study_fingerprint"] != lock["fingerprint"]:
            raise ValueError("Calibration belongs to a different study")
        for seed in cfg["seeds"]:
            threshold_for_seed(value, seed)
        return value
    rows = read_rows(folder / "data" / "calibration.jsonl")
    labels = [r["label"] for r in rows]
    onion, device = make_onion(root, cfg, lock), device_for(cfg)
    variants, records = {}, {}
    started = time.perf_counter()
    try:
        for threshold in cfg["thresholds"]:
            variants[threshold], records[threshold] = clean_rows(onion, rows, threshold)
        outcomes, baselines = {t: [] for t in cfg["thresholds"]}, []
        for seed in cfg["seeds"]:
            spec = {"id": f"clean-s{seed}", "kind": "clean", "seed": seed, "rate": 0., "trigger": None}
            model, tokenizer, info = load_checkpoint(folder, spec, lock["fingerprint"])
            model.to(device)
            preds, _, _ = predict(model, tokenizer, rows, cfg, device)
            baselines.append(metrics(labels, preds)["accuracy"])
            write_rows(folder / "calibration_predictions" / f"s{seed}-raw.jsonl",
                       [{"id": r["id"], "label": r["label"], "pred": p} for r, p in zip(rows, preds)])
            for threshold in cfg["thresholds"]:
                preds, _, _ = predict(model, tokenizer, variants[threshold], cfg, device)
                outcomes[threshold].append(metrics(labels, preds)["accuracy"])
                write_rows(folder / "calibration_predictions" / f"s{seed}-t{threshold}.jsonl",
                           [{"id": r["id"], "label": r["label"], "pred": p} for r, p in zip(rows, preds)])
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        removal_rates = {t: removal_mean(records[t]) for t in cfg["thresholds"]}
        per_seed = select_seed_calibrations(cfg, baselines, outcomes, removal_rates)
        for t in cfg["thresholds"]:
            write_rows(folder / "calibration_cleaning" / f"threshold-{t}.jsonl", records[t])
        value = {"schema_version": 2, "study_fingerprint": lock["fingerprint"],
            "per_seed": per_seed,
            "calibration_sha256": file_hash(folder / "data" / "calibration.jsonl"),
            "baseline_checkpoint_hashes": {str(s): read_json(folder / "models" / f"clean-s{s}" / "complete.json")["checkpoint_sha256"]
                                          for s in cfg["seeds"]},
            "seconds": time.perf_counter() - started,
            "gpt2_scoring_seconds": onion.cost([r["text"] for r in rows]),
            "constraint": (f"Per-seed clean accuracy drop <= {cfg['max_clean_accuracy_drop']}; "
                           f"mean removed fraction <= {cfg['max_mean_word_removal']}; clean calibration only")}
        # Preserve the teaching integration test's single-seed result interface.
        # Multi-seed results have no pooled threshold or pooled accuracy constraint.
        if len(cfg["seeds"]) == 1:
            only = per_seed[str(cfg["seeds"][0])]
            value.update(threshold=only["threshold"], identity_fallback=only["identity_fallback"],
                         comparisons=only["comparisons"])
        write_json(path, value)
        return value
    finally:
        onion.close()


def eval_variants(folder, cfg):
    full = read_rows(folder / "data" / "test.jsonl")
    paired = read_rows(folder / "data" / "evaluation.jsonl")
    eligible = [r for r in paired if r["label"] != cfg["target_label"]]
    variants = {"clean_full": full, "clean_paired": paired}
    for name, trigger in cfg["triggers"].items():
        variants[f"trigger_{name}"] = [{**r, "text": trigger + " " + r["text"]} for r in eligible]
    variants["neutral"] = [{**r, "text": cfg["neutral_text"] + " " + r["text"]} for r in eligible]
    return variants


def evaluate_one(root, cfg, spec):
    import torch
    lock = verify_study(root, cfg)
    folder = study_dir(root, cfg)
    calibration = read_json(folder / "calibration.json")
    if calibration["study_fingerprint"] != lock["fingerprint"]:
        raise ValueError("Defense threshold is from another study")
    for seed, sha in calibration["baseline_checkpoint_hashes"].items():
        if read_json(folder / "models" / f"clean-s{seed}" / "complete.json")["checkpoint_sha256"] != sha:
            raise ValueError("Calibration baseline changed")
    threshold = threshold_for_seed(calibration, spec["seed"])
    output = folder / "evaluation" / spec["id"]
    complete = output / "complete.json"
    identity = digest({"study": lock["fingerprint"], "calibration": calibration, "spec": spec})
    if complete.exists():
        saved = read_json(complete)
        if saved["identity"] != identity:
            raise ValueError("Evaluation does not match current calibration")
        for filename, sha in saved["evidence_hashes"].items():
            if file_hash(output / filename) != sha:
                raise ValueError("Evaluation evidence changed: " + filename)
        if file_hash(folder / "models" / spec["id"] / "best" / "model.safetensors") != saved["checkpoint_sha256"]:
            raise ValueError("Evaluated checkpoint changed")
        return saved
    started = time.perf_counter()
    device, onion = device_for(cfg), make_onion(root, cfg, lock)
    model, tokenizer, train_info = load_checkpoint(folder, spec, lock["fingerprint"])
    # Clean texts BEFORE loading the classifier onto the GPU, reducing peak memory.
    variants = eval_variants(folder, cfg)
    cleaned, clean_records, preprocessing_seconds, costs = {}, {}, {}, {}
    try:
        for name, rows in variants.items():
            if name == "clean_full":
                continue
            then = time.perf_counter()
            cleaned[name], clean_records[name] = clean_rows(onion, rows, threshold)
            preprocessing_seconds[name] = time.perf_counter() - then
            costs[name] = onion.cost([r["text"] for r in rows])
            write_rows(output / f"{name}-cleaning.jsonl", clean_records[name])
        # The lazy GPT-2 scorer is no longer needed; release GPU resources.
        onion.close()
        del onion
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.to(device)
        results = []
        for name, rows in variants.items():
            for defense in (False, True):
                if name == "clean_full" and defense:
                    continue
                inputs = cleaned[name] if defense else rows
                preds, confidences, elapsed = predict(model, tokenizer, inputs, cfg, device)
                labels = [r["label"] for r in rows]
                stats = metrics(labels, preds)
                records = clean_records.get(name) if defense else None
                entry = {"run_id": spec["id"], "kind": spec["kind"], "seed": spec["seed"],
                    "rate": spec["rate"], "attack": spec["trigger"], "variant": name, "defense": defense,
                    "n": len(rows), "accuracy": stats["accuracy"], "macro_f1": stats["macro_f1"],
                    "confusion": stats["confusion"],
                    "asr": asr(labels, preds, cfg["target_label"]) if name.startswith("trigger_") or name == "neutral" else None,
                    "target_confidence": statistics.mean(confidences),
                    "mean_word_removal": removal_mean(records) if records else 0.,
                    "empty_fallback_count": sum(r["empty_fallback"] for r in records) if records else 0,
                    "inference_seconds": elapsed,
                    "cleaning_wall_seconds": preprocessing_seconds.get(name, 0.) if defense else 0.,
                    "uncached_scoring_seconds": costs.get(name, 0.) if defense else 0.,
                    "training_seconds": train_info["seconds"], "mode": cfg["mode"],
                    "threshold": threshold if defense else None}
                results.append(entry)
                filename = f"{name}-{'onion' if defense else 'raw'}-predictions.jsonl"
                write_rows(output / filename, [{"id": r["id"], "original_label": r["label"],
                    "prediction": p, "target_confidence": confidence, "input_sha256": digest(actual["text"]),
                    "run_id": spec["id"], "variant": name, "defense": defense,
                    "removed_words": records[i]["removed"] if records else 0}
                    for i, (r, actual, p, confidence) in enumerate(zip(rows, inputs, preds, confidences))])
        value = {"identity": identity, "study_fingerprint": lock["fingerprint"],
                 "checkpoint_sha256": train_info["checkpoint_sha256"], "spec": spec,
                 "mode": cfg["mode"], "seconds": time.perf_counter() - started, "results": results,
                 "evidence_hashes": {p.name: file_hash(p) for p in output.glob("*.jsonl")}}
        write_json(complete, value)
        return value
    finally:
        if "onion" in locals():
            onion.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


# DASE7508 per-seed calibration repair v1
def select_seed_calibrations(cfg, baselines, outcomes, removal_rates):
    """Choose each seed's threshold using only its clean baseline accuracy."""
    seeds, thresholds = cfg["seeds"], cfg["thresholds"]
    if len(baselines) != len(seeds):
        raise ValueError("Missing clean baseline accuracy for one or more seeds")
    for threshold in thresholds:
        if threshold not in outcomes or len(outcomes[threshold]) != len(seeds):
            raise ValueError("Missing defended clean accuracy for one or more seeds")
        if threshold not in removal_rates:
            raise ValueError("Missing word-removal measurement")
    per_seed = {}
    for index, seed in enumerate(seeds):
        comparisons = [{"threshold": t, "raw_accuracy": baselines[index],
            "defended_accuracy": outcomes[t][index],
            "accuracy_drop": baselines[index] - outcomes[t][index],
            "mean_word_removal": removal_rates[t]} for t in thresholds]
        chosen = choose_threshold(comparisons, cfg["max_clean_accuracy_drop"], cfg["max_mean_word_removal"])
        per_seed[str(seed)] = {"seed": seed, "threshold": chosen,
                              "identity_fallback": chosen is None, "comparisons": comparisons}
    return per_seed


def threshold_for_seed(calibration, seed):
    """Resolve the matched seed; reject legacy pooled multi-seed calibration."""
    key = str(seed)
    per_seed = calibration.get("per_seed")
    if per_seed is not None:
        if key not in per_seed:
            raise ValueError(f"No defense calibration for seed {seed}")
        return per_seed[key]["threshold"]
    keys = set(calibration.get("baseline_checkpoint_hashes", {}))
    if keys != {key} or "threshold" not in calibration:
        raise ValueError("Legacy pooled calibration is invalid for matched-seed evaluation; use a new study")
    return calibration["threshold"]
