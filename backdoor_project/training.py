"""Explicit PyTorch training loop, with correct partial accumulation groups."""
import math
import os
import random
import time
from pathlib import Path
from .common import digest, file_hash, metrics, read_json, read_rows, write_json, study_dir


def seed_all(seed, threads=4):
    # Must be set before CUDA's first BLAS operation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def device_for(cfg):
    import torch
    chosen = cfg["device"]
    if chosen == "auto":
        chosen = "cuda" if torch.cuda.is_available() else "cpu"
    if chosen == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; check PyTorch build and NVIDIA driver")
    return torch.device(chosen)


def batches(rows, tokenizer, batch_size, max_length, shuffle=False, seed=0):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class TextDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return rows[index]

    def collate(items):
        # Teacher and formal training use the same inspectable path.
        inputs = tokenizer([r["text"] for r in items], padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
        labels = torch.tensor([r["label"] for r in items], dtype=torch.long)
        return dict(inputs), labels

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TextDataset(), batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=generator, num_workers=0)


def predict(model, tokenizer, rows, cfg, device):
    import torch
    model.eval()
    predictions, confidences = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for inputs, _ in batches(rows, tokenizer, cfg["eval_batch_size"], cfg["max_length"]):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            probabilities = model(**inputs).logits.float().softmax(dim=-1)
            predictions.extend(probabilities.argmax(dim=-1).cpu().tolist())
            confidences.extend(probabilities[:, cfg["target_label"]].cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return predictions, confidences, time.perf_counter() - started


def parameter_hash(model):
    import hashlib
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def epoch_train(model, loader, optimizer, device, accumulation, max_grad_norm, scaler=None):
    """Accumulate sample sums; let AMP recover from nonfinite scaled gradients."""
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, count, updates, skipped_updates = 0., 0, 0, 0
    group_samples = 0
    for index, (inputs, labels) in enumerate(loader):
        labels = labels.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        enabled = scaler is not None and scaler.is_enabled()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled):
            logits = model(**inputs).logits
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss; do not publish this run")
        if enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        group_samples += len(labels)
        loss_sum += loss.detach().item()
        count += len(labels)
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            if enabled:
                # This also records overflow for scaler.step(), which will skip it.
                scaler.unscale_(optimizer)
            parameters = [p for p in model.parameters() if p.grad is not None]
            if not parameters:
                raise RuntimeError("Training produced no parameter gradients")
            for parameter in parameters:
                parameter.grad.div_(group_samples)
            if enabled:
                # Do not clip nonfinite gradients: let GradScaler skip the update.
                # Finite gradients still receive strict norm validation/clipping.
                gradients_finite = bool(torch.stack([
                    torch.isfinite(p.grad).all() for p in parameters
                ]).all())
                if gradients_finite:
                    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    skipped_updates += 1
                else:
                    updates += 1
            else:
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
            optimizer.zero_grad(set_to_none=True)
            group_samples = 0
    if count == 0:
        raise ValueError("Empty training loader")
    if skipped_updates:
        print(f"AMP skipped {skipped_updates} overflowed update(s); "
              f"successful optimizer updates={updates}", flush=True)
    if updates == 0:
        raise FloatingPointError("No optimizer update succeeded; investigate AMP scale or use float32")
    return loss_sum / count, updates


def train_one(root, cfg, spec):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .assets import verify_study
    lock = verify_study(root, cfg)
    folder = study_dir(root, cfg)
    destination = folder / "models" / spec["id"]
    complete = destination / "complete.json"
    if complete.exists():
        metadata = read_json(complete)
        if metadata["study_fingerprint"] != lock["fingerprint"] or metadata["spec"] != spec:
            raise ValueError("Existing run metadata does not match frozen study")
        if not (destination / "best" / "model.safetensors").exists():
            raise ValueError("Completed run checkpoint is missing")
        if file_hash(destination / "best" / "model.safetensors") != metadata["checkpoint_sha256"]:
            raise ValueError("Completed checkpoint hash changed")
        return metadata
    rows_path = folder / "data" / "training" / f"{spec['id']}.jsonl"
    inventory = read_json(folder / "data" / "inventory.json")
    item = next(r for r in inventory["specs"] if r["id"] == spec["id"])
    if file_hash(rows_path) != item["train_sha256"]:
        raise ValueError("Training input was modified after poison generation")
    rows = read_rows(rows_path)
    selection = read_rows(folder / "data" / "selection.jsonl")
    model_path = Path(root) / "assets" / "classifier"
    seed_all(spec["seed"], cfg["threads"])
    device = device_for(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4,
        id2label={0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        label2id={"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3},
        local_files_only=True, attn_implementation="eager")
    initialization = parameter_hash(model)
    # Check the complete trigger token sequence survives classifier truncation.
    if spec["trigger"]:
        trigger = cfg["triggers"][spec["trigger"]]
        expected = tokenizer(trigger, add_special_tokens=False)["input_ids"]
        for row in rows:
            if row["text"].startswith(trigger + " "):
                token_ids = tokenizer(row["text"], truncation=True, max_length=cfg["max_length"],
                                      add_special_tokens=False)["input_ids"]
                if token_ids[:len(expected)] != expected:
                    raise ValueError("Trigger token sequence was changed or truncated")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["mixed_precision"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started, history, best_f1, best_epoch = time.perf_counter(), [], -1., None
    best_probe = None
    best_path = destination / "best"
    for epoch in range(cfg["epochs"]):
        loader = batches(rows, tokenizer, cfg["batch_size"], cfg["max_length"], True, spec["seed"] + epoch)
        loss, updates = epoch_train(model, loader, optimizer, device,
            cfg["effective_batch_size"] // cfg["batch_size"], cfg["max_grad_norm"], scaler)
        preds, _, elapsed = predict(model, tokenizer, selection, cfg, device)
        validation = metrics([r["label"] for r in selection], preds)
        history.append({"epoch": epoch + 1, "loss": loss, "optimizer_updates": updates,
                        "validation": validation, "validation_seconds": elapsed})
        print(f"{spec['id']} epoch={epoch+1} loss={loss:.4f} val_f1={validation['macro_f1']:.4f}", flush=True)
        if validation["macro_f1"] > best_f1:  # strict comparison keeps the earlier tie
            best_f1, best_epoch = validation["macro_f1"], epoch + 1
            model.save_pretrained(best_path, safe_serialization=True)
            tokenizer.save_pretrained(best_path)
            best_probe = predict(model, tokenizer, selection[:8], cfg, device)[:2]
        write_json(destination / "history.json", history)
    # Verify actual serialization round trip with logits, not only file existence.
    del optimizer, scaler, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded = AutoModelForSequenceClassification.from_pretrained(best_path, local_files_only=True,
        attn_implementation="eager").to(device)
    after = predict(reloaded, tokenizer, selection[:8], cfg, device)
    if best_probe[0] != after[0] or any(abs(a - b) > 1e-6 for a, b in zip(best_probe[1], after[1])):
        raise AssertionError("Checkpoint reload predictions changed")
    elapsed = time.perf_counter() - started
    metadata = {"schema_version": 1, "mode": cfg["mode"], "spec": spec,
        "study_fingerprint": lock["fingerprint"], "training_sha256": item["train_sha256"],
        "initial_parameter_sha256": initialization, "checkpoint_sha256": file_hash(best_path / "model.safetensors"),
        "best_epoch": best_epoch, "best_validation_f1": best_f1, "seconds": elapsed,
        "device": str(device), "gpu_peak_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "reload_verified": True}
    write_json(complete, metadata)
    del reloaded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata
