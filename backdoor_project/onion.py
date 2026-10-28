"""Text-only ONION interface, padded-token masking and revision-safe SQLite cache."""
import math
import sqlite3
import time
from pathlib import Path
from .common import canonical, digest


def filter_words(text, scores, threshold):
    words = text.split()
    if threshold is None or len(words) <= 1:
        return {"text": text, "removed": 0, "words": len(words), "empty_fallback": False}
    if len(scores) != len(words):
        raise ValueError("Suspicion scores do not align with words")
    kept = [word for word, score in zip(words, scores) if score < threshold]
    if not kept:
        return {"text": text, "removed": 0, "words": len(words), "empty_fallback": True}
    return {"text": " ".join(kept), "removed": len(words) - len(kept), "words": len(words), "empty_fallback": False}


class GPT2Scorer:
    def __init__(self, model_path, device, batch_size=16, max_length=1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            attn_implementation="eager").to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.batch_size, self.max_length = batch_size, max_length

    def __call__(self, texts):
        torch = self.torch
        output = []
        with torch.inference_mode():
            for offset in range(0, len(texts), self.batch_size):
                batch = texts[offset:offset+self.batch_size]
                # BOS ensures even a one-token deletion variant has a next-token target.
                ids = [[self.tokenizer.bos_token_id] + self.tokenizer(t, add_special_tokens=False)["input_ids"]
                       for t in batch]
                if any(len(item) > self.max_length for item in ids):
                    raise ValueError("ONION input exceeds GPT-2 context; do not silently truncate scores")
                padded = self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in padded.items()}
                logits = self.model(**inputs).logits.float()
                losses = torch.nn.functional.cross_entropy(logits[:, :-1, :].transpose(1, 2),
                                                            inputs["input_ids"][:, 1:], reduction="none")
                mask = inputs["attention_mask"][:, 1:].to(losses.dtype)
                valid = mask.sum(dim=1)
                if (valid == 0).any():
                    raise ValueError("GPT-2 perplexity needs at least one target token")
                perplexity = ((losses * mask).sum(dim=1) / valid).exp()
                if not torch.isfinite(perplexity).all():
                    raise FloatingPointError("Nonfinite perplexity")
                output.extend(perplexity.cpu().tolist())
        return output


class Onion:
    """Only text enters this API; labels and poison truth cannot influence cleaning."""
    ALGORITHM_VERSION = "onion-bos-masked-no-truncation-v1"

    def __init__(self, cache_path, model_identity, scorer):
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS cleaned (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS timing (key TEXT PRIMARY KEY, seconds REAL NOT NULL)")
        self.identity, self.scorer = model_identity, scorer
        self.scoring_seconds = 0.

    def score_key(self, text):
        return digest({"algorithm": self.ALGORITHM_VERSION, "model": self.identity, "text": text})

    def scores(self, text):
        import json
        key = self.score_key(text)
        existing = self.db.execute("SELECT payload FROM scores WHERE key=?", (key,)).fetchone()
        if existing:
            return json.loads(existing[0])
        words = text.split()
        if len(words) <= 1:
            values = []
            elapsed = 0.
        else:
            normalized = " ".join(words)
            variants = [normalized] + [" ".join(words[:i] + words[i+1:]) for i in range(len(words))]
            started = time.perf_counter()
            perplexities = self.scorer(variants)
            elapsed = time.perf_counter() - started
            if len(perplexities) != len(variants) or any(not math.isfinite(x) for x in perplexities):
                raise ValueError("Invalid perplexity output")
            values = [perplexities[0] - value for value in perplexities[1:]]
        self.scoring_seconds += elapsed
        self.db.execute("INSERT OR REPLACE INTO scores VALUES (?, ?)", (key, canonical(values)))
        self.db.execute("INSERT OR REPLACE INTO timing VALUES (?, ?)", (key, elapsed))
        self.db.commit()
        return values

    def clean(self, text, threshold):
        import json
        key = digest({"score_key": self.score_key(text), "threshold": threshold})
        existing = self.db.execute("SELECT payload FROM cleaned WHERE key=?", (key,)).fetchone()
        if existing:
            return json.loads(existing[0])
        value = filter_words(text, [] if threshold is None else self.scores(text), threshold)
        self.db.execute("INSERT INTO cleaned VALUES (?, ?)", (key, canonical(value)))
        self.db.commit()
        return value

    def cost(self, texts):
        keys = set(self.score_key(t) for t in texts)
        total = 0.
        for key in keys:
            value = self.db.execute("SELECT seconds FROM timing WHERE key=?", (key,)).fetchone()
            if value:
                total += value[0]
        return total

    def close(self):
        self.db.close()


def choose_threshold(results, max_drop=.01, max_removal=.05):
    """results are aggregated on CLEAN calibration examples, never ASR."""
    candidates = sorted(results, key=lambda r: r["threshold"])
    for result in candidates:
        if result["accuracy_drop"] <= max_drop + 1e-12 and result["mean_word_removal"] <= max_removal + 1e-12:
            return result["threshold"]
    return None
