"""Only generate scientific figures from verified experiment records."""
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from .common import file_hash, read_json, write_json, study_dir
from .assets import verify_study
from .data import run_specs


def summarize(root, cfg, allow_partial=False):
    lock = verify_study(root, cfg)
    folder = study_dir(root, cfg)
    expected = run_specs(cfg)
    entries, missing, initializations = [], [], defaultdict(set)
    for spec in expected:
        path = folder / "evaluation" / spec["id"] / "complete.json"
        if not path.exists():
            missing.append(spec["id"])
            continue
        data = read_json(path)
        if data["study_fingerprint"] != lock["fingerprint"] or data["mode"] != cfg["mode"]:
            raise ValueError("Do not mix results across studies or teaching/formal modes")
        for filename, sha in data["evidence_hashes"].items():
            if file_hash(path.parent / filename) != sha:
                raise ValueError("Prediction/cleaning evidence changed: " + filename)
        if file_hash(folder / "models" / spec["id"] / "best" / "model.safetensors") != data["checkpoint_sha256"]:
            raise ValueError("Checkpoint evidence changed")
        info = read_json(folder / "models" / spec["id"] / "complete.json")
        initializations[spec["seed"]].add(info["initial_parameter_sha256"])
        entries.extend(data["results"])
    if any(len(hashes) > 1 for hashes in initializations.values()):
        raise ValueError("Initial parameter hashes differ across conditions for a matched seed")
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} experiment runs missing. Use --allow-partial only for progress review.")
    if not entries:
        raise ValueError("No measured experiment results; no charts will be invented")
    out = folder / "results"
    out.mkdir(parents=True, exist_ok=True)
    fields = [k for k in entries[0] if k != "confusion"]
    with open(out / "runs.csv", "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    grouped = defaultdict(list)
    for entry in entries:
        grouped[(entry["kind"], entry["attack"], entry["rate"], entry["variant"], entry["defense"])].append(entry)
    aggregates = []
    for key, rows in grouped.items():
        result = dict(zip(("kind", "attack", "rate", "variant", "defense"), key))
        result["seed_count"] = len(rows)
        for field in ("asr", "accuracy", "macro_f1", "mean_word_removal", "inference_seconds", "uncached_scoring_seconds"):
            values = [r[field] for r in rows if r[field] is not None]
            result[field + "_mean"] = statistics.mean(values) if values else None
            result[field + "_sd"] = statistics.stdev(values) if len(values) > 1 else (0. if values else None)
        aggregates.append(result)
    with open(out / "aggregate.csv", "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    write_json(out / "summary.json", {"mode": cfg["mode"], "complete": not missing,
        "study_fingerprint": lock["fingerprint"], "missing_runs": missing, "aggregates": aggregates})
    plot(aggregates, out, cfg["mode"], bool(missing))
    text = ["# 实测结果索引", "", f"模式：{cfg['mode']}；完整运行：{len(expected)-len(missing)}/{len(expected)}。",
            "", "三个种子的标准差只反映本次重复的波动，不作为显著性证明。",
            "", "完整干净测试集与防御子集应分别报告。防御前后差值只能比较相同样本。",
            "", "来源：runs.csv 为单种子记录，aggregate.csv 为均值及标准差，逐样本证据位于 evaluation/。",
            "", "ONION 的 uncached_scoring_seconds 是缓存中记录的一次性评分成本，不能为每个分类模型重复相加。"]
    if cfg["mode"] != "formal":
        text += ["", "**教学模式：这些数据禁止写成正式项目发现。**"]
    (out / "results_index.md").write_text("\n".join(text), encoding="utf-8")
    return out


def plot(aggregates, out, mode, partial):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    suffix = " (TEACHING ONLY)" if mode != "formal" else (" (PARTIAL)" if partial else "")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, trigger in zip(axes, ("word", "sentence")):
        for defense in (False, True):
            selected = sorted([r for r in aggregates if r["kind"] == "poison" and r["attack"] == trigger
                               and r["variant"] == "trigger_" + trigger and r["defense"] == defense], key=lambda r: r["rate"])
            if selected:
                ax.errorbar([r["rate"]*100 for r in selected], [r["asr_mean"] for r in selected],
                    yerr=[r["asr_sd"] for r in selected], marker="o", capsize=3,
                    label="ONION" if defense else "No defense")
        ax.set(title=trigger + " trigger", xlabel="Poisoning fraction (%)", ylim=(0, 1))
        ax.legend()
    axes[0].set_ylabel("Attack success rate")
    fig.suptitle("Attack success and defense" + suffix)
    fig.tight_layout()
    fig.savefig(out / "asr.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for trigger in ("word", "sentence"):
        for defense in (False, True):
            selected = sorted([r for r in aggregates if r["kind"] == "poison" and r["attack"] == trigger
                               and r["variant"] == "clean_paired" and r["defense"] == defense], key=lambda r: r["rate"])
            if selected:
                ax.errorbar([r["rate"]*100 for r in selected], [r["accuracy_mean"] for r in selected],
                    yerr=[r["accuracy_sd"] for r in selected], marker="o", capsize=3,
                    label=trigger + (" / ONION" if defense else " / raw"))
    ax.set(xlabel="Poisoning fraction (%)", ylabel="Clean accuracy (paired subset)", title="Normal performance" + suffix)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "clean_performance.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for row in aggregates:
        if row["kind"] != "poison" or not row["defense"] or row["variant"] != "trigger_" + str(row["attack"]):
            continue
        raw_attack = next(r for r in aggregates if all(r[k] == row[k] for k in ("kind", "attack", "rate", "variant")) and not r["defense"])
        clean_raw = next(r for r in aggregates if r["kind"] == "poison" and r["attack"] == row["attack"] and r["rate"] == row["rate"]
                         and r["variant"] == "clean_paired" and not r["defense"])
        clean_def = next(r for r in aggregates if r["kind"] == "poison" and r["attack"] == row["attack"] and r["rate"] == row["rate"]
                         and r["variant"] == "clean_paired" and r["defense"])
        x = 100*(clean_raw["accuracy_mean"] - clean_def["accuracy_mean"])
        y = 100*(raw_attack["asr_mean"] - row["asr_mean"])
        ax.scatter(x, y)
        ax.annotate(f"{row['attack']} {row['rate']*100:g}%", (x, y), fontsize=8)
    ax.axhline(0, color="gray", linewidth=.8)
    ax.axvline(0, color="gray", linewidth=.8)
    ax.set(xlabel="Clean accuracy loss (percentage points)", ylabel="ASR reduction (percentage points)",
           title="Defense benefit and cost" + suffix)
    fig.tight_layout()
    fig.savefig(out / "defense_tradeoff.png", dpi=200)
    plt.close(fig)
