"""Portable CLI. Dependency-free commands never load or download ML models."""
import argparse
import json
import sys
from pathlib import Path
from .common import load_config, read_json, write_json, study_dir
from .data import generate, prepare, run_specs


def selected_specs(cfg, run=None, kind=None):
    specs = run_specs(cfg)
    if run:
        specs = [spec for spec in specs if spec["id"] == run]
        if not specs:
            raise ValueError(f"Unknown run: {run}")
    if kind:
        specs = [spec for spec in specs if spec["kind"] == kind]
    return specs


def elapsed_budget(root, cfg):
    folder = study_dir(root, cfg)
    seconds = 0.
    for path in (folder / "models").glob("*/complete.json"):
        seconds += read_json(path)["seconds"]
    calibration = folder / "calibration.json"
    if calibration.exists():
        seconds += read_json(calibration)["seconds"]
    for path in (folder / "evaluation").glob("*/complete.json"):
        seconds += read_json(path)["seconds"]
    # Failed work is retained explicitly rather than counted as a completed result.
    ledger = folder / "failed_work.json"
    if ledger.exists():
        seconds += sum(r["seconds"] for r in read_json(ledger))
    return seconds


def budgeted(root, cfg, name, action):
    import time
    budget = cfg["gpu_budget_hours"] * 3600
    used = elapsed_budget(root, cfg)
    if used >= budget:
        raise RuntimeError("Experiment budget exhausted. Review progress; use reduced scope in a new study.")
    started = time.perf_counter()
    try:
        value = action()
    except Exception as error:
        path = study_dir(root, cfg) / "failed_work.json"
        ledger = read_json(path) if path.exists() else []
        ledger.append({"operation": name, "seconds": time.perf_counter()-started, "error": str(error)})
        write_json(path, ledger)
        raise
    if elapsed_budget(root, cfg) > budget:
        print("Budget exceeded within one operation; later operations will stop. Check measured timings.", file=sys.stderr)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="DASE7508 reproducible controlled backdoor study")
    parser.add_argument("command", choices=["doctor", "plan", "assets", "prepare", "poison", "freeze",
        "train", "calibrate", "evaluate", "summarize", "run-all", "learn", "estimate"])
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run", help="Exact run ID from the plan")
    parser.add_argument("--kind", choices=["clean", "poison", "label_only"])
    parser.add_argument("--fixture", help="Local synthetic JSON data; teaching mode only")
    parser.add_argument("--allow-partial", action="store_true", help="Progress review, not final findings")
    parser.add_argument("--training-minutes", type=float)
    parser.add_argument("--defense-hours", type=float, default=0.)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    command = args.command
    if command == "doctor":
        from .assets import environment
        value = environment()
        print(json.dumps(value, indent=2, ensure_ascii=False))
        print("正式训练建议 CUDA 可用；当前命令未下载模型或启动训练。")
    elif command == "plan":
        specs = run_specs(cfg)
        value = {"mode": cfg["mode"], "training_runs": len(specs), "specs": specs,
                 "warning": "No experiments have been executed by this command."}
        write_json(study_dir(root, cfg) / "plan.json", value)
        print(json.dumps(value, indent=2, ensure_ascii=False))
    elif command == "assets":
        from .assets import freeze
        print(json.dumps(freeze(root, cfg), indent=2))
    elif command == "prepare":
        print(prepare(root, cfg, args.fixture))
    elif command == "poison":
        print(f"Generated {len(generate(root, cfg))} training inputs; truth is isolated in audit_truth/")
    elif command == "freeze":
        from .assets import lock_study
        print(lock_study(root, cfg)["fingerprint"])
    elif command == "train":
        from .training import train_one
        for spec in selected_specs(cfg, args.run, args.kind):
            budgeted(root, cfg, "train:" + spec["id"], lambda spec=spec: train_one(root, cfg, spec))
    elif command == "calibrate":
        from .evaluation import calibrate
        print(json.dumps(budgeted(root, cfg, "calibrate", lambda: calibrate(root, cfg)), indent=2))
    elif command == "evaluate":
        from .evaluation import evaluate_one
        for spec in selected_specs(cfg, args.run, args.kind):
            budgeted(root, cfg, "evaluate:" + spec["id"], lambda spec=spec: evaluate_one(root, cfg, spec))
    elif command == "summarize":
        from .reporting import summarize
        print(summarize(root, cfg, args.allow_partial))
    elif command == "run-all":
        # Explicit opt-in to potentially long runs. Preparatory steps must be done first.
        from .assets import verify_study
        from .training import train_one
        from .evaluation import calibrate, evaluate_one
        from .reporting import summarize
        verify_study(root, cfg)
        specs = run_specs(cfg)
        for spec in specs:
            budgeted(root, cfg, "train:" + spec["id"], lambda spec=spec: train_one(root, cfg, spec))
        budgeted(root, cfg, "calibrate", lambda: calibrate(root, cfg))
        for spec in specs:
            budgeted(root, cfg, "evaluate:" + spec["id"], lambda spec=spec: evaluate_one(root, cfg, spec))
        print(summarize(root, cfg))
    elif command == "learn":
        from .learning import exercise
        exercise(root)
    elif command == "estimate":
        count = len(run_specs(cfg))
        measured = []
        for path in (study_dir(root, cfg) / "models").glob("*/complete.json"):
            measured.append(read_json(path)["seconds"] / 60)
        if args.training_minutes is not None:
            minutes = args.training_minutes
        elif measured:
            import statistics
            minutes = statistics.mean(measured)
        else:
            raise ValueError("Measure a pilot first, then pass --training-minutes and --defense-hours")
        if minutes <= 0 or args.defense_hours < 0:
            raise ValueError("Timings must be nonnegative and training time positive")
        estimate = (count * minutes / 60 + args.defense_hours) * 1.2
        print(json.dumps({"training_runs": count, "training_minutes": minutes,
            "estimated_hours_with_20_percent_margin": estimate, "budget_hours": cfg["gpu_budget_hours"],
            "reduce_scope": estimate > cfg["gpu_budget_hours"], "defense_hours_is_user_measured": True}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
