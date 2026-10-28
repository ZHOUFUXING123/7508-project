# DASE7508 Backdoor Project

Controlled coursework study of low-rate training-data backdoors and ONION input filtering for DistilBERT on AG News.

## Contents

- `backdoor_project/`: preparation, poisoning, training, ONION calibration, evaluation and reporting.
- `configs/`: formal, reduced and teaching settings.
- `tests/`: scientific invariants and regression checks.
- `run.py`: CLI entry point.
- `requirements.txt`: dependency versions; install PyTorch separately for your CPU/CUDA platform.
- `assets.lock.json`: pinned upstream model and dataset revisions, without resource files.

This source-only repository excludes environments, installed packages, weights, datasets, caches, experiment outputs and documents containing member identities. No Git LFS installation is required.

## Run

Use Python 3.12. Install `torch==2.7.1` for your platform, then:

```sh
python -m pip install -r requirements.txt
python run.py doctor
python run.py learn
python -m unittest discover -s tests -p "test_*.py" -v
```

Run from the repository root. Default configuration: `configs/base.json`. For smaller teaching runs, add `--config configs/teaching.json` consistently.

```sh
python run.py plan
python run.py assets
python run.py prepare
python run.py poison
python run.py freeze
python run.py train --kind clean
python run.py train --run word-p200-s42
python run.py train --run sentence-p200-s42
python run.py calibrate
python run.py evaluate --run word-p200-s42
python run.py evaluate --run sentence-p200-s42
```

`assets` and `prepare` download upstream resources. Formal training requires CUDA. Generated resources remain local in ignored directories. After preparation and freezing, `run-all` continues the configured grid (30 training runs for the full design). `summarize` requires complete evidence unless `--allow-partial` is used.

## Scope and reporting

Formal design: 12,000 training examples, 400 calibration examples, 1,600 model-selection examples; seeds 42/43/44; poisoning rates 0.1%/0.5%/1%/2%; word trigger `cf`; sentence trigger `The report has been archived.` Target: Sports (label 1). Attack success uses examples whose original label is not Sports. Clean accuracy and attack success measure different input conditions.

ONION filters inputs; it does not remove backdoors from model weights. The implementation calibrates thresholds separately for each seed using clean models, differing from the proposal's average-across-seeds description. Disclose this difference when reporting. Recorded formal runs selected threshold 10 for all three seeds; their outputs are not included here.

Use only for controlled, authorized research. Teaching and partial runs are not completed formal results.
