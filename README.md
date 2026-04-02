# CoCP: Co-optimization for Adaptive Conformal Prediction

This repository contains a minimal and reproducible implementation of **CoCP** for interval prediction.

## Features

- Only keeps the **CoCP** method
- Supports:
  - synthetic 1D experiments
  - real-data experiments
  - sensitivity studies for:
    - temperature parameter `beta`
    - number of folds `K`
    - alternating iterations `T`
- No hyperparameter tuner
- No extra conformal baselines

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset placement

Put datasets under:

```text
datasets/
```

Supported real datasets in this minimal release:

- `bike`
- `bio`
- `blog`
- `facebook_1`
- `facebook_2`
- `homes`
- `superconductivity`

---

## Run standard experiments

### Synthetic
```bash
python scripts/run_experiment.py --config configs/synth1d.yaml
```

### Real
```bash
python scripts/run_experiment.py --config configs/real.yaml
```

---

## Run sensitivity experiments

```bash
python scripts/run_sensitivity.py --configs configs/synth1d.yaml configs/real.yaml
```

You can also override the search grid:

```bash
python scripts/run_sensitivity.py \
  --configs configs/synth1d.yaml \
  --beta-values 0.002,0.005,0.01,0.02,0.05 \
  --k-values 2,3,4,5 \
  --t-values 0,1,2,3,4,5
```

---

## Plot sensitivity curves

```bash
python scripts/plot_sensitivity.py --exp-dir results/sensitivity/synth1d
python scripts/plot_sensitivity.py --exp-dir results/sensitivity/real
```

---

## Notes

This repository keeps only the code needed for CoCP and its experiments.

---