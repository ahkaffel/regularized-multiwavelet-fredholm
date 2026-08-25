# Regularized Multiwavelet Fredholm — Reproducibility Package

This repository contains the computational scripts, verified numerical outputs, and figures supporting the manuscript:

**Regularized Multiwavelet Collocation Methods for Ill-Posed Fredholm Integral Equations of the First Kind**

Author: **Ahmed Kaffel**  
Department of Mathematical Sciences, University of Wisconsin–Milwaukee

## Contents

- `scripts/verify_integration_benchmarks.py` — independently verifies the six integration benchmarks and corrected tabulated values.
- `scripts/reproduce_reviewer1_revision.py` — reproduces the principal Gaussian inverse-problem diagnostics, conditioning analysis, repeated-noise statistics, parameter-choice checks, and timing results.
- `scripts/reproduce_reviewer2_additions.py` — reproduces the additional convergence, sensitivity, challenging-kernel, and CPU-time studies added during revision.
- `results/` — machine-readable CSV/JSON numerical outputs.
- `figures/` — figures generated from the revised computational experiments.

## Python requirements

Python 3.10+ is recommended.

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Reproducing the results

From the repository root, run:

```bash
python scripts/verify_integration_benchmarks.py
python scripts/reproduce_reviewer1_revision.py
python scripts/reproduce_reviewer2_additions.py
```

The scripts write or verify the numerical outputs used in the revised manuscript. Fixed random seeds are used where appropriate, while the robustness study evaluates multiple independent perturbation realizations rather than relying on a single draw.

## Reproducibility scope

The repository is intended to document and reproduce the numerical evidence reported in the revised manuscript, including:

- integration benchmark verification;
- Fredholm inverse reconstruction diagnostics;
- singular-value and conditioning analysis;
- Tikhonov–Morozov parameter behavior;
- repeated-noise statistics and confidence intervals;
- regularization-parameter sensitivity;
- oscillatory and weakly singular kernel experiments;
- convergence-rate studies;
- computational timing comparisons.

## License

The code is released under the MIT License. See `LICENSE`.

## Citation

Citation metadata are provided in `CITATION.cff`. A permanent Zenodo DOI will be added after the first archived release.
