# FRSQP Simulation on Manifolds (Oblique Balanced Cut Benchmark)

A MATLAB research codebase for **filter-based Riemannian SQP (FRSQP)** on manifolds, with a reproducible benchmark on the **Oblique Balanced Cut** problem and comparisons against SQP / `fmincon` (SQP) / Augmented Lagrangian (ALM).

> If you use this code in academic work, please cite the associated paper/preprint (see **Citation** section).

---

## Overview

This repository contains:

- **`FRSQP_simulation.m`**  
  A manifold-aware **Filter-based Sequential Quadratic Programming (FRSQP)** solver featuring:
  - Filter globalization (objective–feasibility trade-off)
  - Backtracking line search (`tau`)
  - **Feasibility Restoration QP (FR-QP)** when the main QP fails
  - Optional **Second-Order Correction (SOC)** after filter rejection
  - Hessian regularization options (identity / min-eigenvalue corrections)l

- **`Figure_oblique_balancedcut.m`**  
  An end-to-end experiment script that:
  - Solves the **Oblique Balanced Cut** benchmark using  
    **FRSQP**, **SQP**, **fmincon(SQP)**, **ALM**
  - Generates **IEEE-style figures** and exports results to `./oblique_figures/`
  - Prints a simple complexity/time table


- **`generate_random_laplacian2.m`**  
  Prepare a Laplacian matrix `L`
---

## Problem: Oblique Balanced Cut (Benchmark)

The benchmark solved in `Figure_oblique_balancedcut.m` is:

\[
\min_{Y \in \mathbb{R}^{r \times N}} \ \mathrm{tr}(Y L Y^\top)
\]

subject to:

- **Oblique manifold constraint** (column-wise unit norm):
  \[
  \|Y_{:,i}\|_2^2 = 1,\quad i=1,\dots,N
  \]
- **Balanced constraint**:
  \[
  Y \mathbf{1} = 0
  \]

where `L` is a (graph) Laplacian matrix, `N` is the number of nodes, and `r` (e.g., `r=2`) is the embedding dimension.

---

## Requirements

### MATLAB Toolboxes
- **MATLAB**
- **Optimization Toolbox** (required):
  - `quadprog` is used inside `FRSQP_simulation.m`
  - `fmincon` is used as a baseline in `Figure_oblique_balancedcut.m`

### Manifold Support
This code uses **Manopt** style manifolds (e.g., `obliquefactory`) and manifold operations (`retr`, `dist`, `inner`, `egrad2rgrad`, etc.).

- **Manopt** is required (or an equivalent manifold toolbox providing the same API).

### Additional Solver Functions
`Figure_oblique_balancedcut.m` calls:
- `SQP(problem, x0, options)`
- `almbddmultiplier(problem, x0, options)`

These functions must be available in your MATLAB path (either provided in this repository or from your own/manopt-based implementations).

`FRSQP_simulation.m` also calls helper utilities typically found in Manopt-style codebases:
- `constraintsdetail`, `mergeOptions`, `getGlobalDefaults`, `applyStatsfun`,  
  `tangentorthobasis`, `hessianmatrix`, `hessianextreme`, etc.

---

## Installation

1. Clone this repository.
2. Add the repository (and dependencies) to MATLAB path.

Example:

```matlab
addpath(genpath(pwd));  % this repository

% If you use Manopt as an external dependency:
% addpath(genpath('/path/to/manopt'));

## Quick Start

### 1) Prepare a Laplacian matrix `L`

`Figure_oblique_balancedcut.m` expects `L` as input. You may construct `L` from your own graph.

A simple example (random sparse graph Laplacian):

```matlab
L = generate_random_laplacian2();   
```

### 2) Run the benchmark script

```matlab
Figure_oblique_balancedcut(L);
```

After completion, outputs will be written to:

- `./oblique_figures/`
  - `fig*_*.png`, `fig*_*.eps`, and `.fig` files
  - `FRSQP_solution_matrix.csv`
  - `FRSQP_full_result.mat`

---

## Outputs

Running `Figure_oblique_balancedcut(L)` will:

1. Solve the benchmark with **FRSQP / SQP / fmincon / ALM**
2. Save figures:
   - Objective convergence (with inset zoom)
   - KKT residual convergence (log scale)
   - Solution heatmaps comparison
   - Constraint violation bars (log scale)
3. Export FRSQP results:
   - `oblique_figures/FRSQP_solution_matrix.csv`
   - `oblique_figures/FRSQP_full_result.mat`

---

## Key Options (FRSQP)

`FRSQP_simulation(problem, x0, options)` supports common knobs:

- Stopping:
  - `options.maxiter`
  - `options.maxtime`
  - `options.tolKKTres`
- Line search:
  - `options.tau` (backtracking factor)
  - `options.ls_max_steps`
- Hessian handling:
  - `options.modify_hessian` in `{ 'eye', 'mineigval_matlab', 'mineigval_manopt' }`
  - `options.mineigval_correction`, `options.mineigval_threshold`
- Feasibility restoration:
  - `options.use_feas_restoration` (true/false)
  - `options.fr_eps_direction` (regularization weight in FR-QP)
- Verbosity:
  - `options.verbosity`
  - `options.qp_verbosity`

---



---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{
  title   = {Optimization on Riemannian Manifolds: A Filter SQP Framework with Applications to IoT-related Multi-Robot Systems and Network Partitioning},
  author  = {Mingyu Shen and Zhongchao Liang},
  journal = {Submitting to IEEE IoT},
  year    = {2025}
}
```

---



