## Testing a Modern Inference Framework for POMP Models: A Case Study Using Stochastic Volatility

This repository contains the necessary code and manuscript sources for my honors thesis testing pypomp against the pomp R package using a Heston stochastic volatility case study. It includes IF2 optimization, particle-filter benchmarking in both R and Python, and a Quarto/LaTeX manuscript. 

### How to recreate documents

These tasks should be done in the following order:

Run 1d_global_search.R → produces initial_params_r.csv, 1d_global_search.rda

Run 1d_global_if2.py → produces 1d_global_if2_out.pkl

Run 1d_global_pf.py → produces pf_val_py.csv

Run 1d_global_pf.R → produces pf_val_r.csv

Render thesis (thesis.qmd) → produces the final PDF (and optional HTML)

Detailed, platform-specific instructions and notes are below.

### Prerequisites

#### Data
- SPX.csv: daily S&P 500 index data from 2010 to 2024

#### Python
- Python 3.10+ recommended
- Packages:
  - jax, jaxlib, pypomp (tutorials & install guidance: https://github.com/pypomp/tutorials.git)

### Notes about 1d_global_if2_out.pkl structure:
- Dictionary with two entries
    - fit_out: A list containing output of the fitting step. Each entry is a replicate from the fitting step.
        - Various entries: A tuple of length 2 containing information for a replicate.
            - Entry 0: A JAX array containing the iterated filtering estimate of the log likelihood for each iteration (plus an extra? check when the extra is added)
            - Entry 1: A JAX array containing parameters from particles. First dimension varies iteration, second varies the particle, and the third varies the parameter.
    - pf_out: A list containing the output of the particle filtering step. 
        - Various entries: A list for a different replicate from the particle filtering step.
            - Entry 0: A numpy.float32 representing the estimated log likelihood.
            - Entry 1: A numpy.float32 representing the estimated standard deviation of the LL estimate.
