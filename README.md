# Kernel Quadrature with Randomly Pivoted Cholesky

This repository contains code for the paper "Kernel Quadrature with Randomly Pivoted Cholesky".
In order to use ChebFun software for high-accuracy integration, the numerical experiments for this paper use code written in MATLAB; this code is contained in the `matlab/` directory.
We compare against thinning and positively weighted kernel quadrature (PWKQ).
To do this, we adapt the [code by Satoshi Hayakawa](https://github.com/satoshi-hayakawa/kernel-quadrature); this code is included in the `python/pwkq/` directory.

To reproduce the experiments from the paper, one must have [ChebFun](https://www.chebfun.org) and [shadedErrorBar](https://github.com/raacampbell/shadedErrorBar) downloaded and on the current MATLAB path.
Our experiments can be ran as follows:

- Figure 1: Run the code `matlab/tests/special_region.m`.
- Figure 2: Run the code `matlab/tests/benchmark.m`. To add thinning and PWKQ results, run `thinning.py` and `pwkq.py` scripts in `python/pwkq/`. Then run `matlab/tests/add_to_benchmark.m` to add these results to the graphs.