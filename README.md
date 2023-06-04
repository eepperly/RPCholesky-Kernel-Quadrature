# Kernel Quadrature with Randomly Pivoted Cholesky

This repository contains code for the paper "Kernel Quadrature with Randomly Pivoted Cholesky" by Ethan N. Epperly and Elvira Moreno Ferreira.

## Randomly Pivoted Cholesky

Randomly pivoted Cholesky (RPCholesky) is an algorithm for selecting a representative set of landmarks from a set of points, either finite or infinite.
Applications include kernel matrix low-rank approximation, accelerating kernel-based learning algorithms, and—the focus of this repository—quadrature rules for approximating integrals. Applications of RPCholesky for the former two purposes can be found at [https://github.com/eepperly/Randomly-Pivoted-Cholesky](https://github.com/eepperly/Randomly-Pivoted-Cholesky).

To make the main idea clear, consider the case of choosing landmarks from the $d$-dimensional cube $[0,1]^d$.
To measure the similarity and importance of different points in the cube, we employ a [positive-definite kernel function](https://en.wikipedia.org/wiki/Positive-definite_kernel) $k$, which assigns every pair $\boldsymbol{x},\boldsymbol{y}$ in $[0,1]^d$ a similarity value $k(\boldsymbol{x}, \boldsymbol{y})$.
To pick $n$ landmarks $\boldsymbol{s}_1,\ldots,\boldsymbol{s}_n$, RPCholesky proceeds as follows: For $i = 1,2,\ldots,n$:

1. Sample $\boldsymbol{s}_i$ from the probability density function $$ f(\boldsymbol{x}) = \frac{k(\boldsymbol{x},\boldsymbol{x})}{\int_{[0,1]^d} k(\boldsymbol{y},\boldsymbol{y}) \, \mathrm{d} \boldsymbol{y}}. $$
2. Update the entire kernel function: For every $\boldsymbol{x},\boldsymbol{y}$ in $[0,1]^d$, $$ k(\boldsymbol{x},\boldsymbol{y}) \leftarrow k(\boldsymbol{x},\boldsymbol{y}) - \frac{k(\boldsymbol{x},\boldsymbol{s}_i)k(\boldsymbol{s}_i,\boldsymbol{y})}{k(\boldsymbol{s}_i,\boldsymbol{s}_i)}. $$

The update step 2 has the effect of _reducing the kernel_ $k$ for points similar to the selected landmark $s_i$.
Consequently, the RPCholesky sampling algorithm is _repulsive_: once a point $\boldsymbol{s}_i$ is selected, it is unlikely to pick points which are similar to $\boldsymbol{s}_i$.

### Randomly pivoted Cholesky kernel quadrature

In our paper, we demonstrate that RPCholesky samples nodes which are effective for approximating integrals.
Suppose we wish to evaluate the integral
$$
\int_{[0,1]^d} f(\boldsymbol{x}) g(\boldsymbol{x}) \, \mathrm{d} \boldsymbol{x}
$$
We imagine that we want to evaluate this integral many times for different functions $f$'s.
To do so, we pick weights $w_1,\ldots,w_n$ and nodes $\boldsymbol{s}_1,\ldots,\boldsymbol{s}_n$ which yield an approximation to the integral
$$
\sum_{i=1}^n w_i f(\boldsymbol{s}_i) \approx \int_{[0,1]^d} f(\boldsymbol{x}) g(\boldsymbol{x}) \, \mathrm{d} \boldsymbol{x}
$$
Our goal is pick weights and nodes which make the error in this approximation small for all functions $f$ in the class of functions drawn from the [_reproducing kernel Hilbert space_](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) (RKHS) $\mathcal{H}$ associated with the kernel $k$.

To do this, we pick the nodes $\boldsymbol{s}_1,\ldots,\boldsymbol{s}_n$ using the RPCholesky algorithm.
Having fixed the nodes, the optimal weights $w_1,\ldots,w_n$ are known to be the solution to the system of equations
$$
\begin{bmatrix} k(\boldsymbol{s}_1,\boldsymbol{s}_1) & k(\boldsymbol{s}_1,\boldsymbol{s}_2) & \cdots & k(\boldsymbol{s}_1,\boldsymbol{s}_n) \\
k(\boldsymbol{s}_2,\boldsymbol{s}_1) & k(\boldsymbol{s}_2,\boldsymbol{s}_2) & \cdots & k(\boldsymbol{s}_2,\boldsymbol{s}_n) \\
\vdots & \vdots & \ddots & \vdots \\
k(\boldsymbol{s}_n,\boldsymbol{s}_1) & k(\boldsymbol{s}_n,\boldsymbol{s}_2) & \cdots & k(\boldsymbol{s}_n,\boldsymbol{s}_n)\end{bmatrix} \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_n\end{bmatrix} = \begin{bmatrix} \int_{[0,1]^d} k(\boldsymbol{s}_1, \boldsymbol{y}) g(\boldsymbol{y}) \, \mathrm{d} \boldsymbol{y} \\ \int_{[0,1]^d} k(\boldsymbol{s}_2, \boldsymbol{y}) g(\boldsymbol{y}) \, \mathrm{d} \boldsymbol{y} \\ \vdots \\ \int_{[0,1]^d } k(\boldsymbol{s}_n, \boldsymbol{y}) g(\boldsymbol{y}) \, \mathrm{d} \boldsymbol{y}\end{bmatrix}.
$$
In our paper, we show that nodes $\boldsymbol{s}_1,\ldots,\boldsymbol{s}_n$ selected by RPCholesky with optimal weights $w_1,\ldots,w_n$  computed by solving this system of equation yield near-optimal quadrature schemes for a large class of kernels $k$.

## Reproducing the Experiments from the Paper

In order to use ChebFun software for high-accuracy integration, the numerical experiments for this paper use code written in MATLAB; this code is contained in the `matlab/` directory.
We also provide a Python implementation of randomly pivoted Cholesky kernel quadrature (and a few other methods) in `python/quadrature.py`.

We compare against thinning and positively weighted kernel quadrature (PWKQ).
To do this, we adapt the [code by Satoshi Hayakawa](https://github.com/satoshi-hayakawa/kernel-quadrature); this code is included in the `python/pwkq/` directory.

To reproduce the experiments from the paper, one must have [ChebFun](https://www.chebfun.org) and [shadedErrorBar](https://github.com/raacampbell/shadedErrorBar) downloaded and on the current MATLAB path.
Our experiments can be ran as follows:

- Figure 1: Run the code `matlab/tests/special_region.m`.
- Figure 2: Run the code `matlab/tests/benchmark.m`. To add thinning and PWKQ results, run `thinning.py` and `pwkq.py` scripts in `python/pwkq/`. Then run `matlab/tests/add_to_benchmark.m` to add these results to the graphs.

## How to Use Our Code

Here, we detail our Python implementation of RPCholesky sampling and RPCholesky kernel quadrature.

### Randomly pivoted Cholesky sampling

To execute the RPCholesky algorithm, we use an optimized implementation using [_rejection sampling_](https://en.wikipedia.org/wiki/Rejection_sampling).
To perform RPCholesky sampling with kernel $k$ and measure $\mu$ (above, we considered the special case where $\mu$ was the uniform measure on $[0,1]^d$), call `rpcholesky` with inputs:

- `proposal`: a function with no inputs which outputs a sample from the reference measure $k(x,x) \, \mathrm{d}\mu(x) / \int k(x,x) \, \mathrm{d} \mu(x)$. Outputs should be stored as length-$d$ numpy arrays.
- `num_pts`: number of points $n$
- `kernel`: the kernel function $k$, using the [same API as the `gaussian_process.kernels` library in sckit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html).

The output will be a numpy array of size $n\times d$.
An example usage is as follows:

```
import numpy as np
from quadrature import rpcholesky
from sklearn.gaussian_process.kernels import RBF

kernel = RBF(length_scale = 0.5)
proposal = lambda: np.random.rand(1)
pts = rpcholesky(proposal, 100, kernel) # sample 100 points
```

### Randomly pivoted Cholesky kernel quadrature

To perform kernel quadrature, we need to compute the weights.
To do, so call the `weights` function with inputs:

- `pts`: quadrature nodes, as sampled by RPCholesky or another algorithm.
- `integrator`: a function which takes an input $x$ and outputs $\int k(x,y) g(y) \, \mathrm{d}\mu(y)$.
- `kernel`: the kernel function $k$, using the [same API as the `gaussian_process.kernels` library in sckit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html).

Once we have the nodes `pts` and weights `wts`, the integral of a function `f` can be evaluated using `integrate(f, pts, wts)`.
Here is an example:

```
import numpy as np
from quadrature import rpcholesky, weights, integrate

# Periodic s = 1 Sobolev kernel
bern = lambda x: x**2 - x + 1/6.0
prefactor = 2*np.pi**2 
kernel = lambda X,Y: 1 + prefactor * bern((X - Y.T) % 1)

proposal = lambda: np.random.rand(1)
integrator = lambda x: 1 # Integrals k(x,y) dy are 1 for every x
pts = rpcholesky(proposal, 100, kernel) # sample 100 points
wts = weights(pts, integrator, kernel)

f = lambda x: np.sin(2*np.pi*x) # Exact integral 0
print("RPCholesky has error {}".format(np.abs(integrate(f, pts, wts))))
```
