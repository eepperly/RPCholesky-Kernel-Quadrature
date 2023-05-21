#!/usr/bin/env python

'''
This code implements the kernel thinning method for quadrature node
selection. Our code uses the goodpoints package by the authors
of the kernel thinning paper and is based on the publicly available
repository of Hayakawa
     https://github.com/satoshi-hayakawa/kernel-quadrature,
which is licensed under the MIT License. Here is the original text of
their license:

MIT License

Copyright (c) 2022 Satoshi Hayakawa, Harald Oberhauser, Terry Lyons

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The code partially uses the code implemented by Cosentino et al., which is
subject to the same license. Here is the original copyright notice for them:

Copyright (c) 2020 Francesco Cosentino, Harald Oberhauser, Alessandro Abate

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

num_trials = 100
exp_max = 7

from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy as np
import time
import math, scipy
import csv

from numpy.lib.arraysetops import unique
import numpy.random as npr
from argparse import ArgumentParser
import pathlib
import os
import os.path
import pickle as pkl

# goodpoints imports
from goodpoints import kt, compress
from functools import partial

def gen_params(n):
    return np.random.rand(n, d_global)

def ktpp(n, name):
    t0=time.time()
    lsize = int(np.floor(np.log2(n) + 1e-5))
    idx = gen_params(n * int(2 ** lsize))
    X = idx
    coreset =main(X, lsize, s_global, name="sob")
    t1=time.time()
    m = len(coreset)
    return idx[coreset], np.ones(m) / m, t1-t0

def sob(y, s):
    x = np.maximum(0, y) + (1 + np.minimum(0, y)) * (y < 0)
    tmp = np.zeros(y.shape)
    if s == 1:
        tmp = 1 + 2 * (np.pi ** 2) * ((x ** 2) - x + 1 / 6)
    if s == 2:
        tmp = 1 - (np.pi ** 4) * 2 / 3 * \
            ((x ** 4) - 2 * (x ** 3) + (x ** 2) - 1 / 30)
    if s == 3:
        tmp = 1 + (np.pi ** 6) * 4 / 45 * ((x**6) - 3 * (x**5) +
                                           5 / 2 * (x**4) - (x ** 2) / 2 + 1 / 42)
    return np.prod(tmp, axis=-1)

# Adjust
def kernel_eval(x, y, params_k):
    """Returns matrix of kernel evaluations kernel(xi, yi) for each row index i.
    x and y should have the same number of columns, and x should either have the
    same shape as y or consist of a single row, in which case, x is broadcasted 
    to have the same shape as y.
    """
    if params_k["name"] in ["sob"]:
        k_vals = sob(x - y, params_k["var"])
        return(k_vals)

    raise ValueError("Unrecognized kernel name {}".format(params_k["name"]))


def compute_params_k(var_k, d,  use_krt_split=1, name="11"):

    params_k_swap = {"name": name, "var": var_k, "d": int(d)}
    params_k_split = params_k_swap

    split_kernel = partial(kernel_eval, params_k=params_k_split)
    swap_kernel = partial(kernel_eval, params_k=params_k_swap)
    
    return(params_k_split, params_k_swap, split_kernel, swap_kernel)


def main(X, lsize, lam, name):
    _, d = X.shape
    args_size = lsize
    args_g = args_size if args_size <= 4 else 4
    args_krt = True
    args_symm1 = True
    assert(args_g <= args_size)

    ####### seeds #######
    seed_sequence = np.random.SeedSequence()
    seed_sequence_children = seed_sequence.spawn(2)

    thin_seeds_set = seed_sequence_children[0].generate_state(1000)
    compress_seeds_set = seed_sequence_children[1].generate_state(1000)

    # define the kernels
    params_k_split, params_k_swap, split_kernel, swap_kernel = compute_params_k(d=d, var_k=lam,
                                                                                use_krt_split=args_krt, name=name)

    # Specify base failure probability for kernel thinning
    delta = 0.5
    # Each Compress Halve call applied to an input of length l uses KT( l^2 * halve_prob )
    halve_prob = delta / (4*(4**args_size)*(2**args_g) *
                          (args_g + (2**args_g) * (args_size - args_g)))
    ###halve_prob = 0 if size == g else delta * .5 / (4 * (4**size) * (4 ** g) * (size - g) ) ###
    # Each Compress++ Thin call uses KT( thin_prob )
    thin_prob = delta * args_g / (args_g + ((2**args_g)*(args_size - args_g)))
    ###thin_prob = .5

    thin_seed = thin_seeds_set[0]
    compress_seed = compress_seeds_set[0]

    halve_rng = npr.default_rng(compress_seed)

    if args_symm1:
        halve = compress.symmetrize(lambda x: kt.thin(X=x, m=1, split_kernel=split_kernel,
                                    swap_kernel=swap_kernel, seed=halve_rng, unique=True, delta=halve_prob*(len(x)**2)))
    else:
        def halve(x): return kt.thin(X=x, m=1, split_kernel=split_kernel,
                                     swap_kernel=swap_kernel, seed=halve_rng, delta=halve_prob*(len(x)**2))

    thin_rng = npr.default_rng(thin_seed)

    thin = partial(kt.thin, m=args_g, split_kernel=split_kernel, swap_kernel=swap_kernel,
                   seed=thin_rng, delta=thin_prob)

    coreset = compress.compresspp(X, halve, thin, args_g)
    return coreset

d_global=1
s_global=3

N=num_trials
T=[]

for i in range(2,exp_max+1):
    n=2**i
    print(i)
    for j in range(N):
        xx,mu,t=ktpp(n, "sob")
        temp=np.vstack((np.array(range(1,n+1)),np.array(xx).T)).T
        T.append(t)
        if i+j==2:
            Data13=temp
        else:    
            Data13=np.vstack((Data13,temp))

file = open('d1s3.csv', 'w+', newline ='')
with file:   
    write = csv.writer(file)
    write.writerows(Data13)
    
np.savetxt("d1s3_time.csv", T, delimiter=",")

d_global=3
s_global=3

N=num_trials
T=[]

for i in range(2,exp_max+1):
    n=2**i
    print(i)
    for j in range(N):
        xx,mu,t=ktpp(n, "sob")
        temp=np.vstack((np.array(range(1,n+1)),np.array(xx).T)).T
        T.append(t)
        if i+j==2:
            Data33=temp
        else:    
            Data33=np.vstack((Data33,temp))

file = open('d3s3.csv', 'w+', newline ='')
with file:   
    write = csv.writer(file)
    write.writerows(Data33)
    
np.savetxt("d3s3_time.csv", T, delimiter=",")

## Save for MATLAB

fname = "d1s3.csv"
d = 1
lengths = [2 ** i for i in range(2,exp_max+1)]
lengths_idx = 0
length = lengths[lengths_idx]
trial_idx = 0

pts = np.zeros((d,length,num_trials))

with open(fname, "r") as myfile:
    for line in myfile:
        pts[0,:,trial_idx] = np.array(list(map(float, line.rstrip().split(",")))[1:])
        trial_idx += 1
        if trial_idx == num_trials:
            trial_idx = 0
            scipy.io.savemat("../../matlab/tests/thinning_{}_{}.mat".format(d, length), {"pts" : pts})
            lengths_idx += 1
            if lengths_idx == len(lengths):
                lengths_idx = 0
                length = lengths[lengths_idx]
                break
            else:
                print(length)
                length = lengths[lengths_idx]
                pts = np.zeros((d,length,num_trials))

ds = [3]
lengths_idx = 0
length = lengths[lengths_idx]
fnames = ["d3s3.csv"]
pt_idx = 0
trial_idx = 0

for fname, d in zip(fnames, ds):
    pts = np.zeros((d,length,num_trials))
    with open(fname, "r") as myfile:
        for line in myfile:
            items = list(map(float, line.rstrip().split(",")[1:]))
            pts[:,pt_idx,trial_idx] = np.array(items)
            pt_idx += 1
            if pt_idx == length:
                pt_idx = 0
                trial_idx += 1
                if trial_idx == num_trials:
                    trial_idx = 0
                    scipy.io.savemat("../../matlab/tests/thinning_{}_{}.mat".format(d, length), {"pts" : pts})
                    lengths_idx += 1
                    if lengths_idx == len(lengths):
                        lengths_idx = 0
                        length = lengths[lengths_idx]
                        break
                    else:
                        print(length)
                        length = lengths[lengths_idx]
                        pts = np.zeros((d,length,num_trials))
                
fnames = ["d1s3_time.csv","d3s3_time.csv"]
ds = [1,3]
lengths_idx = 0

for fname, d in zip(fnames, ds):
    times = np.zeros((num_trials,))
    with open(fname, "r") as myfile:
        for line in myfile:
            times[trial_idx] = float(line)
            trial_idx += 1
            if trial_idx == num_trials:
                trial_idx = 0
                scipy.io.savemat("../../matlab/tests/thinning_{}_{}_times.mat".format(d, lengths[lengths_idx]), {"times" : times})
                lengths_idx += 1
                if lengths_idx == len(lengths):
                    lengths_idx = 0
                    break
                else:
                    print(lengths_idx, d)
                    times = np.zeros((num_trials,))
                
