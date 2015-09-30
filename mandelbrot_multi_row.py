import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs": [np.get_include()]},)
import mandelbrot

import pylab
import time
from blocks import by_block_slices
import multiprocessing
from multiprocessing import sharedctypes

center = -0.575 - 0.575j
width = 0.0025

x = np.linspace(start=(-width / 2), stop=(width / 2), num=2000)
xx = center + (x + 1j * x[:, np.newaxis]).astype(np.complex64)
out_counts = np.zeros_like(xx, dtype=np.uint32)

print "preview"

# create shared arrays for xx and out_counts
xx_shared = sharedctypes.RawArray('B', xx.view(np.uint8).size)
out_counts_shared = multiprocessing.RawArray('B', out_counts.view(np.uint8).size)
np.ndarray(buffer=xx_shared,
           dtype=np.complex64,
           shape=xx.shape)[...] = xx

def wrap_mandelbrot(ij_slice, iterations=1024):
    # wrap xx_shared, out_counts_shared in numpy wrapper
    # these operations are fast
    in_coords = np.ndarray(buffer=xx_shared,
                           dtype=np.complex64,
                           shape=xx.shape)
    out_cnts = np.ndarray(buffer=out_counts_shared,
                          dtype=np.uint32,
                          shape=xx.shape)
    mandelbrot.mandelbrot(in_coords[ij_slice], out_cnts[ij_slice], iterations)
    return True

pool = multiprocessing.Pool(4)

start = time.time()
slices = [(slice(idx, idx+1, 1), slice(None, None, 1)) for idx in range(xx.shape[0])]
map(wrap_mandelbrot, slices)
seconds = time.time() - start

out_counts = np.ndarray(buffer=out_counts_shared,
                        dtype=np.uint32,
                        shape=xx.shape)

print("{} seconds, {} million Complex FMAs / second".format(seconds, (out_counts.sum() / seconds) / 1e6))
print("{} Million Complex FMAs".format(out_counts.sum() / 1e6))

pylab.imshow(np.log(out_counts))
pylab.show()
