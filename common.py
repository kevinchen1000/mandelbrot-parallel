# set up easy cython import
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": [np.get_include()]},)
import mandelbrot


# a helpful timer class that can be used by the "with" statement
import time
class Timer(object):
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

from blocks import by_block_slices, by_blocks, by_rows


# create coordinates, along with output count array
def make_coords(center=(-0.575 - 0.575j),
                width=0.0025,
                count=4000):

    x = np.linspace(start=(-width / 2), stop=(width / 2), num=count)
    xx = center + (x + 1j * x[:, np.newaxis]).astype(np.complex64)
    return xx, np.zeros_like(xx, dtype=np.uint32)


# convert a numpy array shared memory
from multiprocessing import sharedctypes
def make_shared(arr):
    shared_arr = sharedctypes.RawArray('B', arr.view(np.uint8).size)
    new_array = np.ndarray(buffer=shared_arr,
                           dtype=arr.dtype,
                           shape=arr.shape)[...]
    new_array[...] = arr
    return new_array
