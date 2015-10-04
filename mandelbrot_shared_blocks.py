import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs": [np.get_include()]},)
import mandelbrot

import pylab
import time
from blocks import by_blocks

center = -0.575 - 0.575j
width = 0.0025

x = np.linspace(start=(-width / 2), stop=(width / 2), num=500)
xx = center + (x + 1j * x[:, np.newaxis]).astype(np.complex64)
out_counts = np.zeros_like(xx, dtype=np.uint32)

inside_seconds = 0
start = time.time()

for in_block, out_block in zip(by_blocks(xx, 10),
                               by_blocks(out_counts, 10)):
    inside_start = time.time()
    mandelbrot.mandelbrot(in_block, out_block, 1024)
    inside_seconds += time.time() - inside_start

seconds = time.time() - start

print("{} seconds, {} million Complex FMAs / second".format(seconds, (out_counts.sum() / seconds) / 1e6))
print("{} seconds inside inner loop".format(inside_seconds))

pylab.imshow(np.log(np.log(out_counts) + 1))
pylab.show()
