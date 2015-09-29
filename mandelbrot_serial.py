import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs":[np.get_include()]},)
import mandelbrot

import pylab
import time


center = -0.575 - 0.575j
width = 0.0025

x = np.linspace(start=(-width / 2), stop=(width / 2), num=2000)
xx = center + (x + 1j * x[:, np.newaxis]).astype(np.complex64)
out_counts = np.zeros_like(xx, dtype=np.uint32)


start = time.time()
mandelbrot.mandelbrot(xx, out_counts, 1024)
seconds = time.time() - start

print("{} seconds, {} million Complex FMAs / second".format(seconds, (out_counts.sum() / seconds) / 1e6))
print("{} Million Complex FMAs".format(out_counts.sum() / 1e6))

pylab.imshow(np.log(out_counts))
pylab.show()
