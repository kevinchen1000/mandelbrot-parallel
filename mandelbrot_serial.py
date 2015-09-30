import numpy as np
import pylab as plt

from common import mandelbrot, Timer, \
    by_block_slices, by_blocks, by_rows, \
    make_coords, make_shared

in_coords, out_counts = make_coords()

with Timer() as t:
    mandelbrot.mandelbrot(in_coords, out_counts, 1024)
seconds = t.interval

print("{} seconds, {} million Complex FMAs / second".format(seconds, (out_counts.sum() / seconds) / 1e6))
print("{} Million Complex FMAs".format(out_counts.sum() / 1e6))

plt.imshow(np.log(out_counts))
plt.show()
