import numpy as np
import pylab as plt
from multiprocessing.pool import ThreadPool
import time

from common import mandelbrot, Timer, \
    by_block_slices, by_blocks, by_rows, \
    make_coords, make_shared


if __name__ == '__main__':
    in_coords, out_counts = make_coords()

    # this has to be here for scoping reasons
    def wrap_mandelbrot(row, iterations=1024):
        mandelbrot.mandelbrot(in_coords[row:(row + 1), :], out_counts[row:(row + 1), :], iterations)

    pool = ThreadPool(4)

    with Timer() as t:
        rows = range(in_coords.shape[0])
        pool.map(wrap_mandelbrot, rows)
    seconds = t.interval

    print("{} seconds, {} million Complex FMAs / second".format(seconds, (out_counts.sum() / seconds) / 1e6))
    print("{} Million Complex FMAs".format(out_counts.sum() / 1e6))

    plt.imshow(np.log(out_counts))
    plt.show()
