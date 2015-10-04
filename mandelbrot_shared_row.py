import numpy as np
import pylab as plt
import multiprocessing
import time

from common import mandelbrot, Timer, \
    by_block_slices, by_blocks, by_rows, \
    make_coords, make_shared


if __name__ == '__main__':
    in_coords, out_counts = make_coords()
    in_coords = make_shared(in_coords)
    out_counts = make_shared(out_counts)

    # this has to be here for scoping reasons
    def wrap_mandelbrot(ij_slice, iterations=1024):
        mandelbrot.mandelbrot(in_coords[ij_slice], out_counts[ij_slice], iterations)

    # pool should be declared after any shared variables
    pool = multiprocessing.Pool(4)

    with Timer() as t:
        slices = by_rows(in_coords)
        pool.map(wrap_mandelbrot, slices)
    seconds = t.interval

    print("{} seconds, {} million Complex FMAs / second".format(seconds, (out_counts.sum() / seconds) / 1e6))
    print("{} Million Complex FMAs".format(out_counts.sum() / 1e6))

    plt.imshow(np.log(out_counts))
    plt.show()
