import numpy as np
import pylab as plt
import multiprocessing

from common import mandelbrot, Timer, \
    by_block_slices, by_blocks, by_rows, \
    make_coords, make_shared

def wrap_mandelbrot(coords, iterations=1024):
    tmp_counts = np.zeros_like(coords, dtype=np.uint32)
    mandelbrot.mandelbrot(coords, tmp_counts, iterations)
    return tmp_counts

if __name__ == '__main__':
    in_coords, out_counts = make_coords()

    pool = multiprocessing.Pool(4)

    with Timer() as t:
        in_blocks = by_blocks(in_coords, 10)
        tmp_blocks = pool.map(wrap_mandelbrot, in_blocks)
        for src, dest in zip(tmp_blocks, by_blocks(out_counts, 10)):
            dest[...] = src
    seconds = t.interval

    print("{} seconds, {} million Complex FMAs / second".format(seconds, (out_counts.sum() / seconds) / 1e6))
    print("{} Million Complex FMAs".format(out_counts.sum() / 1e6))

    plt.imshow(np.log(out_counts))
    plt.show()
