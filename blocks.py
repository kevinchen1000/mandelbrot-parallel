import numpy as np

def by_blocks(array, num):
    '''
    Generator that iterates over blocks in an array with (num x num) total
    blocks.

    Resulting subarrays are views on the original array (shared data).
    '''
    for hblock in np.array_split(array, num, axis=0):
        for vblock in np.array_split(hblock, num, axis=1):
            yield vblock

def by_block_slices(array, num):
    '''
    Generator that iterates over blocks in an array with (num x num) total
    blocks, but returns slice indices rather than actual subarray.

    '''
    i_step = array.shape[0] / num
    j_step = array.shape[1] / num
    for i_lo in range(0, array.shape[0], i_step):
        for j_lo in range(0, array.shape[1], j_step):
            yield (slice(i_lo, i_lo + i_step),
                   slice(j_lo, j_lo + j_step))

def by_rows(array):
    for idx in range(array.shape[0]):
        yield (slice(idx, idx + 1, 1), slice(None, None, 1))
