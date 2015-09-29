import numpy as np

def by_blocks(array, num):
    '''
    Generator that iterates over blocks in an array with (num x num) total blocks.
    Resulting subarrays are views on the original array (shared data).
    '''

    for hblock in np.array_split(array, num, axis=0):
        for vblock in np.array_split(hblock, num, axis=1):
            yield vblock
