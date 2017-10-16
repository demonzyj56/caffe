"""
Generates random blob for testing PredefinedFiller.
"""
import os
import numpy as np
from caffe.io import array_to_blobproto


def generate_random_blob(sz, filename=None):
    """ Generates a random blob and save to directory
    as the binaryproto. """
    assert len(sz) <= 4
    # shifts to [2, 3).
    arr = np.random.rand(*sz) + 2
    blob = array_to_blobproto(arr, diff=None)
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__),
                                'random_blob.binaryproto')
    with open(filename, 'wb') as f:
        f.write(blob.SerializeToString())
    print("Successfully write blob to {}".format(filename))


if __name__ == '__main__':
    generate_random_blob([108, 100])
