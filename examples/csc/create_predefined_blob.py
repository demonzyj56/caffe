#!/usr/bin/env python
""" Create predefined blob from external numpy blob. """
import os
import numpy as np
from caffe.io import array_to_blobproto


def create_predefined_blob(arr, filename=None):
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__),
                                'centroids.binaryproto')
    blob = array_to_blobproto(arr, diff=None)
    with open(filename, 'wb') as f:
        f.write(blob.SerializeToString())
    print "Successfully write blob to {}".format(filename)


if __name__ == '__main__':
    arr = np.load('/home/leoyolo/PycharmProjects/cocoa/data/centroids.npy')
    arr = arr.transpose()
    create_predefined_blob(arr)
