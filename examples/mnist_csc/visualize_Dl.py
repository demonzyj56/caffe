import os
import sys
import numpy as np
from IPython import embed

this_dir = os.path.dirname(__file__)
caffe_path = os.path.join(os.path.dirname(__file__), '..', '..', 'python')
sys.path.insert(0, caffe_path)
import caffe


def visualize_dl(net_path, net_weight):
    """ Visualize the local dictionary Dl in net.
    Dl is of size n x m, where n = c x kernel_h x kernel_w."""
    net = caffe.Net(net_path, net_weight, caffe.TEST)
    return net


if __name__ == '__main__':
    net_path = os.path.join(this_dir, 'small_context_4.prototxt')
    net_weight = os.path.join(this_dir, \
        'snapshots/small_context_4_iter_1200.caffemodel')
    net = visualize_dl(net_path, net_weight)
    embed()
