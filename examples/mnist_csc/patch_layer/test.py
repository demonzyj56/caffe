import os
import sys
import numpy as np
from IPython import embed

caffe_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'python')
patch_module_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, caffe_path)
sys.path.insert(0, patch_module_path)
import caffe


if __name__ == '__main__':
    input_blob = np.random.randn(1, 3, 32, 32).astype(np.float32)
    net = caffe.Net('examples/mnist_csc/patch_layer/dummy_test_net.prototxt', caffe.TEST)
    net.blobs['data'].data[...] = input_blob
    out = net.forward()
    output_blob = net.blobs['gather'].data
    print out['loss']
    print (input_blob == output_blob).all()
    embed()
