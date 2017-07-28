#!/usr/bin/env python


import numpy as np
import sys
import os
__this_dir = os.path.dirname(__file__)
__caffe_path = os.path.join(__this_dir, '..', '..', 'python')
if __caffe_path not in sys.path:
    sys.path.insert(0, __caffe_path)
try:
    import caffe
    print 'Imported caffe version: {}'.format(caffe.__version__)
except:
    print 'Unable to import caffe'
    raise


if __name__ == '__main__':
    caffe.set_mode_gpu()
    caffe.set_device(0)
    solver = caffe.SGDSolver(os.path.join(__this_dir,
                                          'cifar10_csc_solver.prototxt'))
    solver.step(1)

