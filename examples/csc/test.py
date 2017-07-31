#!/usr/bin/env python

from __future__ import division
import os
import _init_path
import caffe


def test_net(test_prototxt, test_iters, pretrained_model=None):
    accuracy = 0.
    net = caffe.Net(test_prototxt, caffe.TEST);
    if pretrained_model is not None:
        net.copy_from(pretrained_model)
    for it in xrange(test_iters):
        acc = net.forward()['accuracy'].tolist()
        #  print 'Iteration {}, accuracy {}'.format(it+1, acc)
        accuracy += acc
    print 'Final test accuracy: {}'.format(accuracy/test_iters)
    return net


if __name__ == '__main__':
    caffe.set_mode_gpu()
    caffe.set_device(0)
    test_prototxt = os.path.join(os.path.dirname(__file__),
                                 'cifar10_csc.prototxt')
    pretrained_model = None
    test_iters = 1
    net = test_net(test_prototxt, test_iters, pretrained_model)
    from IPython import embed; embed()
