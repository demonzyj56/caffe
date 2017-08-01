#!/usr/bin/env python

from __future__ import division
import os
import numpy as np
import _init_path
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import matplotlib.pyplot as plt
from test import test_net
plt.ion()

_DEBUG = True


class SolverWrapper(object):
    def __init__(self, solver_prototxt, pretrained_model=None):
        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            self.solver.net.copy_from(pretrained_model)
        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            try:
                pb2.text_format.Merge(f.read(), self.solver_param)
            except AttributeError:
                from google.protobuf import text_format
                text_format.Merge(f.read(), self.solver_param)
        self._nnz = []
        self._eta = []
        self._train_loss = []
        self._train_acc = []

    def train_model(self, max_iters):
        """ Monitor the following value:
        - nonzeros of alpha
        - eta
        - loss value
        """
        net = self.solver.net
        while self.solver.iter < max_iters:
            self.solver.step(1)
            csc_output = net.blobs['csc'].data
            csc_loss = net.blobs['loss'].data.tolist()
            csc_acc = net.blobs['accuracy'].data.tolist()
            nnz = self._zero_norm(csc_output)
            self._nnz.append(nnz)
            self._train_loss.append(csc_loss)
            self._train_acc.append(csc_acc)
            if _DEBUG:
                from ipdb import set_trace; set_trace()
                print 'Nonzero elements per column: {}'.format(
                    nnz / csc_output.size * csc_output.shape[1])
            #  if self.solver.iter % self.solver_param.display == 0:
            #      self._plot()
            #  if self.solver.iter % 100 == 0:
            #      model_path = str(self.snapshot())
            #      test_prototxt = str(self.solver_param.net)
            #      test_net(test_prototxt, 100, pretrained_model=model_path)


    def _zero_norm(self, blob):
        return np.count_nonzero(blob)

    def _plot(self):
        plt.cla()
        plt.plot(self._train_loss)
        plt.xlabel('Iteration')
        plt.ylabel('Train loss')
        plt.title('Train loss over iterations')
        plt.show()
        plt.pause(0.05)

    def snapshot(self):
        """ snapshot at the same directory"""
        net = self.solver.net
        filename = (self.solver_param.snapshot_prefix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        net.save(str(filename))
        return filename



if __name__ == '__main__':
    caffe.set_mode_gpu()
    caffe.set_device(0)
    if _DEBUG:
        solver_prototxt = os.path.join(os.path.dirname(__file__),
                                       'cifar10_csc_solver_debug.prototxt')
    else:
        solver_prototxt = os.path.join(os.path.dirname(__file__),
                                       'cifar10_csc_solver.prototxt')
    solver = SolverWrapper(solver_prototxt)
    solver.train_model(1000)
    from IPython import embed; embed()

