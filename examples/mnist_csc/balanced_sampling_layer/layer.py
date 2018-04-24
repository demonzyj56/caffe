""" MNIST/CIFAR10 utility loader that samples a fixed number of
samples in each class. """
import os
import numpy as np
import skimage.transform as skt
import caffe
from torchvision import datasets, transforms


__default_data_path = os.path.expanduser('~/PycharmProjects/cocoa/data')


def sample_mnist(samples_per_class, train=True):
    """ Sample mnist. """
    mnist = datasets.MNIST(__default_data_path, train=train, download=True)
    data = mnist.train_data.float().numpy() / 255.0 if train \
        else mnist.test_data.float().numpy() / 255.0
    data = np.expand_dims(data, axis=1)
    labels = mnist.train_labels.numpy() if train else mnist.test_labels.numpy()
    sampled_data, sampled_labels = [], []
    for i in range(10):
        index = np.where(labels == i)[0]
        perm = np.random.permutation(len(index))[:min(samples_per_class,
                                                      len(index))]
        sampled_data.append(data[index[perm]])
        sampled_labels.append(labels[index[perm]])
    sampled_data = np.concatenate(sampled_data, axis=0)
    sampled_labels = np.concatenate(sampled_labels)

    return sampled_data, sampled_labels


def sample_cifar10(samples_per_class, train=True, grayscale=True):
    """ Sample CIFAR10. Notice we transfer cifar10 to grayscale image. """
    cifar10 = datasets.CIFAR10(__default_data_path, train=train, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))
    data = cifar10.train_data.astype(np.float64, copy=False) / 255.0 if train \
        else cifar10.test_data.astype(np.float64, copy=False) / 255.0
    data = data.transpose((0, 3, 1, 2))
    if grayscale:
        data = data[:, 0, :, :] * 0.2989 + data[:, 1, :, :] * 0.5870 + \
            data[:, 2, :, :] * 0.1140
        data = np.expand_dims(data, axis=1)
    labels = cifar10.train_labels if train else cifar10.test_labels
    labels = np.array(labels)
    sampled_data, sampled_labels = [], []
    for i in range(10):
        index = np.where(labels == i)[0]
        perm = np.random.permutation(len(index))[:min(samples_per_class,
                                                      len(index))]
        sampled_data.append(data[index[perm]])
        sampled_labels.append(labels[index[perm]])
    sampled_data = np.concatenate(sampled_data, axis=0)
    sampled_labels = np.concatenate(sampled_labels)

    return sampled_data, sampled_labels


def resize_and_random_crop(img, sz):
    """ Resize the given image into sz, but then randomly crop to
    the original position.
    The input image img is is of order (H, W, C) with values as
    uint8 (0-255) or float (0-1)."""
    if img.shape[0] >= sz[0] or img.shape[1] >= sz[1]:
        return img
    resized = skt.resize(img, sz)
    r = np.random.randint(0, sz[0]-img.shape[0])
    c = np.random.randint(0, sz[1]-img.shape[1])
    return resized[r:r+img.shape[0], c:c+img.shape[1], :]


class BalancedSamplingLayer(caffe.Layer):
    """ Sample the target dataset according to samples per class. """

    def setup(self, bottom, top):
        """
        The sampler takes in the dataset name, number of samples per class,
        and whether it is train/test.
        """
        params = eval(self.param_str)
        self.dataset_name = str(params['name'])
        self.batch_size = int(params['batch_size'])
        self.crop = params.get('crop', False)
        if self.dataset_name == 'mnist':
            self.data, self.labels = sample_mnist(int(params['samples_per_class']),
                                                  train=bool(params['train']))
        elif self.dataset_name == 'cifar10':
            self.data, self.labels = sample_cifar10(int(params['samples_per_class']),
                                                    train=bool(params['train']))
        else:
            raise ValueError("Unknonwn dataset name {}".format(self.dataset_name))
        assert self.data.ndim == 4
        assert self.data.shape[1] in (1, 3)
        assert self.data.shape[2] in (28, 32)  # cifar10 or mnist
        assert self.data.shape[2] == self.data.shape[3]
        assert self.data.shape[0] == len(self.labels)
        self.perm = np.random.permutation(len(self.labels))
        self.cur_idx = 0
        top[0].reshape(self.batch_size, *self.data.shape[1:])
        #  top[0].reshape(self.batch_size, self.data.shape[-1], self.data.shape[1],
        #                 self.data.shape[2])
        top[1].reshape(self.batch_size)

    def reshape(self, bottom, top):
        """ Reshape. """
        if self.cur_idx >= len(self.labels):
            self.cur_idx = 0
            self.perm = np.random.permutation(len(self.labels))
        batch_size = min(self.batch_size, len(self.labels) - self.cur_idx)
        top[0].reshape(batch_size, *self.data.shape[1:])
        #  top[0].reshape(self.batch_size, self.data.shape[-1], self.data.shape[1],
        #                 self.data.shape[2])
        top[1].reshape(batch_size)

    def forward(self, bottom, top):
        """ Samples a batch from data and sample. """
        batch_size = min(self.batch_size, len(self.labels) - self.cur_idx)
        #  input_data = self.data[self.perm[self.cur_idx:self.cur_idx+batch_size], :, :, :]
        #  if self.crop:
        #      input_data = [resize_and_random_crop(input_data[i], self.crop)[np.newaxis, :, :, :]
        #                    for i in range(batch_size)]
        #      input_data = np.concatenate(input_data, axis=0)
        #  input_data = input_data.transpose((0, 3, 1, 2))
        #  top[0].data[...] = input_data
        top[0].data[...] = self.data[self.perm[self.cur_idx:self.cur_idx+batch_size], :, :, :]
        top[1].data[...] = self.labels[self.perm[self.cur_idx:self.cur_idx+batch_size]]
        self.cur_idx += batch_size

    def backward(self, top, propagate_down, bottom):
        """ No need to backward. """
        pass


if __name__ == '__main__':
    data, label = sample_mnist(30, train=True)
    from IPython import embed; embed()
