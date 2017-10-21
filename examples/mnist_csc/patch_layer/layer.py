""" Patch layer that crops the image into non-overlapping patches. """
from __future__ import division
import caffe


class CSCPatchLayer(caffe.Layer):
    """
    Crops the input into [patch_size, patch_size] size of patches.
    The cropped patches are gathered in the first dimension.
    """

    def setup(self, bottom, top):
        """ Loads path_size from param_str.
        Assume that bottom size is fixed to save my life. """
        params = eval(self.param_str)
        self.patch_size = int(params['patch_size'])
        assert bottom[0].data.shape[2] % self.patch_size == 0
        assert bottom[0].data.shape[3] % self.patch_size == 0
        self.num_patches_h = bottom[0].data.shape[2] // self.patch_size
        self.num_patches_w = bottom[0].data.shape[3] // self.patch_size
        top[0].reshape(
            bottom[0].data.shape[0] * self.num_patches_h * self.num_patches_w,
            bottom[0].data.shape[1],
            self.patch_size,
            self.patch_size
        )


    def reshape(self, bottom, top):
        """ Set top shape. """
        pass

    def forward(self, bottom, top):
        """ Just copy the data from bottom to top. """
        for n in range(bottom[0].data.shape[0]):
            for i in range(self.num_patches_h):
                for j in range(self.num_patches_w):
                    patch = bottom[0].data[n, :,
                        i*self.patch_size:(i+1)*self.patch_size,
                        j*self.patch_size:(j+1)*self.patch_size,
                    ]
                    offset = (n * self.num_patches_h + i) * self.num_patches_w + j
                    top[0].data[offset, :, :, :] = patch

    def backward(self, top, propagate_down, bottom):
        """ Copy the diff from top to bottom. """
        if propagate_down[0]:
            for idx in range(top[0].diff.shape[0]):
                n_index = idx // self.num_patches_w
                j = idx % self.num_patches_w
                i = n_index % self.num_patches_h
                n = n_index // self.num_patches_h
                bottom[0].diff[n, :,
                               i*self.patch_size:(i+1)*self.patch_size,
                               j*self.patch_size:(j+1)*self.patch_size,
                              ] = top[0].diff[idx, :, :, :]


class CSCGatherLayer(caffe.Layer):
    """ Gather the patches created before. """

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.num_patches_h = int(params['num_patches_h'])
        self.num_patches_w = int(params['num_patches_w'])
        top[0].reshape(
            bottom[0].data.shape[0] // self.num_patches_h // self.num_patches_w,
            bottom[0].data.shape[1],
            bottom[0].data.shape[2] * self.num_patches_h,
            bottom[0].data.shape[3] * self.num_patches_w
        )

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        height, width = bottom[0].data.shape[2:]
        for idx in range(bottom[0].data.shape[0]):
            n_index = idx // self.num_patches_w
            j = idx % self.num_patches_w
            i = n_index % self.num_patches_h
            n = n_index // self.num_patches_h
            top[0].data[n, :, i*height:(i+1)*height, j*width:(j+1)*width] = \
                bottom[0].data[idx, :, :, :]

    def backward(self, top, propagate_down, bottom):
        height, width = bottom[0].data.shape[2:]
        if propagate_down[0]:
            for n in range(top[0].diff.shape[0]):
                for i in range(self.num_patches_h):
                    for j in range(self.num_patches_w):
                        patch = top[0].diff[n, :,
                            i*height:(i+1)*height,
                            j*width:(j+1)*width
                        ]
                        offset = (n * self.num_patches_h + i) * self.num_patches_w + j
                        bottom[0].diff[offset, ...] = patch
