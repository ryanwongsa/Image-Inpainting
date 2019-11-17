import types
import collections
import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import random

 class ExternalSourcePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_data):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        self.input_images = ops.ExternalSource()
        self.input_masks = ops.ExternalSource()

        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.cast = ops.Cast(device = "gpu", dtype = types.INT32)
        self.res = ops.Resize(device="gpu", resize_x=240, resize_y=240)

        self.external_data = external_data
        self.iterator = iter(self.external_data)

    def define_graph(self):
        self.images = self.input_images()
        self.masks = self.input_masks()

        images = self.decode(self.images)
        images = self.res(images)
        images = self.cast(images)

        masks = self.decode(self.masks)
        masks = self.res(masks)
        masks = self.cast(masks)

        return (images, masks)

    def iter_setup(self):
        try:
            (images, masks) = self.iterator.next()
            self.feed_input(self.images, images)
            self.feed_input(self.masks, masks)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration