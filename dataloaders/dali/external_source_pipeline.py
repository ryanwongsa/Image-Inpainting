import types
import collections
import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import random

import cv2
import numpy as np

def base_mask_augmentation(is_inverted):
  def mask_augmentation(mask):
    rand = np.random.randint(5, 49)
    kernel = np.ones((rand, rand), np.uint8) 
    mask = cv2.erode(mask, kernel, iterations=1)

    if is_inverted:
        return (np.invert(mask > 1)).astype(np.uint8)
    else:
        return (mask > 1).astype(np.uint8)
  return mask_augmentation

class ExternalSourcePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_data, exec_async=False, exec_pipelined=False):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12,
                                      exec_async=exec_async, 
                                      exec_pipelined=exec_pipelined)
        self.input_images = ops.ExternalSource()
        self.input_masks = ops.ExternalSource()

        self.decode = ops.ImageDecoder(device = 'cpu', output_type = types.RGB)
        # self.cast = ops.Cast(device = "gpu", dtype = types.INT32)

        self.image_res = ops.Resize(device="gpu", resize_x=512, resize_y=512)
        self.image_cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(512, 512),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        
        
        self.mask_augmentation = ops.PythonFunction(function=base_mask_augmentation(False), num_outputs=1)
        self.mask_res = ops.Resize(device="gpu", resize_x=512, resize_y=512)
        self.mask_rotate = ops.Rotate(device = 'gpu', interp_type = types.INTERP_LINEAR, fill_value = 1)
        self.mask_rng = ops.Uniform(range = (-180.0, 180.0))
        self.mask_cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(512, 512))


        self.external_data = external_data
        self.iterator = iter(self.external_data)

    def define_graph(self):
        self.images = self.input_images()
        self.masks = self.input_masks()

        images = self.decode(self.images)
        images = self.image_res(images.gpu())
        images = self.image_cmnp(images.gpu())
        # images = self.cast(images)

        masks = self.decode(self.masks)
        masks = self.mask_augmentation(masks)
        masks = self.mask_res(masks.gpu())
        masks_angle = self.mask_rng()
        masks = self.mask_rotate(masks, angle = masks_angle)
        masks = self.mask_cmnp(masks)

        return (images, masks)

    def iter_setup(self):
        try:
            (images, masks) = self.iterator.next()
            self.feed_input(self.images, images)
            self.feed_input(self.masks, masks)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration