# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset
import scipy.io

class KAISTDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KAISTDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.K = np.array([[3.23557313e+03/512, 0, 5.77682065e+02/512, 0],
                           [0, 3.23243823e+03/448, 4.82450866e+02/448, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (512, 448)
        self.side_map = {"2": 2, "3": 3, "l": "LEFT", "r": "RIGHT","t":"THER"}

    def check_depth(self,mode="train"):

        if not mode:
            return True
        else:
            return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    def get_thermal(self, folder, frame_index, side, do_flip):
        thermal = self.loader(self.get_image_path(folder, frame_index, side))
        thermal=thermal.convert("RGB")
        if do_flip:
            thermal = thermal.transpose(pil.FLIP_LEFT_RIGHT)

        return thermal

class KAISTRAWDataset(KAISTDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KAISTRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        if not side=="t":
            f_str = "_%09d%s"%(frame_index, self.img_ext)

            image_path = os.path.join(
                self.data_path, folder,self.side_map[side], self.side_map[side]+f_str)
            return image_path
        else:
            f_str = "_%09d%s"%(frame_index, self.img_ext)
            sp=folder.split("/")
            folder=os.path.join(folder,"THERMAL")
            image_path = os.path.join(
                self.data_path, folder, self.side_map[side]+f_str)
            return image_path
            
    def get_depth(self, folder):
        
        Depth=scipy.io.loadmat(folder)["depth"]
        return Depth

