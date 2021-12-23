# TUM sequence loader
from .util import Rays, Intrin, select_or_shuffle_rays
from .dataset_base import DatasetBase
import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional, Union
from os import path
import imageio
from tqdm import tqdm
import cv2
import json
import numpy as np


# Note,this step converts w2c (Tcw) to c2w (Twc)
def load_K_Rt_from_P(P):
    """
    modified from IDR https://github.com/lioryariv/idr
    """
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # convert from w2c to c2w
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class TUMDataset(DatasetBase):
    """
    NeRF dataset loader
    """

    focal: float
    c2w: torch.Tensor  # (n_images, 4, 4)
    gt: torch.Tensor  # (n_images, h, w, 3)
    h: int
    w: int
    n_images: int
    rays: Optional[Rays]
    split: str

    def __init__(
            self,
            root,
            split,
            device: Union[str, torch.device] = "cpu",
            scene_scale: Optional[float] = 1.0,  # scene scale factor, default 2/3
            time_downsample_factor: int = 25,
            factor: int = 1,
            scale : Optional[float] = None,  # image scale factor
            permutation: bool = True,
            white_bkgd: bool = True,
            n_images = None,
            **kwargs
    ):
        super().__init__()
        assert path.isdir(root), f"'{root}' is not a directory"

        # We don't need this
        if scene_scale is None:
            scene_scale = 1.  # was 2/3
        if scale is None:
            scale = 1.0
        self.device = device
        self.permutation = permutation
        all_c2w = []
        all_K = []
        all_gt = []

        split_name = split if split != "test_train" else "train"
        mod_offset = 0
        if split_name == "test":
            mod_offset = time_downsample_factor // 2

        # root should be tum_sequence
        data_path = path.join(root, "processed")
        cam_file = path.join(data_path, "cameras.npz")
        print("LOAD DATA", data_path)

        # world_mats, normalize_mat
        cam_dict = np.load(cam_file)
        world_mats = cam_dict["world_mats"]  # K @ w2c
        normalize_mat = cam_dict["normalize_mat"]  # w'2w

        # TUM saves camera poses in OpenCV convention
        for i, world_mat in enumerate(tqdm(world_mats)):
            # down-sample number of images
            if i % time_downsample_factor != mod_offset:
                continue
            intrinsics, c2w = load_K_Rt_from_P(world_mat @ normalize_mat)
            c2w = torch.tensor(c2w, dtype=torch.float32)
            # image path
            fpath = path.join(data_path, "rgb/{:04d}.png".format(i))

            im_gt = imageio.imread(fpath)
            if scale < 1.0:
                full_size = list(im_gt.shape[:2])
                rsz_h, rsz_w = [round(hw * scale) for hw in full_size]
                im_gt = cv2.resize(im_gt, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)
                intrinsics[0, 0] *= scale
                intrinsics[1, 1] *= scale
                intrinsics[0, 2] *= scale
                intrinsics[1, 2] *= scale

            all_c2w.append(c2w)
            all_K.append(intrinsics)
            all_gt.append(torch.from_numpy(im_gt))

        K = all_K[0]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.c2w = torch.stack(all_c2w)
        # We don't need this, as normalize_mat already takes care of scaling
        self.c2w[:, :3, 3] *= scene_scale

        self.gt = torch.stack(all_gt).float() / 255.0
        if self.gt.size(-1) == 4:
            if white_bkgd:
                # Apply alpha channel
                self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])
            else:
                self.gt = self.gt[..., :3]

        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        # Choose a subset of training images
        if n_images is not None:
            if n_images > self.n_images:
                print(f'using {self.n_images} available training views instead of the requested {n_images}.')
                n_images = self.n_images
            self.n_images = n_images
            self.gt = self.gt[0:n_images,...]
            self.c2w = self.c2w[0:n_images,...]

        # This need to be computed dynamically
        self.epoch_size = self.n_images * self.h_full * self.w_full

        self.intrins_full : Intrin = Intrin(fx, fy, cx, cy)
        self.split = split
        self.scene_scale = scene_scale
        if self.split == "train":
            self.gen_rays(factor=factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins : Intrin = self.intrins_full

        self.should_use_background = False  # Give warning

