from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import cv2
import joblib
import pickle
import random
import argparse
import collections
import numpy as np
from time import time
from PIL import Image

import torch
from torch.utils.data import IterableDataset


# follow paper Sec. 3.3 of https://arxiv.org/pdf/2003.08400.pdf
# - for chair: use 0.03
# - for Apple stream: use 0.1
Z_DIFF_THRESHOLD = 0.03


class AdvTexIterDataset(IterableDataset):
    def __init__(
        self,
        parent_dir,
        texture_name,
        num_workers,
        z_diff_threshold=Z_DIFF_THRESHOLD,
        Cache=False,
        use_raw_rgb=True,
        data_chair=False,
    ):

        print(texture_name)

        tex_img = np.array(Image.open(texture_name))

        self.cached = Cache
        self.tex_dim_height = tex_img.shape[0]
        self.tex_dim_width = tex_img.shape[1]

        if parent_dir is None or not os.path.exists(parent_dir):
            raise Exception("input_dir does not exist")

        # global view_pairs, intrinsic, kernel
        self.view_pairs = pickle.load(
            open(os.path.join(parent_dir, "pose_pair.pkl"), "rb")
        )

        self.data_chair = data_chair

        if not data_chair:
            self.offset_dict = joblib.load(os.path.join(parent_dir, "offset_dict.p"))
        else:
            self.offset_dict = None

        self.use_raw_rgb = use_raw_rgb
        if not data_chair:
            if self.use_raw_rgb:
                self.color_paths = sorted(
                    glob.glob(os.path.join(parent_dir, "raw_colors/*_raw_color.png"))
                )
            else:
                self.color_paths = sorted(
                    glob.glob(os.path.join(parent_dir, "*_color.png"))
                )
        else:
            self.color_paths = sorted(
                glob.glob(os.path.join(parent_dir, "*_color.png"))
            )

        self.color_paths_for_worker = [[] for _ in range(num_workers)]
        for i, color_path in enumerate(self.color_paths):
            self.color_paths_for_worker[i % num_workers].append(color_path)

        for i in range(len(self.view_pairs)):
            if type(self.view_pairs[i]) == type([]):
                p = self.view_pairs[i].copy()
            else:
                p = self.view_pairs[i].tolist()
            p.append(i)
            self.view_pairs[i] = np.array(p, dtype="int32")

        self.z_diff_threshold = z_diff_threshold

        print("\nz_diff_threshold: ", z_diff_threshold, "\n")

        print("\ncolor_paths: ", len(self.color_paths), "\n")
        print("\nview_pairs: ", len(self.view_pairs), "\n")

        assert len(self.color_paths) == len(self.view_pairs), f"{len(self.color_paths)}, {len(self.view_pairs)}"

    def __len__(self):
        return len(self.view_pairs)

    def load_data_by_id(self, root, index):
        # if not index in dictionary:
        if self.data_chair:
            color_src_img = root + "/%05d_color.png" % (index)
        else:
            if self.use_raw_rgb:
                color_src_img = os.path.join(
                    root, f"raw_colors/{index:05d}_raw_color.png"
                )
            else:
                color_src_img = root + "/%05d_color.png" % (index)
        uv_src_img = root + "/%05d_uv.npz" % (index)
        depth_src_img = root + "/%05d_depth.npz" % (index)
        mask_src_img = root + "/%05d_mask.png" % (index)

        intrinsic_f = root + "/%05d_intrinsic.txt" % (index)
        if not os.path.exists(intrinsic_f):
            intrinsic_f = os.path.join(root, "intrinsic.txt")
        intrinsic = np.loadtxt(intrinsic_f)
        intrinsic = np.reshape(intrinsic, [16])

        pose = root + "/%05d_pose.txt" % (index)
        color_src = np.array(Image.open(color_src_img)) / 255.0
        uv_src = np.load(uv_src_img)["arr_0"]
        depth_src = np.load(depth_src_img)["arr_0"]
        mask_src = np.array(Image.open(mask_src_img)) / 255.0
        world2cam = np.loadtxt(pose)

        assert (
            np.min(uv_src) >= 0.0 and np.max(uv_src) <= 1.0
        ), f"{np.min(uv_src)}, {np.max(uv_src)}"
        return (
            color_src.astype("float32"),
            uv_src.astype("float32"),
            depth_src.astype("float32"),
            mask_src.astype("float32"),
            intrinsic.astype("float32"),
            world2cam.astype("float32"),
        )

    def load_chunk(self, filename):

        if not isinstance(filename, str):
            filename = filename.decode("utf-8")

        if self.use_raw_rgb:
            # XXXXX_raw_color.png
            index = int(os.path.basename(filename).split("_")[0])
            root = os.path.dirname(os.path.dirname(filename))
        else:
            index = int(filename[-15:-10])
            root = filename[:-16]

        (
            raw_color_src,
            uv_src,
            depth_src,
            mask_src,
            intrinsic_src,
            world2cam_src,
        ) = self.load_data_by_id(root, index)

        img_h, img_w, _ = raw_color_src.shape

        rindex = random.choice(self.view_pairs[index])

        if rindex != index:
            (
                color_tar,
                uv_tar,
                depth_tar,
                mask_tar,
                intrinsic_tar,
                world2cam_tar,
            ) = self.load_data_by_id(root, rindex)

            cam2world_src = np.linalg.inv(world2cam_src)
            src2tar = np.transpose(np.dot(world2cam_tar, cam2world_src))

            y = np.linspace(0, img_h - 1, img_h)
            x = np.linspace(0, img_w - 1, img_w)
            xx, yy = np.meshgrid(x, y)

            # NOTE: intrinsic is assumed to be 4x4
            fx_src = intrinsic_src[0]
            cx_src = intrinsic_src[2]
            fy_src = intrinsic_src[5]
            cy_src = intrinsic_src[6]

            fx_tar = intrinsic_tar[0]
            cx_tar = intrinsic_tar[2]
            fy_tar = intrinsic_tar[5]
            cy_tar = intrinsic_tar[6]

            x = (xx - cx_src) / fx_src * depth_src
            y = (yy - cy_src) / fy_src * depth_src
            coords = np.zeros((img_h, img_w, 4))
            coords[:, :, 0] = x
            coords[:, :, 1] = y
            coords[:, :, 2] = depth_src
            coords[:, :, 3] = 1
            coords = np.dot(coords, src2tar)
            z_tar = coords[:, :, 2]
            # pixel coords in target
            x = coords[:, :, 0] / (1e-8 + z_tar) * fx_tar + cx_tar
            y = coords[:, :, 1] / (1e-8 + z_tar) * fy_tar + cy_tar

            mask0 = depth_src == 0
            tar_img_h, tar_img_w, _ = color_tar.shape
            mask1 = (x < 0) + (y < 0) + (x >= tar_img_w - 1) + (y >= tar_img_h - 1)
            lx = np.floor(x).astype("float32")
            ly = np.floor(y).astype("float32")
            rx = (lx + 1).astype("float32")
            ry = (ly + 1).astype("float32")
            sample_z1 = np.abs(z_tar - cv2.remap(depth_tar, lx, ly, cv2.INTER_NEAREST))
            sample_z2 = np.abs(z_tar - cv2.remap(depth_tar, lx, ry, cv2.INTER_NEAREST))
            sample_z3 = np.abs(z_tar - cv2.remap(depth_tar, rx, ly, cv2.INTER_NEAREST))
            sample_z4 = np.abs(z_tar - cv2.remap(depth_tar, rx, ry, cv2.INTER_NEAREST))
            # check whether there is occulusion
            mask2 = (
                np.minimum(
                    np.minimum(sample_z1, sample_z2), np.minimum(sample_z3, sample_z4)
                )
                > self.z_diff_threshold
            )

            mask_remap = (1 - (mask0 + mask1 + mask2 > 0)).astype("float32")

            map_x = x.astype("float32")
            map_y = y.astype("float32")

            raw_color_tar_to_src = cv2.remap(color_tar, map_x, map_y, cv2.INTER_LINEAR)
            mask = (
                cv2.remap(mask_tar, map_x, map_y, cv2.INTER_LINEAR) > 0.99
            ) * mask_remap
            for j in range(3):
                raw_color_tar_to_src[:, :, j] *= mask

        else:
            raw_color_tar_to_src = raw_color_src.copy()
            mask = mask_src.copy()
            # raise ValueError
        
        if np.sum(np.abs(mask_src)) == 0 or ((rindex != index) and np.sum(np.abs(mask_tar)) == 0):
            # This is mainly for ScanNet as it has some poses with value inf.
            # https://github.com/ScanNet/ScanNet/issues/9
            return None

        raw_color_src = torch.FloatTensor(raw_color_src)
        raw_color_tar_to_src = torch.FloatTensor(raw_color_tar_to_src)

        mask = torch.FloatTensor(mask).unsqueeze(2)
        mask_src = torch.FloatTensor(mask_src).unsqueeze(2)

        # NOTE: do not use mask
        # color_src_rendered_area = raw_color_src * mask_src
        color_src_rendered_area = raw_color_src

        color_src = raw_color_src * mask
        color_tar_to_src = raw_color_tar_to_src * mask

        # [0, 1] -> [-1, 1]
        color_src = color_src * 2 - 1
        color_tar_to_src = color_tar_to_src * 2 - 1
        color_src_rendered_area = color_src_rendered_area * 2 - 1

        # In commonly-used definition, we have
        # - U elem is for horizontal, +U right
        # - V elem is for vertical, +V up
        # However, in image pixel coordinates, +V down
        # Ref:
        # - https://stackoverflow.com/a/8851832
        # - https://github.com/hjwdzh/AdversarialTexture/issues/3
        uv_src[:, :, 1] = 1 - uv_src[:, :, 1]

        # NOTE: torch.grid_sample needs grid val in [-1, 1]
        uv_src = torch.FloatTensor(uv_src) * 2 - 1

        if self.offset_dict is not None:
            shift_u, shift_v = self.offset_dict[index]
        else:
            shift_u, shift_v = 0, 0

        return (
            color_src,
            color_tar_to_src,
            color_src_rendered_area,
            uv_src,
            mask,
            mask_src,
            img_h,
            img_w,
            index,
            rindex,
            shift_u,
            shift_v,
        )

    def __iter__(self):

        try:
            worker_info = torch.utils.data.get_worker_info()
        except:
            worker_info = None
        if worker_info is None:
            # single-process data loading, return the full iterator
            worker_rank = 0
        else:
            # in a worker process
            worker_rank = worker_info.id
        
        cur_color_paths = self.color_paths_for_worker[worker_rank]
        random.shuffle(cur_color_paths)
        for color_f in cur_color_paths:
            ret_val = self.load_chunk(color_f)
            if ret_val is not None:
                yield ret_val

    @classmethod
    def collate_func(cls, data):
        batch = list(zip(*data))

        color_src = torch.stack(batch[0], dim=0)
        color_tar_to_src = torch.stack(batch[1], dim=0)
        color_src_rendered_area = torch.stack(batch[2], dim=0)
        uv_src = torch.stack(batch[3], dim=0)
        mask = torch.stack(batch[4], dim=0)
        mask_rendered_area = torch.stack(batch[5], dim=0)
        # img_h = batch[6]
        # img_w = batch[7]
        view_idx = batch[8]
        view_ridx = batch[9]
        shift_u = batch[10]
        shift_v = batch[11]

        return {
            "color_src": color_src,
            "color_tar_to_src": color_tar_to_src,
            "color_src_rendered_area": color_src_rendered_area,
            "uv_src": uv_src,
            "mask": mask,
            "mask_rendered_area": mask_rendered_area,
            "index": view_idx,
            "rindex": view_ridx,
            "shift_u": shift_u,
            "shift_v": shift_v,
        }