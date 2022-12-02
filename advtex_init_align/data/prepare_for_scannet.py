from genericpath import exists
import os
import sys
import copy
import joblib
import pickle
import tqdm
import time
import yaml
import cv2
import trimesh
import shutil
import traceback
import argparse
import numpy as np
import open3d as o3d
import skimage.io as sio
import multiprocessing as mp
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image, ImageOps


import PIL

PIL.Image.MAX_IMAGE_PIXELS = None

import torch

from advtex_init_align.utils.stream_utils import StreamReader
from advtex_init_align.data.common import cam_mat_to_ex_intr_mat


STREAM_FILENAME = "Recv.stream"
OBJ_FILENAME = "TexAlign.obj"
MTL_FILENAME = "TexAlign.mtl"
SINGLE_MTL_IMG_NAME = "mtl0.png"


def resize_depth(depth, rgb):
    new_depth = np.array(
        Image.fromarray(depth, mode="F").resize(
            (rgb.shape[1], rgb.shape[0]), resample=Image.NEAREST
        )
    )
    return new_depth


def write_mtl_for_o3d_for_single_img(mtl_f, mtl_name):
    mtl_save_str = (
        "newmtl {mtl_name}_1\n"
        "  Ka 1.000 1.000 1.000\n"
        "  Kd 1.000 1.000 1.000\n"
        "  Ks 0.000 0.000 0.000\n"
        "  d 1.0\n"
        "  illum 2\n"
    )

    with open(mtl_f, "wb") as f:
        k = 0
        f.write(mtl_save_str.format(mtl_name=mtl_name).encode("utf-8"))
        f.write((f"  map_Ka mtl{k}.png\n" f"  map_Kd mtl{k}.png\n").encode("utf-8"))


def prepare_data(stream_f, stream_type, save_dir):

    stream_reader = StreamReader(stream_type, stream_f)
    stream_reader.read_stream()

    gt_rgbs = stream_reader.rgbs
    gt_depths = stream_reader.depth_maps
    view_matrices = stream_reader.view_matrices
    proj_matrices = stream_reader.proj_matrices

    data_save_dir = os.path.join(save_dir, "data")
    os.makedirs(data_save_dir, exist_ok=True)

    for idx in tqdm.tqdm(range(len(view_matrices))):
        try:

            cur_raw_rgb = gt_rgbs[idx]
            cur_depth = gt_depths[idx]
            cur_view_mat = view_matrices[idx]
            cur_proj_mat = proj_matrices[idx]

            img_h, img_w, _ = cur_raw_rgb.shape

            if stream_type == "apple":
                # images in Apple stream is left-right flipped
                cur_raw_rgb = np.fliplr(cur_raw_rgb)
                cur_depth = np.fliplr(cur_depth)
            elif stream_type == "scannet":
                # images in ScanNet stream is in canonical orientation
                pass
            else:
                raise ValueError

            if (
                cur_depth.shape[0] != cur_raw_rgb.shape[0]
                or cur_depth.shape[1] != cur_raw_rgb.shape[1]
            ):
                cur_depth = resize_depth(cur_depth, cur_raw_rgb)

            K, world2cam_mat = cam_mat_to_ex_intr_mat(
                stream_type, cur_view_mat, cur_proj_mat, img_h, img_w
            )
            new_K = np.eye(4)
            new_K[:3, :3] = K

            Image.fromarray(cur_raw_rgb).save(
                os.path.join(data_save_dir, f"{idx:05d}_color.png")
            )

            # [H, W]
            np.savez_compressed(
                os.path.join(data_save_dir, f"{idx:05d}_depth.npz"), cur_depth
            )

            np.savetxt(os.path.join(data_save_dir, f"{idx:05d}_intrinsic.txt"), new_K)

            # [4, 4]
            np.savetxt(
                os.path.join(data_save_dir, f"{idx:05d}_pose.txt"), world2cam_mat
            )
        except:
            traceback.print_exc()
            err = sys.exc_info()[0]
            print(err)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream_f_list",
        nargs="+",
        type=str,
        required=True,
        help="file path for stream file.",
    )
    parser.add_argument("--save_dir", type=str, required=True, help="save_directory.")
    parser.add_argument(
        "--stream_type", type=str, default="apple", choices=["apple", "scannet"]
    )

    args = parser.parse_args()

    for i, stream_f in enumerate(args.stream_f_list):

        scene_save_dir = args.save_dir
        os.makedirs(scene_save_dir, exist_ok=True)
        print("\nscene_save_dir: ", scene_save_dir, "\n")

        prepare_data(stream_f, args.stream_type, scene_save_dir)


if __name__ == "__main__":
    main()
