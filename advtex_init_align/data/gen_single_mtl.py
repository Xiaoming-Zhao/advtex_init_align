import os
import time
import h5py
import glob
import json
import gzip
import glob
import time
import tqdm
import copy
import argparse
import numpy as np
import multiprocessing as mp
from PIL import Image

import torch

from advtex_init_align.utils.io_utils import load_mtl_imgs, read_obj


MAX_H = 1920
MAX_w = 1440
CHUNK_BYTES = 1920 * 1440 * 3  # RGB channel, uint8

STREAM_FILENAME = "Recv.stream"
OBJ_FILENAME = "TexAlign.obj"
MTL_FILENAME = "TexAlign.mtl"


def gen_single_mtl_in_a_column(obj_f, mtl_f, save_dir):

    v_coords, v_tex_uvs, faces, mtl_name_dict = read_obj(obj_f)
    mtl_imgs = load_mtl_imgs(mtl_f)

    single_mtl_img = []
    n_mtl_imgs = len(mtl_imgs)

    new_uv_cnt = 0
    new_tex_uvs = []

    new_faces = []

    for i, mtl_k in enumerate(mtl_imgs):
        single_mtl_img.append(mtl_imgs[mtl_k])

        old2new_idx_map = {}
        for elem in faces[mtl_k]:
            _, f_p_idxs, f_uv_idxs = elem

            new_f_uv_idxs = []
            for tmp_idx in f_uv_idxs:
                if tmp_idx not in old2new_idx_map:
                    old2new_idx_map[tmp_idx] = new_uv_cnt
                    new_uv_cnt += 1

                new_f_uv_idxs.append(old2new_idx_map[tmp_idx])

            new_faces.append((f_p_idxs, new_f_uv_idxs))

        unique_tex_uv_idxs = np.array(list(old2new_idx_map.keys()))

        cur_tex_uvs = v_tex_uvs[unique_tex_uv_idxs, :]

        # UV's origin is at bottom-left, we build single mtl from bottom to top
        # +U right, +V up
        # https://stackoverflow.com/a/8851832
        cur_tex_uvs[:, 1] = cur_tex_uvs[:, 1] / n_mtl_imgs + i / n_mtl_imgs
        new_tex_uvs.append(cur_tex_uvs)

    new_tex_uvs = np.concatenate(new_tex_uvs, axis=0)
    print(new_tex_uvs.shape)

    with open(os.path.join(save_dir, OBJ_FILENAME), "w") as f:
        f.write(f"mtllib {MTL_FILENAME}\n")
        f.write("usemtl mtl0\n")

        for i in range(v_coords.shape[0]):
            f.write(f"v {v_coords[i, 0]} {v_coords[i, 1]} {v_coords[i, 2]}\n")

        for i in range(new_tex_uvs.shape[0]):
            f.write(f"vt {new_tex_uvs[i, 0]} {new_tex_uvs[i, 1]}\n")

        for f_elem in new_faces:
            v_idxs, uv_idxs = f_elem
            f.write(
                f"f {v_idxs[0] + 1}/{uv_idxs[0] + 1} {v_idxs[1] + 1}/{uv_idxs[1] + 1} {v_idxs[2] + 1}/{uv_idxs[2] + 1}\n"
            )

    with open("./test.mtl", "w") as f:
        mtl_str = (
            "newmtl mtl0\n"
            "    Ka 1.000 1.000 1.000\n"
            "    Kd 1.000 1.000 1.000\n"
            "    Ks 0.000 0.000 0.000\n"
            "    d 1.0\n"
            "    illum 2\n"
            "    map_Ka mtl0.png\n"
            "    map_Kd mtl0.png\n"
        )
        f.write(mtl_str)

    # NOTE: this is important as we build single texture image from bottom to top
    single_mtl_img.reverse()
    single_mtl_img = np.concatenate(single_mtl_img, axis=0)

    Image.fromarray(single_mtl_img).save(os.path.join(save_dir, "mtl0.png"))


def gen_single_mtl_in_square(obj_f, mtl_f, save_dir):

    v_coords, v_tex_uvs, faces, mtl_name_dict = read_obj(obj_f)
    mtl_imgs = load_mtl_imgs(mtl_f)

    # we enforce all mtl images to have same resolution
    all_hs = [_.shape[0] for _ in mtl_imgs.values()]
    all_ws = [_.shape[1] for _ in mtl_imgs.values()]
    max_h = max(all_hs)
    max_w = max(all_ws)

    n_mtl_imgs = len(mtl_imgs)

    n_row = int(np.ceil((np.sqrt(n_mtl_imgs))))
    n_col = int(np.ceil(n_mtl_imgs / n_row))
    print("rows, cols: ", n_row, n_col)

    single_mtl_img = [[] for _ in range(n_row)]
    for i in range(n_row):
        single_mtl_img[i] = [None for _ in range(n_col)]

    new_uv_cnt = 0
    new_tex_uvs = []

    new_faces = []

    for i, mtl_k in enumerate(mtl_imgs):

        tmp_h, tmp_w, _ = mtl_imgs[mtl_k].shape
        if tmp_h != max_h or tmp_w != max_w:
            # we enforce all mtl images to have same resolution
            assert tmp_h < max_h and tmp_w < max_w, f"{tmp_h}, {tmp_w} ({max_h}, {max_w})"
            cur_mtl_img = np.zeros((max_h, max_w, 3), dtype=mtl_imgs[mtl_k].dtype)
            # bottom-left is the origin
            cur_mtl_img[-tmp_h:, :tmp_w] = mtl_imgs[mtl_k]
        else:
            cur_mtl_img = mtl_imgs[mtl_k]

        # NOTE: we build single mtl image:
        # - from bottom to top
        # - from left to right
        row_i = i // n_col
        col_i = i % n_col
        single_mtl_img[n_row - row_i - 1][col_i] = cur_mtl_img

        old2new_idx_map = {}
        for elem in faces[mtl_k]:
            _, f_p_idxs, f_uv_idxs = elem

            new_f_uv_idxs = []
            for tmp_idx in f_uv_idxs:
                if tmp_idx not in old2new_idx_map:
                    old2new_idx_map[tmp_idx] = new_uv_cnt
                    new_uv_cnt += 1

                new_f_uv_idxs.append(old2new_idx_map[tmp_idx])

            new_faces.append((f_p_idxs, new_f_uv_idxs))

        unique_tex_uv_idxs = np.array(list(old2new_idx_map.keys()))

        cur_tex_uvs = v_tex_uvs[unique_tex_uv_idxs, :]

        if tmp_h != max_h or tmp_w != max_w:
            # we need to scale UV
            cur_tex_uvs[:, 0] = cur_tex_uvs[:, 0] * tmp_w / max_w
            cur_tex_uvs[:, 1] = cur_tex_uvs[:, 1] * tmp_h / max_h

        # UV's origin is at bottom-left, we build single mtl from bottom to top and from left to right
        # +U right, +V up
        # https://stackoverflow.com/a/8851832
        cur_tex_uvs[:, 0] = cur_tex_uvs[:, 0] / n_col + col_i / n_col
        cur_tex_uvs[:, 1] = cur_tex_uvs[:, 1] / n_row + row_i / n_row
        new_tex_uvs.append(cur_tex_uvs)

    new_tex_uvs = np.concatenate(new_tex_uvs, axis=0)
    print(new_tex_uvs.shape)

    new_obj_f = os.path.join(save_dir, OBJ_FILENAME)
    with open(new_obj_f, "w") as f:
        f.write(f"mtllib {MTL_FILENAME}\n")
        f.write("usemtl mtl0\n")

        for i in range(v_coords.shape[0]):
            f.write(f"v {v_coords[i, 0]} {v_coords[i, 1]} {v_coords[i, 2]}\n")

        for i in range(new_tex_uvs.shape[0]):
            f.write(f"vt {new_tex_uvs[i, 0]} {new_tex_uvs[i, 1]}\n")

        for f_elem in new_faces:
            v_idxs, uv_idxs = f_elem
            f.write(
                f"f {v_idxs[0] + 1}/{uv_idxs[0] + 1} {v_idxs[1] + 1}/{uv_idxs[1] + 1} {v_idxs[2] + 1}/{uv_idxs[2] + 1}\n"
            )

    new_mtl_f = os.path.join(save_dir, MTL_FILENAME)
    with open(new_mtl_f, "w") as f:
        mtl_str = (
            "newmtl mtl0\n"
            "    Ka 1.000 1.000 1.000\n"
            "    Kd 1.000 1.000 1.000\n"
            "    Ks 0.000 0.000 0.000\n"
            "    d 1.0\n"
            "    illum 2\n"
            "    map_Ka mtl0.png\n"
            "    map_Kd mtl0.png\n"
        )
        f.write(mtl_str)

    old_tex_h, old_tex_w, _ = single_mtl_img[-1][0].shape
    for i in range(n_row):
        for j in range(n_col):
            if single_mtl_img[i][j] is None:
                single_mtl_img[i][j] = np.zeros(
                    (old_tex_h, old_tex_w, 3), dtype=np.uint8
                )

    single_mtl_img = [np.concatenate(_, axis=1) for _ in single_mtl_img]
    single_mtl_img = np.concatenate(single_mtl_img, axis=0)

    new_mtl_img_f = os.path.join(save_dir, "mtl0.png")
    Image.fromarray(single_mtl_img).save(new_mtl_img_f)

    return new_obj_f, new_mtl_f, new_mtl_img_f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj_f_list",
        nargs="+",
        type=str,
        required=True,
        help="file path for obj file.",
    )
    args = parser.parse_args()

    for obj_f in args.obj_f_list:
        base_dir = os.path.dirname(obj_f)
        mtl_f = os.path.join(base_dir, MTL_FILENAME)
        save_dir = os.path.join(base_dir, "single_mtl")
        os.makedirs(save_dir, exist_ok=True)
        gen_single_mtl_in_square(obj_f, mtl_f, save_dir)


if __name__ == "__main__":
    main()