from concurrent.futures import process
import os
import json
import glob
import tqdm
import struct
import trimesh
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from advtex_init_align.utils.stream_utils import StreamReader
from advtex_init_align.data.common import cam_mat_to_ex_intr_mat

STREAM_FILENAME = "Recv.stream"


def main(
    raw_stream_f,
    stream_type,
    save_dir,
):

    stream_reader = StreamReader(stream_type, str(raw_stream_f))

    stream_reader.read_stream()

    gt_rgbs = stream_reader.rgbs
    depth_maps = stream_reader.depth_maps
    view_matrices = stream_reader.view_matrices
    proj_matrices = stream_reader.proj_matrices

    rgb_dir = save_dir / "rgb"
    depth_dir = save_dir / "depth"
    mat_dir = save_dir / "mat"

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(mat_dir, exist_ok=True)

    n_views = len(gt_rgbs)

    for i in tqdm.tqdm(range(n_views)):

        fname = f"{i:05d}"

        cur_rgb = np.fliplr(gt_rgbs[i])
        Image.fromarray(cur_rgb).save(rgb_dir / f"{fname}.png")

        cur_depth = np.fliplr(depth_maps[i])
        plt.imsave(depth_dir / f"{fname}.png", cur_depth, cmap="plasma")
        np.savez(depth_dir / f"{fname}.npz", depth=cur_depth)
        
        cur_h, cur_w, _ = cur_rgb.shape
        cur_view_mat = stream_reader.view_matrices[i, ...]
        cur_proj_mat = stream_reader.proj_matrices[i, ...]

        cur_K, cur_w2c = cam_mat_to_ex_intr_mat("apple", cur_view_mat, cur_proj_mat, cur_h, cur_w)
        np.savez(mat_dir / f"{fname}.npz", K=cur_K, w2c=cur_w2c)


    # Soft link all vertices and faces files
    raw_dir = raw_stream_f.parent.resolve()

    all_vertex_fs = list(glob.glob(os.path.join(raw_dir, "Vertices.*")))
    all_face_fs = list(glob.glob(os.path.join(raw_dir, "Faces.*")))
    assert len(all_vertex_fs) == len(
        all_face_fs
    ), f"{len(all_vertex_fs)}, {len(all_face_fs)}"

    all_verts = []
    all_faces = []

    cum_sum = 0

    for i in tqdm.tqdm(range(len(all_face_fs))):
        cur_vert_f = os.path.join(raw_dir, f"Vertices.{i}")
        cur_face_f = os.path.join(raw_dir, f"Faces.{i}")

        with open(cur_vert_f, "rb") as f:
            # [#vertices, 3]
            cur_verts = np.fromfile(f, np.float32).reshape((-1, 3))
            all_verts.append(cur_verts)
        
        with open(cur_face_f, "rb") as f:
            # [#faces, 3]
            cur_faces = np.fromfile(f, np.uint32).reshape((-1, 3)) + cum_sum
            all_faces.append(cur_faces)
        
        cum_sum += cur_verts.shape[0]
        
    all_verts = np.concatenate(all_verts, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    _ = mesh.export(save_dir / "mesh.ply")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream_dir",
        type=str,
        required=True,
        help="file path for stream file.",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="save_directory.",
    )
    parser.add_argument(
        "--stream_type", type=str, default="apple", choices=["apple"],
    )

    args = parser.parse_args()

    stream_f = pathlib.Path(args.stream_dir) / STREAM_FILENAME
    main(
        stream_f,
        args.stream_type,
        pathlib.Path(args.save_dir),
    )

