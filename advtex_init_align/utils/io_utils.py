import os
import sys
import ctypes
import struct
import traceback
import re
import time
import cv2
import png
import glob
import tqdm
import json
import numpy as np
from PIL import Image
from plyfile import PlyData

try:
    import trimesh
    import pygalmesh
except:
    pass

import torch

from advtex_init_align.utils.rotation_utils import matrix_to_axis_angle
from advtex_init_align.utils.camera_utils import proj_mat_from_K


def write_16bit_single_channel_redwood_depth(fname, depth):
    # https://stackoverflow.com/questions/25696615/can-i-save-a-numpy-array-as-a-16-bit-image-using-normal-enthought-python/25814423#25814423
    # Use pypng to write z as a color PNG.
    # Open3D default stores depth in millimeters: http://www.open3d.org/docs/latest/tutorial/Basic/rgbd_image.html
    z = (depth * 1000).astype(np.uint16)

    assert fname.split(".")[-1] == "png"

    with open(fname, "wb") as f:
        writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=16)
        # Convert z to the Python list of lists expected by the png writer.
        z2list = z.reshape(-1, z.shape[1]).tolist()
        writer.write(f, z2list)


def read_16bit_single_channel_redwood_depth(fname):
    # https://stackoverflow.com/questions/32622658/read-16-bit-png-image-file-using-python
    reader = png.Reader(fname)
    data = reader.asDirect()
    pixels = data[2]
    image = []
    for row in pixels:
        row = np.asarray(row)
        row = np.reshape(row, -1)
        image.append(row)
    image = np.stack(image, 0)
    # print(image.dtype)
    # print(image.shape)

    image = image.astype(np.float32) / 1000

    return image


def read_raw_rgbs(raw_dir):
    """Read raw observations."""

    f_list = glob.glob(os.path.join(raw_dir, "*.png"))
    sort_f_list = sorted(
        f_list, key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[1])
    )

    rgbs = []
    for f in tqdm.tqdm(sort_f_list):
        rgbs.append(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))
    return rgbs


def read_raw_rgbs_vectorize(raw_dir):

    f_list = glob.glob(os.path.join(raw_dir, "*.png"))
    sort_f_list = sorted(
        f_list, key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[1])
    )
    # print(sort_f_list)

    rgbs = []
    rgb_shapes = []
    for f in tqdm.tqdm(sort_f_list):
        rgbs.append(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))
        rgb_shapes.append(rgbs[-1].shape)

    # [#cameras, 3]
    rgb_shapes = np.array(rgb_shapes)

    # [#cameras, max_h, max_w, 3]
    vec_rgbs = np.zeros((len(rgbs), *list(np.max(rgb_shapes, axis=0))), dtype=np.uint8)

    for i, tmp_rgb in tqdm.tqdm(enumerate(rgbs), total=len(rgbs)):
        vec_rgbs[i, : rgb_shapes[i, 0], : rgb_shapes[i, 1], :] = tmp_rgb

    return vec_rgbs, rgb_shapes


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(file_list_ordered, key=alphanum_key)


def get_sorted_file_list(path, extension=None):
    if extension is None:
        file_list = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
    else:
        file_list = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
            and os.path.splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list


def read_ply_file(ply_file):

    with open(ply_file, "r") as f:

        lines = f.readlines()

        color_flag = False
        for i, line in enumerate(lines):
            if "element vertex" in line:
                n_vertex = int(line.split(" ")[-1])
            elif "element face" in line:
                n_faces = int(line.split(" ")[-1])
            elif "property uchar red" in line:
                color_flag = True
            elif "end_header" in line:
                start_line = i + 1
                break
            else:
                pass

        vert_lines = lines[start_line : (start_line + n_vertex)]
        face_lines = lines[start_line + n_vertex :]

        vert_infos = np.array(
            [list(map(float, l.strip().split(" "))) for l in vert_lines]
        ).astype(np.float32)
        verts = vert_infos[:, :3]
        if color_flag:
            vert_colors = vert_infos[:, 3:6] / 255
        else:
            vert_colors = None

        faces = np.array([list(map(int, l.strip().split(" "))) for l in face_lines])[
            :, 1:
        ]

        # print(verts.shape, vert_colors.shape, faces.shape)
        # print(verts.dtype, vert_colors.dtype, faces.dtype)
        print(f"read {n_vertex} vertices, {n_faces} faces.\n")

    return verts, vert_colors, faces


def read_obj(obj_f):

    with open(obj_f, "r") as f:
        lines = f.readlines()

    point_vs = []
    tex_vs = []
    faces = {}
    face_cnt = 0
    mtl_name = ""

    for l in tqdm.tqdm(lines):
        if l.split(" ")[0] == "mtllib":
            mtl_f = os.path.join(os.path.dirname(obj_f), l.split(" ")[1].strip())
        elif l.split(" ")[0] == "v":
            assert len(l.split(" ")) == 4
            point_vs.append([float(_.strip()) for _ in l.split(" ")[1:]])
        elif l.split(" ")[0] == "vt":
            assert len(l.split(" ")) == 3
            tex_vs.append([float(_.strip()) for _ in l.split(" ")[1:]])
        elif l.split(" ")[0] == "usemtl":
            mtl_name = l.split(" ")[1].strip()
            if mtl_name not in faces:
                faces[mtl_name] = []
        elif l.split(" ")[0] == "f":
            assert len(l.split(" ")) == 4
            face_info = [_.split("/") for _ in l.split(" ")[1:]]
            # obj is 1-based, change to 0-based
            point_face = [int(_[0]) - 1 for _ in face_info]
            if len(face_info[0]) > 1:
                tex_face = [int(_[1]) - 1 for _ in face_info]
            else:
                tex_face = []
            faces[mtl_name].append([face_cnt, point_face, tex_face])
            face_cnt += 1

    # read mtl file
    with open(mtl_f, "r") as f:
        lines = f.readlines()

    mtl_name_dict = {}

    for l in lines:
        if l.split(" ")[0] == "newmtl":
            mtl_name = l.split(" ")[1].strip()
        if l.strip().split(" ")[0] == "map_Ka":
            f_name = l.strip().split(" ")[1].split(".")[0].strip()
            mtl_name_dict[mtl_name] = f_name

    return np.array(point_vs), np.array(tex_vs), faces, mtl_name_dict


def read_obj_vectorize(obj_f):

    with open(obj_f, "r") as f:
        lines = f.readlines()

    point_vs = []
    tex_vs = []
    faces = []
    face_cnt = 0
    mtl_name = ""
    mtl_name_list = []

    for l in tqdm.tqdm(lines):
        if l.split(" ")[0] == "mtllib":
            mtl_f = os.path.join(os.path.dirname(obj_f), l.split(" ")[1].strip())
        elif l.split(" ")[0] == "v":
            assert len(l.split(" ")) == 4
            point_vs.append([float(_.strip()) for _ in l.split(" ")[1:]])
        elif l.split(" ")[0] == "vt":
            assert len(l.split(" ")) == 3
            tex_vs.append([float(_.strip()) for _ in l.split(" ")[1:]])
        elif l.split(" ")[0] == "usemtl":
            mtl_name = l.split(" ")[1].strip()
            if mtl_name not in mtl_name_list:
                mtl_name_list.append(mtl_name)
        elif l.split(" ")[0] == "f":
            assert len(l.split(" ")) == 4
            face_info = [_.split("/") for _ in l.split(" ")[1:]]
            # obj is 1-based, change to 0-based
            point_face = [int(_[0]) - 1 for _ in face_info]
            if len(face_info[0]) > 1:
                tex_face = [int(_[1]) - 1 for _ in face_info]
            else:
                tex_face = []
            faces.append([*point_face, *tex_face, mtl_name_list.index(mtl_name)])
            face_cnt += 1

    # read mtl file
    with open(mtl_f, "r") as f:
        lines = f.readlines()

    mtl_name_dict = {}

    for l in lines:
        if l.split(" ")[0] == "newmtl":
            mtl_name = l.split(" ")[1].strip()
        if l.strip().split(" ")[0] == "map_Ka":
            f_name = l.strip().split(" ")[1].split(".")[0].strip()
            mtl_name_dict[mtl_name] = f_name

    return (
        np.array(point_vs),
        np.array(tex_vs),
        np.array(faces),
        mtl_name_list,
        mtl_name_dict,
    )


def gen_obj_f_dummy_tex(save_dir, obj_f_name, mtl_f_name, v_coords, face_v_ids):
    """
    v_coords: [#points, 3];
    face_v_ids: [#faces, 3]
    """

    n_faces = face_v_ids.shape[0]
    n_vertices = v_coords.shape[0]

    obj_f = os.path.join(save_dir, obj_f_name)
    mtl_f = os.path.join(save_dir, mtl_f_name)

    with open(obj_f, "wb") as f:

        f.write(f"mtllib {mtl_f_name}\n".encode("utf-8"))

        # write vertex's coordinates [x, y, z]
        # NOTE: vertices may not be completely unique
        # https://stackoverflow.com/a/46864473/6803039
        for i in range(n_vertices):
            xyz = v_coords[i, :]
            f.write(f"v {xyz[0]} {xyz[1]} {xyz[2]}\n".encode("utf-8"))

        # add three dummy texture vertices
        f.write(f"vt 0.1 0.5\n".encode("utf-8"))
        f.write(f"vt 0.2 0.5\n".encode("utf-8"))
        f.write(f"vt 0.5 0.9\n".encode("utf-8"))

        # write faces
        f.write(f"usemtl mtl0\n".encode("utf-8"))

        for f_id in range(n_faces):
            # NOTE: important !!!
            # .obj file's index is 1-based!
            f.write(
                "f {}/{} {}/{} {}/{}\n".format(
                    face_v_ids[f_id, 0] + 1,
                    1,
                    face_v_ids[f_id, 1] + 1,
                    2,
                    face_v_ids[f_id, 2] + 1,
                    3,
                ).encode("utf-8")
            )

        f.write("s off\n".encode("utf-8"))

    # save material file
    mtl_save_str = (
        "newmtl mtl{}\n"
        "  Ka 1.000 1.000 1.000\n"
        "  Kd 1.000 1.000 1.000\n"
        "  Ks 0.000 0.000 0.000\n"
        "  d 1.0\n"
        "  illum 2\n"
    )

    with open(mtl_f, "wb") as f:
        k = 0
        f.write(mtl_save_str.format(k).encode("utf-8"))
        f.write((f"  map_Ka mtl_{k}.png\n" f"  map_Kd mtl_{k}.png\n").encode("utf-8"))

        img_f_name = os.path.join(save_dir, f"mtl_{k}.png")
        if not os.path.exists(img_f_name):
            # put a placeholder
            Image.fromarray(np.zeros((1000, 1000, 3), dtype=np.uint8)).save(img_f_name)

    return obj_f, mtl_f


def add_container_to_mesh(v_coords, face_v_ids, min_facet_angle=34):
    """
    v_coords: [#points, 3];
    face_v_ids: [#faces, 3]
    """
    scene_center = np.mean(v_coords, axis=0, keepdims=True)

    v_radius = np.linalg.norm(v_coords - scene_center, ord=2, axis=1)
    print(
        "\nv_radius: ",
        v_radius.shape,
        scene_center.shape,
        np.max(np.abs(v_coords - scene_center), axis=0),
    )
    max_radius = float(np.max(v_radius))
    max_radius = max_radius + min(0.1, max_radius * 1.005)
    container_prime = pygalmesh.Ball(scene_center[0, :].tolist(), max_radius)

    # # [3, ]
    # max_val = np.max(np.abs(v_coords - scene_center), axis=0) * 1.005
    # # Ellipsoid https://github.com/nschloe/pygalmesh/blob/7c29fbe49f05e32c97194566c29d07bf55887514/src/primitives.hpp#L133
    # container_prime = pygalmesh.Ellipsoid(scene_center[0, :].tolist(), max_val[0], max_val[1], max_val[2])

    # NOTE: this min_facet_angle is important, which somehow controls how many faces we will generate.
    # The more faces we have, the more accurate while the speed is quite slow.
    # Need some tuning wrt different dataset.
    container_mesh = pygalmesh.generate_surface_mesh(
        container_prime,
        min_facet_angle=min_facet_angle,
        seed=123,
    )

    sphere_vs = container_mesh.points

    # this is just sanity check. I think #cells shoule be 1
    assert len(container_mesh.cells) == 1

    sphere_face_v_ids = container_mesh.cells[0].data

    # NOTE: subdivide to get more fine-grained mesh
    sphere_vs, sphere_face_v_ids = trimesh.remesh.subdivide(
        sphere_vs, sphere_face_v_ids
    )

    sphere_face_v_ids += v_coords.shape[0]

    # add container to scene_mesh
    new_v_coords = np.concatenate((v_coords, sphere_vs), axis=0)
    new_face_v_ids = np.concatenate((face_v_ids, sphere_face_v_ids), axis=0)

    return new_v_coords, new_face_v_ids


def load_mtl_imgs(mtl_f, return_fname=False):

    data_dir = os.path.dirname(mtl_f)
    mtl_imgs = {}

    print("Start reading material images ...")
    tmp_start = time.time()

    with open(mtl_f, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == "newmtl":
                material_name = tokens[1]
            elif tokens[0] == "map_Kd":
                # Diffuse texture map
                # Account for the case where filenames might have spaces
                filename = os.path.join(data_dir, line.strip()[7:])
                if return_fname:
                    mtl_imgs[material_name] = (
                        filename,
                        cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB),
                    )
                else:
                    mtl_imgs[material_name] = cv2.cvtColor(
                        cv2.imread(filename), cv2.COLOR_BGR2RGB
                    )
    print(
        f"... complete reading {len(mtl_imgs.keys())} material images in {time.time() - tmp_start:.2f} s.\n"
    )

    return mtl_imgs


def load_mtl_imgs_vectorize(mtl_f, return_fname=False):

    data_dir = os.path.dirname(mtl_f)
    tex_imgs = []
    tex_img_shapes = []
    fnames = []

    print("Start reading material images ...")
    tmp_start = time.time()

    with open(mtl_f, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == "newmtl":
                material_name = tokens[1]
            elif tokens[0] == "map_Kd":
                # Diffuse texture map
                # Account for the case where filenames might have spaces
                filename = os.path.join(data_dir, line.strip()[7:])
                tmp_img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                tex_imgs.append(tmp_img)
                tex_img_shapes.append(tmp_img.shape)
                if return_fname:
                    fnames.append(filename)

    # [#cameras, 3]
    tex_img_shapes = np.array(tex_img_shapes)

    # [#cameras, max_h, max_w, 3]
    vec_tex_imgs = np.zeros(
        (len(tex_imgs), *list(np.max(tex_img_shapes, axis=0))), dtype=np.uint8
    )

    for i, tmp_tex_img in tqdm.tqdm(enumerate(tex_imgs), total=len(tex_imgs)):
        vec_tex_imgs[i, : tex_img_shapes[i, 0], : tex_img_shapes[i, 1], :] = tmp_tex_img

    print(
        f"... complete reading {vec_tex_imgs.shape[0]} material images in {time.time() - tmp_start:.2f} s.\n"
    )

    if return_fname:
        return vec_tex_imgs, fnames
    else:
        return vec_tex_imgs


# ===============================================================================
# read binary file


def read_depth_bin(bin_f, n_cams):

    raw_depths = np.fromfile(bin_f, dtype=np.float32)

    # first two elements are for values of #rows and #cols
    rows = int(raw_depths[0])
    cols = int(raw_depths[1])

    raw_depths = raw_depths[2:]

    # [h * #cameras, w]
    raw_depths = raw_depths.reshape((rows * n_cams, cols))

    # [#cameras, H, W]
    depths = np.concatenate(
        [raw_depths[rows * i : rows * (i + 1), :][None, ...] for i in range(n_cams)],
        axis=0,
    )

    depth_shapes = np.array([[rows, cols] for _ in range(n_cams)])

    return depths, depth_shapes


def old_read_face_cam_pair_bin(bin_f):
    """The storage format:
    for every face: [#valid_cams, valid_1, valid_2, ...], score[valid_i] > score[valid_j] if i < j.

    All values are stored in unsigned short.
    """
    face_cam_pairs = []
    face_cam_pair_cnts = []

    with open(bin_f, "rb") as f:
        while True:
            try:
                n_valid_cams = struct.unpack("H", f.read(2))[0]

                if n_valid_cams != 0:
                    all_valid_idxs = struct.unpack(
                        "H" * n_valid_cams, f.read(2 * n_valid_cams)
                    )
                    face_cam_pairs.append(list(all_valid_idxs))
                else:
                    face_cam_pairs.append([])

                face_cam_pair_cnts.append(n_valid_cams)
            except:
                # traceback.print_exc()
                # err = sys.exc_info()[0]
                # print(err)
                break

    return face_cam_pairs, face_cam_pair_cnts


def read_face_cam_pair_bin(bin_f):
    """The storage format:
    for every face: [#valid_cams, valid_1, valid_2, ...], score[valid_i] > score[valid_j] if i < j.

    All values are stored in unsigned short.
    """
    face_cam_pairs = []
    face_cam_pair_cnts = []

    all_infos = np.fromfile(bin_f, dtype=np.ushort)

    idx = 0
    while True:
        n_valid_cams = all_infos[idx]
        idx += 1

        if n_valid_cams != 0:
            face_cam_pairs.append(list(all_infos[idx : (idx + n_valid_cams)]))
            idx = idx + n_valid_cams
        else:
            face_cam_pairs.append([])

        face_cam_pair_cnts.append(n_valid_cams)

        if idx >= all_infos.shape[0]:
            break

    return face_cam_pairs, face_cam_pair_cnts


def read_face_cam_pair_bin_vectorize(bin_f, n_cams):
    """The storage format:
    for every face: [#valid_cams, valid_1, valid_2, ...], score[valid_i] > score[valid_j] if i < j.

    All values are stored in unsigned short, 0-based index.

    Please note, in CPP, we store face_came_pairs according to obj's face IDs.
    """

    face_cam_pairs, face_cam_pair_cnts = read_face_cam_pair_bin(
        bin_f
    )  # old_read_face_cam_pair_bin(bin_f)

    # max_n = max([len(_) for _ in face_cam_pairs])
    max_n = max(face_cam_pair_cnts)
    vec_face_cam_pairs = np.ones((len(face_cam_pairs), max_n), dtype=np.int) * n_cams

    for i, elem in tqdm.tqdm(enumerate(face_cam_pairs), total=len(face_cam_pairs)):
        vec_face_cam_pairs[i, : len(elem)] = np.array(elem)

    return vec_face_cam_pairs, np.array(face_cam_pair_cnts)


def old_read_ndc_bin(bin_f):
    """The storage format:
    - #rows, #col: unsigned int
    - NDC: float
    """

    NDC_list = []

    with open(bin_f, "rb") as f:
        while True:
            try:
                n_rows, n_cols = struct.unpack("I" * 2, f.read(4 * 2))

                sub_NDC = struct.unpack(
                    "f" * n_rows * n_cols, f.read(4 * n_rows * n_cols)
                )
                # [2 * #cameras, #points]
                sub_NDC = np.array(sub_NDC, dtype=np.float).reshape(n_rows, n_cols)
                sub_NDC1 = sub_NDC[0::2, ...]
                sub_NDC2 = sub_NDC[1::2, ...]
                sub_NDC = np.concatenate(
                    (sub_NDC1[..., None], sub_NDC2[..., None]), axis=2
                )
                NDC_list.append(sub_NDC)
            except:
                # traceback.print_exc()
                # err = sys.exc_info()[0]
                # print(err)
                break

    NDC = np.concatenate(NDC_list, axis=1)

    return NDC


def read_ndc_bin(bin_f, n_cams, c_order=False):
    ndc = np.fromfile(bin_f, dtype=np.float32)

    # [2 * #cameras, #points]
    if c_order:
        ndc = ndc.reshape((2 * n_cams, -1))
    else:
        ndc = np.transpose(ndc.reshape((-1, 2 * n_cams)))

    # [#cameras, #points, 2]
    ndc = np.concatenate([ndc[0::2, :, None], ndc[1::2, :, None]], axis=-1)

    return ndc


def read_cues_bin(
    bin_f,
    face_id_convert_f,
    n_cams,
    c_order=False,
    dummy_truncate=True,
    dummy_val=float("inf"),
    dummy_val_dim=1,
    add_dummy_cam=True,
):
    """return: [#faces, #cameras]."""
    raw_data = np.fromfile(bin_f, dtype=np.float32)

    # [#faces, #cameras]
    if not c_order:
        vec_data = raw_data.reshape((-1, n_cams))
    else:
        vec_data = np.transpose(raw_data.reshape((n_cams, -1)))

    assert vec_data.shape[1] == n_cams

    if dummy_truncate:
        vec_data[vec_data >= dummy_val] = dummy_val

    # NOTE: up till now, face ID is mesh's ID.
    # However, obj file has its own face ID, which is sorted by mtl index.
    # We need to convert to that.
    with open(face_id_convert_f, "r") as f:
        mesh_face_id_to_obj_face_id = json.load(f)
    re_index = np.argsort(
        np.array(
            [mesh_face_id_to_obj_face_id[str(_)] for _ in np.arange(vec_data.shape[0])]
        )
    )
    vec_data = vec_data[re_index, :]

    if add_dummy_cam:
        vec_dummy = np.ones((vec_data.shape[0], dummy_val_dim)) * dummy_val
        vec_data = np.concatenate((vec_data, vec_dummy), axis=1)

    return vec_data


def read_mtl_bin(bin_f):
    """The storage format:
    - #rows, #col: unsigned int
    - tri_local_coords: float
    """

    with open(bin_f, "rb") as f:
        while True:
            try:
                n_rows, n_cols = struct.unpack("I" * 2, f.read(4 * 2))

                local_coords1 = struct.unpack(
                    "f" * n_rows * n_cols, f.read(4 * n_rows * n_cols)
                )
                local_coords2 = struct.unpack(
                    "f" * n_rows * n_cols, f.read(4 * n_rows * n_cols)
                )

                local_coords1 = np.array(local_coords1, dtype=np.float).reshape(
                    n_rows, n_cols
                )
                local_coords2 = np.array(local_coords2, dtype=np.float).reshape(
                    n_rows, n_cols
                )
                local_coords = np.concatenate(
                    (local_coords1[..., None], local_coords2[..., None]), axis=2
                )
            except:
                # traceback.print_exc()
                # err = sys.exc_info()[0]
                # print(err)
                break

    return local_coords


def load_mtl_bins(base_dir, mtl_name_dict):

    mtl_arrs = {}

    for mtl_name in tqdm.tqdm(mtl_name_dict):
        # read binary
        mtl_bin_f = os.path.join(base_dir, f"bin/{mtl_name_dict[mtl_name]}.bin")
        mtl_arrs[mtl_name] = read_mtl_bin(mtl_bin_f)

        img_f_name = os.path.join(base_dir, f"{mtl_name_dict[mtl_name]}.png")
        if not os.path.exists(img_f_name):
            # put a placeholder
            Image.fromarray(
                np.zeros((*mtl_arrs[mtl_name].shape[:2], 3), dtype=np.uint8)
            ).save(img_f_name)

    return mtl_arrs


def load_mtl_bins2(base_dir, mtl_name_ordered_list, mtl_name_dict):

    mtl_arrs = []

    for i, mtl_name in tqdm.tqdm(
        enumerate(mtl_name_ordered_list), total=len(mtl_name_ordered_list)
    ):
        # read binary
        mtl_bin_f = os.path.join(base_dir, f"bin/{mtl_name_dict[mtl_name]}.bin")
        mtl_arrs.append(read_mtl_bin(mtl_bin_f))

        img_f_name = os.path.join(base_dir, f"{mtl_name_dict[mtl_name]}.png")
        if not os.path.exists(img_f_name):
            # put a placeholder
            Image.fromarray(
                np.zeros((*mtl_arrs[-1].shape[:2], 3), dtype=np.uint8)
            ).save(img_f_name)

    return mtl_arrs


def load_mtl_bins_vectorize(base_dir, mtl_name_ordered_list, mtl_name_dict):

    # mtl_arrs = []
    mtl_arrs = None

    for i, mtl_name in tqdm.tqdm(
        enumerate(mtl_name_ordered_list), total=len(mtl_name_ordered_list)
    ):
        assert mtl_name in mtl_name_dict
        # read binary
        mtl_bin_f = os.path.join(base_dir, f"bin/{mtl_name_dict[mtl_name]}.bin")

        if mtl_arrs is None:
            tmp_arr = read_mtl_bin(mtl_bin_f)
            mtl_arrs = np.zeros((len(mtl_name_ordered_list), *tmp_arr.shape))
            mtl_arrs[i, ...] = tmp_arr
        else:
            mtl_arrs[i, ...] = read_mtl_bin(mtl_bin_f)

        img_f_name = os.path.join(base_dir, f"{mtl_name_dict[mtl_name]}.png")
        if not os.path.exists(img_f_name):
            # put a placeholder
            Image.fromarray(
                np.zeros((*mtl_arrs[0, :].shape[:2], 3), dtype=np.uint8)
            ).save(img_f_name)

    # [#mtl, mtl_h, mtl_w, 2]
    # return np.array(mtl_arrs)
    return mtl_arrs


def old_read_cam_mat_bin(bin_f):
    """
    The storage format: view_mat, proj_mat, transform_mat. [4 * #cameras, 4] for each mat
    """
    with open(bin_f, "rb") as f:
        while True:
            try:
                n_rows, n_cols = struct.unpack("f" * 2, f.read(4 * 2))
                n_rows = int(n_rows)
                n_cols = int(n_cols)

                view_mat = struct.unpack(
                    "f" * n_rows * n_cols, f.read(4 * n_rows * n_cols)
                )
                view_mat = np.array(view_mat, dtype=np.float).reshape(
                    (int(n_rows / 4), 4, n_cols)
                )

                proj_mat = struct.unpack(
                    "f" * n_rows * n_cols, f.read(4 * n_rows * n_cols)
                )
                proj_mat = np.array(proj_mat, dtype=np.float).reshape(
                    (int(n_rows / 4), 4, n_cols)
                )

                transform_mat = struct.unpack(
                    "f" * n_rows * n_cols, f.read(4 * n_rows * n_cols)
                )
                transform_mat = np.array(transform_mat, dtype=np.float).reshape(
                    (int(n_rows / 4), 4, n_cols)
                )
            except:
                # traceback.print_exc()
                # err = sys.exc_info()[0]
                # print(err)
                break

    # diff = 0
    # for i in range(view_mat.shape[0]):
    #     tmp_transform_mat = np.matmul(proj_mat[i, :], view_mat[i, :])
    #     diff += np.mean(np.abs(tmp_transform_mat - transform_mat[i, :]))

    # print(n_rows, diff)

    return view_mat, proj_mat, transform_mat


def read_cam_mat_bin(bin_f):
    """
    The storage format: view_mat, proj_mat, transform_mat. [4 * #cameras, 4] for each mat
    """
    raw_data = np.fromfile(bin_f, dtype=np.float32)

    idx = 0
    while True:
        n_rows = int(raw_data[idx])
        n_cols = int(raw_data[idx + 1])
        idx += 2

        view_mat = raw_data[idx : (idx + n_rows * n_cols)].reshape(
            (int(n_rows / 4), 4, n_cols)
        )
        idx += n_rows * n_cols

        proj_mat = raw_data[idx : (idx + n_rows * n_cols)].reshape(
            (int(n_rows / 4), 4, n_cols)
        )
        idx += n_rows * n_cols

        transform_mat = raw_data[idx : (idx + n_rows * n_cols)].reshape(
            (int(n_rows / 4), 4, n_cols)
        )
        idx += n_rows * n_cols

        if idx >= raw_data.shape[0]:
            break

    return view_mat, proj_mat, transform_mat


def read_fvs_processed_data(scene_dense_dir, scale=0.25, only_shapes=False):

    if not only_shapes:
        print("start reading delaunay mesh ...")
        pdata = PlyData.read(os.path.join(scene_dense_dir, "delaunay_photometric.ply"))
        v_coords = np.array(pdata["vertex"][:].tolist()).astype(np.float32)
        face_v_ids = np.array(pdata["face"][:].tolist()).squeeze().astype(np.int)
        print("v_coords: ", v_coords.shape, v_coords.dtype)
        print("face_v_ids: ", face_v_ids.shape, face_v_ids.dtype)
        print("... done.")
    else:
        v_coords = None
        face_v_ids = None

    img_root_dir = os.path.join(scene_dense_dir, f"ibr3d_pw_{scale:.2f}")

    # read RGBs
    rgb_fs = sorted(glob.glob(os.path.join(img_root_dir, "im_*.jpg")))
    raw_rgbs = []
    raw_rgb_shapes = []
    for tmp_f in rgb_fs:
        tmp_rgb = np.array(Image.open(tmp_f))
        raw_rgbs.append(tmp_rgb)
        raw_rgb_shapes.append(tmp_rgb.shape)

    # read depths
    depth_fs = sorted(glob.glob(os.path.join(img_root_dir, "dm_*.npy")))
    raw_depths = []
    raw_depth_shapes = []
    for tmp_f in depth_fs:
        tmp_depth = np.load(tmp_f)
        raw_depths.append(tmp_depth)
        raw_depth_shapes.append(tmp_depth.shape)

    # [#raw_cameras, 3]
    raw_rgb_shapes = np.array(raw_rgb_shapes)
    if not only_shapes:
        # [#raw_cameras, max_h, max_w, 3], uint8
        vec_rgbs = np.zeros(
            (len(raw_rgbs), *list(np.max(raw_rgb_shapes, axis=0))), dtype=np.uint8
        )
        for i, tmp_rgb in tqdm.tqdm(enumerate(raw_rgbs), total=len(raw_rgbs)):
            vec_rgbs[i, : raw_rgb_shapes[i, 0], : raw_rgb_shapes[i, 1], :] = tmp_rgb
    else:
        vec_rgbs = None

    # [#raw_cameras, 2]
    raw_depth_shapes = np.array(raw_depth_shapes)
    if not only_shapes:
        # [#raw_cameras, max_h, max_w], float32
        vec_depths = np.zeros(
            (len(raw_depths), *list(np.max(raw_depth_shapes, axis=0))), dtype=np.float32
        )
        for i, tmp_depth in tqdm.tqdm(enumerate(raw_depths), total=len(raw_depths)):
            vec_depths[
                i, : raw_depth_shapes[i, 0], : raw_depth_shapes[i, 1]
            ] = tmp_depth
    else:
        vec_depths = None

    if not only_shapes:
        # read camera poses and intrinsics
        # [#cam, 3, 3]
        Rs = np.load(os.path.join(img_root_dir, "Rs.npy"))
        # [#cam, 3]
        ts = np.load(os.path.join(img_root_dir, "ts.npy"))
        # [#cam, 3, 3]
        Ks = np.load(os.path.join(img_root_dir, "Ks.npy"))

        # [#cam, 3, 4]
        view_matrices = np.concatenate((Rs, ts[..., np.newaxis]), axis=2)
        # [#cam, 1, 4]
        homo_ones = np.zeros((view_matrices.shape[0], 1, 4))
        homo_ones[..., 3] = 1
        # [#cam, 4, 4]
        view_matrices = np.concatenate((view_matrices, homo_ones), axis=1)

        proj_matrices = []
        for i in range(Ks.shape[0]):
            proj_matrices.append(
                proj_mat_from_K(Ks[i, ...], raw_rgb_shapes[i, 0], raw_rgb_shapes[i, 1])
            )
        # [#cam, 4, 4]
        proj_matrices = np.array(proj_matrices)

        # make 1st elem for height, 2nd elem for width
        proj_matrices = proj_matrices[:, (1, 0, 2, 3), :]
        # flip left-right to align with Apple format's convention
        proj_matrices[:, 1, :] = -proj_matrices[:, 1, :]

        transform_matrices = np.matmul(proj_matrices, view_matrices)
    else:
        view_matrices = None
        proj_matrices = None
        transform_matrices = None

    return (
        v_coords,
        face_v_ids,
        vec_rgbs,
        raw_rgb_shapes,
        vec_depths,
        raw_depth_shapes,
        view_matrices,
        proj_matrices,
        transform_matrices,
    )
