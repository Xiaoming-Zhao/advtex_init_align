# for adversarial texture optimization at https://github.com/hjwdzh/AdversarialTexture

import os
import sys
import copy
import joblib
import pickle
import tqdm
import time
import yaml
import cv2
import json
import glob
import shutil
import traceback
import argparse
import numpy as np
import skimage.io as sio
import multiprocessing as mp
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image, ImageOps

import PIL

PIL.Image.MAX_IMAGE_PIXELS = None

import torch

from advtex_init_align.utils.stream_utils import StreamReader
from advtex_init_align.utils.io_utils import load_mtl_imgs, load_mtl_imgs_vectorize, read_obj
from advtex_init_align.utils.renderer_utils import (
    load_objs_as_meshes,
    batch_render_img,
    batch_render_img_torch,
)
from advtex_init_align.utils.img_utils import compute_offset_fft
from advtex_init_align.data.gen_single_mtl import gen_single_mtl_in_square
from advtex_init_align.data.common import cam_mat_to_ex_intr_mat, read_scannet_data


STREAM_FILENAME = "Recv.stream"
OBJ_FILENAME = "TexAlign.obj"
MTL_FILENAME = "TexAlign.mtl"
SINGLE_MTL_IMG_NAME = "mtl0.png"

TEXTURE_ATLAS_SIZE = 60

RENDER_BATCHSIZE = 1

OVERLAP_ANGLE_THRESHOLD = 15


def resize_depth(depth, rgb):
    new_depth = np.array(
        Image.fromarray(depth, mode="F").resize(
            (rgb.shape[1], rgb.shape[0]), resample=Image.NEAREST  # Image.BILINEAR,  # Image.NEAREST
        )
    )
    return new_depth


def save_info_to_disk(save_dir, stream_type, idx, cur_view_mat, cur_proj_mat, img_h, img_w,
    cur_rgb=None, cur_raw_rgb=None, cur_depth=None, cur_mask=None, cur_uvs=None, raw_color_dir=None
):
    K, world2cam_mat = cam_mat_to_ex_intr_mat(
        stream_type, cur_view_mat, cur_proj_mat, img_h, img_w
    )
    # [4, 4]
    np.savetxt(os.path.join(save_dir, f"{idx:05d}_pose.txt"), world2cam_mat)

    new_K = np.eye(4)
    new_K[:3, :3] = K
    # flat_K = np.ravel(new_K)

    np.savetxt(os.path.join(save_dir, f"{idx:05d}_intrinsic.txt"), new_K)

    if cur_mask is not None:
        # [H, W]
        Image.fromarray(cur_mask).save(
            os.path.join(save_dir, f"{idx:05d}_mask.png")
        )
    if cur_rgb is not None:
        # [H, W, 3]
        Image.fromarray(cur_rgb.astype("uint8")).save(
            os.path.join(save_dir, f"{idx:05d}_color.png")
        )
    if cur_raw_rgb is not None:
        assert raw_color_dir is not None
        # [H, W, 3]
        Image.fromarray(cur_raw_rgb.astype("uint8")).save(
            os.path.join(raw_color_dir, f"{idx:05d}_raw_color.png")
        )
    if cur_depth is not None:
        # [H, W]
        np.savez_compressed(
            os.path.join(save_dir, f"{idx:05d}_depth.npz"), cur_depth
        )
    if cur_uvs is not None:
        # [H, W, 2]
        np.savez_compressed(
            os.path.join(save_dir, f"{idx:05d}_uv.npz"), cur_uvs
        )
    
    return world2cam_mat


def render_one_scene(
    device,
    obj_f,
    mtl_f,
    atlas_size,
    gt_rgbs,
    view_matrices,
    proj_matrices,
    stream_type="apple",
    flag_return_mtl_uvs=True,
):

    print("\nStart reading obj file ...")
    t1 = time.time()
    mesh, mesh_aux, face_to_mtl_idxs = load_objs_as_meshes(
        [obj_f],
        device=device,
        create_texture_atlas=True,
        texture_atlas_size=atlas_size,
    )
    print(f"\n... done reading obj in {time.time() - t1} seconds.\n")

    # mtl_f = os.path.join(obj_dir, MTL_FILENAME)
    mtl_imgs = load_mtl_imgs_vectorize(mtl_f)
    mtl_imgs = torch.ByteTensor(mtl_imgs).to(device)

    n_views = view_matrices.shape[0]

    render_size_h_list = []
    render_size_w_list = []

    for tmp_rgb in gt_rgbs:
        tmp_h, tmp_w = tmp_rgb.shape[:2]
        render_size_h_list.append(tmp_h)
        render_size_w_list.append(tmp_w)

    assert len(render_size_h_list) == n_views
    assert len(render_size_w_list) == n_views

    render_batch_size = RENDER_BATCHSIZE

    nbatches = int(n_views / render_batch_size)
    pbar = tqdm.tqdm(total=nbatches)
    batch_cnt = 0

    render_list = []

    start_id = 0
    while start_id < n_views:

        if batch_cnt >= nbatches:
            nbatches += 1
            pbar.total = nbatches
            pbar.refresh()

        end_id = n_views

        # find chunk with same image size
        render_size_h = render_size_h_list[start_id]
        render_size_w = render_size_w_list[start_id]
        for i in np.arange(start_id, n_views, 1):
            if (render_size_h_list[i] != render_size_h) or (
                render_size_w_list[i] != render_size_w
            ):
                end_id = i
                break

        for tmp_start_i in np.arange(start_id, end_id, render_batch_size):

            tmp_end_i = min(tmp_start_i + render_batch_size, end_id)
            batch_cnt += 1

            batch_proj_matrices = copy.deepcopy(
                proj_matrices[tmp_start_i:tmp_end_i, ...]
            )
            if stream_type == "scannet":
                # NOTE: when creating stream file from ScanNet output to Apple's format, we flip the 2nd row.
                # We now flip it back to reconstruct ScanNet's original output.
                # If the conversion script for ScanNet changes, this part must be modified accordingly.
                batch_proj_matrices[:, 1, :] = -1 * batch_proj_matrices[:, 1, :]
            elif stream_type == "apple":
                pass
            else:
                raise ValueError

            # NOTE: important!
            # PyTorch3D's vector conventions are row-vector.
            # However, in raw stream, the matrices are used for column-vector.
            # Therefore, we must transpose them.
            # https://github.com/facebookresearch/pytorch3d/blob/14f015d8bf01e8c870b26a28aebbe97460cc398f/pytorch3d/transforms/transform3d.py#L114
            batch_view_matrices = np.transpose(
                view_matrices[tmp_start_i:tmp_end_i, ...], (0, 2, 1)
            )
            batch_proj_matrices = np.transpose(batch_proj_matrices, (0, 2, 1))

            new_imgs, new_masks, new_extras = batch_render_img_torch(
                stream_type=stream_type,
                device1=device,
                device2=device,
                mesh=mesh,
                texture_atlas_uv_pos=mesh_aux.texture_atlas_uv_pos,
                face_to_mtl_idxs=face_to_mtl_idxs,
                mtl_imgs=mtl_imgs,
                view_matrices=batch_view_matrices,
                proj_matrices=batch_proj_matrices,
                render_size_w=render_size_w,
                render_size_h=render_size_h,
                bin_flag=False,
                timing=True,
                flag_return_mtl_uvs=flag_return_mtl_uvs,
                flag_post_process=False,
                flag_return_extra=True,
            )

            if stream_type == "scannet":
                # [B, H, W, 3]
                # NOTE: post-process will produce images with orientation aligned with images in stream
                # - for Apple stream, post-process will produce left-right flipped one
                # - for ScanNet stream, post-process will produce correct orientation
                # So we need to manually flip image here
                new_imgs = torch.flip(torch.flip(new_imgs, dims=(2,)), dims=(1,))
                new_masks = torch.flip(torch.flip(new_masks, dims=(2,)), dims=(1,))

            if flag_return_mtl_uvs:
                # [B, H, W, 3], 1st elem is for mtl index
                new_imgs = new_imgs.cpu().numpy()
                # NOTE: in the renderer, we have already aligned UV to pixel coordinates. Namely,
                # - U elem is for vertical, +U down, so u * img_h = row
                # - V elem is for horizontal, +V right, so v * img_w = col
                # However, to align with more commonly-used definition, we convert it to
                # - U elem is for horizontal, +U right
                # - V elem is for vertical, +V up
                # Ref: https://stackoverflow.com/a/8851832
                new_imgs[..., 1] = 1 - new_imgs[..., 1]
                new_imgs = new_imgs[..., [0, 2, 1]]

                new_masks = new_masks.byte().cpu().numpy()
            else:
                new_imgs = new_imgs.byte().cpu().numpy()
                new_masks = new_masks.byte().cpu().numpy()

            # In rendering, the value of one indicates non-rendered pixels, we convert it.
            # new_masks = ((1 - new_masks.astype(np.float32)) * 255).astype(np.uint8)
            new_masks = 1 - new_masks.astype(np.float32)

            view_ids = np.arange(tmp_start_i, tmp_end_i)
            for i in range(len(view_ids)):
                # rgb: [H, W, 3]; mask: [H, W]
                render_list.append((view_ids[i], new_imgs[i, ...], new_masks[i, ...]))

            pbar.update()

        start_id = end_id

    try:
        mesh = mesh.no_clone_to(torch.device("cpu"))
        mtl_imgs = mtl_imgs.to(torch.device("cpu"))
        torch.cuda.empty_cache()
    except:
        traceback.print_exc()
        err = sys.exc_info()[0]
        print(err)
        sys.exit(1)

    return render_list


def prepare_subproc(subproc_input):

    (
        proc_id,
        stream_type,
        idx_list,
        rgb_list,
        rendered_rgb_list,
        depth_list,
        view_mat_list,
        proj_mat_list,
        mask_list,
        uv_list,
        save_dir,
        debug_dir,
        device,
        for_train,
    ) = subproc_input

    if for_train:
        raw_color_dir = os.path.join(save_dir, "raw_colors")
        os.makedirs(raw_color_dir, exist_ok=True)

    return_info_list = []

    cm = plt.get_cmap("viridis")

    for i in tqdm.tqdm(range(len(rgb_list))):
        try:
            # print(f"\n{proc_id} Start\n")

            idx = idx_list[i]
            cur_raw_rgb = rgb_list[i]
            cur_depth = depth_list[i]
            cur_view_mat = view_mat_list[i]
            cur_proj_mat = proj_mat_list[i]

            # print("\nsave_dir: ", save_dir, "\n")

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

            cur_depth = resize_depth(cur_depth, cur_raw_rgb)

            mask_idx, cur_mask = mask_list[i]
            assert mask_idx == idx, f"{idx}, {mask_idx}"

            cur_rgb = (cur_raw_rgb.astype(np.float32) * cur_mask[..., None]).astype(
                np.uint8
            )
            cur_depth = cur_depth * cur_mask
            cur_mask = (cur_mask * 255).astype(np.uint8)

            uv_idx, cur_uvs = uv_list[i]
            assert uv_idx == idx, f"{idx}, {uv_idx}"

            rendered_rgb_idx, cur_rendered_rgb = rendered_rgb_list[i]
            assert rendered_rgb_idx == idx, f"{idx}, {rendered_rgb_idx}"

            # print(f"\n{proc_id} Done flip\n")

            if for_train:

                # compute mislignment offset
                cur_rgb_torch = (
                    torch.FloatTensor(cur_rgb).unsqueeze(0).permute(0, 3, 1, 2)
                )
                cur_rendered_rgb_torch = (
                    torch.FloatTensor(cur_rendered_rgb).unsqueeze(0).permute(0, 3, 1, 2)
                )
                misalign_offset = compute_offset_fft(
                    cur_rgb_torch, cur_rendered_rgb_torch
                ).cpu()
                shift_u = misalign_offset[0, 0].item()
                shift_v = misalign_offset[0, 1].item()

                world2cam_mat = save_info_to_disk(save_dir, stream_type, idx, cur_view_mat, cur_proj_mat, img_h, img_w,
                    cur_rgb=cur_rgb, cur_raw_rgb=cur_raw_rgb, cur_depth=cur_depth,
                    cur_mask=cur_mask, cur_uvs=cur_uvs, raw_color_dir=raw_color_dir
                )

                return_info_list.append((idx, world2cam_mat, (shift_u, shift_v)))
            else:
                return_info_list.append((idx, None, (None, None)))

            # save visualization for debug
            cur_mask = np.tile(cur_mask[..., None], (1, 1, 3))

            cur_depth = (cur_depth - np.min(cur_depth)) / (
                np.max(cur_depth) - np.min(cur_depth) + 1e-8
            )
            cur_depth = cm(cur_depth)
            # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
            # But we want to convert to RGB in uint8 and save it:
            cur_depth = (cur_depth[..., :3] * 255).astype(np.uint8)

            cat_img = np.concatenate(
                (cur_rgb, cur_rendered_rgb, cur_depth, cur_mask), axis=1
            )

            Image.fromarray(cat_img).save(os.path.join(debug_dir, f"{idx:05d}.png"))
        except:
            traceback.print_exc()
            err = sys.exc_info()[0]
            print(err)
            sys.exit(1)

    return return_info_list


def prepare_one_scene(
    scene_id,
    stream_f,
    stream_type,
    obj_f,
    mtl_f,
    atlas_size,
    save_dir,
    debug_dir,
    flag_already_single_mtl=False,
    for_train=True,
    pure_save_gt_info=False,
    pure_save_gt_info_rgb_only=True,
    # mainly for scannet
    scannet_data_dir=None,
    copy_processed_dir=None,
    prepare_for_test_only=False,
    train_idx_to_raw_idx_map_f=None
):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    print("\nflag_already_single_mtl: ", flag_already_single_mtl, "\n")

    if for_train:
        obj_dir = os.path.join(save_dir, "shape")
        os.makedirs(obj_dir, exist_ok=True)

        if not flag_already_single_mtl:
            # merge separate mtl images into a single one
            print("\nStart merging separate mtl images into a single one.\n")
            new_obj_f, new_mtl_f, new_mtl_img_f = gen_single_mtl_in_square(
                obj_f, mtl_f, obj_dir
            )
        else:
            obj_fname = os.path.basename(obj_f)
            mtl_fname = os.path.basename(mtl_f)
            old_obj_dir = os.path.dirname(obj_f)
            new_obj_f = os.path.join(obj_dir, obj_fname)
            new_mtl_f = os.path.join(obj_dir, mtl_fname)
            shutil.copyfile(obj_f, new_obj_f)
            shutil.copyfile(mtl_f, new_mtl_f)
            shutil.copyfile(
                os.path.join(old_obj_dir, SINGLE_MTL_IMG_NAME),
                os.path.join(obj_dir, SINGLE_MTL_IMG_NAME),
            )
    else:
        new_obj_f = obj_f
        new_mtl_f = mtl_f

    print("\nrender from obj: ", new_obj_f, "\n")
    print("\nrender from mtl: ", new_mtl_f, "\n")

    if copy_processed_dir:

        # make sure obj file is the same
        processed_obj_f = os.path.join(copy_processed_dir, "shape", OBJ_FILENAME)
        tmp_v_coords1, tmp_v_tex_uvs1, tmp_faces1, tmp_mtl_name_dict1 = read_obj(processed_obj_f)
        tmp_v_coords2, tmp_v_tex_uvs2, tmp_faces2, tmp_mtl_name_dict2 = read_obj(new_obj_f)
        assert np.sum(np.abs(tmp_v_tex_uvs1 - tmp_v_tex_uvs2)) == 0

        os.makedirs(os.path.join(save_dir, "raw_colors"), exist_ok=True)
        all_depth_fs = sorted(list(glob.glob(os.path.join(copy_processed_dir, f"*_depth.npz"))))

        def tmp_link(src_f, tar_f):
            assert os.path.exists(src_f), src_f
            if os.path.exists(tar_f) and os.path.islink(tar_f):
                os.unlink(tar_f)
            os.symlink(src_f, tar_f)
        
        tmp_link(os.path.join(copy_processed_dir, f"pose_pair.pkl"), os.path.join(save_dir, f"pose_pair.pkl"))
        tmp_link(os.path.join(copy_processed_dir, f"offset_dict.p"), os.path.join(save_dir, f"offset_dict.p"))

        for i in tqdm.tqdm(range(len(all_depth_fs))):
            tmp_link(os.path.join(copy_processed_dir, f"{i:05d}_color.png"), os.path.join(save_dir, f"{i:05d}_color.png"))
            tmp_link(os.path.join(copy_processed_dir, f"{i:05d}_depth.npz"), os.path.join(save_dir, f"{i:05d}_depth.npz"))
            tmp_link(os.path.join(copy_processed_dir, f"{i:05d}_intrinsic.txt"), os.path.join(save_dir, f"{i:05d}_intrinsic.txt"))
            tmp_link(os.path.join(copy_processed_dir, f"{i:05d}_mask.png"), os.path.join(save_dir, f"{i:05d}_mask.png"))
            tmp_link(os.path.join(copy_processed_dir, f"{i:05d}_pose.txt"), os.path.join(save_dir, f"{i:05d}_pose.txt"))
            tmp_link(os.path.join(copy_processed_dir, f"{i:05d}_uv.npz"), os.path.join(save_dir, f"{i:05d}_uv.npz"))

            tmp_link(os.path.join(copy_processed_dir, f"raw_colors/{i:05d}_raw_color.png"), os.path.join(save_dir, f"raw_colors/{i:05d}_raw_color.png"))
        return

    if scannet_data_dir is None:
        stream_reader = StreamReader(stream_type, stream_f)
        stream_reader.read_stream()
    else:
        # ScanNet's scene has thousands of high-res images.
        # It is too slow to read with struct.unpack.
        # We directly read from disk.
        print("\nscannet_data_dir: ", scannet_data_dir, "\n")
        stream_reader = read_scannet_data(stream_type, scannet_data_dir, read_depth=True, for_train=for_train)
    
    gt_rgbs = stream_reader.rgbs
    depth_maps = stream_reader.depth_maps
    view_matrices = stream_reader.view_matrices
    proj_matrices = stream_reader.proj_matrices

    if hasattr(stream_reader, "raw_idx_list"):
        raw_idx_list = stream_reader.raw_idx_list
    else:
        raw_idx_list = np.arange(len(gt_rgbs)).tolist()

    if pure_save_gt_info:
        print("\nStart saving GT rgbs\n")
        if prepare_for_test_only:
            assert train_idx_to_raw_idx_map_f is not None

            with open(train_idx_to_raw_idx_map_f, "r") as f:
                train_idx_to_raw_idx_map = json.load(f)
            train_idxs = list(train_idx_to_raw_idx_map.values())
            idx_list = [_ for _ in range(len(gt_rgbs)) if _ not in train_idxs]
        else:
            idx_list = np.arange(len(gt_rgbs)).tolist()

        for idx in tqdm.tqdm(idx_list):
            if stream_type == "apple":
                tmp_rgb = np.fliplr(gt_rgbs[idx])
            elif stream_type == "scannet":
                tmp_rgb = gt_rgbs[idx]
            else:
                raise ValueError
            Image.fromarray(tmp_rgb).save(
                os.path.join(save_dir, f"{idx:05d}_raw_color.png"),
            )

            if not pure_save_gt_info_rgb_only:
                # We need to save other information for rendering from texture later
                img_h, img_w, _ = tmp_rgb.shape
                save_info_to_disk(save_dir, stream_type, idx, view_matrices[idx, ...], proj_matrices[idx, ...], img_h, img_w, cur_depth=depth_maps[idx])
    else:
        # get rendered RGB
        print("\nStart rendering to get rendered RGB.\n")
        flag_return_mtl_uvs = False
        rgb_rendered_list = render_one_scene(
            device,
            new_obj_f,
            new_mtl_f,
            atlas_size,
            gt_rgbs,
            view_matrices,
            proj_matrices,
            stream_type=stream_type,
            flag_return_mtl_uvs=flag_return_mtl_uvs,
        )

        if for_train:
            # get UV src coords
            print("\nStart rendering to get UV coords.\n")
            flag_return_mtl_uvs = True
            uv_rendered_list = render_one_scene(
                device,
                new_obj_f,
                new_mtl_f,
                atlas_size,
                gt_rgbs,
                view_matrices,
                proj_matrices,
                stream_type=stream_type,
                flag_return_mtl_uvs=flag_return_mtl_uvs,
            )
        else:
            uv_rendered_list = [None for _ in range(len(rgb_rendered_list))]

        all_masks = []
        all_rendered_rgbs = []
        all_uvs = []

        print("\nStart saving UV coords to disk.\n")
        for elem_uv, elem_rgb in tqdm.tqdm(
            zip(uv_rendered_list, rgb_rendered_list), total=len(uv_rendered_list)
        ):
            tmp_idx, tmp_rendered_rgb, tmp_mask = elem_rgb
            all_rendered_rgbs.append((raw_idx_list[tmp_idx], tmp_rendered_rgb))

            all_masks.append((raw_idx_list[tmp_idx], tmp_mask))

            if for_train:
                # tmp_mask: [H, W]
                tmp_idx, tmp_rendered_uv, tmp_mask = elem_uv

                # we only have single mtl image
                assert np.sum(tmp_rendered_uv[..., 0]) == 0
                tmp_uvs = tmp_rendered_uv[..., 1:]
                all_uvs.append((raw_idx_list[tmp_idx], tmp_uvs))
            else:
                all_uvs.append((raw_idx_list[tmp_idx], None))

        print("\nStart saving imgs parallelly.\n")

        nproc = 10

        idx_list = [[] for i in range(nproc)]
        rgb_list = [[] for i in range(nproc)]
        depth_list = [[] for i in range(nproc)]
        view_mat_list = [[] for i in range(nproc)]
        proj_mat_list = [[] for i in range(nproc)]
        rendered_rgb_list = [[] for i in range(nproc)]
        mask_list = [[] for i in range(nproc)]
        uv_list = [[] for i in range(nproc)]

        for i in range(len(stream_reader.rgbs)):
            idx_list[i % nproc].append(raw_idx_list[i])
            rgb_list[i % nproc].append(stream_reader.rgbs[i])
            depth_list[i % nproc].append(stream_reader.depth_maps[i])
            view_mat_list[i % nproc].append(stream_reader.view_matrices[i])
            proj_mat_list[i % nproc].append(stream_reader.proj_matrices[i])
            rendered_rgb_list[i % nproc].append(all_rendered_rgbs[i])
            mask_list[i % nproc].append(all_masks[i])
            uv_list[i % nproc].append(all_uvs[i])

        with mp.get_context("spawn").Pool(nproc) as pool:
            gather_output = pool.map(
                prepare_subproc,
                zip(
                    range(nproc),
                    [stream_type for _ in range(nproc)],
                    idx_list,
                    rgb_list,
                    rendered_rgb_list,
                    depth_list,
                    view_mat_list,
                    proj_mat_list,
                    mask_list,
                    uv_list,
                    [save_dir for _ in range(nproc)],
                    [debug_dir for _ in range(nproc)],
                    [device for _ in range(nproc)],
                    [for_train for _ in range(nproc)],
                ),
            )
            pool.close()
            pool.join()

        if for_train:
            # process world2cam mat
            all_infos = []
            for elem in gather_output:
                all_infos.extend(elem)

            all_infos = sorted(all_infos, key=lambda x: x[0])

            n_pairs = 0
            t = []
            min_len = 10000
            for i in range(len(all_infos)):
                assert all_infos[i][0] == i, f"{all_infos[i][0]}, {i}"
                a = []
                for j in range(len(all_infos)):
                    angle = np.dot(all_infos[i][1][:, 2], all_infos[j][1][:, 2])
                    if angle > np.cos(OVERLAP_ANGLE_THRESHOLD / 180.0 * np.pi):
                        a.append(j)
                if len(a) < min_len:
                    min_len = len(a)
                t.append(a)
                n_pairs += len(a)
            print(f"\nFound {n_pairs} pairs of camera poses.\n")

            with open(os.path.join(save_dir, "pose_pair.pkl"), "wb") as fp:
                pickle.dump(t, fp)

            # save misalignment offset
            misalign_offset_dict = {}
            for i in range(len(all_infos)):
                view_idx, _, shift_offset = all_infos[i]
                misalign_offset_dict[view_idx] = shift_offset

            with open(os.path.join(save_dir, f"offset_dict.p"), "wb") as f:
                joblib.dump(misalign_offset_dict, f, compress="lz4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream_f_list",
        nargs="+",
        type=str,
        required=True,
        help="file path for stream file.",
    )
    parser.add_argument(
        "--obj_f_list",
        nargs="+",
        type=str,
        required=True,
        help="file path for obj file.",
    )
    parser.add_argument(
        "--mtl_fname", type=str, default=MTL_FILENAME,
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="save_directory.",
    )
    parser.add_argument(
        "--atlas_size", type=int, required=True, default=TEXTURE_ATLAS_SIZE,
    )
    parser.add_argument(
        "--already_single_mtl",
        type=int,
        required=True,
        help="whether mtl is already a single one.",
    )
    parser.add_argument(
        "--for_train",
        type=int,
        required=True,
        help="whether generating data for training.",
    )
    parser.add_argument(
        "--pure_save_gt_info", type=int, default=0, help="whether only save GT infos.",
    )
    parser.add_argument(
        "--pure_save_gt_info_rgb_only", type=int, default=0, help="whether only save GT rgbs.",
    )
    parser.add_argument(
        "--stream_type", type=str, default="apple", choices=["apple", "scannet"],
    )
    parser.add_argument(
        "--scannet_data_dir", type=str, default=None,
    )
    parser.add_argument(
        "--copy_processed_dir", type=str, default=None,
    )
    parser.add_argument(
        "--prepare_for_test_only", type=int, default=0,
    )
    parser.add_argument(
        "--train_idx_to_raw_idx_map_f", type=str, default=None,
    )

    args = parser.parse_args()

    for i, obj_f in enumerate(args.obj_f_list):
        stream_f = args.stream_f_list[i]

        scene_save_dir = args.save_dir
        os.makedirs(scene_save_dir, exist_ok=True)
        print("\nscene_save_dir: ", scene_save_dir, "\n")

        if args.pure_save_gt_info == 0:
            base_dir = os.path.dirname(obj_f)
            mtl_f = os.path.join(base_dir, args.mtl_fname)

            scene_id = os.path.basename(os.path.dirname(stream_f))
            # scene_save_dir = os.path.join(args.save_dir, scene_id)

            scene_debug_dir = os.path.join(scene_save_dir, "debug_vis")
            os.makedirs(scene_debug_dir, exist_ok=True)
        else:
            scene_id = None
            mtl_f = None
            scene_debug_dir = None

        prepare_one_scene(
            scene_id,
            stream_f,
            args.stream_type,
            obj_f,
            mtl_f,
            args.atlas_size,
            scene_save_dir,
            scene_debug_dir,
            flag_already_single_mtl=bool(args.already_single_mtl),
            for_train=bool(args.for_train),
            pure_save_gt_info=bool(args.pure_save_gt_info),
            pure_save_gt_info_rgb_only=bool(args.pure_save_gt_info_rgb_only),
            scannet_data_dir=args.scannet_data_dir,
            copy_processed_dir=args.copy_processed_dir,
            prepare_for_test_only=bool(args.prepare_for_test_only),
            train_idx_to_raw_idx_map_f=args.train_idx_to_raw_idx_map_f,
        )


if __name__ == "__main__":
    main()
