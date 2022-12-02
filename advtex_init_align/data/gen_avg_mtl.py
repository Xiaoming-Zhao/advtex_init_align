import os
import time
import tqdm
import copy
import glob
import shutil
import argparse
import numpy as np
from PIL import Image

import torch

from advtex_init_align.utils.stream_utils import StreamReader
from advtex_init_align.utils.io_utils import load_mtl_imgs, load_mtl_imgs_vectorize
from advtex_init_align.utils.renderer_utils import (
    load_objs_as_meshes,
    batch_render_img,
    batch_render_img_torch,
)
from advtex_init_align.data.fuse_mrf_and_avg_mtl import fuse_mrf_and_avg_mtl
from advtex_init_align.data.common import ex_tri_mat_to_view_proj_mat, read_scannet_data


STREAM_FILENAME = "Recv.stream"
OBJ_FILENAME = "TexAlign.obj"
MTL_FILENAME = "TexAlign.mtl"

TEXTURE_ATLAS_SIZE = 45

RENDER_BATCHSIZE = 1

EPS = 1e-4


def compute_splatting_weight(val, val_ceil, val_floor, round_type):
    assert val.ndim == 1, f"{val.shape}"
    w = torch.zeros(val.shape, device=val.device)
    ret_val = torch.zeros(val.shape, device=val.device)

    if round_type == "ceil":
        equal_idxs = torch.where(val == val_ceil)[0]
        w[equal_idxs] = 1.0
        ret_val[equal_idxs] = val[equal_idxs]

        non_equal_idxs = torch.where(val != val_ceil)[0]
        w[non_equal_idxs] = val[non_equal_idxs] - val_floor[non_equal_idxs]
        ret_val[non_equal_idxs] = val_ceil[non_equal_idxs]
    elif round_type == "floor":
        equal_idxs = torch.where(val == val_floor)[0]
        w[equal_idxs] = 1.0
        ret_val[equal_idxs] = val[equal_idxs]

        non_equal_idxs = torch.where(val != val_floor)[0]
        w[non_equal_idxs] = val_ceil[non_equal_idxs] - val[non_equal_idxs]
        ret_val[non_equal_idxs] = val_floor[non_equal_idxs]
    else:
        raise ValueError
    if torch.numel(w) > 0:
        assert (
            torch.min(w) >= 0.0 and torch.max(w) <= 1.0
        ), f"{torch.min(w)}, {torch.max(w)}"
        return ret_val, w
    else:
        return None, None


def update_mtl(
    mtl_tensors,
    mtl_mask_tensors,
    mtl_val_cnts,
    mtl_h,
    mtl_w,
    gt_imgs,
    rendered_to_mtl_pix_coords,
    rendered_masks,
    use_all_white_imgs=False,
):

    # we use white RGB to indicate pixels that have been used by renderer
    mask_imgs = torch.ones(gt_imgs.shape, device=gt_imgs.device)

    # [#valid, ]
    valid_bs, valid_rows, valid_cols, _ = torch.where(rendered_masks == 1)

    # print("\nvalid_bs: ", valid_bs.shape, valid_rows.shape, valid_cols.shape)

    # [#valid, 3]
    gt_rgbs = gt_imgs[valid_bs, valid_rows, valid_cols, :]
    mask_rgbs = mask_imgs[valid_bs, valid_rows, valid_cols, :]
    # print("gt_rgbs: ", gt_rgbs.shape)

    # # NOTE: DEBUG
    # gt_rgbs = torch.ones(gt_rgbs.shape, device=gt_rgbs.device)

    # print("\nvalid ratio: ", 3 * valid_bs.shape[0] / torch.numel(gt_imgs))

    rgb_cnts = torch.ones(gt_rgbs.shape, device=gt_rgbs.device)

    # [#valid, 3]
    valid_rendered_to_mtl_pix_coords = rendered_to_mtl_pix_coords[
        valid_bs, valid_rows, valid_cols, :
    ]
    # [#valid, ]
    mtl_idxs = valid_rendered_to_mtl_pix_coords[:, 0]  # .long()
    mtl_rows = valid_rendered_to_mtl_pix_coords[:, 1]  # .long()
    mtl_cols = valid_rendered_to_mtl_pix_coords[:, 2]  # .long()
    # print("mtl_idxs: ", mtl_idxs.shape, mtl_rows.shape, mtl_cols.shape)

    # We do splatting here
    mtl_rows_floor = torch.floor(mtl_rows)
    mtl_rows_ceil = torch.ceil(mtl_rows)
    mtl_cols_floor = torch.floor(mtl_cols)
    mtl_cols_ceil = torch.ceil(mtl_cols)

    for row_round_type in ["ceil", "floor"]:
        for col_round_type in ["ceil", "floor"]:

            mtl_idxs = mtl_idxs.long()

            tmp_mtl_rows, w_row = compute_splatting_weight(
                mtl_rows, mtl_rows_ceil, mtl_rows_floor, row_round_type
            )
            tmp_mtl_cols, w_col = compute_splatting_weight(
                mtl_cols, mtl_cols_ceil, mtl_cols_floor, col_round_type
            )

            if (tmp_mtl_rows is not None) and (tmp_mtl_cols is not None):

                tmp_mtl_rows[tmp_mtl_rows < 0] = 0
                tmp_mtl_rows[tmp_mtl_rows >= mtl_h - 1] = mtl_h - 1
                tmp_mtl_cols[tmp_mtl_cols < 0] = 0
                tmp_mtl_cols[tmp_mtl_cols >= mtl_w - 1] = mtl_w - 1

                tmp_mtl_rows = tmp_mtl_rows.long()
                tmp_mtl_cols = tmp_mtl_cols.long()

                tmp_mtl_ws = (w_row * w_col).unsqueeze(1).expand(-1, 3)
                # print("\nw: ", torch.min(w_row), torch.max(w_row), torch.min(w_col), torch.max(w_col), torch.min(tmp_mtl_ws), torch.max(tmp_mtl_ws), "\n")
                # print("\nrow: ", torch.sum(tmp_mtl_rows > 0), torch.sum(tmp_mtl_cols > 0), "\n")

                tmp_rgbs = gt_rgbs * tmp_mtl_ws
                tmp_masks = mask_rgbs * tmp_mtl_ws

                # print("\ntmp_mtl_ws: ", rgb_cnts.shape, tmp_mtl_ws.shape, "\n")

                # [#valid, ]
                mtl_flat_idxs = (
                    mtl_idxs * mtl_h * mtl_w + tmp_mtl_rows * mtl_w + tmp_mtl_cols
                )
                # [#valid, 3]
                mtl_flat_idxs = mtl_flat_idxs.unsqueeze(1).expand(-1, 3)
                # print("mtl_flat_idxs: ", mtl_flat_idxs.shape, rendered_to_mtl_pix_coords.dtype, mtl_flat_idxs.dtype)

                # https://discuss.pytorch.org/t/why-does-index-add-and-scatter-add-induce-non-deterministic-behavior-on-the-cuda-backend/45544/2
                mtl_tensors.scatter_add_(0, mtl_flat_idxs, tmp_rgbs)
                mtl_mask_tensors.scatter_add_(0, mtl_flat_idxs, tmp_masks)
                mtl_val_cnts.scatter_add_(0, mtl_flat_idxs, tmp_mtl_ws)

    return mtl_tensors, mtl_mask_tensors, mtl_val_cnts


def gen_avg_mtl(
    obj_f,
    mtl_f,
    gt_rgbs,
    view_matrices,
    proj_matrices,
    atlas_size,
    flag_update_mtl=True,
    stream_type="apple",
):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

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
    raw_mtl_imgs, raw_mtl_fnames = load_mtl_imgs_vectorize(mtl_f, return_fname=True)
    raw_mtl_imgs = torch.ByteTensor(raw_mtl_imgs).to(device)
    print(f"\nDone reading mtl imgs with {raw_mtl_imgs.shape}.\n")

    # NHWC
    assert raw_mtl_imgs.ndim == 4 and raw_mtl_imgs.shape[3] == 3

    n_mtl_imgs, mtl_h, mtl_w, _ = raw_mtl_imgs.shape

    if flag_update_mtl:
        mtl_tensors = torch.zeros(raw_mtl_imgs.shape, device=device)
        mtl_mask_tensors = torch.zeros(raw_mtl_imgs.shape, device=device)
        mtl_val_cnts = torch.zeros(raw_mtl_imgs.shape, device=device)

        mtl_tensors = mtl_tensors.reshape((-1, 3))
        mtl_mask_tensors = mtl_mask_tensors.reshape((-1, 3))
        mtl_val_cnts = mtl_val_cnts.reshape((-1, 3))

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

            # Get mtl pix coords
            new_imgs, new_masks, new_extras = batch_render_img_torch(
                stream_type=stream_type,
                device1=device,
                device2=device,
                mesh=mesh,
                texture_atlas_uv_pos=mesh_aux.texture_atlas_uv_pos,
                face_to_mtl_idxs=face_to_mtl_idxs,
                mtl_imgs=raw_mtl_imgs,
                view_matrices=batch_view_matrices,
                proj_matrices=batch_proj_matrices,
                render_size_w=render_size_w,
                render_size_h=render_size_h,
                bin_flag=False,
                timing=True,
                flag_return_idxs=flag_update_mtl,
                flag_return_idxs_float=flag_update_mtl,
                flag_post_process=True,
                flag_return_extra=True,
            )

            # In rendering, the value of one indicates non-rendered pixels, we convert it.
            # new_masks = ((1 - new_masks.astype(np.float32)) * 255).astype(np.uint8)
            rendered_masks = 1 - new_masks.float()

            # print("\nnew_imgs: ", rendered_masks.shape, rendered_masks.dtype, new_imgs.shape, new_imgs.dtype, "\n")

            view_ids = np.arange(tmp_start_i, tmp_end_i)

            if flag_update_mtl:
                cur_gt_rbgs = (
                    torch.FloatTensor(
                        np.stack([gt_rgbs[tmp_i] for tmp_i in view_ids], axis=0)
                    ).to(device)
                    / 255.0
                )

                rendered_to_mtl_pix_coords = new_imgs

                mtl_tensors, mtl_mask_tensors, mtl_val_cnts = update_mtl(
                    mtl_tensors,
                    mtl_mask_tensors,
                    mtl_val_cnts,
                    mtl_h,
                    mtl_w,
                    cur_gt_rbgs,
                    rendered_to_mtl_pix_coords,
                    rendered_masks,
                    use_all_white_imgs=False,
                )
            else:
                for i in range(len(view_ids)):
                    # rgb: [H, W, 3]; mask: [H, W]
                    render_list.append(
                        (
                            view_ids[i],
                            gt_rgbs[view_ids[i]],
                            new_imgs[i, ...].byte().cpu().numpy(),
                        )
                    )

            pbar.update()

        start_id = end_id

    mesh = mesh.no_clone_to(torch.device("cpu"))
    raw_mtl_imgs = raw_mtl_imgs.to(torch.device("cpu"))
    torch.cuda.empty_cache()

    if flag_update_mtl:
        # print("\nmtl_tensors: ", torch.min(mtl_tensors), torch.max(mtl_tensors), torch.sum(mtl_tensors > 0), "\n")
        # print("\nmtl_val_cnts: ", torch.min(mtl_val_cnts), torch.max(mtl_val_cnts), torch.sum(mtl_val_cnts > 0), "\n")

        # average RGB values
        non_zero_flat_idx, _ = torch.where(mtl_val_cnts != 0)
        mtl_tensors[non_zero_flat_idx, :] = (
            mtl_tensors[non_zero_flat_idx, :] / mtl_val_cnts[non_zero_flat_idx, :]
        )
        mtl_mask_tensors[non_zero_flat_idx, :] = (
            mtl_mask_tensors[non_zero_flat_idx, :] / mtl_val_cnts[non_zero_flat_idx, :]
        )

        assert (
            torch.min(mtl_tensors) >= 0.0 - EPS and torch.max(mtl_tensors) <= 1.0 + EPS
        ), f"{torch.min(mtl_tensors)}, {torch.max(mtl_tensors)}"
        assert (
            torch.min(mtl_mask_tensors) >= 0.0 - EPS
            and torch.max(mtl_mask_tensors) <= 1.0 + EPS
        ), f"{torch.min(mtl_mask_tensors)}, {torch.max(mtl_mask_tensors)}"

        avg_mtl_imgs = mtl_tensors.reshape((n_mtl_imgs, mtl_h, mtl_w, 3)).cpu().numpy()
        mtl_mask_imgs = (
            mtl_mask_tensors.reshape((n_mtl_imgs, mtl_h, mtl_w, 3)).cpu().numpy()
        )
        # float32 -> uint8
        avg_mtl_imgs = (avg_mtl_imgs * 255).astype(np.uint8)
        mtl_mask_imgs = (mtl_mask_imgs * 255).astype(np.uint8)
    else:
        avg_mtl_imgs = None
        mtl_mask_imgs = None

    return avg_mtl_imgs, mtl_mask_imgs, render_list, raw_mtl_fnames


def gen_avg_mtl_one_scene(
    scene_id,
    stream_f,
    stream_type,
    obj_f,
    mtl_f,
    atlas_size,
    save_dir,
    debug_dir,
    flag_fuse=True,
    directly_fuse=False,
    scannet_data_dir=None,
    debug_vis=True,
):

    tar_obj_f = os.path.join(save_dir, OBJ_FILENAME)
    tar_mtl_f = os.path.join(save_dir, MTL_FILENAME)

    if not directly_fuse:
        shutil.copyfile(obj_f, tar_obj_f)
        shutil.copyfile(mtl_f, tar_mtl_f)

        print("\n", mtl_f, tar_mtl_f, "\n")

        if scannet_data_dir is None:
            stream_reader = StreamReader(stream_type, stream_f)
            stream_reader.read_stream()
        else:
            # ScanNet's scene has thousands of high-res images.
            # It is too slow to read with struct.unpack.
            # We directly read from disk.
            stream_reader = read_scannet_data(stream_type, scannet_data_dir, read_depth=False)

        gt_rgbs = stream_reader.rgbs
        view_matrices = stream_reader.view_matrices
        proj_matrices = stream_reader.proj_matrices

        tmp_shapes = [_.shape for _ in gt_rgbs]
        tmp_shapes = set(tmp_shapes)
        print("\ngt_rgbs: ", tmp_shapes, "\n")
        print("\n#gt_rgbs: ", len(gt_rgbs), "\n")

        avg_mtl_imgs, mtl_mask_imgs, _, raw_mtl_fnames = gen_avg_mtl(
            obj_f,
            mtl_f,
            gt_rgbs,
            view_matrices,
            proj_matrices,
            atlas_size,
            flag_update_mtl=True,
            stream_type=stream_type,
        )

        for i in range(avg_mtl_imgs.shape[0]):

            # mtl_X_X.png
            fname = os.path.basename(raw_mtl_fnames[i])
            f_idx = "_".join(fname.split(".")[0].split("_")[1:])

            tmp_f = os.path.join(save_dir, f"mtl_{f_idx}.png")
            Image.fromarray(avg_mtl_imgs[i, ...]).save(tmp_f)

            tmp_mask_f = os.path.join(save_dir, f"mtl_mask_{f_idx}.png")
            Image.fromarray(mtl_mask_imgs[i, ...]).save(tmp_mask_f)

        if debug_vis:
            _, _, render_list, _ = gen_avg_mtl(
                obj_f,
                mtl_f,
                gt_rgbs,
                view_matrices,
                proj_matrices,
                atlas_size,
                flag_update_mtl=False,
                stream_type=stream_type,
            )

            for elem in tqdm.tqdm(render_list):
                view_idx, gt_rgb, rendered_rgb = elem
                # print("\ngt_rgb: ", gt_rgb.dtype, gt_rgb.shape, rendered_rgb.dtype, rendered_rgb.shape, "\n")
                cat_img = np.concatenate((gt_rgb, rendered_rgb), axis=1)
                Image.fromarray(cat_img).save(
                    os.path.join(debug_dir, f"{view_idx:05d}.png")
                )

    if flag_fuse:
        # fuse MRF and average mtl images
        print("\nStart fusing MRF and L2 averaged mtl images")
        print("\n", mtl_f, tar_mtl_f, "\n")
        fuse_mrf_and_avg_mtl(mtl_f, tar_mtl_f)


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
        "--save_dir", type=str, required=True, help="save_directory.",
    )
    parser.add_argument(
        "--atlas_size", type=int, required=True, default=TEXTURE_ATLAS_SIZE,
    )
    parser.add_argument(
        "--fuse", type=int, default=1,
    )
    parser.add_argument(
        "--directly_fuse", type=int, required=True, default=0,
    )
    parser.add_argument(
        "--debug_vis", type=int, default=1,
    )
    parser.add_argument(
        "--stream_type", type=str, default="apple", choices=["apple", "scannet"],
    )
    parser.add_argument(
        "--scannet_data_dir", type=str, default=None,
    )

    args = parser.parse_args()

    for i, obj_f in enumerate(args.obj_f_list):
        stream_f = args.stream_f_list[i]

        base_dir = os.path.dirname(obj_f)
        mtl_f = os.path.join(base_dir, MTL_FILENAME)

        scene_id = os.path.basename(os.path.dirname(stream_f))
        # scene_save_dir = os.path.join(args.save_dir, scene_id)
        scene_save_dir = args.save_dir
        os.makedirs(scene_save_dir, exist_ok=True)
        print("\nsave_dir: ", scene_save_dir, "\n")

        if args.debug_vis == 1:
            scene_debug_dir = os.path.join(scene_save_dir, "debug_vis")
            os.makedirs(scene_debug_dir, exist_ok=True)
        else:
            scene_debug_dir = None

        gen_avg_mtl_one_scene(
            scene_id,
            stream_f,
            args.stream_type,
            obj_f,
            mtl_f,
            args.atlas_size,
            scene_save_dir,
            scene_debug_dir,
            flag_fuse=bool(args.fuse),
            directly_fuse=bool(args.directly_fuse),
            scannet_data_dir=args.scannet_data_dir,
            debug_vis=bool(args.debug_vis),
        )


if __name__ == "__main__":
    main()