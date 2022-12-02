import os
import time
import cv2
import ctypes
import h5py
import copy
import tqdm

# import line_profiler
import numpy as np
import multiprocessing as mp
from PIL import Image, ImageOps
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler

from advtex_init_align.utils.logging import dummy_context_mgr
from advtex_init_align.utils.cue_utils import world_to_cam_coords


# --------------------------------------------------------------------------------------------------------------------------
# Numpy Version


def post_render_process(stream_type, target_view, mask, render_size_w, render_size_h):
    img = Image.fromarray(np.uint8(target_view))
    if stream_type == "apple":
        new_img = np.array(
            ImageOps.mirror(img),
            dtype=np.uint8,
        )
    elif stream_type == "scannet":
        new_img = np.array(
            ImageOps.flip(ImageOps.mirror(img)),
            dtype=np.uint8,
        )
    else:
        raise ValueError

    mask = Image.fromarray(np.uint8(mask))
    if stream_type == "apple":
        new_mask = np.array(
            ImageOps.mirror(mask),
            dtype=np.uint8,
        )
    elif stream_type == "scannet":
        new_mask = np.array(
            ImageOps.flip(ImageOps.mirror(mask)),
            dtype=np.uint8,
        )
    else:
        raise ValueError

    return new_img, new_mask


def batch_uv_to_pixel_coords(atlas_uv_pos, fragments, mtl_size_h, mtl_size_w):
    """modified from https://github.com/facebookresearch/pytorch3d/blob/aa4cc0adbce5f6277a862f5cedacd1c4555bb66e/pytorch3d/renderer/mesh/textures.py#L480"""
    N, H, W, K = fragments.pix_to_face.shape
    R = atlas_uv_pos.shape[1]
    bary = fragments.bary_coords
    pix_to_face = fragments.pix_to_face

    bary_w01 = bary[..., :2]
    # pyre-fixme[16]: `bool` has no attribute `__getitem__`.
    mask = (pix_to_face < 0)[..., None]
    bary_w01 = torch.where(mask, torch.zeros_like(bary_w01), bary_w01)
    w_xy = (bary_w01 * R).to(torch.int64)  # (N, H, W, K, 2)

    below_diag = (
        bary_w01.sum(dim=-1) * R - w_xy.float().sum(dim=-1)
    ) <= 1.0  # (N, H, W, K)
    w_x, w_y = w_xy.unbind(-1)
    w_x = torch.where(below_diag, w_x, (R - 1 - w_x))
    w_y = torch.where(below_diag, w_y, (R - 1 - w_y))

    # https://github.com/facebookresearch/pytorch3d/blob/aa4cc0adbce5f6277a862f5cedacd1c4555bb66e/pytorch3d/renderer/mesh/textures.py#L516
    # follow the mechanism of sample_texture, we need to pack uv_pos into [N, #faces, texture_atlas_size, texture_atlas_size, 2]
    cat_atlas_uv_pos = torch.cat([atlas_uv_pos for i in range(N)], axis=0)

    # [(N, mtl_size_h, mtl_size_w, 2)]
    mtl_uv_coords = cat_atlas_uv_pos[pix_to_face, w_y, w_x].squeeze(3).cpu().numpy()

    # NOTE: must flip the v coordinates.
    # since when computing uv coordinates, the image is flipped: https://github.com/facebookresearch/pytorch3d/blob/f5383a7e5a79bd9e912af0fb0199c557b9987877/pytorch3d/io/mtl_io.py#L110
    mtl_uv_coords[..., 1] = 1.0 - mtl_uv_coords[..., 1]

    # in uv, u is for horizontal. However, in pixel coords, first elem is for vertical
    mtl_pixel_coords = mtl_uv_coords[..., [1, 0]]
    mtl_pixel_coords = np.floor(
        mtl_pixel_coords * np.array([mtl_size_h - 1, mtl_size_w - 1])[np.newaxis, :]
    ).astype(np.int)

    return mtl_uv_coords, mtl_pixel_coords


# -------------------------------------------------------------------------------------------------
# Torch Version


def post_pix_assign_process_torch(
    stream_type,
    target_view,
    mask,
    extra_infos,
    render_size_w,
    render_size_h,
):
    def flip_func(input_tensor, stream_type):
        if stream_type == "apple":
            input_tensor = torch.flip(input_tensor, dims=(2,))
        elif stream_type == "scannet":
            input_tensor = torch.flip(torch.flip(input_tensor, dims=(2,)), dims=(1,))
        else:
            raise ValueError
        return input_tensor

    if target_view is not None:
        # new_imgs = F.interpolate(
        #     target_view.float().permute(0, 3, 1, 2),
        #     size=(render_size_h, render_size_w),
        #     mode="bilinear",
        # ).permute(0, 2, 3, 1)
        if target_view.dtype == torch.long:
            new_imgs = target_view
        else:
            new_imgs = target_view.float()
        new_imgs = flip_func(new_imgs, stream_type)
    else:
        new_imgs = None

    if mask is not None:
        # new_masks = F.interpolate(
        #     mask.float().unsqueeze(1), size=(render_size_h, render_size_w), mode="bilinear"
        # ).permute(0, 2, 3, 1)
        new_masks = mask.float().unsqueeze(-1)
        new_masks = flip_func(new_masks, stream_type)
    else:
        new_masks = None

    new_extra_infos = {}
    for k, v in extra_infos.items():
        if v is not None:
            ori_dtype = v.dtype
            new_extra_infos[k] = flip_func(v.float(), stream_type).to(ori_dtype)

    return new_imgs, new_masks, new_extra_infos


# @profile
def batch_uv_to_pixel_coords_torch(
    device,
    atlas_uv_pos,
    fragments,
    mtl_size_h,
    mtl_size_w,
    save_mem=False,
    return_idxs_float=False,
):
    """modified from https://github.com/facebookresearch/pytorch3d/blob/aa4cc0adbce5f6277a862f5cedacd1c4555bb66e/pytorch3d/renderer/mesh/textures.py#L480"""
    N, H, W, K = fragments.pix_to_face.shape  # K = pixel's #top_faces
    R = atlas_uv_pos.shape[1]
    if atlas_uv_pos.device != device:
        atlas_uv_pos = atlas_uv_pos.to(device)
    bary = fragments.bary_coords
    pix_to_face = fragments.pix_to_face
    if bary.device != device:
        bary = bary.to(device)
        pix_to_face = pix_to_face.to(device)

    # print(fragments.pix_to_face.shape, bary.shape)

    bary_w01 = bary[..., :2]
    # pyre-fixme[16]: `bool` has no attribute `__getitem__`.
    mask = (pix_to_face < 0)[..., None]
    bary_w01 = torch.where(mask, torch.zeros_like(bary_w01), bary_w01)
    w_xy = (bary_w01 * R).to(torch.int64)  # (N, render_H, render_W, K, 2)

    below_diag = (
        bary_w01.sum(dim=-1) * R - w_xy.float().sum(dim=-1)
    ) <= 1.0  # (N, render_H, render_W, K)
    w_x, w_y = w_xy.unbind(-1)
    w_x = torch.where(below_diag, w_x, (R - 1 - w_x))  # (N, render_H, render_W, K)
    w_y = torch.where(below_diag, w_y, (R - 1 - w_y))  # (N, render_H, render_W, K)

    # print(below_diag.size(), w_xy.size(), w_x.size(), w_y.size())

    if not save_mem:
        # https://github.com/facebookresearch/pytorch3d/blob/aa4cc0adbce5f6277a862f5cedacd1c4555bb66e/pytorch3d/renderer/mesh/textures.py#L516
        # follow the mechanism of sample_texture, we need to pack uv_pos into [N * #faces, texture_atlas_size, texture_atlas_size, 2]
        cat_atlas_uv_pos = torch.cat([atlas_uv_pos for i in range(N)], axis=0)

        # print("\ncat_atlas_uv: ", atlas_uv_pos.shape, cat_atlas_uv_pos.shape, w_y.shape, "\n")

        # [(N, render_H, render_w, 2)]
        mtl_uv_coords = cat_atlas_uv_pos[pix_to_face, w_y, w_x].squeeze(3)
    else:
        assert K == 1
        n_faces = atlas_uv_pos.size(0)
        mtl_uv_coords = torch.zeros((N, H, W, 2)).to(device)
        # mtl_uv_coords = []
        for i in range(N):
            cur_pixel_to_face = pix_to_face[i, ...] - i * n_faces
            cur_pixel_to_face[cur_pixel_to_face < 0] = -1

            # cur_pixel_to_face[torch.where(cur_pixel_to_face < 0)] = -1
            mtl_uv_coords[i, :] = atlas_uv_pos[
                cur_pixel_to_face, w_y[i, :], w_x[i, :]
            ].squeeze()
            mtl_uv_coords[i, :] = mtl_uv_coords[i, :] * (cur_pixel_to_face >= 0).float()

    # NOTE: must flip the v coordinates.
    # since when computing uv coordinates, the image is flipped:
    # https://github.com/facebookresearch/pytorch3d/blob/f5383a7e5a79bd9e912af0fb0199c557b9987877/pytorch3d/io/mtl_io.py#L110
    mtl_uv_coords[..., 1] = 1.0 - mtl_uv_coords[..., 1]
    mtl_uv_coords = mtl_uv_coords[..., [1, 0]]

    # NOTE: in CPP: when computing mtl UV, we include the minimum-cover-rectangle's top/left and exclude bottom/right.
    # Namely, we should use floor for both row and column.
    mtl_pixel_coords = mtl_uv_coords * torch.FloatTensor([mtl_size_h, mtl_size_w]).unsqueeze(0).to(device)
    if not return_idxs_float:
        mtl_pixel_coords = torch.floor(mtl_pixel_coords).long()
    mtl_pixel_coords[
        torch.where(mtl_pixel_coords[..., 0] >= mtl_size_h)[0],
        torch.where(mtl_pixel_coords[..., 0] >= mtl_size_h)[1],
        torch.where(mtl_pixel_coords[..., 0] >= mtl_size_h)[2],
        0,
    ] = (
        mtl_size_h - 1
    )
    mtl_pixel_coords[
        torch.where(mtl_pixel_coords[..., 1] >= mtl_size_w)[0],
        torch.where(mtl_pixel_coords[..., 1] >= mtl_size_w)[1],
        torch.where(mtl_pixel_coords[..., 1] >= mtl_size_w)[2],
        1,
    ] = (
        mtl_size_w - 1
    )

    return mtl_uv_coords, mtl_pixel_coords


def batch_uv_to_pixel_coords_adaptive_torch(
    device,
    atlas_uv_pos,
    atlas_tex_size,
    atlas_start_idx,
    fragments,
    mtl_size_h,
    mtl_size_w,
    save_mem=True,
):
    """modified from https://github.com/facebookresearch/pytorch3d/blob/aa4cc0adbce5f6277a862f5cedacd1c4555bb66e/pytorch3d/renderer/mesh/textures.py#L480"""
    N, H, W, K = fragments.pix_to_face.shape  # K = pixel's #top_faces
    # R = atlas_uv_pos.shape[1]
    if atlas_uv_pos.device != device:
        atlas_uv_pos = atlas_uv_pos.to(device)
    bary = fragments.bary_coords
    pix_to_face = fragments.pix_to_face
    if bary.device != device:
        # [N, render_h, render_w, K, 3]
        bary = bary.to(device)
        pix_to_face = pix_to_face.to(device)

    mtl_uv_coords = torch.zeros((N, H, W, 2)).to(device)

    for i in range(N):

        assert K == 1
        n_faces = atlas_tex_size.size(0)

        # [render_h, render_w, K]
        cur_pixel_to_face = pix_to_face[i, ...] - i * n_faces

        # [render_h, render_w, K]
        face_R = atlas_tex_size[cur_pixel_to_face]
        face_tex_start_idx = atlas_start_idx[cur_pixel_to_face]

        # print(fragments.pix_to_face.shape, bary.shape)

        # [render_h, render_w, K, 2]
        bary_w01 = bary[i, ..., :2]
        # pyre-fixme[16]: `bool` has no attribute `__getitem__`.
        # [render_h, render_w, K, 1]
        mask = (cur_pixel_to_face < 0)[..., None]
        bary_w01 = torch.where(mask, torch.zeros_like(bary_w01), bary_w01)
        w_xy = (bary_w01 * face_R.unsqueeze(-1)).to(
            torch.int64
        )  # (render_H, render_W, K, 2)

        below_diag = (
            bary_w01.sum(dim=-1) * face_R - w_xy.float().sum(dim=-1)
        ) <= 1.0  # (render_H, render_W, K)
        w_x, w_y = w_xy.unbind(-1)
        w_x = torch.where(
            below_diag, w_x, (face_R - 1 - w_x)
        )  # (render_H, render_W, K)
        w_y = torch.where(
            below_diag, w_y, (face_R - 1 - w_y)
        )  # (render_H, render_W, K)

        mtl_uv_coords[i, :] = atlas_uv_pos[
            face_tex_start_idx + w_y * face_R + w_x
        ].squeeze()
        mtl_uv_coords[i, :] = mtl_uv_coords[i, :] * (cur_pixel_to_face >= 0).float()

    # NOTE: must flip the v coordinates.
    # since when computing uv coordinates, the image is flipped: https://github.com/facebookresearch/pytorch3d/blob/f5383a7e5a79bd9e912af0fb0199c557b9987877/pytorch3d/io/mtl_io.py#L110
    mtl_uv_coords[..., 1] = 1.0 - mtl_uv_coords[..., 1]
    mtl_uv_coords = mtl_uv_coords[..., [1, 0]]

    # NOTE: in CPP: when computing mtl UV, we include the minimum-cover-rectangle's top/left and exclude bottom/right.
    # Namely, we should use floor for both row and column.
    mtl_pixel_coords = torch.floor(
        mtl_uv_coords
        * torch.FloatTensor([mtl_size_h, mtl_size_w]).unsqueeze(0).to(device)
    ).long()
    mtl_pixel_coords[
        torch.where(mtl_pixel_coords[..., 0] >= mtl_size_h)[0],
        torch.where(mtl_pixel_coords[..., 0] >= mtl_size_h)[1],
        torch.where(mtl_pixel_coords[..., 0] >= mtl_size_h)[2],
        0,
    ] = (
        mtl_size_h - 1
    )
    mtl_pixel_coords[
        torch.where(mtl_pixel_coords[..., 1] >= mtl_size_w)[0],
        torch.where(mtl_pixel_coords[..., 1] >= mtl_size_w)[1],
        torch.where(mtl_pixel_coords[..., 1] >= mtl_size_w)[2],
        1,
    ] = (
        mtl_size_w - 1
    )

    return mtl_uv_coords, mtl_pixel_coords


def retrieve_pixel_val_torch(
    *,
    device,
    stream_type,
    mtl_imgs,
    pix_to_face,
    mtl_pixel_coords,
    face_to_mtl_idxs,
    return_idxs=False,
    return_uvs=False,
):
    """
    pix_to_face: [batch, render_h, render_w, #top_k_faces];
    mtl_pixel_coords: [batch, render_h, render_w, 2];
    face_to_mtl_idxs: [#faces, ];
    mtl_imgs: [#mtl, mtl_h, mtl_w, 3].
    """

    # [N, size_h, size_w, 1]
    pixel_to_mtl_idxs = face_to_mtl_idxs[pix_to_face]
    # print("\n", pixel_to_mtl_idxs.shape)

    # keep info for mask
    pixel_to_mtl_idxs[torch.where(pix_to_face <= -1)] = -1

    if return_idxs:
        target_view = torch.zeros(
            (
                mtl_pixel_coords.shape[0],
                mtl_pixel_coords.shape[1],
                mtl_pixel_coords.shape[2],
                3,
            ),
            dtype=mtl_pixel_coords.dtype,
            device=device,
        )
    elif return_uvs:
        target_view = torch.zeros(
            (
                mtl_pixel_coords.shape[0],
                mtl_pixel_coords.shape[1],
                mtl_pixel_coords.shape[2],
                3,
            ),
            dtype=torch.float32,
            device=device,
        )
    else:
        target_view = torch.zeros(
            (
                mtl_pixel_coords.shape[0],
                mtl_pixel_coords.shape[1],
                mtl_pixel_coords.shape[2],
                3,
            ),
            dtype=mtl_imgs.dtype,
            device=device,
        )
    mask = torch.zeros(mtl_pixel_coords.shape[:3], dtype=torch.uint8, device=device)

    # [#render_pixels, ]
    target_view_batchs, target_view_rows, target_view_cols = torch.where(
        pixel_to_mtl_idxs != -1
    )[:3]

    # [#render_pixels, ]
    target_mtl_idxs = pixel_to_mtl_idxs[
        target_view_batchs, target_view_rows, target_view_cols, 0
    ]
    # print("\n", target_mtl_idxs.shape)

    # [#render_pixels, 2]
    src_mtl_pixel_coords = mtl_pixel_coords[
        target_view_batchs, target_view_rows, target_view_cols, :
    ]
    # print("\n", src_mtl_pixel_coords.shape)

    if return_idxs or return_uvs:
        target_view[
            target_view_batchs, target_view_rows, target_view_cols, :
        ] = torch.stack(
            (
                target_mtl_idxs,
                src_mtl_pixel_coords[..., 0],
                src_mtl_pixel_coords[..., 1],
            ),
            dim=1,
        )
    else:
        target_view[
            target_view_batchs, target_view_rows, target_view_cols, :
        ] = mtl_imgs[
            target_mtl_idxs,
            src_mtl_pixel_coords[..., 0],
            src_mtl_pixel_coords[..., 1],
            :,
        ]

    non_render_batch, non_render_rows, non_render_cols = torch.where(
        pixel_to_mtl_idxs == -1
    )[:3]
    if non_render_batch.size != 0:
        if (not return_idxs) and (not return_uvs):
            if stream_type == "apple":
                target_view[
                    non_render_batch, non_render_rows, non_render_cols, :
                ] = torch.zeros(3, dtype=mtl_imgs.dtype, device=device)
            elif stream_type == "scannet":
                target_view[
                    non_render_batch, non_render_rows, non_render_cols, :
                ] = 0 * torch.ones(3, dtype=mtl_imgs.dtype, device=device)
            else:
                raise ValueError
        mask[non_render_batch, non_render_rows, non_render_cols] = 1

    return target_view, mask


def get_pix_to_cam_z_from_v_coords_chunk(
    face_v_ids, v_coords, view_mats, pix_to_cam, pix_to_face, pix_to_bary_coords, n_cams
):
    """
    face_v_ids: [#faces, 3]
    pix_to_face: [H, W]
    pix_to_cam: [H, W, #max_cams]
    pix_to_cam: [H, W, #max_cams]
    v_coords: [#points, 3]
    pix_to_bary_coords: [H, W, 3]

    NOTE: n_cams not always equals to view_mats.shape[0].
    - n_cams: maximum camera ID for the whole scene.
    - view_mats: a subset of some selected cameras.

    return: [H, W, view_mats.shape[0]]
    """
    h, w = pix_to_face.shape

    rendered_rows, rendered_cols = torch.where(pix_to_face >= 0)

    # get face's vertex IDs, [H, W, 3]
    pix_to_face_v_ids = face_v_ids[pix_to_face, :3]
    # [#rendered, 3]
    valid_face_v_ids = pix_to_face_v_ids[rendered_rows, rendered_cols, :]
    # [#rendered, 3, 3] --> [#rendered x 3, 3]
    rendered_v_coords = v_coords[valid_face_v_ids, :].view((-1, 3))
    # [#cameras, 3, #rendered x 3]
    rendered_v_cam_coords = world_to_cam_coords(
        view_mats, rendered_v_coords, save_mem=False
    )
    # [#cameras, 3, #rendered x 3] --> [#cameras, #rendered x 3, 3] --> [#cameras, #rendered, 3, 3]
    # 2nd 3 is for 3-dim coordinates
    rendered_v_cam_coords = rendered_v_cam_coords.permute(0, 2, 1).view(
        (view_mats.shape[0], rendered_rows.shape[0], 3, 3)
    )

    # [#cameras, #rendered, ]
    v1_z = rendered_v_cam_coords[..., 0, 2]
    v2_z = rendered_v_cam_coords[..., 1, 2]
    v3_z = rendered_v_cam_coords[..., 2, 2]

    # [#cameras, #rendered, 3]
    vs_z = torch.stack((v1_z, v2_z, v3_z), dim=2)

    # [#cameras, #rendered] --> [#rendered, #cameras]
    cam_z = torch.sum(
        vs_z * pix_to_bary_coords[rendered_rows, rendered_cols, :].unsqueeze(0), dim=2
    ).permute(1, 0)

    pix_to_cam_z = torch.zeros((h, w, view_mats.shape[0]), device=pix_to_face.device)
    pix_to_cam_z[rendered_rows, rendered_cols, :] = cam_z

    # NOTE: this is important
    pix_to_cam_z[pix_to_cam >= n_cams] = 0

    # [H, W, #cam]
    return pix_to_cam_z


def get_pix_to_cam_z_from_v_coords(
    face_v_ids,
    v_coords,
    view_mats,
    pix_to_cam,
    pix_to_face,
    pix_to_bary_coords,
    n_cams,
    save_mem=False,
):
    """
    face_v_ids: [#faces, 3]
    pix_to_face: [H, W]
    pix_to_cam: [H, W, #max_cams]
    pix_to_cam: [H, W, #max_cams]
    v_coords: [#points, 3]
    pix_to_bary_coords: [H, W, 3]

    NOTE: n_cams not always equals to view_mats.shape[0].
    - n_cams: maximum camera ID for the whole scene.
    - view_mats: a subset of some selected cameras.
    """

    if save_mem:
        h, w, _ = pix_to_cam.shape
        pix_to_cam_z = torch.zeros((h, w, view_mats.shape[0]))
        n_rows_per_chunk = 50
        start_row = 0
        for start_row in np.arange(0, h, n_rows_per_chunk):
            end_row = min(start_row + n_rows_per_chunk, h)
            cur_pix_to_cam = pix_to_cam[start_row:end_row, ...]
            cur_pix_to_face = pix_to_face[start_row:end_row, :]
            cur_pix_to_bary_coords = pix_to_bary_coords[start_row:end_row, ...]
            pix_to_cam_z[start_row:end_row, :] = get_pix_to_cam_z_from_v_coords_chunk(
                face_v_ids,
                v_coords,
                view_mats,
                cur_pix_to_cam,
                cur_pix_to_face,
                cur_pix_to_bary_coords,
                n_cams,
            ).to(torch.device("cpu"))
        pix_to_cam_z = pix_to_cam_z.to(pix_to_cam.device)
    else:
        pix_to_cam_z = get_pix_to_cam_z_from_v_coords_chunk(
            face_v_ids,
            v_coords,
            view_mats,
            pix_to_cam,
            pix_to_face,
            pix_to_bary_coords,
            n_cams,
        )

    return pix_to_cam_z


def get_pix_to_cam_z(v_cam_coords, pix_to_vs, pix_to_cam, pix_to_bary_coords, n_cams):
    """
    v_cam_coords: [#cameras, 3, #points];
    pix_to_vs: [H, W, 3];
    pix_to_cam: [H, W, #max_cams];
    pix_to_bary_coords: [H, W, 3]
    """
    H, W, _ = pix_to_vs.shape

    pix_to_cam_z = torch.zeros(pix_to_cam.shape, device=pix_to_cam.device)

    valid_rows, valid_cols, valid_channels = torch.where(pix_to_cam < n_cams)

    # [#valid, ]
    v1_z = v_cam_coords[
        pix_to_cam[valid_rows, valid_cols, valid_channels],
        2,
        pix_to_vs[valid_rows, valid_cols, 0],
    ]
    v2_z = v_cam_coords[
        pix_to_cam[valid_rows, valid_cols, valid_channels],
        2,
        pix_to_vs[valid_rows, valid_cols, 1],
    ]
    v3_z = v_cam_coords[
        pix_to_cam[valid_rows, valid_cols, valid_channels],
        2,
        pix_to_vs[valid_rows, valid_cols, 2],
    ]

    # [#valid, 3]
    vs_z = torch.stack((v1_z, v2_z, v3_z), dim=1)

    # [H, W, #max_cams]
    pix_to_cam_z[valid_rows, valid_cols, valid_channels] = torch.sum(
        vs_z * pix_to_bary_coords[valid_rows, valid_cols, :], dim=1
    )

    # NOTE: in COLMAP, positive-Z is the forward direction
    # https://colmap.github.io/format.html
    # pix_to_cam_z = -1 * pix_to_cam_z

    return pix_to_cam_z


def old_get_pix_to_cam_z(
    v_cam_coords, pix_to_vs, pix_to_cam, pix_to_bary_coords, n_cams
):
    """
    v_cam_coords: [#cameras, 3, #points];
    pix_to_vs: [H, W, 3];
    pix_to_cam: [H, W];
    pix_to_bary_coords: [H, W, 3]
    """
    H, W, _ = pix_to_vs.shape

    pix_to_cam_z = torch.zeros((H, W)).to(pix_to_cam.device)

    valid_rows, valid_cols = torch.where(pix_to_cam < n_cams)

    # [#valid, ]
    v1_z = v_cam_coords[
        pix_to_cam[valid_rows, valid_cols], 2, pix_to_vs[valid_rows, valid_cols, 0]
    ]
    v2_z = v_cam_coords[
        pix_to_cam[valid_rows, valid_cols], 2, pix_to_vs[valid_rows, valid_cols, 1]
    ]
    v3_z = v_cam_coords[
        pix_to_cam[valid_rows, valid_cols], 2, pix_to_vs[valid_rows, valid_cols, 2]
    ]

    # [#valid, 3]
    vs_z = torch.stack((v1_z, v2_z, v3_z), dim=1)

    # [H, W]
    pix_to_cam_z[valid_rows, valid_cols] = torch.sum(
        vs_z * pix_to_bary_coords[valid_rows, valid_cols], dim=1
    )

    # # negative-Z is the forward direction
    # pix_to_cam_z = -1 * pix_to_cam_z

    return pix_to_cam_z