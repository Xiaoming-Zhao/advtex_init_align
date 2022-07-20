import os
import copy
import cv2
import tqdm
import png
import glob
import numpy as np
import multiprocessing as mp
from typing import List

import torch

# https://github.com/pytorch/vision/issues/1439
# format: NCHW
IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))


def compute_offset_fft(inp, gt):
    """Note, the order of input matters.
    The offset computed by this function means: inp needs to shift with the offset to align with gt.
    """

    # inp_p = torch.nn.functional.pad(inp,(2,2,2,2,0,0))
    # gt_p = torch.nn.functional.pad(gt,(2,2,2,2,0,0))

    inp_p = inp
    gt_p = gt
    # print(inp.shape, gt.shape)

    # [B, C, H, W]
    inp_p_fft = torch.fft.fft2(inp_p, dim=(-2, -1))
    gt_p_fft = torch.fft.fft2(gt_p, dim=(-2, -1))

    # [B, C, H, W]
    res_ifft = torch.fft.ifft2(inp_p_fft * torch.conj(gt_p_fft))
    # print(inp_p_fft.shape, gt_p_fft.shape, res_ifft.shape)

    # [B, C, W]
    v0, a0 = torch.max(res_ifft.real, dim=2)
    # [B, C]
    v1, a1 = torch.max(v0, dim=2)
    # print(v0.shape, a0.shape, v1.shape, a1.shape)

    # [B, C]
    tmp1 = torch.arange(a1.shape[0]).unsqueeze(1).expand(-1, a1.shape[1])
    tmp2 = torch.arange(a1.shape[1]).unsqueeze(0).expand(a1.shape[0], -1)
    # print(tmp1.shape, tmp1)
    # print(tmp2.shape, tmp2)
    max_idx0 = a0[tmp1, tmp2, a1]
    # print(max_idx0.shape, max_idx0)

    # [H,]
    f0 = torch.fft.fftfreq(inp_p.shape[2]) * inp_p.shape[2]
    # [W,]
    f1 = torch.fft.fftfreq(inp_p.shape[3]) * inp_p.shape[3]
    # print(f0.shape, f1.shape)
    # print(f0[max_idx0])
    # print(f1[a1])

    # [B, C, 2]
    offset_init = torch.stack((f0[max_idx0], f1[a1]), dim=2)
    # print(offset_init)

    # # [B, 2]
    # mean_offset_init = torch.mean(offset_init, dim=1).long()

    # NOTE: filter out invalid values first
    # we filter out value that are too unreasonable
    _, _, img_h, img_w = inp.shape
    img_res = torch.FloatTensor([img_h, img_w]).reshape((1, 1, 2)).to(offset_init.device)
    # [B, C, 2]
    valid_mask = (torch.abs(offset_init) <= img_res * 0.05).float().to(offset_init.device)
    masked_offset = valid_mask * offset_init

    # [B, 2], 1st elem is for row, 2nd elem is for col
    mean_offset_init = torch.sum(masked_offset, dim=1) / (torch.sum(valid_mask, dim=1) + 1e-8)
    mean_offset_init = mean_offset_init.long()

    return mean_offset_init


def compute_boundary_after_shift(ref_img_size, rendered_img_size, shift_u, shift_v):
    """
    v is for horizontal and u is for vertical.
    +U points to vertical down; +V points to horizontal right.

    img_size: (#rows, #cols)
    """
    if shift_v <= 0:
        # move rendered image left
        ref_min_v, ref_max_v = 0, ref_img_size[1] + shift_v
        render_min_v, render_max_v = np.abs(shift_v), rendered_img_size[1]
    else:
        # move rendered image right
        ref_min_v, ref_max_v = shift_v, rendered_img_size[1]
        render_min_v, render_max_v = 0, ref_img_size[1] - shift_v

    if shift_u <= 0:
        # move rendered image up
        ref_min_u, ref_max_u = 0, ref_img_size[0] + shift_u
        render_min_u, render_max_u = np.abs(shift_u), rendered_img_size[0]
    else:
        # move rendered image down
        ref_min_u, ref_max_u = shift_u, rendered_img_size[0]
        render_min_u, render_max_u = 0, ref_img_size[0] - shift_u

    return (
        [ref_min_u, ref_max_u, ref_min_v, ref_max_v],
        [render_min_u, render_max_u, render_min_v, render_max_v],
    )


def shift_imgs(img, shift_u, shift_v, gt=True):

    if img.ndim == 3:
        assert img.shape[2] == 3 or img.shape[2] == 1
        img_size = img.shape[:2]
    elif img.ndim == 4:
        assert img.shape[3] == 3 or img.shape[3] == 1
        img_size = img.shape[1:3]
    else:
        raise ValueError(f"{img.shape}")

    # We assume imgs with shape [H, W, 3].
    # +U points to vertical down; +V points to horizontal right.
    img_boundary, new_img_boundary = compute_boundary_after_shift(
        img_size, img_size, shift_u, shift_v
    )
    ref_min_u, ref_max_u, ref_min_v, ref_max_v = img_boundary
    render_min_u, render_max_u, render_min_v, render_max_v = new_img_boundary

    if gt:
        min_u, max_u, min_v, max_v = ref_min_u, ref_max_u, ref_min_v, ref_max_v
    else:
        min_u, max_u, min_v, max_v = (
            render_min_u,
            render_max_u,
            render_min_v,
            render_max_v,
        )

    if img.ndim == 3:
        new_img = img[min_u:max_u, min_v:max_v, :]
    else:
        new_img = img[:, min_u:max_u, min_v:max_v, :]

    return new_img


def compute_diff_after_shift(ref_img, rendered_img, shift_u, shift_v):

    ref_boundary_info, render_boundary_info = compute_boundary_after_shift(
        ref_img.shape[:2], rendered_img.shape[:2], shift_u, shift_v
    )

    ref_min_u, ref_max_u, ref_min_v, ref_max_v = ref_boundary_info
    render_min_u, render_max_u, render_min_v, render_max_v = render_boundary_info

    # all images must be float to avoid weird errors
    diff_sum = np.sum(
        np.abs(
            ref_img[ref_min_u:ref_max_u, ref_min_v:ref_max_v]
            - rendered_img[render_min_u:render_max_u, render_min_v:render_max_v]
        )
    )
    diff_mean = diff_sum / (
        (ref_img.shape[0] - np.abs(shift_u))
        * (ref_img.shape[1] - np.abs(shift_v))
        * ref_img.shape[2]
    )

    return diff_mean


def find_best_shift_subproc(subproc_input):
    ref_img, rendered_img, shift_list = subproc_input

    mean_diff_list = []

    for (shift_u, shift_v) in tqdm.tqdm(shift_list):
        # print(shift_u, shift_v)
        diff_mean = compute_diff_after_shift(ref_img, rendered_img, shift_u, shift_v)
        mean_diff_list.append((shift_u, shift_v, diff_mean))

    return mean_diff_list


def find_best_shift(*, nproc, ref_img, rendered_img, horizontal_range, vertical_range):
    """This function finds best shift for rendered image.
    Namely, we keep ref_img static, shift rendered image to find best alignment.
    """

    assert (
        ref_img.shape == rendered_img.shape
    ), f"ref: {ref_img.shape}, render: {rendered_img.shape}"

    ref_img = ref_img.astype(np.float)
    rendered_img = rendered_img.astype(np.float)

    n_elem_subproc = int(np.ceil(len(vertical_range) * len(horizontal_range) / nproc))

    shift_list = []
    subproc_list = []
    cnt = 0
    for shift_u in tqdm.tqdm(vertical_range):
        for shift_v in horizontal_range:
            subproc_list.append([shift_u, shift_v])
            cnt += 1
            if cnt >= n_elem_subproc:
                shift_list.append(subproc_list)
                subproc_list = []
                cnt = 0

    if len(subproc_list) != 0:
        shift_list.append(subproc_list)

    assert len(vertical_range) * len(horizontal_range) == np.sum(
        [len(_) for _ in shift_list]
    ), f"{len(vertical_range) * len(horizontal_range)}, {np.sum([len(_) for _ in shift_list])}"

    with mp.Pool(nproc) as pool:

        import time

        start = time.time()

        gathered_mean_diff = pool.map(
            find_best_shift_subproc,
            zip(
                [ref_img for _ in range(nproc)],
                [rendered_img for _ in range(nproc)],
                shift_list,
            ),
        )

        print(time.time() - start)

    gathered_mean_diff = np.concatenate(
        [np.array(_) for _ in gathered_mean_diff], axis=0
    )
    # print(gathered_mean_diff.shape)

    best_idx = np.argmin(gathered_mean_diff[:, 2])
    best_shift_u, best_shift_v, min_shift_diff = gathered_mean_diff[best_idx, :]
    # print(gathered_mean_diff[best_idx, :])

    return int(best_shift_u), int(best_shift_v), min_shift_diff


def fill_area(
    small_patch_top_left_pixel_coord,
    large_patch_h,
    large_patch_w,
    small_patch_h,
    small_patch_w,
    img_h,
    img_w,
):
    """This function fills cropped area with original image's pixel coordinates.
    It does not handle any contraints, assuming all pixel coordinates are valid.
    It follows the Overlap-tile strategy mentioned in U-net publication.
    Namely, it crops a large_patch which will be used as input to U-net.
    Within this large_patch, a centered small_patch will be used as ground-truth for comparison with output of U-net.

    - tl: top-left
    - tr: top-right
    - bl: bottom-left
    - br: bottom-right

     ------------------------------
    | tl |    pad top        | tr  |
    |----|-------------------|-----|
    |    |                   |     |
    |pad |    small patch    |pad  |
    |left|                   |right|
    |----|-------------------|-----|
    | bl |    pad bottom     | br  |
     ------------------------------
    """
    small_patch_top_row, small_patch_left_col = small_patch_top_left_pixel_coord
    assert small_patch_top_row >= 0 and small_patch_top_row + small_patch_h <= img_h
    assert small_patch_left_col >= 0 and small_patch_left_col + small_patch_w <= img_w

    patch_pixel_coords = np.zeros((large_patch_h, large_patch_w, 2), dtype=np.int)

    pad_top = int((large_patch_h - small_patch_h) / 2)
    pad_bottom = large_patch_h - small_patch_h - pad_top
    pad_left = int((large_patch_w - small_patch_w) / 2)
    pad_right = large_patch_w - small_patch_w - pad_left

    large_patch_left_col = small_patch_left_col - pad_left
    large_patch_right_col = small_patch_left_col + small_patch_w + pad_right
    large_patch_top_row = small_patch_top_row - pad_top
    large_patch_bottom_row = small_patch_top_row + small_patch_h + pad_bottom

    exceed_left = 0
    exceed_right = 0
    exceed_top = 0
    exceed_bottom = 0

    if large_patch_left_col < 0:
        exceed_left = -1 * large_patch_left_col
    if large_patch_right_col > img_w:
        exceed_right = large_patch_right_col - img_w
    if large_patch_top_row < 0:
        exceed_top = -1 * large_patch_top_row
    if large_patch_bottom_row > img_h:
        exceed_bottom = large_patch_bottom_row - img_h

    # fill area which is covered by original image
    rows, cols = np.meshgrid(
        np.arange(large_patch_h - exceed_top - exceed_bottom),
        np.arange(large_patch_w - exceed_left - exceed_right),
        sparse=False,
        indexing="ij",
    )
    patch_pixel_coords[
        exceed_top : (large_patch_h - exceed_bottom),
        exceed_left : (large_patch_w - exceed_right),
        0,
    ] = rows + max(large_patch_top_row, 0)
    patch_pixel_coords[
        exceed_top : (large_patch_h - exceed_bottom),
        exceed_left : (large_patch_w - exceed_right),
        1,
    ] = cols + max(large_patch_left_col, 0)

    # fill exceed-left
    if exceed_left > 0:
        patch_pixel_coords[
            exceed_top : (large_patch_h - exceed_bottom), :exceed_left, :
        ] = np.fliplr(
            patch_pixel_coords[
                exceed_top : (large_patch_h - exceed_bottom),
                exceed_left : 2 * exceed_left,
                :,
            ]
        )

    # fill exceed-right
    if exceed_right > 0:
        patch_pixel_coords[
            exceed_top : (large_patch_h - exceed_bottom),
            (large_patch_w - exceed_right) :,
            :,
        ] = np.fliplr(
            patch_pixel_coords[
                exceed_top : (large_patch_h - exceed_bottom),
                (large_patch_w - 2 * exceed_right) : (large_patch_w - exceed_right),
                :,
            ]
        )

    # fill exceed-top
    if exceed_top > 0:
        patch_pixel_coords[
            :exceed_top, exceed_left : (large_patch_w - exceed_right), :
        ] = np.flipud(
            patch_pixel_coords[
                exceed_top : 2 * exceed_top,
                exceed_left : (large_patch_w - exceed_right),
                :,
            ]
        )

    # fill exceed-bottom
    if exceed_bottom > 0:
        patch_pixel_coords[
            (large_patch_h - exceed_bottom) :,
            exceed_left : (large_patch_w - exceed_right),
            :,
        ] = np.flipud(
            patch_pixel_coords[
                (large_patch_h - 2 * exceed_bottom) : (large_patch_h - exceed_bottom),
                exceed_left : (large_patch_w - exceed_right),
                :,
            ]
        )

    # fill exceed top-left
    if exceed_left > 0 and exceed_top > 0:
        patch_pixel_coords[:exceed_top, :exceed_left, :] = np.flipud(
            np.fliplr(
                patch_pixel_coords[
                    exceed_top : 2 * exceed_top, exceed_left : 2 * exceed_left, :
                ]
            )
        )

    # fill exceed top-right
    if exceed_right > 0 and exceed_top > 0:
        patch_pixel_coords[
            :exceed_top, (large_patch_w - exceed_right) :, :
        ] = np.flipud(
            np.fliplr(
                patch_pixel_coords[
                    exceed_top : 2 * exceed_top,
                    (large_patch_w - 2 * exceed_right) : (large_patch_w - exceed_right),
                    :,
                ]
            )
        )

    # fill exceed bottom-left
    if exceed_left > 0 and exceed_bottom > 0:
        patch_pixel_coords[
            (large_patch_h - exceed_bottom) :, :exceed_left, :
        ] = np.flipud(
            np.fliplr(
                patch_pixel_coords[
                    (large_patch_h - 2 * exceed_bottom) : (
                        large_patch_h - exceed_bottom
                    ),
                    exceed_left : 2 * exceed_left,
                    :,
                ]
            )
        )

    # fill exceed bottom-right
    if exceed_right > 0 and exceed_bottom > 0:
        patch_pixel_coords[
            (large_patch_h - exceed_bottom) :, (large_patch_w - exceed_right) :, :
        ] = np.flipud(
            np.fliplr(
                patch_pixel_coords[
                    (large_patch_h - 2 * exceed_bottom) : (
                        large_patch_h - exceed_bottom
                    ),
                    (large_patch_w - 2 * exceed_right) : (large_patch_w - exceed_right),
                    :,
                ]
            )
        )

    return patch_pixel_coords, (pad_top, pad_bottom, pad_left, pad_right)


def get_base_n_patch_per_img(img_h, img_w, shift_u, shift_v, patch_h, patch_w):
    img_h_after_shift = img_h - np.abs(shift_u)
    img_w_after_shift = img_w - np.abs(shift_v)

    n_seg_row = np.ceil(img_h_after_shift / patch_h)
    n_seg_col = np.ceil(img_w_after_shift / patch_w)
    n_base = n_seg_row * n_seg_col
    return n_base, img_h_after_shift, img_w_after_shift


def sample_patches(
    img_h, img_w, small_patch_h, small_patch_w, shift_u, shift_v, multiplier=1
):
    """This function returns a list of top_left pixel coordinates of cropped regions
    as well as corresponding top_left pixel coordinates of referred patch.

    Small patch must locate within the valid region after shift.

    In pixel coordinate (u, v), u is for vertical and v is for horizontal.
    +U points to vertical down; +V points to horizontal right.

    Assume N patches could cover the whole valid region after shift.
    The number of patches (N) this function returns is `multiplier x N`.
    """

    n_base, img_h_after_shift, img_w_after_shift = get_base_n_patch_per_img(
        img_h, img_w, shift_u, shift_v, small_patch_h, small_patch_w
    )

    # First, we sample patches cover the whole image.
    # We lay patches one-by-one w/o overlapping until the rest of the height/width could not fit a patch.
    # Then we deliberately add one more patch to cover the rest height/width
    row_coord = list(np.arange(0, img_h_after_shift, small_patch_h))
    if img_h_after_shift - row_coord[-1] < small_patch_h:
        row_coord[-1] = img_h_after_shift - small_patch_h
    # print(row_coord)

    col_coord = list(np.arange(0, img_w_after_shift, small_patch_w))
    if img_w_after_shift - col_coord[-1] < small_patch_w:
        col_coord[-1] = img_w_after_shift - small_patch_w

    # [4, #samples], order of row: [ref_row, ref_col, render_row, render_col]
    mesh_rows, mesh_cols = np.meshgrid(
        row_coord, col_coord, sparse=False, indexing="ij"
    )
    mesh_rows = mesh_rows.reshape(-1)
    mesh_cols = mesh_cols.reshape(-1)

    # add more samples
    if multiplier > 1:
        extra_n = int(n_base * (multiplier - 1))

        extra_rows_pool = np.arange(img_h_after_shift - small_patch_h)
        extra_cols_pool = np.arange(img_w_after_shift - small_patch_w)

        cnt = 0
        while True:
            extra_row = np.random.choice(extra_rows_pool, size=1)
            extra_col = np.random.choice(extra_cols_pool, size=1)

            if not np.any((mesh_rows - extra_row == 0) & (mesh_cols - extra_col == 0)):
                mesh_rows = np.append(mesh_rows, extra_row)
                mesh_cols = np.append(mesh_cols, extra_col)
                cnt += 1

            if cnt >= extra_n:
                break

    top_left_pixel_coords = np.array([mesh_rows, mesh_cols, mesh_rows, mesh_cols])

    # get boundary info after shift
    ref_boundary_info, render_boundary_info = compute_boundary_after_shift(
        (img_h, img_w), (img_h, img_w), shift_u, shift_v
    )

    ref_min_u, ref_max_u, ref_min_v, ref_max_v = ref_boundary_info
    render_min_u, render_max_u, render_min_v, render_max_v = render_boundary_info

    # get pixel coordinates on image before shift
    top_left_pixel_coords += np.array(
        [ref_min_u, ref_min_v, render_min_u, render_min_v]
    ).reshape((-1, 1))

    return top_left_pixel_coords


def crop_out_small_patch(batch_imgs, batch_pad_infos):
    assert batch_imgs.ndim == 4
    assert batch_imgs.size(1) in [3, 1]  # ensure format NCHW
    pad_top, pad_bottom, pad_left, pad_right = batch_pad_infos[0, :]
    cropped_imgs = batch_imgs[
        :,
        :,
        pad_top : (batch_imgs.size(2) - pad_bottom),
        pad_left : (batch_imgs.size(3) - pad_right),
    ]
    return cropped_imgs


def normalize_imgs_imagenet(batch_imgs):
    return (batch_imgs - IMAGENET_MEAN.to(batch_imgs.device)) / IMAGENET_STD.to(
        batch_imgs.device
    )


def denormalize_imgs_imagenet(batch_imgs):
    return batch_imgs * IMAGENET_STD.to(batch_imgs.device) + IMAGENET_MEAN.to(
        batch_imgs.device
    )


def normalize_imgs_to_range_pm_one(batch_imgs):
    """[0, 1] --> [-1, 1]"""
    assert torch.min(batch_imgs) >= 0
    assert torch.max(batch_imgs) <= 1
    return (batch_imgs - 0.5) / 0.5


def denormalize_imgs_from_range_pm_one(batch_imgs):
    """[-1, 1] --> [0, 1]"""
    assert torch.min(batch_imgs) >= -1, f"{torch.min(batch_imgs)}"
    assert torch.max(batch_imgs) <= 1
    return batch_imgs * 0.5 + 0.5


def create_input_for_lpips(in_img):
    assert in_img.dtype == np.uint8 or in_img.dtype == torch.uint8

    out_img = torch.FloatTensor(in_img).unsqueeze(0) / 255.0
    out_img = out_img.permute(0, 3, 1, 2)

    # normalize to [-1, 1]
    out_img = (out_img - 0.5) / 0.5
    assert torch.min(out_img) >= -1
    assert torch.max(out_img) <= 1

    return out_img


def crop_img(img, shift_u, shift_v, gt=False):
    img_boundary, new_img_boundary = compute_boundary_after_shift(
        img.shape[:2], img.shape[:2], shift_u, shift_v
    )
    ref_min_u, ref_max_u, ref_min_v, ref_max_v = img_boundary
    render_min_u, render_max_u, render_min_v, render_max_v = new_img_boundary

    if gt:
        img = img[ref_min_u:ref_max_u, ref_min_v:ref_max_v, :]
    else:
        img = img[render_min_u:render_max_u, render_min_v:render_max_v, :]

    return img
