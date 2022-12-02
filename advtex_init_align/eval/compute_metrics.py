import os
import glob
import h5py
import tqdm
import cv2
import pprint
import joblib
import datetime
import argparse
import numpy as np
import scipy.io as sio
import multiprocessing as mp
from collections import defaultdict
from PIL import Image, ImageOps

from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from advtex_init_align.utils.cpbd.compute import compute as cpbd_compute
from advtex_init_align.utils.metric_utils import metric_avg_grad_intensity

import torch

import lpips

loss_fn_alex = lpips.LPIPS(net="alex")  # best forward scores suggested by author

from advtex_init_align.utils.img_utils import (
    crop_img,
    compute_offset_fft,
    create_input_for_lpips,
    compute_boundary_after_shift,
)
from advtex_init_align.eval.exp_list import (
    SEED,
    # UofI Texture Scenes
    GT,
    MTL_ATLAS_SIZE,
    MTL_RES_DICT,
    # ScanNet
    SCANNET_GT,
    SCANNET_MTL_ATLAS_SIZE,
    SCANNET_MTL_RES,
    SCANNET_N_PARTS,
    METHOD_ID_ABBREVIATION,
    ALL_EXPS,
)


def split_img(img_f):
    # we assume the image is a concatenated one consisting of
    # [GT | rendered | depth | mask]
    cat_img = np.array(Image.open(img_f))
    h, cat_w, _ = cat_img.shape
    assert cat_w % 4 == 0, f"{cat_img.shape}"
    w = cat_w // 4
    img_gt = cat_img[:, :w, :]
    img_eval = cat_img[:, w : 2 * w, :]
    mask = cat_img[:, -w:, :].astype(np.float32) / 255
    return img_gt, img_eval, mask


def compute_s3_val(s3_mat, quant_val=0.99):
    # S3 uses top 1% values
    # Ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6030937
    quant_val = np.quantile(s3_mat, quant_val)
    select_rows, select_cols = np.where(s3_mat > quant_val)
    select_rows.shape, select_cols.shape
    selected_vals = s3_mat[select_rows, select_cols]
    return np.mean(selected_vals)


def align_imgs(
    img_gt,
    img_rendered_dict,
    mask_rendered_dict,
    s3_gt=None,
    s3_rendered_dict=None,
    existed_offset_dict=None,
    existed_agg_mask=None,
):

    img_h, img_w, _ = img_gt.shape

    gt_boundary_dict = {}
    rendered_boundary_dict = {}
    offset_dict = {}

    img_gt_torch = torch.FloatTensor(img_gt).unsqueeze(0).permute(0, 3, 1, 2)

    for k, img_rendered in img_rendered_dict.items():
        img_rendered_torch = (
            torch.FloatTensor(img_rendered).unsqueeze(0).permute(0, 3, 1, 2)
        )

        if existed_offset_dict is None:
            mean_offset = compute_offset_fft(img_gt_torch, img_rendered_torch)
            shift_row = mean_offset[0, 0].item()
            shift_col = mean_offset[0, 1].item()
        else:
            raise NotImplementedError

        # NOTE: DEBUG
        if (np.abs(shift_row) > img_h * 0.05) or (np.abs(shift_col) > img_w * 0.05):
            shift_row, shift_col = 0, 0

        offset_dict[k] = (shift_row, shift_col)

        gt_img_boundary, rendered_img_boundary = compute_boundary_after_shift(
            (img_h, img_w), (img_h, img_w), shift_row, shift_col
        )

        gt_boundary_dict[k] = gt_img_boundary
        rendered_boundary_dict[k] = rendered_img_boundary

    gt_boundaries = np.array(list(gt_boundary_dict.values()))

    # Aggregated
    if existed_offset_dict is None:
        agg_ref_min_row = np.max(gt_boundaries[:, 0])
        agg_ref_max_row = np.min(gt_boundaries[:, 1])
        agg_ref_min_col = np.max(gt_boundaries[:, 2])
        agg_ref_max_col = np.min(gt_boundaries[:, 3])
    else:
        (
            agg_ref_min_row,
            agg_ref_max_row,
            agg_ref_min_col,
            agg_ref_max_col,
        ) = existed_offset_dict["agg_ref_boundary"]

    offset_dict["agg_ref_boundary"] = (
        agg_ref_min_row,
        agg_ref_max_row,
        agg_ref_min_col,
        agg_ref_max_col,
    )

    new_img_gt = img_gt[
        agg_ref_min_row:agg_ref_max_row, agg_ref_min_col:agg_ref_max_col, :
    ]
    if s3_gt is not None:
        new_s3_gt = s3_gt[
            agg_ref_min_row:agg_ref_max_row, agg_ref_min_col:agg_ref_max_col
        ]
    else:
        new_s3_gt = None

    new_img_rendered_dict = {}
    new_mask_rendered_dict = {}
    new_s3_rendered_dict = {}

    for k, img_rendered in img_rendered_dict.items():
        ref_min_row, ref_max_row, ref_min_col, ref_max_col = gt_boundary_dict[k]
        (
            rendered_min_row,
            rendered_max_row,
            rendered_min_col,
            rendered_max_col,
        ) = rendered_boundary_dict[k]

        tmp_min_row = agg_ref_min_row - ref_min_row
        tmp_max_row = agg_ref_max_row - ref_min_row
        tmp_min_col = agg_ref_min_col - ref_min_col
        tmp_max_col = agg_ref_max_col - ref_min_col

        tmp_new = img_rendered[
            rendered_min_row:rendered_max_row, rendered_min_col:rendered_max_col, :
        ]
        tmp_cropped = tmp_new[tmp_min_row:tmp_max_row, tmp_min_col:tmp_max_col, :]
        new_img_rendered_dict[k] = tmp_cropped

        tmp_new_mask = mask_rendered_dict[k][
            rendered_min_row:rendered_max_row, rendered_min_col:rendered_max_col, :
        ]
        tmp_cropped_mask = tmp_new_mask[
            tmp_min_row:tmp_max_row, tmp_min_col:tmp_max_col, :
        ]
        new_mask_rendered_dict[k] = tmp_cropped_mask

        if s3_gt is not None:
            tmp_new_s3 = s3_rendered_dict[k][
                rendered_min_row:rendered_max_row, rendered_min_col:rendered_max_col
            ]
            tmp_cropped_s3 = tmp_new_s3[
                tmp_min_row:tmp_max_row, tmp_min_col:tmp_max_col
            ]
            new_s3_rendered_dict[k] = tmp_cropped_s3

    # We take a union over all masks.
    # This will eliminate the potentials that metrics differences come from mask's different positions
    # [H, W, 3]
    if existed_offset_dict is None:
        agg_mask = None
        for tmp_mask in new_mask_rendered_dict.values():
            if agg_mask is None:
                # initialize it
                agg_mask = np.ones(tmp_mask.shape)
            agg_mask[(agg_mask == 0) | (tmp_mask == 0)] = 0.0
    else:
        agg_mask = existed_agg_mask

    for k in new_img_rendered_dict:
        tmp_rgb = new_img_rendered_dict[k]
        tmp_rgb = (tmp_rgb.astype(np.float32) * agg_mask).astype(np.uint8)
        new_img_rendered_dict[k] = tmp_rgb

        if s3_gt is not None:
            # agg_mask: [H, W, 3], range [0, 1]
            tmp_s3 = new_s3_rendered_dict[k]
            tmp_s3 = tmp_s3 * agg_mask[..., 0]
            new_s3_rendered_dict[k] = tmp_s3

    new_img_gt = (new_img_gt.astype(np.float32) * agg_mask).astype(np.uint8)

    return (
        new_img_gt,
        new_s3_gt,
        new_img_rendered_dict,
        new_mask_rendered_dict,
        new_s3_rendered_dict,
        offset_dict,
        agg_mask,
    )


def compute_metrics_single_view(
    dataset,
    scene_id,
    view_id,
    input_dir_dict,
    input_k_list,
    debug_dir,
    flag_compute_s3=False,
    existed_offset_dict=None,
    existed_debug_dir=None,
    sample_freq=None,
):

    # read image
    img_rendered_dict = {}
    mask_rendered_dict = {}
    s3_mat_rendered_dict = {}
    for i, input_k in enumerate(input_k_list):
        input_dir = input_dir_dict[input_k]
        input_f = os.path.join(input_dir, f"{view_id:05d}.png")
        _, img_rendered, mask = split_img(input_f)
        img_rendered_dict[input_k] = img_rendered
        mask_rendered_dict[input_k] = mask
        if flag_compute_s3:
            tmp_root = os.path.dirname(input_dir)
            tmp_s3_mat_f = os.path.join(tmp_root, f"s3_mats/{view_id:05d}/s3.mat")
            tmp_s3_mat = sio.loadmat(tmp_s3_mat_f)["s3"]
            s3_mat_rendered_dict[input_k] = tmp_s3_mat

    if "scannet" in input_k_list[0]:
        gt_f = os.path.join(
            SCANNET_GT.format(
                dataset=dataset, scene_id=scene_id, sample_freq=sample_freq
            ),
            f"{view_id:05d}_raw_color.png",
        )
    else:
        gt_f = os.path.join(
            GT.format(dataset=dataset, scene_id=scene_id, sample_freq=sample_freq),
            f"raw_rgbs/{view_id:05d}_raw_color.png",
        )
    img_gt = np.array(Image.open(gt_f))

    raw_h, raw_w, _ = img_gt.shape

    if flag_compute_s3:
        if "scannet" in input_k_list[0]:
            tmp_gt = SCANNET_GT.format(
                dataset=dataset, scene_id=scene_id, sample_freq=sample_freq
            )
            s3_mat_gt_f = os.path.join(
                os.path.dirname(tmp_gt), f"s3_mats/{view_id:05d}/s3.mat"
            )
        else:
            s3_mat_gt_f = os.path.join(
                GT.format(scene_id=scene_id), f"s3_mats/{view_id:05d}/s3.mat"
            )
        s3_mat_gt = sio.loadmat(s3_mat_gt_f)["s3"]
    else:
        s3_mat_gt = None

    if existed_offset_dict is None:
        existed_agg_mask = None
    else:
        tmp_f = os.path.join(existed_debug_dir, f"agg_masks/{view_id:05d}.png")
        existed_agg_mask = np.array(Image.open(tmp_f)).astype(np.float32) / 255.0

    (
        aligned_img_gt,
        aligned_s3_mat_gt,
        aligned_img_rendered_dict,
        aligned_mask_rendered_dict,
        aligned_s3_mat_rendered_dict,
        offset_dict,
        agg_mask,
    ) = align_imgs(
        img_gt,
        img_rendered_dict,
        mask_rendered_dict,
        s3_gt=s3_mat_gt,
        s3_rendered_dict=s3_mat_rendered_dict,
        existed_offset_dict=existed_offset_dict,
        existed_agg_mask=existed_agg_mask,
    )

    metric_dict = {}

    if flag_compute_s3:
        s3_gt_val = compute_s3_val(aligned_s3_mat_gt)

    debug_cat_img = [aligned_img_gt]

    for input_k in input_k_list:

        aligned_img_rendered = aligned_img_rendered_dict[input_k]

        debug_cat_img.append(aligned_img_rendered)

        mse_metric = mean_squared_error(
            aligned_img_rendered / 255.0, aligned_img_gt / 255.0
        )
        psnr_metric = psnr(aligned_img_gt, aligned_img_rendered, data_range=255)
        ssim_metric = ssim(
            aligned_img_gt, aligned_img_rendered, data_range=255, multichannel=True
        )

        aligned_img_gt_gray = (
            np.array(Image.fromarray(aligned_img_gt).convert("L")).astype(np.float32)
            / 255.0
        )
        aligned_img_rendered_gray = (
            np.array(Image.fromarray(aligned_img_rendered).convert("L")).astype(
                np.float32
            )
            / 255.0
        )

        cpbd_ref = cpbd_compute(aligned_img_gt_gray)
        cpbd_eval = cpbd_compute(aligned_img_rendered_gray)
        cpbd_metric = np.abs(cpbd_ref - cpbd_eval)

        grad_map_ref, avg_grad_ref = metric_avg_grad_intensity(aligned_img_gt)
        grad_map_eval, avg_grad_eval = metric_avg_grad_intensity(aligned_img_rendered)
        grad_map_diff = np.mean(np.abs(grad_map_ref - grad_map_eval))
        avg_grad_metric = np.abs(avg_grad_ref - avg_grad_eval)

        # image should be RGB, IMPORTANT: normalized to [-1,1]
        with torch.no_grad():
            lpips_metric = float(
                loss_fn_alex(
                    create_input_for_lpips(aligned_img_gt),
                    create_input_for_lpips(aligned_img_rendered),
                ).numpy()
            )

        if flag_compute_s3:
            # for S3
            # difference between value
            s3_rendered_val = compute_s3_val(aligned_s3_mat_rendered_dict[input_k])
            s3_val_diff = np.abs(s3_gt_val - s3_rendered_val)
            # difference between map
            s3_mat_diff = np.mean(
                np.abs(aligned_s3_mat_gt - aligned_s3_mat_rendered_dict[input_k])
            )
        else:
            s3_val_diff = 0.0
            s3_mat_diff = 0.0

        metric_dict[input_k] = [
            view_id,
            raw_h,
            raw_w,
            mse_metric,
            psnr_metric,
            ssim_metric,
            lpips_metric,
            grad_map_diff,
            avg_grad_metric,
            s3_mat_diff,
            s3_val_diff,
            cpbd_metric,
        ]

    debug_cat_img = np.concatenate(debug_cat_img, axis=1)
    Image.fromarray(debug_cat_img).save(os.path.join(debug_dir, f"{view_id:05d}.png"))

    # save aggregated mask
    agg_mask = (agg_mask * 255).astype(np.uint8)
    Image.fromarray(agg_mask).save(
        os.path.join(debug_dir, f"agg_masks/{view_id:05d}.png")
    )

    return metric_dict, offset_dict


def compute_metrics_single_scene_subproc(subproc_input):
    """
    base_h5_f: for getting shift_u, shift_v.
    """

    (
        worker_id,
        dataset,
        scene_id,
        view_ids,
        input_dir_dict,
        input_k_list,
        debug_dir,
        flag_compute_s3,
        existed_offset_dict,
        existed_debug_dir,
        sample_freq,
    ) = subproc_input

    metric_placeholder = [0, 0, 0, 0]

    all_metric_dict = {}
    for k in input_dir_dict:
        all_metric_dict[k] = []
    all_offset_dict = {}

    for view_id in tqdm.tqdm(view_ids):

        if existed_offset_dict is None:
            tmp_offset_dict = None
        else:
            tmp_offset_dict = existed_offset_dict[view_id]

        view_metric_dict, view_offset_dict = compute_metrics_single_view(
            dataset,
            scene_id,
            view_id,
            input_dir_dict,
            input_k_list,
            debug_dir,
            flag_compute_s3=flag_compute_s3,
            existed_offset_dict=tmp_offset_dict,
            existed_debug_dir=existed_debug_dir,
            sample_freq=sample_freq,
        )

        for input_k in input_k_list:
            all_metric_dict[input_k].append(view_metric_dict[input_k])

        all_offset_dict[view_id] = view_offset_dict

    for k in all_metric_dict:
        all_metric_dict[k] = np.array(all_metric_dict[k])

    return all_metric_dict, all_offset_dict


def compute_metrics_single_scene_mp(
    nproc,
    save_dir,
    debug_dir,
    dataset,
    scene_id,
    input_dir_dict,
    input_k_list,
    flag_compute_s3=False,
    existed_offset_dict=None,
    existed_debug_dir=None,
    sample_freq=None,
):

    n_imgs = None
    for input_k in input_k_list:
        input_dir = input_dir_dict[input_k]
        all_img_fs = list(glob.glob(os.path.join(input_dir, "*.png")))
        if n_imgs is None:
            n_imgs = len(all_img_fs)
        assert n_imgs == len(
            all_img_fs
        ), f"{input_k} with {len(all_img_fs)} images != {n_imgs}: {input_dir}"

    print(f"\nFind {len(all_img_fs)} images.\n")

    all_view_ids = sorted(
        [
            int(os.path.basename(_.rstrip("/")).split(".")[0].split("_")[0])
            for _ in all_img_fs
        ]
    )

    view_id_list = [[] for _ in range(nproc)]
    for i, view_id in enumerate(all_view_ids):
        view_id_list[i % nproc].append(view_id)

    # NOTE: np.matmul may freeze when using default "fork"
    # https://github.com/ModelOriented/DALEX/issues/412
    with mp.get_context("spawn").Pool(nproc) as pool:
        gathered_outputs = pool.map(
            compute_metrics_single_scene_subproc,
            zip(
                range(nproc),
                [dataset for _ in range(nproc)],
                [scene_id for _ in range(nproc)],
                view_id_list,
                [input_dir_dict for _ in range(nproc)],
                [input_k_list for _ in range(nproc)],
                [debug_dir for _ in range(nproc)],
                [flag_compute_s3 for _ in range(nproc)],
                [existed_offset_dict for _ in range(nproc)],
                [existed_debug_dir for _ in range(nproc)],
                [sample_freq for _ in range(nproc)],
            ),
        )
        pool.close()
        pool.join()

    all_metric_dict = {}
    for k in gathered_outputs[0][0].keys():
        all_metric_dict[k] = np.concatenate([_[0][k] for _ in gathered_outputs], axis=0)

    all_offset_dict = {}
    for output in gathered_outputs:
        all_offset_dict.update(output[1])

    with open(os.path.join(save_dir, "eval_metrics.p"), "wb") as f:
        joblib.dump(all_metric_dict, f, compress="lz4")

    with open(os.path.join(save_dir, "offset_dict.p"), "wb") as f:
        joblib.dump(all_offset_dict, f, compress="lz4")


def process_single_scene(
    nproc,
    dataset,
    scene_id,
    method_id_list,
    sample_freq_list,
    save_dir,
    flag_compute_s3=False,
    existed_offset_dict_dir=None,
):

    all_input_keys = []
    all_input_dirs = {}

    if "scannet" in method_id_list[0]:
        scene_mtl_h = SCANNET_MTL_RES
        scene_mtl_w = SCANNET_MTL_RES
    else:
        scene_mtl_h = MTL_RES_DICT[scene_id]
        scene_mtl_w = MTL_RES_DICT[scene_id]

    for sample_freq in sample_freq_list:
        for method_id in method_id_list:
            tmp_k = f"{sample_freq}_{method_id}"
            all_input_keys.append(tmp_k)

            if "scannet" in method_id:
                all_input_dirs[tmp_k] = ALL_EXPS[method_id].format(
                    dataset=dataset,
                    scene_id=scene_id,
                    sample_freq=sample_freq,
                    mtl_h=scene_mtl_h,
                    mtl_w=scene_mtl_w,
                    atlas_size=SCANNET_MTL_ATLAS_SIZE,
                    n_splitted_meshes=SCANNET_N_PARTS,
                    seed=SEED,
                )
            else:
                all_input_dirs[tmp_k] = ALL_EXPS[method_id].format(
                    dataset=dataset,
                    scene_id=scene_id,
                    sample_freq=sample_freq,
                    mtl_h=scene_mtl_h,
                    mtl_w=scene_mtl_w,
                    atlas_size=MTL_ATLAS_SIZE,
                    seed=SEED,
                )

    print("\n")
    pprint.pprint(all_input_dirs)

    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    str_freq_list = "_".join([str(_) for _ in sample_freq_list])
    str_method_list = "-".join([METHOD_ID_ABBREVIATION[_] for _ in method_id_list])
    folder_name = f"freq_{str_freq_list}-{str_method_list}"
    # eval_save_dir = os.path.join(save_dir, folder_name, scene_id, cur_time)
    eval_save_dir = os.path.join(save_dir, folder_name, cur_time)

    debug_dir = os.path.join(eval_save_dir, "debug_vis")
    os.makedirs(debug_dir, exist_ok=True)
    mask_dir = os.path.join(debug_dir, "agg_masks")
    os.makedirs(mask_dir, exist_ok=True)
    with open(os.path.join(debug_dir, "cat_img_order.txt"), "w") as f:
        for input_k in all_input_keys:
            f.write(f"{input_k}\n")

    if existed_offset_dict_dir is None:
        existed_offset_dict = None
        existed_debug_dir = None
    else:
        existed_offset_dict_f_list = list(
            glob.glob(
                os.path.join(existed_offset_dict_dir, scene_id, "20*/offset_dict.p")
            )
        )
        assert len(existed_offset_dict_f_list) == 1, f"{existed_offset_dict_f_list}"
        existed_offset_dict_f = existed_offset_dict_f_list[0]
        existed_offset_dict = joblib.load(existed_offset_dict_f)
        existed_debug_dir = os.path.join(
            os.path.dirname(existed_offset_dict_f), "debug_vis"
        )

    compute_metrics_single_scene_mp(
        nproc,
        eval_save_dir,
        debug_dir,
        dataset,
        scene_id,
        all_input_dirs,
        all_input_keys,
        flag_compute_s3=flag_compute_s3,
        existed_offset_dict=existed_offset_dict,
        existed_debug_dir=existed_debug_dir,
        sample_freq=sample_freq,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nproc",
        type=int,
        required=True,
        default=10,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="uofi",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        default=".",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        required=True,
        default=".",
    )
    parser.add_argument(
        "--sample_freq_list",
        nargs="+",
        type=str,
        required=True,
        default="test_1_10",
    )
    parser.add_argument(
        "--method_id_list",
        nargs="+",
        type=str,
        choices=list(ALL_EXPS.keys()),
        required=True,
        default=".",
    )
    parser.add_argument(
        "--compute_s3",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--existed_offset_dict_dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    process_single_scene(
        args.nproc,
        args.dataset,
        args.scene_id,
        args.method_id_list,
        args.sample_freq_list,
        args.save_dir,
        flag_compute_s3=bool(args.compute_s3),
        existed_offset_dict_dir=args.existed_offset_dict_dir,
    )
