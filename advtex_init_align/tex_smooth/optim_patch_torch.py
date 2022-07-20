from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import cv2
import time
import pickle
import shutil
import tqdm
import joblib
import traceback
import random
import argparse
import collections
import numpy as np
import multiprocessing as mp
from PIL import Image

import PIL
PIL.Image.MAX_IMAGE_PIXELS = None

import torch
import torchvision
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from advtex_init_align.tex_smooth.dataset_torch import AdvTexIterDataset
from advtex_init_align.tex_smooth.model_torch import TexG, TexD
from advtex_init_align.tex_smooth.utils import compute_offset_fft, compute_boundary_after_shift


MAX_STEPS = 4001

BATCH_SIZE = 1

EPS = 1e-12

STREAM_FILENAME = "Recv.stream"
OBJ_FILENAME = "TexAlign.obj"
MTL_FILENAME = "TexAlign.mtl"

VALID_OFFSET_THRESHOLD = {
    1: 0.05,
    2: 0.1,
    4: 0.2,
}


def save_tex_img(cur_tex, step, output_dir, mp_queue):
    cur_tex = (np.clip(cur_tex * 0.5 + 0.5, 0, 1) * 255).astype("uint8")
    save_f = os.path.join(output_dir, f"{step:06d}.png")
    Image.fromarray(cur_tex).save(save_f)
    mp_queue.put("Finished")


def masked_out_img(outputs, mask):
    outputs = outputs * mask
    # We make non-rendered area to be black, namely -1 in range [-1, 1]
    mask_addon = torch.zeros(outputs.shape, device=outputs.device)
    non_rendered_bs, non_rendered_rows, non_rendered_cols, _ = torch.where(mask == 0)
    mask_addon[non_rendered_bs, non_rendered_rows, non_rendered_cols, :] = -1
    outputs = outputs + mask_addon
    return outputs


def render_from_tex(model_tex, placeholder, uv_src, mask, mask_rendered_area):
    # [1, H, W, 3]
    cur_tex = model_tex(placeholder).permute(0, 3, 1, 2)

    # [B, 3, H, W] -> [B, H, W, 3]
    raw_outputs = torch.nn.functional.grid_sample(
        cur_tex, uv_src, mode="bilinear", align_corners=True
    ).permute(0, 2, 3, 1)

    # [B, H, W, 3]
    outputs = masked_out_img(raw_outputs, mask)

    with torch.no_grad():
        outputs_rendered_area = masked_out_img(raw_outputs, mask_rendered_area)

    return outputs, outputs_rendered_area


def compute_g_loss(predict_fake, outputs, color_tar_to_src, mask, sx, sy, l1_weight):
    gen_loss_GAN = torch.mean(-1 * torch.log(predict_fake + EPS))
    gen_loss_L1 = torch.sum(
        torch.sum(
            torch.abs(color_tar_to_src[:, sy:, sx:, :] - outputs[:, sy:, sx:, :]), dim=3
        )
        * mask[:, sy:, sx:, 0]
    ) / (torch.sum(mask[:, sy:, sx:, 0]) * 3.0 + EPS)

    gen_loss = gen_loss_L1 * l1_weight + gen_loss_GAN
    return gen_loss, gen_loss_L1, gen_loss_GAN


def compute_d_loss_real(model_disc, color_src, color_tar_to_src, mask, sx, sy):
    input_real = torch.cat([color_src, color_tar_to_src - color_src], dim=3).permute(
        0, 3, 1, 2
    )

    # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
    predict_real, mask_real = model_disc(
        input_real[:, :, sy:, sx:], mask.permute(0, 3, 1, 2)[:, :, sy:, sx:]
    )
    predict_real = predict_real.permute(0, 2, 3, 1)
    mask_real = mask_real.permute(0, 2, 3, 1)

    return predict_real, mask_real


def compute_d_loss_fake(model_disc, color_src, outputs, mask, sx, sy):
    # [B, 6, H, W]
    input_fake = torch.cat([color_src, outputs - color_src], dim=3).permute(0, 3, 1, 2)

    # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
    predict_fake, mask_fake = model_disc(
        input_fake[:, :, sy:, sx:], mask.permute(0, 3, 1, 2)[:, :, sy:, sx:]
    )
    predict_fake = predict_fake.permute(0, 2, 3, 1)
    mask_fake = mask_fake.permute(0, 2, 3, 1)

    return predict_fake, mask_fake


def train(args):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    filename = args.input_dir.rstrip("/").split("/")[-1]

    if args.data_chair == 1:
        initial_file = os.path.join(args.input_dir, "texture.png")
    else:
        initial_file = os.path.join(args.input_dir, "shape/mtl0.png")
    print("\ninitial_file: ", initial_file, "\n")

    # set up dataset
    collate_func = AdvTexIterDataset.collate_func

    train_dataset = AdvTexIterDataset(
        args.input_dir,
        initial_file,
        num_workers=args.num_workers,
        data_chair=bool(args.data_chair),
        z_diff_threshold=args.z_diff_threshold,
    )

    shuffle_flag = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle_flag,
        collate_fn=collate_func,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=False,
    )

    model_tex = TexG(mtl_f=initial_file, from_scratch=bool(args.from_scratch))
    model_disc = TexD()

    model_tex.to(device)
    model_disc.to(device)

    print("\nG: ", model_tex, "\n")
    print("\nD: ", model_disc, "\n")

    optimizer_tex = optim.Adam(
        list(filter(lambda p: p.requires_grad, model_tex.parameters())),
        lr=args.lr_G,
        betas=(args.beta1, 0.999),
    )
    optimizer_disc = optim.Adam(
        list(filter(lambda p: p.requires_grad, model_disc.parameters())),
        lr=args.lr_D,
        betas=(args.beta1, 0.999),
    )

    if args.from_scratch == 1:
        max_steps = 50001
        save_interval = 200
    else:
        if args.scannet == 1:
            max_steps = 50001
            save_interval = 200
        else:
            max_steps = MAX_STEPS
            save_interval = 100

    global_step = -1

    nbatches = np.ceil(len(train_loader.dataset) / BATCH_SIZE)

    placeholder = torch.zeros(1).to(device)

    pbar = tqdm.tqdm(total=max_steps)

    misalign_offset_dict = {}

    img_writer_procs = {}
    img_writer_complete_cnts = 0

    n_patches_h = args.n_patches_h
    n_patches_w = args.n_patches_w

    assert n_patches_h == n_patches_w, f"{n_patches_h}, {n_patches_w}"
    cur_valid_offset_threshold = VALID_OFFSET_THRESHOLD[n_patches_h]

    while global_step < max_steps:

        train_iter = iter(train_loader)
        batch_i = 0

        while True:

            try:
                batch_data = next(train_iter)
            except StopIteration:
                break

            global_step += 1
            if global_step >= max_steps:
                break

            pbar.update()

            optimizer_disc.zero_grad()
            optimizer_tex.zero_grad()

            l1_weight = 10.0 * (0.8 ** float(global_step // 960))

            # color_X: [B, H, W, 3]
            # uv: [B, H, W, 2]
            # mask: [B, H, W, 1]
            color_src = batch_data["color_src"].to(device)
            color_tar_to_src = batch_data["color_tar_to_src"].to(device)
            color_src_rendered_area = batch_data["color_src_rendered_area"].to(device)
            uv_src = batch_data["uv_src"].to(device)
            mask = batch_data["mask"].to(device)
            mask_rendered_area = batch_data["mask_rendered_area"].to(device)
            src_index = batch_data["index"]
            from_tar_index = batch_data["rindex"]
            fixed_shift_u = batch_data["shift_u"][0]
            fixed_shift_v = batch_data["shift_v"][0]

            assert torch.sum(torch.abs(mask_rendered_area)) != 0.0, f"{src_index}, {from_tar_index}"

            assert mask.shape[3] == 1, f"{mask.shape}"

            outputs, outputs_rendered_area = render_from_tex(
                model_tex, placeholder, uv_src, mask, mask_rendered_area
            )

            if args.use_mislaign_offset == 1:

                all_color_src = []
                all_color_tar_to_src = []
                all_outputs = []
                all_masks = []

                _, img_h, img_w, _ = color_src.shape

                cur_patch_h = int(np.ceil(img_h / n_patches_h))
                cur_patch_w = int(np.ceil(img_w / n_patches_w))

                # print("\ncur_patch_h: ", cur_patch_h, cur_patch_w, "\n")

                cur_max_crop_h = -1
                cur_max_crop_w = -1

                for p_i in range(n_patches_h):
                    for p_j in range(n_patches_w):

                        cur_start_row = p_i * cur_patch_h
                        cur_end_row = min(img_h, (p_i + 1) * cur_patch_h)
                        cur_start_col = p_j * cur_patch_w
                        cur_end_col = min(img_w, (p_j + 1) * cur_patch_w)

                        tmp_patch_h = cur_end_row - cur_start_row
                        tmp_patch_w = cur_end_col - cur_start_col

                        patch_color_src = color_src[:, cur_start_row:cur_end_row, cur_start_col:cur_end_col, :]
                        patch_color_tar_to_src = color_tar_to_src[:, cur_start_row:cur_end_row, cur_start_col:cur_end_col, :]
                        patch_outputs = outputs[:, cur_start_row:cur_end_row, cur_start_col:cur_end_col, :]
                        patch_mask = mask[:, cur_start_row:cur_end_row, cur_start_col:cur_end_col, :]

                        patch_color_src_rendered_area = color_src_rendered_area[:, cur_start_row:cur_end_row, cur_start_col:cur_end_col, :]
                        patch_outputs_rendered_area = outputs_rendered_area[:, cur_start_row:cur_end_row, cur_start_col:cur_end_col, :]

                        # we compute the offset between rendered image and GT view.
                        # NOTE:
                        # 1. the order of input matters! Put GT first, rendered second.
                        # 2. We need to compute offset based on pure non-rendered mask instead union mask from tar2src etc.
                        # [1, 2]
                        patch_misalign_offset = compute_offset_fft(
                            patch_color_src_rendered_area.permute(0, 3, 1, 2),
                            patch_outputs_rendered_area.permute(0, 3, 1, 2),
                            use_valid_mask=False
                        )

                        patch_shift_row = patch_misalign_offset[0, 0]
                        patch_shift_col = patch_misalign_offset[0, 1]

                        # print(
                        #     f"\n index: {src_index}, {from_tar_index}; shift: {patch_shift_row}, {patch_shift_col}; img: {tmp_patch_h}, {tmp_patch_w}\n"
                        # )

                        if (np.abs(patch_shift_row) > tmp_patch_h * cur_valid_offset_threshold) or (
                            np.abs(patch_shift_col) > tmp_patch_w * cur_valid_offset_threshold
                        ):
                            patch_shift_row, patch_shift_col = 0, 0

                        patch_gt_img_boundary, patch_rendered_img_boundary = compute_boundary_after_shift(
                            (tmp_patch_h, tmp_patch_w), (tmp_patch_h, tmp_patch_w), patch_shift_row, patch_shift_col
                        )
                        ref_min_row, ref_max_row, ref_min_col, ref_max_col = patch_gt_img_boundary
                        (
                            rendered_min_row,
                            rendered_max_row,
                            rendered_min_col,
                            rendered_max_col,
                        ) = patch_rendered_img_boundary

                        patch_color_src = patch_color_src[
                            :, ref_min_row:ref_max_row, ref_min_col:ref_max_col, :
                        ]
                        # NOTE: we need to treat color_tar_to_src as GT
                        patch_color_tar_to_src = patch_color_tar_to_src[
                            :, ref_min_row:ref_max_row, ref_min_col:ref_max_col, :
                        ]

                        patch_outputs = patch_outputs[
                            :,
                            rendered_min_row:rendered_max_row,
                            rendered_min_col:rendered_max_col,
                            :,
                        ]

                        # NOTE: we try to make sure the areas to be compared are same
                        patch_mask1 = patch_mask[:, ref_min_row:ref_max_row, ref_min_col:ref_max_col, :]
                        patch_mask2 = patch_mask[
                            :,
                            rendered_min_row:rendered_max_row,
                            rendered_min_col:rendered_max_col,
                            :,
                        ]
                        patch_mask = ((patch_mask1 > 0) & (patch_mask2 > 0)).float()

                        patch_color_src = masked_out_img(patch_color_src, patch_mask)
                        patch_color_tar_to_src = masked_out_img(patch_color_tar_to_src, patch_mask)
                        patch_outputs = masked_out_img(patch_outputs, patch_mask)

                        all_color_src.append(patch_color_src)
                        all_color_tar_to_src.append(patch_color_tar_to_src)
                        all_outputs.append(patch_outputs)
                        all_masks.append(patch_mask)

                        if cur_max_crop_h < patch_mask.shape[1]:
                            cur_max_crop_h = patch_mask.shape[1]
                        if cur_max_crop_w < patch_mask.shape[2]:
                            cur_max_crop_w = patch_mask.shape[2]
                
                # concatenate all patches together
                cur_n_patches = len(all_color_src)
                cat_color_src = torch.zeros((cur_n_patches, cur_max_crop_h, cur_max_crop_w, 3), device=all_color_src[0].device)
                cat_color_tar_to_src = torch.zeros((cur_n_patches, cur_max_crop_h, cur_max_crop_w, 3), device=all_color_src[0].device)
                cat_outputs = torch.zeros((cur_n_patches, cur_max_crop_h, cur_max_crop_w, 3), device=all_color_src[0].device)
                cat_masks = torch.zeros((cur_n_patches, cur_max_crop_h, cur_max_crop_w, 1), device=all_color_src[0].device)

                for elem_i in range(len(all_color_src)):
                    _, tmp_h, tmp_w, _ = all_masks[elem_i].shape
                    cat_color_src[elem_i, :tmp_h, :tmp_w, :] = all_color_src[elem_i]
                    cat_color_tar_to_src[elem_i, :tmp_h, :tmp_w, :] = all_color_tar_to_src[elem_i]
                    cat_outputs[elem_i, :tmp_h, :tmp_w, :] = all_outputs[elem_i]
                    cat_masks[elem_i, :tmp_h, :tmp_w, :] = all_masks[elem_i]
                
                color_src = cat_color_src
                color_tar_to_src = cat_color_tar_to_src
                outputs = cat_outputs
                mask = cat_masks

            # offsets = torch.randint(low=0, high=70, size=[2])
            tmp_high = min(70, int(outputs.shape[1] * 0.5), int(outputs.shape[2] * 0.5))
            offsets = torch.randint(low=0, high=tmp_high, size=[2])
            sy = offsets[0]
            sx = offsets[1]

            # Train D

            predict_real, mask_real = compute_d_loss_real(
                model_disc, color_src, color_tar_to_src, mask, sx, sy
            )

            # NOTE: we detach outputs
            predict_fake, mask_fake = compute_d_loss_fake(
                model_disc, color_src, outputs.detach(), mask, sx, sy
            )

            discrim_loss = torch.sum(
                (-(torch.log(predict_real + EPS) + torch.log(1 - predict_fake + EPS)))
                * mask_real
            ) / (torch.sum(mask_real) + EPS)

            with torch.no_grad():
                # NOTE: when training D, G's loss should not have gradient
                gen_loss, gen_loss_L1, gen_loss_GAN = compute_g_loss(
                    predict_fake,
                    outputs.detach(),
                    color_tar_to_src,
                    mask,
                    sx,
                    sy,
                    l1_weight,
                )

            if discrim_loss > gen_loss_GAN:
                # https://github.com/hjwdzh/AdversarialTexture/issues/3
                # NOTE: looks like it is different from what the author states
                discrim_loss_final = discrim_loss
            else:
                discrim_loss_final = discrim_loss * 0

            # print("\ndiscrim_loss: ", discrim_loss, gen_loss_GAN, "\n")

            discrim_loss_final.backward()
            optimizer_disc.step()

            # Train G:
            predict_fake, mask_fake = compute_d_loss_fake(
                model_disc, color_src, outputs, mask, sx, sy
            )
            gen_loss, gen_loss_L1, gen_loss_GAN = compute_g_loss(
                predict_fake, outputs, color_tar_to_src, mask, sx, sy, l1_weight
            )

            gen_loss.backward()
            optimizer_tex.step()

            if global_step % save_interval == 0:
                cur_tex = model_tex.tex.detach().cpu().numpy()[0, ...]
                # Make it daemon
                tmp_step = global_step // 100
                tmp_mp_queue = mp.Queue()
                writer_proc = mp.Process(
                    target=save_tex_img,
                    args=(cur_tex, tmp_step, args.output_dir, tmp_mp_queue,),
                )
                writer_proc.daemon = True
                writer_proc.start()
                img_writer_procs[tmp_step] = (writer_proc, tmp_mp_queue)

                print("\n")
                tmp_to_del = []
                for tmp_k in img_writer_procs:
                    try:
                        # No blocking
                        tmp_msg = img_writer_procs[tmp_k][1].get(block=False)
                        assert tmp_msg == "Finished", f"{tmp_msg}"
                        img_writer_procs[tmp_k][0].join()
                        img_writer_procs[tmp_k][1].close()
                        img_writer_procs[tmp_k][1].join_thread()
                        tmp_to_del.append(tmp_k)
                        print(f"tex_{tmp_k:06d} has been saved.")
                    except:
                        print(f"tex_{tmp_k:06d} has NOT been saved yet.")
                        pass
                print("\n")

                for tmp_k in tmp_to_del:
                    del img_writer_procs[tmp_k]

            if global_step % 10 == 0:
                print(
                    "iter=%d, lossL1=%.4f lossG=%.4f lossD=%.4f"
                    % (global_step, gen_loss_L1, gen_loss_GAN, discrim_loss,)
                )

        del train_iter
    
    tmp_to_del = []
    for tmp_k in img_writer_procs:
        tmp_msg = img_writer_procs[tmp_k][1].get(block=True)
        assert tmp_msg == "Finished", f"{tmp_msg}"
        img_writer_procs[tmp_k][0].join()
        img_writer_procs[tmp_k][1].close()
        img_writer_procs[tmp_k][1].join_thread()
        tmp_to_del.append(tmp_k)
        print(f"tex_{tmp_k:06d} has been saved.")
    
    for tmp_k in tmp_to_del:
        del img_writer_procs[tmp_k]

    final_tex = model_tex.tex.detach().cpu().numpy()[0, ...]
    # [-1, 1] -> [0, 1]
    final_tex = (np.clip(final_tex * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    final_tex_f = os.path.join(args.output_dir, f"{filename}.png")
    Image.fromarray(final_tex).save(final_tex_f)

    tar_f = os.path.join(args.output_dir, "shape/mtl0.png")
    shutil.copyfile(final_tex_f, tar_f)
    print(f"\nComplete copy file from {final_tex_f} to {tar_f}.\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lr_G", type=float, default=1e-3)
    parser.add_argument("--lr_D", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--data_chair", type=int, default=0)
    parser.add_argument("--use_mislaign_offset", type=int, default=0)
    parser.add_argument("--from_scratch", type=int, default=0)
    parser.add_argument("--scannet", type=int, default=0)
    parser.add_argument("--z_diff_threshold", type=float, default=0.1)
    parser.add_argument("--n_patches_h", type=int, default=1)
    parser.add_argument("--n_patches_w", type=int, default=1)

    args = parser.parse_args()

    folder_name = "seed_{}-scratch_{}-offset_{}_n_patch_h_{}_w_{}".format(
        args.seed,
        args.from_scratch,
        args.use_mislaign_offset,
        args.n_patches_h,
        args.n_patches_w,
    )

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input_dir.rstrip("/"))
    save_dir = os.path.join(args.output_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    args.output_dir = save_dir
    print("\ninput_dir: ", args.input_dir, "\n")
    print("\nsave_dir: ", save_dir, "\n")

    # copy obj file etc
    try:
        os.makedirs(os.path.join(args.output_dir, "shape"), exist_ok=True)

        src_obj_f = os.path.join(args.input_dir, "shape", OBJ_FILENAME)
        tar_obj_f = os.path.join(args.output_dir, "shape", OBJ_FILENAME)
        print("\nobj: ", src_obj_f, tar_obj_f, "\n")
        shutil.copyfile(src_obj_f, tar_obj_f)

        src_mtl_f = os.path.join(os.path.join(args.input_dir, "shape", MTL_FILENAME))
        tar_mtl_f = os.path.join(os.path.join(args.output_dir, "shape", MTL_FILENAME))
        print("\nmtl: ", src_mtl_f, tar_mtl_f, "\n")
        shutil.copyfile(src_mtl_f, tar_mtl_f)
    except:
        traceback.print_exc()
        err = sys.exc_info()[0]
        print(err)

    # reproducibility set up
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train(args)
