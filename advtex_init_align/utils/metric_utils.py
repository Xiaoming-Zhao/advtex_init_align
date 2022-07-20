import tqdm
import numpy as np
from PIL import Image

import torch


def metric_avg_grad_intensity(img):
    img = np.array(Image.fromarray(img).convert('L')).astype(np.float32) / 255.0
    # print(img.shape, img.dtype)
    grad_x, grad_y = np.gradient(img)
    # print(grad_x.shape, grad_y.shape)
    gnorm = np.sqrt(grad_x ** 2 + grad_y ** 2)
    avg_intensity = np.mean(gnorm)
    return gnorm, avg_intensity


# ---------------------------------------------------------------------------------------------------------------------
# For patch distance error mentioned in Sec. 4 of https://arxiv.org/pdf/2003.08400.pdf

def get_flat_patch_pixs(img, patch_size, start_row, end_row, start_col, end_col, patch_flat_rel_row_idx, patch_flat_rel_col_idx):
    
    # [batch, batch]
    batch_rows, batch_cols = torch.meshgrid(torch.arange(start_row, end_row), torch.arange(start_col, end_col))
    # print("batch_rows: ", batch_rows.shape)
    
    # [batch x batch, 1]
    flat_batch_rows = batch_rows.reshape((-1, 1))
    flat_batch_cols = batch_cols.reshape((-1, 1))
    # print("flat_batch_rows: ", flat_batch_rows.shape)
            
    # [batch x batch, patch x patch]
    flat_patch_rows = flat_batch_rows + patch_flat_rel_row_idx
    flat_patch_cols = flat_batch_cols + patch_flat_rel_col_idx
    # print("flat_patch_rows: ", flat_patch_rows.shape)
    
    # [batch x batch, patch x patch, 3]
    batch_rgbs = img[flat_patch_rows, flat_patch_cols, :]
    # print("batch_rgbs: ", batch_rgbs.shape)
    
    # [batch x batch, patch x patch x 3]
    flat_batch_rgbs = batch_rgbs.reshape((-1, patch_size * patch_size * 3))
    # print("flat_batch_rgbs: ", flat_batch_rgbs.shape)
    
    flat_batch_rgbs = flat_batch_rgbs.float()
    
    return flat_batch_rgbs, flat_batch_rows, flat_batch_cols


def get_patch_rel_idx(patch_size):
    pad_front = int((patch_size - 1) / 2.0)
    pad_back = patch_size - 1 - pad_front
    
    patch_rel_row_idx, patch_rel_col_idx = torch.meshgrid(torch.arange(-pad_front, pad_back + 1), torch.arange(-pad_front, pad_back + 1))
    
    # [1, patch x patch]
    patch_flat_rel_row_idx = patch_rel_row_idx.reshape((1, -1))
    patch_flat_rel_col_idx = patch_rel_col_idx.reshape((1, -1))
    
    return pad_front, pad_back, patch_flat_rel_row_idx, patch_flat_rel_col_idx


def metric_patch_l2_distance(img_ref, img_gen, patch_size, ref_search_size_one_side=25):
    
    device = img_ref.device
    
    h, w, _ = img_ref.shape
    
    pad_front, pad_back, gen_patch_flat_rel_row_idx, gen_patch_flat_rel_col_idx = get_patch_rel_idx(patch_size)
    # _, _, ref_patch_flat_rel_row_idx, ref_patch_flat_rel_col_idx = get_patch_rel_idx(ref_search_size)
    # print("ref_patch_flat_rel_row_idx: ", ref_patch_flat_rel_row_idx.shape)
    
    start_row = pad_front
    end_row = h - pad_back
    start_col = pad_front
    end_col = w - pad_back
    
    # # [#pixels, 3 x patch x patch]
    # flat_img_ref_rgbs = get_flat_patch_pixs(img_ref, patch_size, start_row, end_row, start_col, end_col, patch_flat_rel_row_idx, patch_flat_rel_col_idx)
    # print("flat_img_ref_rgbs: ", flat_img_ref_rgbs.shape)
    
    all_pix_coord_diff = []
    
    batch = 10
    for tmp_start_row in tqdm.tqdm(range(start_row, end_row, batch)):
        for tmp_start_col in range(start_col, end_col, batch):
            tmp_end_row = min(tmp_start_row + batch, end_row)
            tmp_end_col = min(tmp_start_col + batch, end_col)
            
            tmp_ref_start_row = max(start_row, tmp_start_row - ref_search_size_one_side)
            tmp_ref_end_row = min(end_row, tmp_end_row + ref_search_size_one_side)
            tmp_ref_start_col = max(start_col, tmp_start_col - ref_search_size_one_side)
            tmp_ref_end_col = min(end_col, tmp_end_col + ref_search_size_one_side)
            
            # [batch x batch, 3 x ref_patch x ref_patch]
            tmp_flat_img_ref_rgbs, tmp_flat_ref_rows, tmp_flat_ref_cols = get_flat_patch_pixs(img_ref, patch_size, tmp_ref_start_row, tmp_ref_end_row, tmp_ref_start_col, tmp_ref_end_col, gen_patch_flat_rel_row_idx, gen_patch_flat_rel_col_idx)
            # print("tmp_flat_img_ref_rgbs: ", tmp_flat_img_ref_rgbs.shape)
            
            # [batch x batch, 3 x patch x patch]
            tmp_flat_img_gen_rgbs, tmp_flat_gen_rows, tmp_flat_gen_cols = get_flat_patch_pixs(img_gen, patch_size, tmp_start_row, tmp_end_row, tmp_start_col, tmp_end_col, gen_patch_flat_rel_row_idx, gen_patch_flat_rel_col_idx)
            # print("tmp_flat_img_gen_rgbs: ", tmp_flat_img_gen_rgbs.shape)
            
            # [batch x batch, #ref_pixels, 3 x patch x patch]
            tmp_rgb_diff = tmp_flat_img_gen_rgbs.unsqueeze(1) - tmp_flat_img_ref_rgbs.unsqueeze(0)
            # print("tmp_rgb_diff: ", tmp_rgb_diff.shape)
            
            # [batch x batch, #ref_pixels
            tmp_rgb_diff_norm = torch.norm(tmp_rgb_diff, p=2, dim=2)
            # print("tmp_rgb_diff_norm: ", tmp_rgb_diff_norm.shape)
            
            # [batch x batch, ]
            min_idx = torch.min(tmp_rgb_diff_norm, dim=1)[1]
            
            # [batch x batch, 1]
            tmp_min_ref_rows = tmp_flat_ref_rows[min_idx]
            tmp_min_ref_cols = tmp_flat_ref_cols[min_idx]
            # print("tmp_min_ref_rows: ", tmp_min_ref_rows.shape)
            
            # [batch x batch, 1]
            tmp_row_diff = (tmp_flat_gen_rows - tmp_min_ref_rows).float()
            tmp_col_diff = (tmp_flat_gen_cols - tmp_min_ref_cols).float()
            
            # [batch x batch, 1], we compute distance in [0, 1] range to make it resolution-invariant
            tmp_pix_coord_diff = torch.sqrt((tmp_row_diff / h) ** 2 + (tmp_col_diff / w) ** 2)
            
            all_pix_coord_diff.append(tmp_pix_coord_diff)
    
    all_pix_coord_diff = torch.cat(all_pix_coord_diff, dim=0)
    avg_pix_coord_diff = torch.mean(all_pix_coord_diff)
    
    return avg_pix_coord_diff
