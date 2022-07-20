import os
import tqdm
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def world_to_cam_coords_chunk(view_mats, v_coords):
    """
    view_mats: [#cameras, 4, 4];
    v_coords: [#points, 3], world coordinates
    """
    # [4, #points]
    homo_v_coords = torch.cat(
        (
            torch.transpose(v_coords, 0, 1),
            torch.ones((1, v_coords.shape[0]), device=v_coords.device),
        ),
        dim=0,
    )
    # [#cameras, 4, #points]
    v_cam_coords = torch.matmul(view_mats, homo_v_coords)
    # [#cameras, 3, #points]
    v_cam_coords = v_cam_coords[:, :3, :] / (v_cam_coords[:, 3:, :] + 1e-8)

    return v_cam_coords


def world_to_cam_coords(view_mats, v_coords, save_mem=False):
    """
    view_mats: [#cameras, 4, 4];
    v_coords: [#points, 3], world coordinates

    return: [#cameras, 3, #points]
    """
    n_cams = int(view_mats.shape[0])
    n_verts = int(v_coords.shape[0])

    if save_mem:
        v_cam_coords = torch.zeros((n_cams, 3, n_verts))

        batch = 100000
        start_id = 0
        for start_id in tqdm.tqdm(np.arange(0, n_verts, batch)):
            end_id = min(start_id + batch, n_verts)
            cur_v_coords = v_coords[start_id:end_id, :]
            cur_v_cam_coords = world_to_cam_coords_chunk(view_mats, cur_v_coords)
            v_cam_coords[..., start_id:end_id] = cur_v_cam_coords
    else:
        v_cam_coords = world_to_cam_coords_chunk(view_mats, v_coords)

    return v_cam_coords


def world_to_cam_coords_pix(view_mats, pix_to_v_coords, save_mem=False):
    """
    view_mats: [#cameras, 4, 4];
    pix_to_v_coords: [H, W, 3, 3], world coordinates

    return: [H, W, 3, #cam, 3]
    """
    h, w, _, _ = pix_to_v_coords.shape

    # [#points, 3]
    v_coords = pix_to_v_coords.view((-1, 3))

    # [#cameras, 3, #points]
    v_cam_coords = world_to_cam_coords(view_mats, v_coords, save_mem=save_mem)

    # [#cameras, 3, #points] --> [#cameras, #points, 3] --> [#cameras, H, W, 3, 3], 2nd 3 is for 3-dim coordinates
    pix_to_v_coords = v_cam_coords.permute(0, 2, 1).view(
        (v_cam_coords.shape[0], h, w, 3, 3)
    )

    return pix_to_v_coords


def compute_ndc_chunk(v_coords, transform_matrices):
    r"""This function computes the normalized device coordinates (NDC).
    However, this function does not do clipping.
    We intentionally postpone clipping until where NDC will be used.
    This allows more flexibilities and conveniency.
    """

    n_vertices = int(v_coords.shape[0])

    # get Homogeneous coordinates, shape [#vertices, 4], [x, y, z, w]
    v_homo_coords = torch.cat(
        (v_coords, torch.ones((n_vertices, 1), device=v_coords.device)), dim=1
    )
    # print(v_homo_coords.shape)

    # [#cameras, 4, #vertices] -->[#cameras, #vertices, 4]
    ndc = torch.matmul(
        transform_matrices, torch.transpose(v_homo_coords, 1, 0)
    ).permute((0, 2, 1))

    # if save_mem:
    #     ndc = ndc.to(torch.device("cpu"))
    #     torch.cuda.empty_cache()

    # perspective projection
    ndc[..., :3] = ndc[..., :3] / (ndc[..., -1:] + 1e-8)

    ndc_z = ndc[..., 2:3]
    ndc = ndc[..., :2]

    # NOTE: this is because Apple's format has left-right flip
    ndc[..., 1] = -1 * ndc[..., 1]
    # change value range from [-1, 1] to [0, 1]
    ndc = (ndc + 1) / 2

    return ndc, ndc_z


def compute_ndc(v_coords, transform_matrices, save_mem=False):
    """
    v_coords: [#points, 3];
    transform_matrices: [#cams, 4, 4].
    """

    n_cams = int(transform_matrices.shape[0])
    n_verts = int(v_coords.shape[0])

    if save_mem:
        ndc = torch.zeros((n_cams, n_verts, 2), device=v_coords.device)
        ndc_z = torch.zeros((n_cams, n_verts, 1), device=v_coords.device)

        batch = 100000
        start_id = 0
        for start_id in tqdm.tqdm(np.arange(0, n_verts, batch)):
            end_id = min(start_id + batch, n_verts)
            cur_v_coords = v_coords[start_id:end_id, :]
            cur_ndc, cur_ndc_z = compute_ndc_chunk(cur_v_coords, transform_matrices)
            ndc[:, start_id:end_id, :] = cur_ndc
            ndc_z[:, start_id:end_id, :] = cur_ndc_z
    else:
        ndc, ndc_z = compute_ndc_chunk(v_coords, transform_matrices)

    return ndc, ndc_z


def compute_face_area_chunk(ndc, face_v_ids):
    """
    ndc: [#cameras, #vertices, 2];
    face_v_ids: [#faces, 3]
    """
    # [#cameras, #faces, 3, 2], the 2 is for [x, y]
    face_v_ndc_coords = ndc[:, face_v_ids, :]
    # print(face_v_ndc_coords.shape)

    # [#cameras, #faces, 2]
    dist1 = face_v_ndc_coords[..., 1, :] - face_v_ndc_coords[..., 0, :]
    dist2 = face_v_ndc_coords[..., 2, :] - face_v_ndc_coords[..., 0, :]

    # [#cameras, #faces], area_of_face = area_of_trapezoid - area_of_two_triangles
    cue_matrix = (
        torch.abs(dist1[..., 0] * dist2[..., 1] - dist1[..., 1] * dist2[..., 0]) / 2
    )

    # [#cameras, #faces]
    full_visibility = torch.all(
        (face_v_ndc_coords[..., 0] >= 0)
        & (face_v_ndc_coords[..., 0] <= 1)
        & (face_v_ndc_coords[..., 1] >= 0)
        & (face_v_ndc_coords[..., 1] <= 1),
        dim=-1,
    )
    # print(full_visibility.shape)
    cue_matrix[~full_visibility] = 0.0

    cue_matrix = cue_matrix.permute((1, 0))

    # [#faces, #cam]
    return cue_matrix


def compute_face_area(ndc, face_v_ids, save_mem=False):
    """
    ndc: [#cameras, #vertices, 2];
    face_v_ids: [#faces, 3]
    """
    n_cams = int(ndc.shape[0])
    n_faces = int(face_v_ids.shape[0])
    if save_mem:
        face_area_mat = torch.zeros((n_faces, n_cams))
        batch = 100000
        start_id = 0
        for start_id in tqdm.tqdm(np.arange(0, n_faces, batch)):
            end_id = min(start_id + batch, n_faces)
            cur_face_v_ids_torch = face_v_ids[start_id:end_id, :]
            face_area_mat[start_id:end_id, :] = compute_face_area_chunk(
                ndc, cur_face_v_ids_torch
            ).to(torch.device("cpu"))
    else:
        face_area_mat = compute_face_area_chunk(ndc, face_v_ids).to(torch.device("cpu"))

    # [#faces, #cam]
    return face_area_mat


def compute_face_cam_pairs_chunk(ndc, ndc_z, face_v_ids):
    """
    ndc: [#cameras, #vertices, 2];
    ndc_z: [#cameras, #vertices, 1];
    face_v_ids: [#faces, 3]
    """
    n_cams = ndc.shape[0]
    n_faces = face_v_ids.shape[0]

    # [#cameras, #faces, 3, 1]
    face_v_ndc_z_coords = ndc_z[:, face_v_ids, :]

    # ndc_z = ndc_z.to(torch.device("cpu"))
    del ndc_z
    torch.cuda.empty_cache()

    # [#cameras, #faces]
    positive_z = torch.all((face_v_ndc_z_coords[..., 0] >= 0), dim=2)
    # print(positive_z.shape)

    # face_v_ndc_z_coords = face_v_ndc_z_coords.to(torch.device("cpu"))
    del face_v_ndc_z_coords
    torch.cuda.empty_cache()

    # [#cameras, #faces, 3, 2], the 2 is for [x, y]
    face_v_ndc_xy_coords = ndc[:, face_v_ids, :]

    # ndc = ndc.to(torch.device("cpu"))
    del ndc
    torch.cuda.empty_cache()

    # [#cameras, #faces]
    partial_visibility = torch.any(
        (face_v_ndc_xy_coords[..., 0] >= 0)
        & (face_v_ndc_xy_coords[..., 0] <= 1)
        & (face_v_ndc_xy_coords[..., 1] >= 0)
        & (face_v_ndc_xy_coords[..., 1] <= 1),
        dim=-1,
    )
    # print(partial_visibility.shape)

    # [#cameras, #faces]
    n_visible_verts = torch.sum(
        (face_v_ndc_xy_coords[..., 0] >= 0)
        & (face_v_ndc_xy_coords[..., 0] <= 1)
        & (face_v_ndc_xy_coords[..., 1] >= 0)
        & (face_v_ndc_xy_coords[..., 1] <= 1),
        dim=-1,
    )

    # face_v_ndc_xy_coords = face_v_ndc_xy_coords.to(torch.device("cpu"))
    del face_v_ndc_xy_coords
    torch.cuda.empty_cache()

    # [#cameras, #faces]
    valid_bool = positive_z & partial_visibility

    n_visible_verts[~valid_bool] = 0
    n_visible_verts = n_visible_verts.permute(1, 0)
    # maximum val is three, uint8 is enough
    n_visible_verts = n_visible_verts.byte()

    """
    # [#cameras, #faces, 3]
    face_v_ndc_x_coords = ndc[:, face_v_ids, 0]

    partial_visibility_x = torch.any(
        (face_v_ndc_x_coords >= 0)
        & (face_v_ndc_x_coords <= 1),
        dim=-1,
    )
    # print("\n", partial_visibility_x.shape)

    face_v_ndc_x_coords = face_v_ndc_x_coords.to(torch.device("cpu"))
    torch.cuda.empty_cache()
    
    # [#cameras, #faces]
    face_v_ndc_y_coords = ndc[:, face_v_ids, 1]

    partial_visibility_y = torch.any(
        (face_v_ndc_y_coords >= 0)
        & (face_v_ndc_y_coords <= 1),
        dim=-1,
    )
    # print("\n", partial_visibility_y.shape)
    
    # [#cameras, #faces]
    valid_bool = (positive_z & partial_visibility_x & partial_visibility_y)
    """

    # [#faces, #cameras]
    valid_bool = valid_bool.permute((1, 0))
    # print(valid_bool.shape)

    # convert boolena to camera indices
    # [#faces, #cam]
    valid_id = torch.arange(n_cams).repeat(n_faces, 1)

    valid_id[~valid_bool] = n_cams

    sorted_idxs = torch.argsort(valid_id, dim=1, descending=False)

    valid_pairs = valid_id[torch.arange(n_faces).unsqueeze(-1), sorted_idxs]

    if False:
        # NOTE: the following check is too costly and slow.
        # I have checked on all 21 scenes of T2 already.
        rows, cols = torch.where(valid_pairs < n_cams)
        ori_cols = valid_pairs[rows, cols]
        assert torch.all(valid_bool[rows, ori_cols])

        assert torch.sum(valid_bool) == torch.sum(
            valid_pairs < n_cams
        ), f"{torch.sum(valid_bool)}, {torch.sum(valid_pairs < n_cams)}"

    return valid_pairs, n_visible_verts


def compute_face_cam_pairs(ndc, ndc_z, face_v_ids, n_raw_scene_fs=None, save_mem=False):
    """
    ndc: [#cameras, #vertices, 2];
    ndc_z: [#cameras, #vertices, 1];
    face_v_ids: [#faces, 3]
    """
    n_cams = ndc.shape[0]
    n_faces = face_v_ids.shape[0]
    if save_mem:
        valid_face_cam_pairs = torch.zeros((n_faces, n_cams)).long()
        n_visible_verts = torch.zeros((n_faces, n_cams)).byte()
        batch = 100000
        start_id = 0
        for start_id in tqdm.tqdm(np.arange(0, n_faces, batch)):
            end_id = min(start_id + batch, n_faces)
            cur_face_v_ids_torch = face_v_ids[start_id:end_id, :]
            (
                cur_valid_face_cam_pairs,
                cur_n_visible_verts,
            ) = compute_face_cam_pairs_chunk(ndc, ndc_z, cur_face_v_ids_torch)
            valid_face_cam_pairs[start_id:end_id, :] = cur_valid_face_cam_pairs.to(
                torch.device("cpu")
            )
            n_visible_verts[start_id:end_id, :] = cur_n_visible_verts.to(
                torch.device("cpu")
            )
    else:
        valid_face_cam_pairs, n_visible_verts = compute_face_cam_pairs_chunk(
            ndc, ndc_z, face_v_ids
        )
        valid_face_cam_pairs = valid_face_cam_pairs.to(torch.device("cpu"))
        n_visible_verts = n_visible_verts.to(torch.device("cpu"))

    # if n_raw_scene_fs is not None:
    #     # make newly-added container visible to all cameras
    #     valid_face_cam_pairs[n_raw_scene_fs:, :] = torch.arange(n_cams).view((1, -1))

    return valid_face_cam_pairs, n_visible_verts


# ----------------------------------------------------------------------
# Positional encoding

# modified from https://github.com/yenchenlin/nerf-pytorch/blob/f38216f43d1939976b440b71290c18ffaab278fa/run_nerf_helpers.py#L18
class PosEncoder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def encode(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_pos_encoder(n_freqs, input_dim=1):

    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dim,
        "max_freq_log2": n_freqs - 1,
        "num_freqs": n_freqs,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    pos_encoder_obj = PosEncoder(**embed_kwargs)
    encoder = lambda x, eo=pos_encoder_obj: eo.encode(x)
    return encoder, pos_encoder_obj.out_dim


# ----------------------------------------------------------------------
# classical RGB/depth gradient

# modified from https://github.com/kornia/kornia/blob/3606cf9c3d1eb3aabd65ca36a0e7cb98944c01ba/kornia/filters/filter.py#L32


def _compute_padding(kernel_size: List[int]) -> List[int]:
    """Computes padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) >= 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymetric padding :(

    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp
    return out_padding


def filter2D(in_tensor, kernel):
    """
    in_tensor: [B, in_C, H, W]
    kernel: [B, kH, kW]
    """
    b, c, h, w = in_tensor.shape
    tmp_kernel = kernel.unsqueeze(1).to(in_tensor)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1).contiguous()
    # print("tmp_kernel: ", tmp_kernel.shape)

    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape = _compute_padding([height, width])
    input_pad = F.pad(in_tensor, padding_shape, mode="reflect")
    # print("input_pad: ", input_pad.shape)

    out_tensor = F.conv2d(input_pad, tmp_kernel, padding=0, stride=1)
    # print("out_tensor: ", out_tensor.shape)

    return out_tensor


Sobel_X = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
Sobel_Y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# NOTE: it is a little bit tricky to construct LoG kernel ourselves
# since we must ensure the sum of the kernel matrix is 0 to enable homogeneous

# https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
LoG_3_1 = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
LoG_3_2 = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
LoG_5 = torch.tensor(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -16, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0],
    ]
)

# https://math.stackexchange.com/questions/2445994/discrete-laplacian-of-gaussian-log
LoG_7 = torch.tensor(
    [
        [0, 0, 3, 2, 2, 2, 3, 0, 0],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [2, 5, 3, -12, -24, -12, 3, 5, 2],
        [2, 5, 0, -24, -40, -24, 0, 5, 2],
        [2, 5, 3, -12, -24, -12, 3, 5, 2],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [0, 0, 3, 2, 2, 2, 3, 0, 0],
    ]
)


# every elem: [kernel, divider]
# divider is used to try to make all kernel results have similar absolute value range
CLASSICAL_KERNELS = {
    "Sobel": {"x": [Sobel_X, 1], "y": [Sobel_Y, 1]},
    "LoG_3_1": [LoG_3_1, 4],
    "LoG_3_2": [LoG_3_2, 8],
    "LoG_5": [LoG_5, 16],
    "LoG_7": [LoG_7, 40],
}

CLASSICAL_KERNELS_ABBR = {
    "Sobel": "s",
    "LoG_3_1": "lg31",
    "LoG_3_2": "lg32",
    "LoG_5": "lg5",
    "LoG_7": "lg7",
}


# ----------------------------------------------------------------------
# learnt RGB/depth gradient


class PerceptDepthGrad(nn.Module):

    out_dim = 3

    def __init__(
        self,
        n_in_channels=3,
        n_out_channels=1,
        # bias=False,
        dilation=1,
        activate=nn.ReLU(inplace=True),
    ):
        # fmt: off
        super(PerceptDepthGrad, self).__init__()
        self.conv1 = nn.Conv2d(n_in_channels, n_out_channels, kernel_size=3, dilation=dilation, stride=1, padding=dilation, bias=False)
        self.conv2 = nn.Conv2d(n_in_channels, n_out_channels, kernel_size=5, dilation=dilation, stride=1, padding=2 * dilation, bias=False)
        self.conv3 = nn.Conv2d(n_in_channels, n_out_channels, kernel_size=7, dilation=dilation, stride=1, padding=3 * dilation, bias=False)
    
        self._weight_init()

        # fmt: on

    def _weight_init(self):
        for m in self.children():
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x):
        grad1 = self.conv1(x)
        grad2 = self.conv2(x)
        grad3 = self.conv3(x)
        # [B, 3 x C, H, W]
        out = torch.cat((grad1, grad2, grad3), dim=1)
        return out
