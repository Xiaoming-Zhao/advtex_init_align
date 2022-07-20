import numpy as np

import torch


def compute_offset_fft(inp, gt, use_valid_mask=True):
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

    # NOTE: filter our invalid values first
    # we filter out value that are too unreasonable
    _, _, img_h, img_w = inp.shape
    img_res = (
        torch.FloatTensor([img_h, img_w]).reshape((1, 1, 2)).to(offset_init.device)
    )

    if use_valid_mask:
        # [B, C, 2]
        valid_mask = (
            (torch.abs(offset_init) <= img_res * 0.05).float().to(offset_init.device)
        )
        masked_offset = valid_mask * offset_init
    else:
        valid_mask = torch.ones_like(offset_init)
        masked_offset = offset_init

    # [B, 2], 1st elem is for row, 2nd elem is for col
    mean_offset_init = torch.sum(masked_offset, dim=1) / (
        torch.sum(valid_mask, dim=1) + 1e-8
    )
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
