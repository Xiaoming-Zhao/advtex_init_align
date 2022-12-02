import os
import copy
import glob
import tqdm
import numpy as np
from PIL import Image

from advtex_init_align.utils.logging import EasyDict


def cam_mat_to_ex_intr_mat(stream_type, view_mat, proj_mat, img_h, img_w):

    K = np.eye(3)

    if stream_type == "apple":
        # in Apple stream, X represents vertical axis.
        # However, in Open3D, X represents horizontal:
        #      https://github.com/intel-isl/Open3D/blob/ae4178f/cpp/open3d/pipelines/color_map/ColorMapUtils.cpp#L40
        # we must re-order the view matrix
        pose_mat = copy.deepcopy(view_mat[(1, 0, 2, 3), :])

        # Open3D treats +Z from camera to object, differs from Apple's principle of z-towards-viewer.
        # - Open3D: https://github.com/intel-isl/Open3D/blob/ae4178f/cpp/open3d/pipelines/color_map/ColorMapUtils.cpp#L159
        # - Apple:
        #   - https://developer.apple.com/documentation/arkit/world_tracking/understanding_world_tracking
        #   - https://developer.apple.com/documentation/arkit/arconfiguration/worldalignment/gravity
        #   - https://developer.apple.com/documentation/arkit/arconfiguration/worldalignment/camera
        #
        # Please note, in CPP, we compute transform_mat via proj_mat * view_mat,
        # in this way (as OpenGL), we will have +Z points from camera to object in NDC.
        # More details appear in http://www.songho.ca/opengl/gl_projectionmatrix.html
        # In short, the 4th row of proj_mat add negative sign on Z axis.
        #
        # However, in Open3D, we compute image coordinates w/ intrinsic matrix as
        # https://github.com/intel-isl/Open3D/blob/ae4178f/cpp/open3d/pipelines/color_map/ColorMapUtils.cpp#L48
        # Therefore, we must manually change the sign of Z.
        pose_mat[2, :] = -1 * pose_mat[2, :]

        # Y-down: https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
        # However,In Apple stream's pixel coordinate system, +X is for height (vertical),
        # Therefore, fx comes from 2nd row of proj matrix
        K[0, 0] = float(proj_mat[1, 1] / 2 * img_w)
        K[1, 1] = float(proj_mat[0, 0] / 2 * img_h)

        # Image in stream is left-right flipped, we need to convert it back.
        K[0, 2] = float((1 - proj_mat[1, 2]) / 2 * img_w)
        K[1, 2] = float((1 - proj_mat[0, 2]) / 2 * img_h)
    elif stream_type == "scannet":
        # Essentially, we just reverse the operation we used for converting to Apple stream
        pose_mat = copy.deepcopy(view_mat)

        # when converting to Apple stream format, we mannually switch 1st and 2nd row in projection matrix.
        # We need to switch it back.
        proj_mat[1, :] = -1 * proj_mat[1, :]
        proj_mat = proj_mat[(1, 0, 2, 3), :]

        K[0, 0] = float(proj_mat[0, 0] / 2 * img_w)
        K[1, 1] = float(proj_mat[1, 1] / 2 * img_h)

        K[0, 2] = float((proj_mat[1, 2] + 1) / 2 * img_w)
        K[1, 2] = float((proj_mat[0, 2] + 1) / 2 * img_h)
    else:
        raise ValueError

    return K, pose_mat


def ex_tri_mat_to_view_proj_mat(K, world2cam_mat, img_w, img_h, stream_type):

    assert stream_type == "scannet", f"Currently only support ScanNet"

    view_mat = copy.deepcopy(world2cam_mat)

    # NOTE: start processing projection matrix
    # originally, projection matrix maps to NDC with range [0, 1]
    # to align with our CPP implementation, we modify it to make points mapped to NDC with range [-1, 1].
    # Specifically, assume original projection matrix is the following:
    # M1 = [[fu, 0, u],
    #       [0, fv, v],
    #       [0, 0,  1]]
    # where fu, fv are focal lengths and (u, v) marks the principal point.
    # Now we change the projection matrix to:
    # M2 = [[2fu, 0,   2u - 1],
    #       [0,   2fv, 2v - 1],
    #       [0,   0,   1]]
    #
    # The validity can be verified as following:
    # a) left end value:
    # assume point p0 = (h0, w0, 1)^T is mapped to (0, 0, 1), namely:
    # M1 * p0 = (0, 0, 1)^T
    # ==> h0 = -u / fu, w0 = -v / fv
    # ==> M2 * p0 = (-1, -1, 1)
    #
    # b) right end value:
    # assume point p1 = (h1, w1, 1)^T is mapped to (1, 1, 1), namely:
    # M1 * p1 = (1, 1, 1)^T
    # ==> h1 = (1 - u) / fu, w0 = (1 - v) / fv
    # ==> M2 * p1 = (1, 1, 1)
    proj_mat = np.eye(4)
    proj_mat[0, 0] = 2 * K[0, 0] / img_w
    proj_mat[1, 1] = 2 * K[1, 1] / img_h
    proj_mat[0, 2] = 2 * K[0, 2] / img_w - 1
    proj_mat[1, 2] = 2 * K[1, 2] / img_h - 1
    proj_mat[2, 2] = 1
    proj_mat[3, 3] = 0
    proj_mat[3, 2] = 1

    # make 1st elem for height, 2nd elem for width
    proj_mat = proj_mat[(1, 0, 2, 3), :]

    # NOTE: we need to flip left-right to align with Apple format's convention
    # However, since we alreay left-right flipped image, whose projection matrix should be flipped.
    # After fliping the "flipped" projection matrix, the projection matrix remain the same.
    proj_mat[1, :] = -proj_mat[1, :]

    return view_mat, proj_mat


def read_scannet_data(stream_type, scannet_data_dir, read_depth=False, for_train=True):
    # ScanNet's scene has thousands of high-res images.
    # It is too slow to read with struct.unpack.
    # We directly read from disk.
    if for_train:
        gt_rgb_fs = sorted(list(glob.glob(os.path.join(scannet_data_dir, "*_color.png"))))
    else:
        gt_rgb_fs = sorted(list(glob.glob(os.path.join(scannet_data_dir, "*_raw_color.png"))))
    gt_rgbs = [np.array(Image.open(_)) for _ in tqdm.tqdm(gt_rgb_fs)]

    raw_idx_list = [int(os.path.basename(_).split("_")[0]) for _ in gt_rgb_fs]

    if read_depth:
        gt_depth_fs = [os.path.join(scannet_data_dir, f"{i:05d}_depth.npz") for i in raw_idx_list]
        gt_depths = [np.load(_)["arr_0"] for _ in tqdm.tqdm(gt_depth_fs)]
    else:
        gt_depths = None

    intri_mat_fs = [os.path.join(scannet_data_dir, f"{i:05d}_intrinsic.txt") for i in raw_idx_list]
    intri_mats = [np.loadtxt(_) for _ in intri_mat_fs]

    extri_mat_fs = [os.path.join(scannet_data_dir, f"{i:05d}_pose.txt") for i in raw_idx_list]
    extri_mats = [np.loadtxt(_) for _ in extri_mat_fs]

    view_matrices = []
    proj_matrices = []
    for i in range(len(gt_rgb_fs)):
        tmp_h, tmp_w, _ = gt_rgbs[i].shape
        tmp_view_mat, tmp_proj_mat = ex_tri_mat_to_view_proj_mat(intri_mats[i], extri_mats[i], tmp_w, tmp_h, stream_type)
        view_matrices.append(tmp_view_mat)
        proj_matrices.append(tmp_proj_mat)

    view_matrices = np.array(view_matrices)
    proj_matrices = np.array(proj_matrices)

    data_dict = EasyDict(
        rgbs=gt_rgbs,
        depth_maps=gt_depths,
        view_matrices=view_matrices,
        proj_matrices=proj_matrices,
        raw_idx_list=raw_idx_list,
    )

    return data_dict