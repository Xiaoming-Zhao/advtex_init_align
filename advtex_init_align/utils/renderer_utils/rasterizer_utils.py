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

from pytorch3d.transforms.transform3d import Transform3d
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)

# https://github.com/facebookresearch/pytorch3d/issues/112
# https://github.com/facebookresearch/pytorch3d/issues/315
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesVertex

from advtex_init_align.utils.renderer_utils.assign_pix_val import *


class MyCamera(FoVPerspectiveCameras):
    def __init__(self, device=None, view_matrices=None, proj_matrices=None):
        """view_matrix and proj_matrix: [N, 4, 4]"""
        super().__init__()
        self._device = device
        self._view_matrices = view_matrices
        self._proj_matrices = proj_matrices

    def compute_projection_matrix(self):
        return torch.Tensor(self._proj_matrices).to(self._device)

    def get_world_to_view_transform(self):
        transform = Transform3d(device=self._device)
        transform._matrix = torch.Tensor(self._view_matrices).to(self._device)
        return transform

    def get_projection_transform(self):
        transform = Transform3d(device=self._device)
        transform._matrix = torch.Tensor(self._proj_matrices).to(self._device)
        return transform


class MyMeshRasterizer(MeshRasterizer):
    def __init__(self, stream_type="apple", cameras=None, raster_settings=None):
        super().__init__(cameras=cameras, raster_settings=raster_settings)
        self._stream_type = stream_type

    def transform(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                vertex coordinates in world space.
        Returns:
            meshes_screen: a Meshes object with the vertex positions in screen
            space
        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of MeshRasterizer"
            raise ValueError(msg)

        n_cameras = len(cameras)
        if n_cameras != 1 and n_cameras != len(meshes_world):
            msg = "Wrong number (%r) of cameras for %r meshes"
            raise ValueError(msg % (n_cameras, len(meshes_world)))

        verts_world = meshes_world.verts_padded()

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Revisit whether or not to transform z coordinate to [-1, 1] or
        # [0, 1] range.
        verts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(
            verts_world
        )

        if self._stream_type == "apple":
            # NOTE: compare camera view coordinate system in:
            # - https://github.com/facebookresearch/pytorch3d/blob/d565032/docs/notes/cameras.md
            # - http://www.songho.ca/opengl/gl_projectionmatrix.html
            # +Z points from camera to object left in PyTorch3D while +Z points from object to camera in OpenGL
            verts_view[..., 2] = -1 * verts_view[..., 2]
            # verts_view = verts_view[..., [1, 0, 2]]
        elif self._stream_type == "scannet":
            # NOTE: seems like in COLMAP, +Z points from camera to object: https://colmap.github.io/format.html#images-txt
            # which is aligned with PyTorch3D's settings
            pass
        else:
            raise ValueError

        verts_screen = cameras.get_projection_transform(**kwargs).transform_points(
            verts_view
        )

        # NOTE: if we uncomment the following line:
        # - apple: fail
        # - ScanNet: succeed. However, after uncommnet, we do not need to flip the final rendered image
        # not sure why ScanNet will word, it seems like ScanNet and PyTorch3D have same pixel coordinate system definition:
        # - COLMAP: https://github.com/colmap/colmap/blob/d3a29e2/src/base/camera_models.h#L76
        # - PyTorch3D: https://github.com/facebookresearch/pytorch3d/blob/3b035f57f08295efc9af076ea60f62ad26d88b91/docs/notes/cameras.md
        # verts_screen[..., 0:2] = -1 * verts_screen[..., 0:2]

        # NOTE: in raw stream, 1st elem in coordinate refers to vertical axis and 2nd elem refers to horizontal axis
        # we need to change the order to align with PyTorch's NDC order
        verts_screen = verts_screen[..., [1, 0, 2]]

        verts_screen[..., 2] = verts_view[..., 2]
        meshes_screen = meshes_world.update_padded(new_verts_padded=verts_screen)

        return meshes_screen


def batch_render_img(
    *,
    stream_type,
    device,
    mesh,
    mesh_aux,
    face_to_mtl_idxs,
    mtl_imgs,
    view_matrices,
    proj_matrices,
    render_size_w,
    render_size_h,
    mp_flag=True,
    bin_flag=False,
    return_flag=False,
    dataset_f=None,
    scene_id=None,
    view_ids=None,
    raw_rgbs=None,
    bin_info_dict={},
):

    # print("\nStart rendering...")
    timing_start = time.time()

    mtl_size_h, mtl_size_w, _ = mtl_imgs[list(mtl_imgs.keys())[0]].shape

    # mesh = copy.deepcopy(mesh)
    if view_matrices.shape[0] != 1:
        mesh = mesh.extend(view_matrices.shape[0])

    cameras = MyCamera(
        device=device, view_matrices=view_matrices, proj_matrices=proj_matrices
    )

    raster_settings = RasterizationSettings(
        image_size=(
            render_size_h,
            render_size_w,
        ),  # pytorch3d supports non-square renderring from d07307a
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=True,
    )

    rasterizer = MyMeshRasterizer(
        stream_type=stream_type, cameras=cameras, raster_settings=raster_settings
    )

    fragments = rasterizer(mesh)

    mtl_uv_coords, mtl_pixel_coords = batch_uv_to_pixel_coords(
        mesh_aux.texture_atlas_uv_pos, fragments, mtl_size_h, mtl_size_w
    )

    n_faces = mesh_aux.texture_atlas_uv_pos.size(0)

    pix_to_face = fragments.pix_to_face.cpu().numpy()

    # timing_ckpt2 = time.time()
    # print("\ndone getting pixel coords, ", timing_ckpt2 - timing_ckpt1)

    pix_to_face_list = []
    for i in range(view_matrices.shape[0]):
        # NOTE: Get face index local to each mesh.
        # Originally, in PyTorch3D, face indices are summed up,
        # see https://github.com/facebookresearch/pytorch3d/blob/3b035f5/pytorch3d/renderer/mesh/textures.py#L498
        # Namely, if mesh1 has N1 faces, mesh2 has N2 faces, then the starting face index for mesh2 will be N1 instead of 0.
        cur_pixel_to_face = pix_to_face[i, ...] - i * n_faces
        cur_pixel_to_face[np.where(cur_pixel_to_face < 0)] = -1
        pix_to_face_list.append(cur_pixel_to_face)
    cat_pix_to_face = np.array(pix_to_face_list)

    if mp_flag:
        nproc = view_matrices.shape[0]
        new_imgs, new_masks = retrieve_pixel_val_mp(
            nproc=nproc,
            stream_type=stream_type,
            mtl_imgs=mtl_imgs,
            view_ids=view_ids,
            render_size_h=render_size_h,
            render_size_w=render_size_w,
            pix_to_face=cat_pix_to_face,
            mtl_pixel_coords=mtl_pixel_coords,
            face_to_mtl_idxs=face_to_mtl_idxs,
            material_names=mesh_aux.material_names,
        )
    else:
        pass

    print(f"\nRendering takes {time.time() - timing_start} seconds.")

    if dataset_f is not None:
        with h5py.File(dataset_f, "r+") as f:
            for i in view_ids:
                f[scene_id]["render_rgbs"].create_dataset(
                    str(i), data=new_imgs[i], compression="lzf"
                )
                f[scene_id]["render_masks"].create_dataset(
                    str(i), data=new_masks[i], compression="lzf"
                )

    if return_flag:
        return new_imgs, new_masks


def batch_render_img_torch(
    *,
    stream_type,
    device1,
    device2,
    mesh,
    texture_atlas_uv_pos=None,
    texture_atlas_tex_size=None,
    texture_atlas_start_idx=None,
    # mesh_aux,
    face_to_mtl_idxs=None,
    mtl_imgs=None,
    view_matrices=None,
    proj_matrices=None,
    render_size_w=None,
    render_size_h=None,
    bin_flag=False,
    # return_flag=False,
    n_faces=None,
    n_cams=None,
    scene_id=None,
    view_ids=None,
    raw_rgbs=None,
    raw_rgb_shapes=None,
    raw_depths=None,
    raw_depth_shapes=None,
    raw_cam_zs=None,
    raw_cam_z_shapes=None,
    view_poses=None,
    ndc=None,
    face_cam_pairs=None,
    all_face_infos=None,
    timing=False,
    extrude_abs_tol=0.1,
    backface_cull=True,
    bypass_mtl=False,
    rgb_bilinear=False,
    depth_bilinear=False,
    cat_pix_to_face=None,
    pix_to_bary_coords=None,
    pix_to_cams=None,
    faces_per_pixel=1,
    flag_pix_assign=True,
    flag_return_none=True,
    flag_save_src_pix_coords=False,
    flag_return_idxs=False,
    flag_return_idxs_float=False,
    flag_return_mtl_uvs=False,
    flag_post_process=True,
    flag_return_extra=False,
):
    """
    face_cam_pairs: [#faces, top_k]
    """

    if timing:
        # print("\nStart rendering...")
        timing_start = time.time()

    if not bypass_mtl:
        assert mtl_imgs.ndim == 4
        mtl_size_h = mtl_imgs.size(1)
        mtl_size_w = mtl_imgs.size(2)

    if cat_pix_to_face is None:

        # mesh = copy.deepcopy(mesh)
        if view_matrices.shape[0] != 1:
            mesh = mesh.extend(view_matrices.shape[0])

        cameras = MyCamera(
            device=device1, view_matrices=view_matrices, proj_matrices=proj_matrices
        )

        raster_settings = RasterizationSettings(
            # image_size=max((render_size_w, render_size_h)),
            image_size=(
                render_size_h,
                render_size_w,
            ),  # pytorch3d supports non-square renderring from d07307a
            blur_radius=0.0,
            faces_per_pixel=faces_per_pixel,
            cull_backfaces=backface_cull,
        )

        rasterizer = MyMeshRasterizer(
            stream_type=stream_type, cameras=cameras, raster_settings=raster_settings
        )

        # Check definition of variables from https://github.com/facebookresearch/pytorch3d/blob/d07307a/pytorch3d/renderer/mesh/rasterize_meshes.py#L80
        fragments = rasterizer(mesh)

        if timing:
            timing_ckpt1 = time.time()
            print("done pytorch3d rasterization, ", timing_ckpt1 - timing_start)

        if not bypass_mtl:
            if texture_atlas_tex_size is not None:
                (
                    mtl_uv_coords,
                    mtl_pixel_coords,
                ) = batch_uv_to_pixel_coords_adaptive_torch(
                    device1,
                    texture_atlas_uv_pos,
                    texture_atlas_tex_size,
                    texture_atlas_start_idx,
                    fragments,
                    mtl_size_h,
                    mtl_size_w,
                )
            else:
                mtl_uv_coords, mtl_pixel_coords = batch_uv_to_pixel_coords_torch(
                    device1,
                    # mesh_aux.texture_atlas_uv_pos.to(device2),
                    texture_atlas_uv_pos,
                    fragments,
                    mtl_size_h,
                    mtl_size_w,
                    return_idxs_float=flag_return_idxs_float,
                )
        else:
            mtl_pixel_coords = None

        if timing:
            timing_ckpt2 = time.time()
            print("done getting pixel coords, ", timing_ckpt2 - timing_ckpt1)

        pix_to_face = fragments.pix_to_face
        pix_to_face_list = []
        # n_faces = mesh_aux.texture_atlas_uv_pos.size(0)
        if n_faces is None:
            n_faces = texture_atlas_uv_pos.size(0)

        for i in range(view_matrices.shape[0]):
            # NOTE: Get face index local to each mesh.
            # Originally, in PyTorch3D, face indices are summed up,
            # see https://github.com/facebookresearch/pytorch3d/blob/3b035f5/pytorch3d/renderer/mesh/textures.py#L498
            # Namely, if mesh1 has N1 faces, mesh2 has N2 faces, then the starting face index for mesh2 will be N1 instead of 0.
            cur_pixel_to_face = pix_to_face[i, ...] - i * n_faces
            cur_pixel_to_face[torch.where(cur_pixel_to_face < 0)] = -1
            pix_to_face_list.append(cur_pixel_to_face)
        cat_pix_to_face = torch.stack(pix_to_face_list)

        pix_to_bary_coords = fragments.bary_coords
    else:
        if timing:
            timing_ckpt1 = time.time()

    if not flag_pix_assign:
        target_extra_dict = {
            "pix_to_face": cat_pix_to_face,
            "bary_coords": pix_to_bary_coords,
        }

        if flag_post_process:
            _, _, new_extras = post_pix_assign_process_torch(
                stream_type,
                None,
                None,
                target_extra_dict,
                render_size_w,
                render_size_h,
            )
        else:
            new_extras = target_extra_dict
        return None, None, new_extras, []
    else:

        new_imgs_list = []
        new_masks_list = []
        new_extra_dict = {}

        n_iters = 1
        if bin_flag:
            if face_cam_pairs is not None:
                n_iters = face_cam_pairs.shape[1]
            elif pix_to_cams is not None:
                n_iters = pix_to_cams.shape[-1]
            else:
                raise ValueError

        # for i in tqdm.tqdm(torch.arange(n_iters)):
        for i in torch.arange(n_iters):

            timing_ckpt2 = time.time()

            target_extra_dict = {}
            if flag_return_mtl_uvs:
                target_views, target_masks = retrieve_pixel_val_torch(
                    device=mtl_imgs.device,
                    stream_type=stream_type,
                    mtl_imgs=mtl_imgs,
                    pix_to_face=cat_pix_to_face.to(mtl_imgs.device),
                    mtl_pixel_coords=mtl_uv_coords.to(mtl_imgs.device),
                    face_to_mtl_idxs=face_to_mtl_idxs.to(mtl_imgs.device),
                    return_idxs=flag_return_idxs,
                    return_uvs=True,
                )
            else:
                target_views, target_masks = retrieve_pixel_val_torch(
                    device=mtl_imgs.device,
                    stream_type=stream_type,
                    mtl_imgs=mtl_imgs,
                    pix_to_face=cat_pix_to_face.to(mtl_imgs.device),
                    mtl_pixel_coords=mtl_pixel_coords.to(mtl_imgs.device),
                    face_to_mtl_idxs=face_to_mtl_idxs.to(mtl_imgs.device),
                    return_idxs=flag_return_idxs,
                )

            if timing and i == 0:
                timing_ckpt3 = time.time()
                print("done retriving pixel val, ", timing_ckpt3 - timing_ckpt2)

            if target_views is not None:

                target_extra_dict["pix_to_face"] = cat_pix_to_face
                target_extra_dict["bary_coords"] = pix_to_bary_coords

                if flag_post_process:
                    new_imgs, new_masks, new_extras = post_pix_assign_process_torch(
                        stream_type,
                        target_views,
                        target_masks,
                        target_extra_dict,
                        render_size_w,
                        render_size_h,
                    )
                else:
                    new_imgs = target_views
                    new_masks = target_masks
                    new_extras = target_extra_dict

                new_imgs_list.append(new_imgs)
                new_masks_list.append(new_masks)
                for k, v in new_extras.items():
                    if k not in new_extra_dict:
                        new_extra_dict[k] = [v]
                    else:
                        new_extra_dict[k].append(v)

            if timing and i == 0:
                timing_ckpt4 = time.time()
                print(f"Done post processing {timing_ckpt4 - timing_ckpt3} seconds.\n")

        if timing:
            timings = [
                timing_ckpt1 - timing_start,
                timing_ckpt2 - timing_ckpt1,
                timing_ckpt3 - timing_ckpt2,
                timing_ckpt4 - timing_ckpt3,
            ]
        else:
            timings = []

        if bin_flag:
            return (
                new_imgs_list,
                new_masks_list,
                new_extra_dict,
                timings,
            )
        else:
            if flag_return_extra:
                return (
                    torch.cat(new_imgs_list, dim=-1),
                    torch.cat(new_masks_list, dim=-1),
                    new_extra_dict,
                )
            else:
                return (
                    torch.cat(new_imgs_list, dim=-1),
                    torch.cat(new_masks_list, dim=-1),
                )


# ------------------------------------------------------------------------------------------------
# render colored vertex


def render_texture_vertex(
    verts,
    faces,
    vert_colors,
    device,
    view_matrices,
    proj_matrices,
    render_size_w,
    render_size_h,
    stream_type="apple",
):

    tex = TexturesVertex(verts_features=[torch.from_numpy(vert_colors).to(device)])
    mesh = Meshes(
        verts=[torch.from_numpy(verts).to(device)],
        faces=[torch.from_numpy(faces).to(device)],
        textures=tex,
    )

    # print(view_matrix.shape[0], len(mesh._verts_list), len(mesh._faces_list), mesh._N, mesh._V, mesh._F)

    if view_matrices.shape[0] != 1:
        mesh = mesh.extend(view_matrices.shape[0])

    cameras = MyCamera(
        device=device, view_matrices=view_matrices, proj_matrices=proj_matrices
    )

    raster_settings = RasterizationSettings(
        image_size=(
            render_size_h,
            render_size_w,
        ),  # pytorch3d supports non-square renderring from d07307a
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=True,
    )

    rasterizer = MyMeshRasterizer(
        stream_type=stream_type, cameras=cameras, raster_settings=raster_settings
    )

    fragments = rasterizer(mesh)

    texels = mesh.sample_textures(fragments)

    new_imgs = []
    for i in range(texels.size(0)):
        new_imgs.append(np.uint8(texels[i, ..., 0, :3].cpu().numpy() * 255))

    return new_imgs
