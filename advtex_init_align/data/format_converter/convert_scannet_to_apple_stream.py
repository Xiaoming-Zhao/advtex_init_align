import os
import cv2
import struct
import shutil
import copy
import glob
import argparse
import trimesh
import numpy as np
import open3d as o3d
from tqdm import tqdm
from PIL import Image


def resize_depth(depth, rgb):
    new_depth = np.array(
        Image.fromarray(depth, mode="F").resize(
            (rgb.shape[1], rgb.shape[0]), resample=Image.Resampling.NEAREST
        )
    )
    return new_depth


def convert_to_apple_stream(scene_id, data_dir, mesh_f, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    print("start reading mesh ...")

    mesh = o3d.io.read_triangle_mesh(mesh_f)
    vert = np.array(mesh.vertices)
    face_id = np.array(mesh.triangles)
    # tex_vs = np.array(mesh.triangle_uvs)
    print("\nmesh: ", vert.shape, face_id.shape, "\n")
    print("... done.")

    for i in tqdm(range(face_id.shape[0])):
        assert (
            (face_id[i, 0] != face_id[i, 1])
            and (face_id[i, 0] != face_id[i, 2])
            and (face_id[i, 1] != face_id[i, 2])
        ), f"{face_id[i, :]}"

    with open(os.path.join(out_dir, "Vertices.0"), "wb") as f:
        f.write(vert.astype(np.float32).tobytes())
    with open(os.path.join(out_dir, "Faces.0"), "wb") as f:
        f.write(face_id.astype(np.uint32).tobytes())

    print("start reading rgb-d ...")

    depth_image_paths = list(glob.glob(os.path.join(data_dir, "depth/*.png")))
    color_image_paths = list(glob.glob(os.path.join(data_dir, "color/*.jpg")))
    cam_pose_paths = list(glob.glob(os.path.join(data_dir, "pose/*.txt")))
    assert len(depth_image_paths) == len(
        color_image_paths
    ), f"{len(depth_image_paths)}, {len(color_image_paths)}"
    assert len(depth_image_paths) == len(
        cam_pose_paths
    ), f"{len(depth_image_paths)}, {len(cam_pose_paths)}"

    n_views = len(depth_image_paths)

    print("... done.")

    # NOTE: since we will resize depth to the same resolution as RGB,
    # we direclty use RGB's intrinsics.
    K = np.loadtxt(os.path.join(data_dir, "intrinsic/intrinsic_color.txt"))

    print("start writing stream file ...")
    cnt = 0
    newFile = open(os.path.join(out_dir, "Recv.stream"), "wb")

    # NOTE: we need to keep the order of image indices
    for i in tqdm(range(n_views)):

        # change from millimeter to meter
        raw_depth_map = (
            cv2.imread(
                os.path.join(data_dir, f"depth/{i:06d}.png"), cv2.IMREAD_ANYDEPTH
            ).astype(np.float32)
            / 1000.0
        )
        image = np.array(Image.open(os.path.join(data_dir, f"color/{i:06d}.jpg")))

        depth_map = resize_depth(raw_depth_map, image)

        # image[mask == 0] = 255

        # when writing to file, we transpose data to mimic Apple's Fortran order
        cnt = cnt + 1
        newFile.write(struct.pack("3I", *image.shape))
        newFile.write(image.transpose((2, 1, 0)).astype(np.uint8).tobytes())
        newFile.write(struct.pack("2I", *depth_map.shape))
        newFile.write(depth_map.transpose((1, 0)).astype(np.float32).tobytes())

        # NOTE: start processing camera poses
        # This is camera-to-world: https://github.com/ScanNet/ScanNet/tree/488e5ba/SensReader/python
        cam2world_mat = np.loadtxt(os.path.join(data_dir, f"pose/{i:06d}.txt"))
        viewMatrix = np.linalg.inv(cam2world_mat)

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
        img_h, img_w, _ = image.shape
        prjMatrix = np.eye(4)
        prjMatrix[0, 0] = 2 * K[0, 0] / img_w
        prjMatrix[1, 1] = 2 * K[1, 1] / img_h
        prjMatrix[0, 2] = 2 * K[0, 2] / img_w - 1
        prjMatrix[1, 2] = 2 * K[1, 2] / img_h - 1
        prjMatrix[2, 2] = 1
        prjMatrix[3, 3] = 0
        prjMatrix[3, 2] = 1

        # make 1st elem for height, 2nd elem for width
        prjMatrix = prjMatrix[(1, 0, 2, 3), :]

        # NOTE: we need to flip left-right to align with Apple format's convention
        prjMatrix[1, :] = -prjMatrix[1, :]

        # to align with Apple's Fortran order
        newFile.write(viewMatrix.transpose((1, 0)).astype(np.float32).tobytes())
        newFile.write(prjMatrix.transpose((1, 0)).astype(np.float32).tobytes())

    newFile.close()

    print("... done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--mesh_f", type=str)
    parser.add_argument("--scene_id", type=str)
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory for saving processed data",
    )
    args = parser.parse_args()

    # out_dir = os.path.join(args.input_dense_dir, "apple_format")

    print(f"\ndata_dir: {args.data_dir}\n")
    print(f"\nout_dir: {args.out_dir}\n")

    convert_to_apple_stream(args.scene_id, args.data_dir, args.mesh_f, args.out_dir)
