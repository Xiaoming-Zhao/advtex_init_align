import joblib
import h5py
import os
import sys
import glob
import tqdm
import cv2
import random
import traceback
import argparse
import subprocess
import numpy as np
import multiprocessing as mp
import scipy.io as sio
from collections import defaultdict
from PIL import Image, ImageOps


S3_REPO = "/data02/xz23/3d_texture_align/code/adv_texture/s3_sharpness/S3codes_Oct2011"
S3_CMD = '''cd {s3_repo}; matlab -nosplash -nodesktop -singleCompThread -r "s3_map('{img_f}', '{mat_save_dir}', 0);exit;"'''


def compute_metrics_single_scene_subproc(subproc_input):
    """
    base_h5_f: for getting shift_u, shift_v.
    """

    (worker_id, scene_id, view_ids, scene_dir, s3_dir, flag_gt_dir, flag_split_img, flag_test_img_only) = subproc_input

    for view_id in tqdm.tqdm(view_ids):

        if flag_gt_dir:
            if flag_test_img_only:
                img_f = os.path.join(scene_dir, f"raw_infos_for_test/{view_id:05d}_raw_color.png")
            else:
                img_f = os.path.join(scene_dir, f"raw_rgbs/{view_id:05d}_raw_color.png")
        else:
            tmp_str = os.path.join(scene_dir, f"debug_vis/{view_id:05d}*.png")
            tmp_list = list(glob.glob(tmp_str))
            assert len(tmp_list) == 1, f"{tmp_list}"
            img_f = tmp_list[0]

        if flag_gt_dir:
            # assume XXXXX_raw_color.png
            img_to_use = np.array(Image.open(img_f))
        else:
            cat_img = np.array(Image.open(img_f))

            print("\nflag_split_img: ", flag_split_img, "\n")

            if flag_split_img:
                h, cat_w, _ = cat_img.shape
                assert cat_w % 4 == 0, f"{cat_w}"
    
                w = cat_w // 4
                # the rendered one
                img_to_use = cat_img[:, w : 2 * w, :]
            else:
                img_to_use = cat_img
        
        print("\nimg_to_use: ", img_to_use.shape, "\n")

        tmp_view_dir = os.path.join(s3_dir, f"{view_id:05d}")
        os.makedirs(tmp_view_dir, exist_ok=True)

        try:
            s3_f = os.path.join(tmp_view_dir, "s3.mat")
            s3_dict = sio.loadmat(s3_f)
            s3_mat = s3_dict["s3"]
            assert s3_mat.shape[0] == img_to_use.shape[0]
            assert s3_mat.shape[1] == img_to_use.shape[1]

            print(f"\n{s3_f} already exists, ignore it.\n")
        except:
            # traceback.print_exc()
            # err = sys.exc_info()[0]
            # print(err)

            tmp_f = os.path.join(tmp_view_dir, "color.png")
            Image.fromarray(img_to_use).save(tmp_f)

            tmp_cmd = S3_CMD.format(
                s3_repo=S3_REPO, img_f=tmp_f, mat_save_dir=tmp_view_dir
            )
            print("\n", worker_id, tmp_cmd, "\n")

            # child = subprocess.Popen(
            #     tmp_cmd,
            #     shell=True,
            #     stdout=subprocess.PIPE,    # NOTE: we cannot use PIPE here as it may generate lots of output, see https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait
            #     stderr=subprocess.STDOUT,
            #     universal_newlines=True,
            # )

            # while True:
            #    output = child.stdout.readline()
            #    print(output.strip())
            #    return_code = child.poll()
            #    if return_code is not None:
            #        print("S3 RETURN CODE", return_code)
            #        # Process has finished, read rest of the output
            #        for output in child.stdout.readlines():
            #            print(output.strip())
            #        break

            try:
                subprocess.run(
                    tmp_cmd,
                    shell=True,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as e:
                print(f"\n{tmp_f} exited with exit status {e.returncode}: {e.stderr}\n")


def compute_metrics_single_scene_mp(
    nproc, scene_id, scene_dir, flag_gt_dir=False, flag_split_img=True, flag_test_img_only=False,
):

    if flag_gt_dir:
        if flag_test_img_only:
            all_img_fs = list(
                glob.glob(os.path.join(scene_dir, "raw_infos_for_test/*_raw_color.png"))
            )
        else:
            all_img_fs = list(
                glob.glob(os.path.join(scene_dir, "raw_rgbs/*_raw_color.png"))
            )
    else:
        all_img_fs = list(glob.glob(os.path.join(scene_dir, "debug_vis/*.png")))

    # all_view_ids = np.arange(len(all_img_fs)).tolist()
    all_view_ids = sorted([int(os.path.basename(_.rstrip("/")).split(".")[0].split("_")[0]) for _ in all_img_fs])
    for i in all_view_ids:
        if flag_gt_dir:
            if flag_test_img_only:
                tmp_f = os.path.join(scene_dir, f"raw_infos_for_test/{i:05d}_raw_color.png")
            else:
                tmp_f = os.path.join(scene_dir, f"raw_rgbs/{i:05d}_raw_color.png")
            assert os.path.exists(tmp_f), f"{tmp_f}"
        else:
            tmp_str = os.path.join(scene_dir, f"debug_vis/{i:05d}*.png")
            tmp_list = list(glob.glob(tmp_str))
            assert len(tmp_list) == 1, f"{tmp_str}, {tmp_list}"
    
    print("\n", all_view_ids, "\n")
    print(f"\nFind {len(all_view_ids)} images.\n")

    s3_dir = os.path.join(scene_dir, "s3_mats")

    # random.shuffle(all_view_ids)
    view_id_list = [[] for _ in range(nproc)]
    for i, view_id in enumerate(all_view_ids):
        view_id_list[i % nproc].append(view_id)

    # NOTE: np.matmul may freeze when using default "fork"
    # https://github.com/ModelOriented/DALEX/issues/412
    with mp.get_context("spawn").Pool(nproc) as pool:
        _ = pool.map(
            compute_metrics_single_scene_subproc,
            zip(
                range(nproc),
                [scene_id for _ in range(nproc)],
                view_id_list,
                [scene_dir for _ in range(nproc)],
                [s3_dir for _ in range(nproc)],
                [flag_gt_dir for _ in range(nproc)],
                [flag_split_img for _ in range(nproc)],
                [flag_test_img_only for _ in range(nproc)],
            ),
        )
        pool.close()
        pool.join()


def process_single_scene(scene_id, scene_dir, args):

    compute_metrics_single_scene_mp(
        args.nproc, scene_id, scene_dir,
        flag_gt_dir=bool(args.gt_dir), flag_split_img=bool(args.split_img), flag_test_img_only=bool(args.test_img_only),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nproc", type=int, required=True, default=10,
    )
    parser.add_argument(
        "--scene_dir", type=str, required=True, default=".",
    )
    parser.add_argument(
        "--gt_dir", type=int, default=0,
    )
    parser.add_argument(
        "--split_img", type=int, default=1,
    )
    parser.add_argument(
        "--test_img_only", type=int, default=0,
    )
    args = parser.parse_args()

    scene_id = os.path.basename(args.scene_dir)
    process_single_scene(scene_id, args.scene_dir, args)
