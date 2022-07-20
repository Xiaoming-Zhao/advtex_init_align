import os
import tqdm
import shutil
import argparse
import numpy as np
from PIL import Image

from advtex_init_align.utils.io_utils import load_mtl_imgs_vectorize


OBJ_FILENAME = "TexAlign.obj"
MTL_FILENAME = "TexAlign.mtl"


def fuse_mrf_and_avg_mtl(mrf_mtl_f, avg_mtl_f):

    mrf_dir = os.path.dirname(mrf_mtl_f)
    avg_dir = os.path.dirname(avg_mtl_f)

    # [max_h, max_w, 3]
    mrf_mtl_imgs, mrf_mtl_names = load_mtl_imgs_vectorize(mrf_mtl_f, return_fname=True)
    avg_mtl_imgs, avg_mtl_names = load_mtl_imgs_vectorize(avg_mtl_f, return_fname=True)

    assert np.all(
        mrf_mtl_imgs.shape == avg_mtl_imgs.shape
    ), f"{mrf_mtl_imgs.shape}, {avg_mtl_imgs.shape}"

    new_mrf_mtl_imgs = mrf_mtl_imgs.copy()

    f_idxs = []

    for i in tqdm.tqdm(range(mrf_mtl_imgs.shape[0])):

        mrf_img_f = mrf_mtl_names[i]
        avg_img_f = avg_mtl_names[i]
        assert os.path.basename(mrf_img_f) == os.path.basename(
            avg_img_f
        ), f"{mrf_img_f}, {avg_img_f}"
        fname = os.path.basename(mrf_img_f)

        # mtl_X_X.png
        tmp_f_idx = "_".join(fname.split(".")[0].split("_")[1:])
        f_idxs.append(tmp_f_idx)

        mrf_mask_f = os.path.join(mrf_dir, f"mtl_mask_{tmp_f_idx}.png")
        mrf_mask = np.array(Image.open(mrf_mask_f))

        avg_mask_f = os.path.join(avg_dir, f"mtl_mask_{tmp_f_idx}.png")
        avg_mask = np.array(Image.open(avg_mask_f))

        # We need to find pixels that are
        # - covered by average mtl
        # - not covered by MRF mtl
        rows_to_fill, cols_to_fill, _ = np.where((mrf_mask == 0) & (avg_mask > 0))
        # mask_to_fill = np.zeros(cur_mrf_img.shape, dtype=np.uint8)
        # mask_to_fill[rows_to_fill, cols_to_fill, :] = np.array((255, 255, 255), dtype=np.uint8)

        new_mrf_mtl_imgs[i, rows_to_fill, cols_to_fill, :] = avg_mtl_imgs[
            i, rows_to_fill, cols_to_fill, :
        ]

    new_mrf_dir = os.path.join(mrf_dir, "fused")
    os.makedirs(new_mrf_dir, exist_ok=True)
    for i in range(mrf_mtl_imgs.shape[0]):
        Image.fromarray(new_mrf_mtl_imgs[i, ...]).save(
            os.path.join(new_mrf_dir, f"mtl_{f_idxs[i]}.png")
        )

    shutil.copyfile(mrf_mtl_f, os.path.join(new_mrf_dir, MTL_FILENAME))
    shutil.copyfile(
        os.path.join(mrf_dir, OBJ_FILENAME), os.path.join(new_mrf_dir, OBJ_FILENAME)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mrf_mtl_f",
        # nargs="+",
        type=str,
        required=True,
        help="file path for MRF mtl file.",
    )
    parser.add_argument(
        "--avg_mtl_f",
        # nargs="+",
        type=str,
        required=True,
        help="file path for average texture mtl file.",
    )
    args = parser.parse_args()

    fuse_mrf_and_avg_mtl(args.mrf_mtl_f, args.avg_mtl_f)


if __name__ == "__main__":
    main()
