import os
import json
import glob
import tqdm
import struct
import argparse
import numpy as np
from PIL import Image

from advtex_init_align.utils.stream_utils import StreamReader

STREAM_FILENAME = "Recv.stream"


def gen_train_stream(
    raw_stream_f,
    stream_type,
    n_views,
    save_dir,
    sample_freq,
    sample_freq_for_train=False,
    use_existed_uvs=False,
    # not_from_full_stream=False,
    add_noise=False,
    noise_ratio=0.05,
):

    if sample_freq_for_train:
        train_view_idxs = np.arange(0, n_views, sample_freq).tolist()
    else:
        # NOTE: we sample sparse views for test
        test_view_idxs = np.arange(0, n_views, sample_freq).tolist()
        train_view_idxs = sorted(
            list(set(np.arange(n_views).tolist()) - set(test_view_idxs))
        )

    print(f"\n#train_views: {len(train_view_idxs)}, train_view_idxs: {train_view_idxs} \n")

    save_stream_f = os.path.join(save_dir, STREAM_FILENAME)

    stream_reader = StreamReader(stream_type, raw_stream_f)

    if add_noise:
        train_idx_to_raw_idx_map = stream_reader.read_write_stream_data_add_noise(
            save_stream_f, train_view_idxs, noise_ratio=noise_ratio,
        )
    else:
        train_idx_to_raw_idx_map = stream_reader.read_write_stream_data(
            save_stream_f, train_view_idxs
        )

    # Soft link all vertices and faces files
    raw_dir = os.path.dirname(raw_stream_f)
    all_vertex_fs = list(glob.glob(os.path.join(raw_dir, "Vertices.*")))
    all_face_fs = list(glob.glob(os.path.join(raw_dir, "Faces.*")))
    assert len(all_vertex_fs) == len(
        all_face_fs
    ), f"{len(all_vertex_fs)}, {len(all_face_fs)}"

    for i in tqdm.tqdm(range(len(all_face_fs))):
        src_vertex_f = os.path.join(raw_dir, f"Vertices.{i}")
        tar_vertex_f = os.path.join(save_dir, f"Vertices.{i}")
        os.system(f"ln -s {src_vertex_f} {tar_vertex_f}")

        src_face_f = os.path.join(raw_dir, f"Faces.{i}")
        tar_face_f = os.path.join(save_dir, f"Faces.{i}")
        os.system(f"ln -s {src_face_f} {tar_face_f}")

        if use_existed_uvs:
            src_uv_f = os.path.join(raw_dir, f"TexVertices.{i}")
            tar_uv_f = os.path.join(save_dir, f"TexVertices.{i}")
            os.system(f"ln -s {src_uv_f} {tar_uv_f}")

    with open(os.path.join(save_dir, "train_idx_to_raw_idx_map.json"), "w") as f:
        json.dump(train_idx_to_raw_idx_map, f)


def main(
    raw_stream_f,
    stream_type,
    save_dir,
    sample_freq_list,
    sample_freq_for_train=False,
    use_existed_uvs=False,
    add_noise=False,
    noise_ratio=0.05,
):

    print("\nraw stream file: ", raw_stream_f, "\n")

    stream_reader = StreamReader(stream_type, raw_stream_f)
    stream_reader.read_stream()
    n_views = len(stream_reader.rgbs)

    tmp_shapes = [_.shape for _ in stream_reader.rgbs]
    tmp_shapes = set(tmp_shapes)
    print("\ngt_rgbs image resolutions: ", tmp_shapes, "\n")
    print(f"#views: {len(stream_reader.rgbs)}")

    scene_id = os.path.basename(os.path.dirname(raw_stream_f))

    for sample_freq in tqdm.tqdm(sample_freq_list):
        if sample_freq_for_train:
            freq_save_dir = os.path.join(save_dir, f"train_1_{sample_freq}")
        else:
            freq_save_dir = os.path.join(save_dir, f"test_1_{sample_freq}")
        print("\nfreq_save_dir: ", freq_save_dir, "\n")
        os.makedirs(freq_save_dir, exist_ok=True)
        gen_train_stream(
            raw_stream_f,
            stream_type,
            n_views,
            freq_save_dir,
            sample_freq,
            sample_freq_for_train=sample_freq_for_train,
            use_existed_uvs=use_existed_uvs,
            add_noise=add_noise,
            noise_ratio=noise_ratio,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream_dir_list",
        nargs="+",
        type=str,
        required=True,
        help="file path for stream file.",
    )
    parser.add_argument(
        "--sample_freq_list",
        nargs="+",
        type=int,
        required=True,
        help="sample frequencies.",
    )
    parser.add_argument(
        "--sample_freq_for_train", type=int, default=0, choices=[0, 1],
    )
    parser.add_argument(
        "--use_existed_uvs", type=int, default=0, choices=[0, 1],
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="save_directory.",
    )
    parser.add_argument(
        "--stream_type", type=str, default="apple", choices=["apple", "scannet"],
    )
    parser.add_argument(
        "--add_noise", type=int, default=0, choices=[0, 1],
    )
    parser.add_argument(
        "--noise_ratio", type=float, default=0.05,
    )

    args = parser.parse_args()

    for stream_dir in args.stream_dir_list:
        stream_f = os.path.join(stream_dir, STREAM_FILENAME)
        main(
            stream_f,
            args.stream_type,
            args.save_dir,
            args.sample_freq_list,
            sample_freq_for_train=bool(args.sample_freq_for_train),
            use_existed_uvs=bool(args.use_existed_uvs),
            # not_from_full_stream=bool(args.not_from_full_stream),
            add_noise=bool(args.add_noise),
            noise_ratio=args.noise_ratio,
        )

