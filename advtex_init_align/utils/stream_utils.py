import os
import sys
import cv2
import copy
import struct
import traceback
import h5py
import time
import numpy as np
from scipy.spatial.transform import Rotation


def add_noises_to_view_mat(view_mat, noise_ratio=0.05, euler_seq="zxy"):
    
    rot_mat = view_mat[:3, :3]
    trans_vec = view_mat[:3, 3]
    
    rot_mat_scipy = Rotation.from_matrix(rot_mat)
    angles = rot_mat_scipy.as_euler(euler_seq, degrees=False)

    angles_with_noises = []
    for tmp in angles:
        tmp_range = noise_ratio * tmp
        tmp_noise = np.random.uniform(-tmp_range, tmp_range)
        angles_with_noises.append(tmp + tmp_noise)

    rot_mat_with_noises = Rotation.from_euler(euler_seq, angles_with_noises, degrees=False).as_matrix()

    trans_with_noises = []
    for tmp in trans_vec:
        tmp_range = noise_ratio * tmp
        tmp_noise = np.random.uniform(-tmp_range, tmp_range)
        trans_with_noises.append(tmp + tmp_noise) 
    
    view_mat_with_noises = np.eye(4)
    view_mat_with_noises[:3, :3] = rot_mat_with_noises
    view_mat_with_noises[:3, 3] = trans_with_noises

    return view_mat_with_noises


class StreamReader:

    _apple_depth_h: int = 256
    _apple_depth_w: int = 192
    _mat_dim: int = 4

    def __init__(self, stream_type, stream_f):
        self._stream_type = stream_type
        self._stream_f = stream_f

    @property
    def rgbs(self):
        return self._rgbs

    @property
    def depth_maps(self):
        return self._depth_maps

    @property
    def view_matrices(self):
        return self._view_matrices

    @property
    def proj_matrices(self):
        return self._proj_matrices

    @property
    def transform_matrices(self):
        return self._transform_matrices

    def __len__(self):
        return len(self.rgbs)

    def read_stream(self):
        if self._stream_type == "apple":
            self._read_apple_stream_data()
        elif self._stream_type == "scannet":
            self._read_scannet_stream_data()
        else:
            raise ValueError
    
    def read_write_stream_data(self, save_f, valid_idxs):
        if self._stream_type == "apple":
            new_idx_to_raw_idx_map = self._read_write_apple_stream_data(save_f, valid_idxs)
        elif self._stream_type == "scannet":
            new_idx_to_raw_idx_map = self._read_write_scannet_stream_data(save_f, valid_idxs)
        else:
            raise ValueError
        
        return new_idx_to_raw_idx_map

    def _read_apple_stream_data(self):

        self._rgbs = []
        self._depth_maps = []
        self._view_matrices = []
        self._proj_matrices = []
        self._transform_matrices = []

        self._num_cameras = 0

        tmp_start = time.time()

        with open(self._stream_f, "rb") as f:
            # NOTE: Important !!!
            # It seems like the stream data is stored in Fortan's order.
            # Namely, it contains the data with the order of (1st column, 2nd column, ...).
            # Therefore, we must first reshape the original data in Fortan's order then permute the dimensions.

            while True:

                try:
                    # uint32
                    Y_height, Y_width, Y_trunc_height = struct.unpack(
                        "I" * 3, f.read(4 * 3)
                    )
                    Y_size = Y_height * Y_width

                    # uint8
                    Y = struct.unpack("B" * Y_size, f.read(1 * Y_size))
                    Y = np.array(Y, dtype=np.uint8).reshape((Y_width, Y_height, 1))
                    # [Y_width, Y_trunc_height, 1]
                    Y = Y[:, :Y_trunc_height, :]

                    # uint32
                    CbCr_height, CbCr_width, CbCr_trunc_height = struct.unpack(
                        "I" * 3, f.read(4 * 3)
                    )
                    CbCr_height //= 2
                    CbCr_size = 2 * CbCr_height * CbCr_width

                    # uint8
                    CbCr = struct.unpack("B" * CbCr_size, f.read(1 * CbCr_size))
                    CbCr = np.array(CbCr, dtype=np.uint8).reshape(
                        (CbCr_width, CbCr_height, 2)
                    )
                    # [CbCr_width, CbCr_trunc_height, 2]
                    CbCr = CbCr[:, :CbCr_trunc_height, :]

                    # python's resize comparison: https://stackoverflow.com/questions/29958670/how-to-use-matlabs-imresize-in-python
                    # NOTE: cv2 accept target size in the order of [columns, rows]
                    # Since we read the data in Fortan's order, #columns is height
                    Cb = cv2.resize(CbCr[:, :, 0], (Y_trunc_height, Y_width))[
                        :, :, np.newaxis
                    ]
                    Cr = cv2.resize(CbCr[:, :, 1], (Y_trunc_height, Y_width))[
                        :, :, np.newaxis
                    ]

                    rgb = cv2.cvtColor(
                        np.concatenate([Y, Cr, Cb], axis=2), cv2.COLOR_YCrCb2RGB
                    )
                    # [Y_trunc_height, Y_width, 3]
                    rgb = np.transpose(rgb, (1, 0, 2))

                    # the last 4 is for float data type
                    depth_size = self._apple_depth_h * self._apple_depth_w * 4
                    depth_map = struct.unpack("B" * depth_size, f.read(1 * depth_size))
                    # TODO: view() is in-place. Not sure whether need a deepcopy?
                    depth_map = np.array(depth_map, dtype=np.uint8).view(np.float32)
                    # estimated depth in meters from camera
                    depth_map = np.transpose(
                        depth_map.reshape((self._apple_depth_w, self._apple_depth_h)),
                        (1, 0),
                    )

                    # the last 4 is for float data type
                    mat_size = self._mat_dim * self._mat_dim * 2 * 4
                    two_matrices = struct.unpack("B" * mat_size, f.read(1 * mat_size))
                    two_matrices = np.array(two_matrices, dtype=np.uint8).view(
                        np.float32
                    )

                    view_matrix = np.transpose(
                        two_matrices[:16].reshape((self._mat_dim, self._mat_dim)),
                        (1, 0),
                    )
                    proj_matrix = np.transpose(
                        two_matrices[16:].reshape((self._mat_dim, self._mat_dim)),
                        (1, 0),
                    )

                    transform_matrix = np.matmul(proj_matrix, view_matrix)

                    # TODO: Remove this when the raw data changes.
                    # Currently, the view/transform matrix in raw data treats x-axis as the row of image.
                    # We hack it to make x-axis for columns and y-axis for rows.
                    # transform_matrix = transform_matrix[[1, 0, 2, 3], :]

                    self._num_cameras += 1
                    if self._num_cameras % 10 == 0:
                        print(
                            f"[Stream Reader] Already load data from {self._num_cameras} cameras."
                        )

                    self._rgbs.append(rgb)
                    self._depth_maps.append(depth_map)
                    self._view_matrices.append(view_matrix)
                    self._proj_matrices.append(proj_matrix)
                    self._transform_matrices.append(transform_matrix)

                except:
                    # traceback.print_exc()
                    # err = sys.exc_info()[0]
                    # print(err)
                    break

        print(
            f"[Stream Reader] Load data from {self._num_cameras} cameras in {time.time() - tmp_start:.2f} s."
        )

        # [#cameras, 4, 4], float32
        self._view_matrices = np.array(self._view_matrices)
        self._proj_matrices = np.array(self._proj_matrices)
        self._transform_matrices = np.array(self._transform_matrices)
    
    def _read_write_apple_stream_data(self, save_f, valid_idxs):

        tmp_start = time.time()

        new_idx_to_raw_idx_map = {}
        
        new_idx = -1
        cnt = -1

        with open(save_f, "wb") as out_f:

            with open(self._stream_f, "rb") as f:
                # NOTE: Important !!!
                # It seems like the stream data is stored in Fortan's order.
                # Namely, it contains the data with the order of (1st column, 2nd column, ...).
                # Therefore, we must first reshape the original data in Fortan's order then permute the dimensions.
    
                while True:
                    
                    cnt += 1
                    if cnt in valid_idxs:
                        flag_write = True
                    else:
                        flag_write = False
    
                    try:
                        # uint32
                        Y_height, Y_width, Y_trunc_height = struct.unpack(
                            "I" * 3, f.read(4 * 3)
                        )
                        if flag_write:
                            out_f.write(struct.pack("I" * 3, Y_height, Y_width, Y_trunc_height))
                            
                        Y_size = Y_height * Y_width
    
                        # uint8
                        Y = struct.unpack("B" * Y_size, f.read(1 * Y_size))
                        if flag_write:
                            out_f.write(struct.pack("B" * Y_size, *Y))
                            
                        Y = np.array(Y, dtype=np.uint8).reshape((Y_width, Y_height, 1))
                        # [Y_width, Y_trunc_height, 1]
                        Y = Y[:, :Y_trunc_height, :]
    
                        # uint32
                        CbCr_height, CbCr_width, CbCr_trunc_height = struct.unpack(
                            "I" * 3, f.read(4 * 3)
                        )
                        if flag_write:
                            out_f.write(struct.pack("I" * 3, CbCr_height, CbCr_width, CbCr_trunc_height))
                            
                        CbCr_height //= 2
                        CbCr_size = 2 * CbCr_height * CbCr_width
    
                        # uint8
                        CbCr = struct.unpack("B" * CbCr_size, f.read(1 * CbCr_size))
                        if flag_write:
                            out_f.write(struct.pack("B" * CbCr_size, *CbCr))
                        
                        CbCr = np.array(CbCr, dtype=np.uint8).reshape(
                            (CbCr_width, CbCr_height, 2)
                        )
                        # [CbCr_width, CbCr_trunc_height, 2]
                        CbCr = CbCr[:, :CbCr_trunc_height, :]
    
                        # python's resize comparison: https://stackoverflow.com/questions/29958670/how-to-use-matlabs-imresize-in-python
                        # NOTE: cv2 accept target size in the order of [columns, rows]
                        # Since we read the data in Fortan's order, #columns is height
                        Cb = cv2.resize(CbCr[:, :, 0], (Y_trunc_height, Y_width))[
                            :, :, np.newaxis
                        ]
                        Cr = cv2.resize(CbCr[:, :, 1], (Y_trunc_height, Y_width))[
                            :, :, np.newaxis
                        ]
    
                        rgb = cv2.cvtColor(
                            np.concatenate([Y, Cr, Cb], axis=2), cv2.COLOR_YCrCb2RGB
                        )
                        # [Y_trunc_height, Y_width, 3]
                        rgb = np.transpose(rgb, (1, 0, 2))
    
                        # the last 4 is for float data type
                        depth_size = self._apple_depth_h * self._apple_depth_w * 4
                        depth_map = struct.unpack("B" * depth_size, f.read(1 * depth_size))
                        if flag_write:
                            out_f.write(struct.pack("B" * depth_size, *depth_map))
                            
                        # TODO: view() is in-place. Not sure whether need a deepcopy?
                        depth_map = np.array(depth_map, dtype=np.uint8).view(np.float32)
                        # estimated depth in meters from camera
                        depth_map = np.transpose(
                            depth_map.reshape((self._apple_depth_w, self._apple_depth_h)),
                            (1, 0),
                        )
    
                        # the last 4 is for float data type
                        mat_size = self._mat_dim * self._mat_dim * 2 * 4
                        two_matrices = struct.unpack("B" * mat_size, f.read(1 * mat_size))
                        if flag_write:
                            out_f.write(struct.pack("B" * mat_size, *two_matrices))
                            
                        two_matrices = np.array(two_matrices, dtype=np.uint8).view(
                            np.float32
                        )
    
                        view_matrix = np.transpose(
                            two_matrices[:16].reshape((self._mat_dim, self._mat_dim)),
                            (1, 0),
                        )
                        proj_matrix = np.transpose(
                            two_matrices[16:].reshape((self._mat_dim, self._mat_dim)),
                            (1, 0),
                        )
    
                        transform_matrix = np.matmul(proj_matrix, view_matrix)

                        if cnt % 10 == 0:
                            print(
                                f"[Stream Reader] Already load data from {cnt} cameras."
                            )
                        
                        if flag_write:
                            new_idx += 1
                            new_idx_to_raw_idx_map[new_idx] = cnt
    
                    except:
                        # traceback.print_exc()
                        # err = sys.exc_info()[0]
                        # print(err)
                        break

        print(
            f"[Stream Reader] Load data from {cnt} cameras in {time.time() - tmp_start:.2f} s."
        )

        return new_idx_to_raw_idx_map
    
    def _read_scannet_stream_data(self):

        self._rgbs = []
        self._depth_maps = []
        self._view_matrices = []
        self._proj_matrices = []
        self._transform_matrices = []

        self._num_cameras = 0

        tmp_start = time.time()

        with open(self._stream_f, "rb") as f:
            # NOTE: Important !!!
            # It seems like the stream data is stored in Fortan's order.
            # Namely, it contains the data with the order of (1st column, 2nd column, ...).
            # Therefore, we must first reshape the original data in Fortan's order then permute the dimensions.

            while True:

                try:
                    # uint32
                    rgb_height, rgb_width, rgb_n_channels = struct.unpack(
                        "I" * 3, f.read(4 * 3)
                    )
                    rgb_size = rgb_height * rgb_width * rgb_n_channels

                    # uint8
                    rgb = struct.unpack("B" * rgb_size, f.read(1 * rgb_size))
                    # Fortran order
                    rgb = np.array(rgb, dtype=np.uint8).reshape(
                        (rgb_n_channels, rgb_width, rgb_height)
                    )

                    # [rgb_height, rgb_width, 3]
                    rgb = np.transpose(rgb, (2, 1, 0))

                    # the last 4 is for float data type
                    depth_height, depth_width = struct.unpack("I" * 2, f.read(4 * 2))
                    depth_size = depth_height * depth_width * 4
                    depth_map = struct.unpack("B" * depth_size, f.read(1 * depth_size))
                    # TODO: view() is in-place. Not sure whether need a deepcopy?
                    depth_map = np.array(depth_map, dtype=np.uint8).view(np.float32)
                    # estimated depth in meters from camera
                    depth_map = np.transpose(
                        depth_map.reshape((depth_width, depth_height)), (1, 0)
                    )

                    # the last 4 is for float data type
                    mat_size = self._mat_dim * self._mat_dim * 2 * 4
                    two_matrices = struct.unpack("B" * mat_size, f.read(1 * mat_size))
                    two_matrices = np.array(two_matrices, dtype=np.uint8).view(
                        np.float32
                    )

                    view_matrix = np.transpose(
                        two_matrices[:16].reshape((self._mat_dim, self._mat_dim)),
                        (1, 0),
                    )
                    proj_matrix = np.transpose(
                        two_matrices[16:].reshape((self._mat_dim, self._mat_dim)),
                        (1, 0),
                    )

                    transform_matrix = np.matmul(proj_matrix, view_matrix)

                    # TODO: Remove this when the raw data changes.
                    # Currently, the view/transform matrix in raw data treats x-axis as the row of image.
                    # We hack it to make x-axis for columns and y-axis for rows.
                    # transform_matrix = transform_matrix[[1, 0, 2, 3], :]

                    self._num_cameras += 1
                    if self._num_cameras % 10 == 0:
                        print(
                            f"[daemon] Already load data from {self._num_cameras} cameras."
                        )

                    self._rgbs.append(rgb)
                    self._depth_maps.append(depth_map)
                    self._view_matrices.append(view_matrix)
                    self._proj_matrices.append(proj_matrix)
                    self._transform_matrices.append(transform_matrix)
                except:
                    # traceback.print_exc()
                    # err = sys.exc_info()[0]
                    # print(err)
                    break

        print(
            f"[daemon] Load data from {self._num_cameras} cameras in {time.time() - tmp_start:.2f} s."
        )

        # [#cameras, 4, 4], float32
        self._view_matrices = np.array(self._view_matrices)
        self._proj_matrices = np.array(self._proj_matrices)
        self._transform_matrices = np.array(self._transform_matrices)
    
    def _read_write_scannet_stream_data(self, save_f, valid_idxs):

        tmp_start = time.time()

        new_idx_to_raw_idx_map = {}
        
        new_idx = -1
        cnt = -1

        with open(save_f, "wb") as out_f:

            with open(self._stream_f, "rb") as f:
                # NOTE: Important !!!
                # It seems like the stream data is stored in Fortan's order.
                # Namely, it contains the data with the order of (1st column, 2nd column, ...).
                # Therefore, we must first reshape the original data in Fortan's order then permute the dimensions.
    
                while True:

                    cnt += 1
                    if cnt in valid_idxs:
                        flag_write = True
                    else:
                        flag_write = False
    
                    try:
                        # uint32
                        rgb_height, rgb_width, rgb_n_channels = struct.unpack(
                            "I" * 3, f.read(4 * 3)
                        )
                        if flag_write:
                            out_f.write(struct.pack("I" * 3, rgb_height, rgb_width, rgb_n_channels))

                        rgb_size = rgb_height * rgb_width * rgb_n_channels
    
                        # uint8
                        rgb = struct.unpack("B" * rgb_size, f.read(1 * rgb_size))
                        if flag_write:
                            out_f.write(struct.pack("B" * rgb_size, *rgb))

                        # Fortran order
                        rgb = np.array(rgb, dtype=np.uint8).reshape(
                            (rgb_n_channels, rgb_width, rgb_height)
                        )
    
                        # [rgb_height, rgb_width, 3]
                        rgb = np.transpose(rgb, (2, 1, 0))
    
                        # the last 4 is for float data type
                        depth_height, depth_width = struct.unpack("I" * 2, f.read(4 * 2))
                        if flag_write:
                            out_f.write(struct.pack("I" * 2, depth_height, depth_width))

                        depth_size = depth_height * depth_width * 4

                        depth_map = struct.unpack("B" * depth_size, f.read(1 * depth_size))
                        if flag_write:
                            out_f.write(struct.pack("B" * depth_size, *depth_map))

                        # TODO: view() is in-place. Not sure whether need a deepcopy?
                        depth_map = np.array(depth_map, dtype=np.uint8).view(np.float32)
                        # estimated depth in meters from camera
                        depth_map = np.transpose(
                            depth_map.reshape((depth_width, depth_height)), (1, 0)
                        )
    
                        # the last 4 is for float data type
                        mat_size = self._mat_dim * self._mat_dim * 2 * 4
                        two_matrices = struct.unpack("B" * mat_size, f.read(1 * mat_size))
                        if flag_write:
                            out_f.write(struct.pack("B" * mat_size, *two_matrices))

                        two_matrices = np.array(two_matrices, dtype=np.uint8).view(
                            np.float32
                        )
    
                        view_matrix = np.transpose(
                            two_matrices[:16].reshape((self._mat_dim, self._mat_dim)),
                            (1, 0),
                        )
                        proj_matrix = np.transpose(
                            two_matrices[16:].reshape((self._mat_dim, self._mat_dim)),
                            (1, 0),
                        )
    
                        transform_matrix = np.matmul(proj_matrix, view_matrix)
    
                        # TODO: Remove this when the raw data changes.
                        # Currently, the view/transform matrix in raw data treats x-axis as the row of image.
                        # We hack it to make x-axis for columns and y-axis for rows.
                        # transform_matrix = transform_matrix[[1, 0, 2, 3], :]
    
                        if cnt % 10 == 0:
                            print(
                                f"[daemon] Already load data from {cnt} cameras."
                            )
                        
                        if flag_write:
                            new_idx += 1
                            new_idx_to_raw_idx_map[new_idx] = cnt
                    except:
                        # traceback.print_exc()
                        # err = sys.exc_info()[0]
                        # print(err)
                        break
    
            print(
                f"[daemon] Load data from {cnt} cameras in {time.time() - tmp_start:.2f} s."
            )
            
        return new_idx_to_raw_idx_map
    

def stream_reader_wrapper(
    new_flag, stream_type, dataset_f, stream_f, chunk_bytes, mp_queue
):

    scene_id = os.path.basename(os.path.dirname(stream_f))

    # read stream infos
    stream_reader = StreamReader(stream_type, stream_f)
    stream_reader.read_stream()

    n_views = len(stream_reader)

    print(f"[daemon] Save {scene_id}'s data to disk ...")
    tmp_start = time.time()

    if not new_flag:
        mode = "r+"
    else:
        mode = "w"

    # The volume of stream's content is too large to be communicated between process,
    # we need to save them to disk.
    with h5py.File(
        dataset_f, mode, libver="latest", rdcc_nbytes=chunk_bytes, rdcc_nslots=1e7
    ) as f:
        group = f.create_group(scene_id)

        # each RGB image's shape
        shapes = np.array([_.shape for _ in stream_reader.rgbs])
        group.create_dataset("rgb_shapes", data=shapes, compression="lzf")

        rgb_group = group.create_group("stream_rgbs")
        for i in range(n_views):
            assert stream_reader.rgbs[i].dtype == np.uint8
            rgb_group.create_dataset(
                str(i), data=stream_reader.rgbs[i], compression="lzf"
            )

        group.create_dataset(
            "view_matrices", data=stream_reader.view_matrices, compression="lzf"
        )
        group.create_dataset(
            "proj_matrices", data=stream_reader.proj_matrices, compression="lzf"
        )
        group.create_dataset(
            "transform_matrices",
            data=stream_reader.transform_matrices,
            compression="lzf",
        )

    print(f"[daemon] ... done saving to disk ({time.time() - tmp_start:.2f}s)\n")

    mp_queue.put("Finished")
