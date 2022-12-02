import os
import pathlib
from collections import OrderedDict


# fmt: off
SEED = 123


SCENES = [
    "scene_01",
    "scene_02",
    "scene_03",
    "scene_04",
    "scene_05",
    "scene_06",
    "scene_07",
    "scene_08",
    "scene_09",
    "scene_10",
    "scene_11",
]

MTL_ATLAS_SIZE = 60

MTL_RES_DICT = {
    "scene_01": 2048,
    "scene_02": 1024,
    "scene_03": 512,
    "scene_04": 1024,
    "scene_05": 1024,
    "scene_06": 2048,
    "scene_07": 1024,
    "scene_08": 1024,
    "scene_09": 1024,
    "scene_10": 2048,
    "scene_11": 1024,
}

SCANNET_MTL_RES = 2048

SCANNET_MTL_ATLAS_SIZE = 15

SCANNET_N_PARTS = 30

SCANNET_SCENES = [
    "scene0000_00",
    "scene0001_00",
    "scene0002_00",
    "scene0003_00",
    "scene0004_00",
    "scene0005_00",
    "scene0006_00",
    "scene0007_00",
    "scene0008_00",
    "scene0009_00",
    "scene0010_00",
    "scene0011_00",
    "scene0012_00",
    "scene0013_00",
    "scene0014_00",
    "scene0015_00",
    "scene0016_00",
    "scene0017_00",
    "scene0018_00",
    "scene0019_00",
    "scene0020_00",
]

EXPS_BASE_DIR = (pathlib.Path(__file__).parent.parent.parent / "experiments").resolve()

# ---------------------------------------------------------------------------------------------------------------------
# UofI Texture Scenes

GT = os.path.join(
    EXPS_BASE_DIR,
    "{dataset}",
    "{scene_id}",
    "{sample_freq}"
)

# for patchwise

ADV_OPTIM_MRF_UNARY_OFFSET_1 = os.path.join(
    EXPS_BASE_DIR,
    "{dataset}",
    "{scene_id}/{sample_freq}",
    "optim",
    "unary_{mtl_h}_{mtl_w}_atlas_{atlas_size}/seed_{seed}-scratch_0-offset_1_n_patch_h_1_w_1/debug_vis"
)


# ---------------------------------------------------------------------------------------------------------------------
# ScanNet

SCANNET_GT = os.path.join(
    EXPS_BASE_DIR,
    "{dataset}",
    "{scene_id}",
    "{sample_freq}/raw_infos_for_test"
)

SCANNET_ADV_OPTIM_MRF_UNARY_OFFSET_1 = os.path.join(
    EXPS_BASE_DIR,
    "{dataset}",
    "{scene_id}/{sample_freq}/splitted_mesh_{n_splitted_meshes}",
    "optim",
    "unary_{mtl_h}_{mtl_w}_atlas_{atlas_size}/seed_{seed}-scratch_0-offset_1_n_patch_h_1_w_1/debug_vis"
)

# ---------------------------------------------------------------------------------------------------------------------

METHOD_ID_ABBREVIATION = {
    "adv_optim_mrf_unary_offset_patch_1x1": "adv_u_off_p1x1",
    "scannet_adv_optim_mrf_unary_offset_patch_1x1": "scannet_adv_u_off_p1x1",
}


ALL_EXPS = {
    "adv_optim_mrf_unary_offset_patch_1x1": ADV_OPTIM_MRF_UNARY_OFFSET_1,
    "scannet_adv_optim_mrf_unary_offset_patch_1x1": SCANNET_ADV_OPTIM_MRF_UNARY_OFFSET_1,
}

# fmt: on