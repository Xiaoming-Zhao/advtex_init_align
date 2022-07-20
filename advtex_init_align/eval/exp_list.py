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


# ---------------------------------------------------------------------------------------------------------------------

EXPS_BASE_DIR = (pathlib.Path(__file__).parent.parent.parent / "experiments").resolve()


GT = os.path.join(
    EXPS_BASE_DIR,
    "{scene_id}",
    "{sample_freq}"
)

# for patchwise

ADV_OPTIM_MRF_UNARY_OFFSET_1 = os.path.join(
    EXPS_BASE_DIR,
    "{scene_id}/{sample_freq}",
    "optim",
    "unary_{mtl_h}_{mtl_w}_atlas_{atlas_size}/seed_{seed}-scratch_0-offset_1_n_patch_h_1_w_1/debug_vis"
)


METHOD_ID_ABBREVIATION = {
    "adv_optim_mrf_unary_offset_patch_1x1": "adv_u_off_p1x1",
}


ALL_EXPS = {
    "adv_optim_mrf_unary_offset_patch_1x1": ADV_OPTIM_MRF_UNARY_OFFSET_1,
}

# fmt: on