#!/bin/bash
{

REPO_DIR="$1"
SCENE_ID="$2"
RUN_TRAIN="$3"

# conda activate ${CONDA_ENV_NAME}
export PYTHONPATH=${REPO_DIR}:$PYTHONPATH

SEED=123

N_PATCHES_H=1
N_PATCHES_W=1

# MRF related
UNARY=1
FACE_AREA_PENALTY=1e-3
DEPTH_PENALTY=-10
PERCEPT_PENALTY=-1
DUMMY_PENALTY=-15

DATA_DIR=${REPO_DIR}/dataset
EXP_DIR=${REPO_DIR}/experiments/scannet

MRF_BIN=${REPO_DIR}/advtex_init_align/tex_init/tex_init

MTL_ATLAS_SIZE=15

N_ITERS_MP=0

N_SUBDIV=0

N_MESH_SPLITS=30

MTL_RES=2048

if [ "${UNARY}" == "1" ]; then
    MRF_NAME="unary"  
    MRF_DIR_NAME="output_obj_argmax_${MTL_RES}_${MTL_RES}_500"
else
    MRF_NAME="pairwise"
    MRF_DIR_NAME="output_obj_mp_only_adj_${MTL_RES}_${MTL_RES}_500"
fi

printf '\nmrf output folder %s\n' ${MRF_DIR_NAME}

SAMPLE_FREQ_FOR_TRAIN=0

if [ "${SAMPLE_FREQ_FOR_TRAIN}" == "1" ]; then
    DIR_PREFIX="train"
else
    DIR_PREFIX="test"
fi

ulimit -n 65000;
MKL_THREADING_LAYER=GNU;

SAMPLE_FREQS=(10)

printf '\nall freqs %s\n' "${SAMPLE_FREQS[@]}"

for sample_freq in "${SAMPLE_FREQS[@]}";
do
    total_start="$(date -u +%s)"

    if [ "${RUN_TRAIN}" == "run_train" ]; then
    
        printf "\nprocess sampling frequency: %s\n" ${sample_freq}

        # read data from .sens file
        printf "\nstart reading raw data from .sens file ...\n"

        python ${REPO_DIR}/advtex_init_align/data/scannet/reader.py \
        --filename ${DATA_DIR}/scannet_raw/${SCENE_ID}/${SCENE_ID}.sens \
        --output_path ${DATA_DIR}/scannet/${SCENE_ID} \
        --export_depth_images \
        --export_color_images \
        --export_poses \
        --export_intrinsics

        printf "\n... done reading raw data.\n"

        # convert raw data to Apple's stream file
        printf "\nstart converting to Apple's stream file ...\n"

        python ${REPO_DIR}/advtex_init_align/data/format_converter/convert_scannet_to_apple_stream.py \
        --data_dir ${DATA_DIR}/scannet/${SCENE_ID} \
        --mesh_f ${DATA_DIR}/scannet/${SCENE_ID}/${SCENE_ID}_vh_clean_2.ply \
        --out_dir ${EXP_DIR}/${SCENE_ID}/full

        printf "\n... done converting to stream file.\n"

        # subsample training view indexs
        printf "\nstart generating new stream file ...\n"

        python ${REPO_DIR}/advtex_init_align/data/gen_train_stream.py \
        --stream_dir_list ${EXP_DIR}/${SCENE_ID}/full \
        --save_dir ${EXP_DIR}/${SCENE_ID} \
        --sample_freq_list ${sample_freq} \
        --sample_freq_for_train ${SAMPLE_FREQ_FOR_TRAIN} \
        --stream_type scannet

        printf "\n... done generating new stream file\n"

        # split mesh into sub-meshes
        export OMP_NUM_THREADS=10 && \
        ${MRF_BIN} \
        --data_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq} \
        --debug 0 \
        --debug_mesh_shape 0 \
        --align_arg_max ${UNARY} \
        --n_iter_mp ${N_ITERS_MP} \
        --n_workers 10 --mtl_width ${MTL_RES} --mtl_height ${MTL_RES} \
        --n_extrude_pixels 0 \
        --iter_subdiv ${N_SUBDIV} --stream_type 2 \
        --preprocess_split_mesh ${N_MESH_SPLITS}

        ln -s ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/Recv.stream ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/splitted_mesh_${N_MESH_SPLITS}/Recv.stream

        # run MRF
        printf "\nstart running MRF ...\n"

        export OMP_NUM_THREADS=10 && \
        ${MRF_BIN} \
        --data_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/splitted_mesh_${N_MESH_SPLITS} \
        --debug 1 \
        --debug_mesh_shape 0 \
        --align_arg_max ${UNARY} \
        --n_iter_mp ${N_ITERS_MP} \
        --n_workers 10 --mtl_width ${MTL_RES} --mtl_height ${MTL_RES} \
        --n_extrude_pixels 0 \
        --iter_subdiv ${N_SUBDIV} --stream_type 2 \
        --unary_potential_dummy ${DUMMY_PENALTY} \
        --remesh 0 \
        --bin_pack_type 3 --n_areas_per_plate_bin_pack 500 \
        --debug_mesh_shape 0 --conformal_map 1 --compact_mtl 1 --top_one_mtl 1 \
        --penalty_face_area ${FACE_AREA_PENALTY} \
        --penalty_face_cam_dist ${DEPTH_PENALTY} \
        --penalty_face_v_perception_consist ${PERCEPT_PENALTY} \
        --pair_potential_mp 1 \
        --pair_potential_off_diag_scale_depth 0 \
        --pair_potential_off_diag_scale_percept 0

        printf "\n... done running MRF\n"

        # ScanNet has thousands of high-res images. We save them on disk for later fast IO.
        python ${REPO_DIR}/advtex_init_align/data/prepare_for_scannet.py \
        --stream_f_list ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/Recv.stream \
        --save_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/prepare \
        --stream_type scannet

        # generate L2 averaged texture
        printf "\nstart generating L2 averaged texture ...\n"

        python ${REPO_DIR}/advtex_init_align/data/gen_avg_mtl.py \
        --stream_f_list ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/Recv.stream \
        --obj_f_list ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/splitted_mesh_${N_MESH_SPLITS}/${MRF_DIR_NAME}/TexAlign.obj \
        --save_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/avg_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE} \
        --atlas_size ${MTL_ATLAS_SIZE} \
        --debug_vis 0 \
        --fuse 1 \
        --directly_fuse 0 \
        --stream_type scannet \
        --scannet_data_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/prepare/data

        printf "\n... done generating L2 averaged texture\n"

        # start format converting
        printf "\nstart converting results into AdvOptim format ...\n"

        python ${REPO_DIR}/advtex_init_align/data/format_converter/convert_mrf_result_to_adv_tex.py \
        --stream_f_list ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/Recv.stream \
        --obj_f_list ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/splitted_mesh_${N_MESH_SPLITS}/${MRF_DIR_NAME}/fused/TexAlign.obj \
        --save_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/splitted_mesh_${N_MESH_SPLITS}/optim/${MRF_NAME}_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE}/fused \
        --atlas_size ${MTL_ATLAS_SIZE} \
        --already_single_mtl 0 \
        --for_train 1 \
        --stream_type scannet \
        --scannet_data_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/prepare/data

        printf "\n... done converting our MRF into AdvOptim format.\n"

        # Run AdvTex
        python ${REPO_DIR}/advtex_init_align/tex_smooth/optim_patch_torch.py \
        --seed ${SEED} \
        --use_mislaign_offset 1 \
        --from_scratch 0 \
        --n_patches_h 1 \
        --n_patches_w 1 \
        --input_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/splitted_mesh_${N_MESH_SPLITS}/optim/${MRF_NAME}_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE}/fused 

        cp -r ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/splitted_mesh_${N_MESH_SPLITS}/optim/${MRF_NAME}_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE}/seed_${SEED}-scratch_0-offset_1_n_patch_h_${N_PATCHES_H}_w_${N_PATCHES_W}/shape/ ${EXP_DIR}/${SCENE_ID}/optimized_texture_${DIR_PREFIX}_1_${sample_freq}
    
    else

        # save GT images to disk
        # We set prepare_for_test_only to 1 since it will be really costly to render all (train + test) images for ScanNet
        python ${REPO_DIR}/advtex_init_align/data/format_converter/convert_mrf_result_to_adv_tex.py \
        --stream_f_list ${EXP_DIR}/${SCENE_ID}/full/Recv.stream \
        --obj_f_list ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/splitted_mesh_${N_MESH_SPLITS}/${MRF_DIR_NAME}/fused/TexAlign.obj \
        --save_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/raw_infos_for_test \
        --atlas_size ${MTL_ATLAS_SIZE} \
        --already_single_mtl 1 \
        --for_train 0 \
        --stream_type scannet \
        --pure_save_gt_info 1 \
        --pure_save_gt_info_rgb_only 0 \
        --prepare_for_test_only 1 \
        --train_idx_to_raw_idx_map_f ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/train_idx_to_raw_idx_map.json

        # Render from optimized texture
        python ${REPO_DIR}/advtex_init_align/data/format_converter/convert_mrf_result_to_adv_tex.py \
        --stream_f_list ${EXP_DIR}/${SCENE_ID}/full/Recv.stream \
        --obj_f_list ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/splitted_mesh_${N_MESH_SPLITS}/optim/${MRF_NAME}_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE}/seed_${SEED}-scratch_0-offset_1_n_patch_h_${N_PATCHES_H}_w_${N_PATCHES_W}/shape/TexAlign.obj \
        --save_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/splitted_mesh_${N_MESH_SPLITS}/optim/${MRF_NAME}_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE}/seed_${SEED}-scratch_0-offset_1_n_patch_h_${N_PATCHES_H}_w_${N_PATCHES_W} \
        --atlas_size ${MTL_ATLAS_SIZE} \
        --already_single_mtl 1 \
        --for_train 0 \
        --stream_type scannet \
        --scannet_data_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/raw_infos_for_test

        # Compute metrics
        python ${REPO_DIR}/advtex_init_align/eval/compute_metrics.py \
        --nproc 10 \
        --dataset scannet \
        --scene_id ${SCENE_ID} \
        --save_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/eval_results \
        --sample_freq_list ${DIR_PREFIX}_1_${sample_freq} \
        --method_id_list scannet_adv_optim_mrf_unary_offset_patch_1x1 \
        --compute_s3 0
    
    fi


    total_end="$(date -u +%s)"
    elapsed="$(($total_end-$total_start))"
    printf "\nTotal time elapsed %f\n" $elapsed
done


exit;
}