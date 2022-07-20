#!/bin/bash
{

REPO_DIR="$1"
SCENE_ID="$2"
RUN_TRAIN="$3"

DATA_DIR=${REPO_DIR}/dataset
EXP_DIR=${REPO_DIR}/experiments

MRF_BIN=${REPO_DIR}/advtex_init_align/tex_init/tex_init

eval "$(conda shell.bash hook)"
conda activate advtex_init_align
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

MTL_ATLAS_SIZE=60

if [ "${SCENE_ID}" == "scene_06" ]; then
    N_ITERS_MP=1
elif [ "${SCENE_ID}" == "scene_05" ]; then
    N_ITERS_MP=1
else
    N_ITERS_MP=10
fi

if [ "${SCENE_ID}" == "scene_07" ]; then
    N_SUBDIV=0 # due to memory issue
else
    N_SUBDIV=1
fi

# We determine the resolution of texture based on the image's resolution.
# Please see Appendix of the paper for details.
if [ "${SCENE_ID}" == "scene_03" ]; then
    MTL_RES=512
elif [ "${SCENE_ID}" == "scene_06" ]; then
    MTL_RES=2048
elif [ "${SCENE_ID}" == "scene_01" ]; then
    MTL_RES=2048
elif [ "${SCENE_ID}" == "scene_10" ]; then
    MTL_RES=2048
else
    MTL_RES=1024
fi

if [ "${UNARY}" == "1" ]; then
    MRF_NAME="unary"  
    MRF_DIR_NAME="output_obj_argmax_${MTL_RES}_${MTL_RES}_500"
else
    MRF_NAME="pairwise"
    MRF_DIR_NAME="output_obj_mp_only_adj_${MTL_RES}_${MTL_RES}_500"
fi

printf '\nmrf output folder %s\n' ${MRF_DIR_NAME}

# NOTE: this file only samples for test.
SAMPLE_FREQ_FOR_TRAIN=0

if [ "${SAMPLE_FREQ_FOR_TRAIN}" == "1" ]; then
    DIR_PREFIX="train"
else
    DIR_PREFIX="test"
fi

# We reserve 10% data for evaluation
SAMPLE_FREQS=(10)

printf '\nall freqs %s\n' "${SAMPLE_FREQS[@]}"

for sample_freq in "${SAMPLE_FREQS[@]}";
do
    total_start="$(date -u +%s)"
    
    if [ "${RUN_TRAIN}" == "run_train" ]; then

        printf "\nRun training\n" 

        printf "\nprocess sampling frequency: %s\n" ${sample_freq}

        # subsample training view indexs
        printf "\nstart generating new stream file ...\n"

        python ${REPO_DIR}/advtex_init_align/data/gen_train_stream.py \
        --stream_dir_list ${DATA_DIR}/raw/${SCENE_ID} \
        --save_dir ${EXP_DIR}/${SCENE_ID} \
        --sample_freq_list ${sample_freq} \
        --sample_freq_for_train 0

        printf "\n... done generating new stream file\n"

        # run MRF
        printf "\nstart running MRF ...\n"

        export OMP_NUM_THREADS=10 && \
        ${MRF_BIN} \
        --data_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq} \
        --debug 0 \
        --debug_mesh_shape 0 \
        --align_arg_max ${UNARY} \
        --n_iter_mp ${N_ITERS_MP} \
        --n_workers 10 --mtl_width ${MTL_RES} --mtl_height ${MTL_RES} \
        --n_extrude_pixels 0 \
        --iter_subdiv ${N_SUBDIV} --stream_type 1 \
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

        # generate L2 averaged texture
        printf "\nstart generating L2 averaged texture ...\n"

        python ${REPO_DIR}/advtex_init_align/data/gen_avg_mtl.py \
        --stream_f_list ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/Recv.stream \
        --obj_f_list ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/${MRF_DIR_NAME}/TexAlign.obj \
        --save_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/avg_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE} \
        --atlas_size ${MTL_ATLAS_SIZE} \
        --debug_vis 1 \
        --fuse 1 \
        --directly_fuse 0

        printf "\n... done generating L2 averaged texture\n"

        # start format converting
        printf "\nstart converting MRF results into AdvOptim format ...\n"

        # For MRF
        python ${REPO_DIR}/advtex_init_align/data/format_converter/convert_mrf_result_to_adv_tex.py \
        --stream_f_list ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/Recv.stream \
        --obj_f_list ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/${MRF_DIR_NAME}/fused/TexAlign.obj \
        --save_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/optim/${MRF_NAME}_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE}/fused \
        --atlas_size ${MTL_ATLAS_SIZE} \
        --already_single_mtl 0 \
        --for_train 1

        # Run AdvTex
        python ${REPO_DIR}/advtex_init_align/tex_smooth/optim_patch_torch.py \
        --use_mislaign_offset 1 \
        --from_scratch 0 \
        --n_patches_h 1 \
        --n_patches_w 1 \
        --input_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/optim/${MRF_NAME}_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE}/fused 

        cp -r ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/optim/${MRF_NAME}_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE}/seed_${SEED}-scratch_0-offset_1_n_patch_h_${N_PATCHES_H}_w_${N_PATCHES_W}/shape/ ${EXP_DIR}/${SCENE_ID}/optimized_texture_${DIR_PREFIX}_1_${sample_freq}

    else

        # save GT images to disk
        # Set prepare_for_test_only to 0 if you want to save all GT images (train + test)
        python ${REPO_DIR}/advtex_init_align/data/format_converter/convert_mrf_result_to_adv_tex.py \
        --stream_f_list ${DATA_DIR}/raw/${SCENE_ID}/Recv.stream \
        --obj_f_list ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/${MRF_DIR_NAME}/fused/TexAlign.obj \
        --save_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/raw_rgbs \
        --atlas_size ${MTL_ATLAS_SIZE} \
        --already_single_mtl 1 \
        --for_train 0 \
        --pure_save_gt_info 1 \
        --prepare_for_test_only 0 \
        --train_idx_to_raw_idx_map_f ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/train_idx_to_raw_idx_map.json

        # Render from optimized texture
        python ${REPO_DIR}/advtex_init_align/data/format_converter/convert_mrf_result_to_adv_tex.py \
        --stream_f_list ${DATA_DIR}/raw/${SCENE_ID}/Recv.stream \
        --obj_f_list ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/optim/${MRF_NAME}_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE}/seed_${SEED}-scratch_0-offset_1_n_patch_h_${N_PATCHES_H}_w_${N_PATCHES_W}/shape/TexAlign.obj \
        --save_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/optim/${MRF_NAME}_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE}/seed_${SEED}-scratch_0-offset_1_n_patch_h_${N_PATCHES_H}_w_${N_PATCHES_W} \
        --atlas_size ${MTL_ATLAS_SIZE} \
        --already_single_mtl 1 \
        --for_train 0

        # Compute metrics
        python ${REPO_DIR}/advtex_init_align/eval/compute_metrics.py \
        --nproc 10 \
        --scene_id ${SCENE_ID} \
        --save_dir ${EXP_DIR}/${SCENE_ID}/${DIR_PREFIX}_1_${sample_freq}/eval_results \
        --sample_freq_list ${DIR_PREFIX}_1_${sample_freq} \
        --method_id_list adv_optim_mrf_unary_offset_patch_1x1 \
        --compute_s3 0

    fi

    total_end="$(date -u +%s)"
    elapsed="$(($total_end-$total_start))"
    printf "\nTotal time elapsed %f seconds.\n" $elapsed
done


exit;
}