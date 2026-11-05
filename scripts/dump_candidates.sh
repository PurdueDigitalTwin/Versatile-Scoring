#!/usr/bin/env bash
# Step 2: the candidate pool of one frozen planner on one split, in three stages (1 and 3 on GPU).
#   1 predictions  run the planner once over the split and store its trajectories
#                  mf: 8 proposals + the ARM trajectory   -> plain_dataset_<split>/meanfuser_pdms_89.0.pkl
#                  dd: 20 trajectory modes + the cls pick -> plain_dataset_dd_<split>/diffusiondrive_navsim_88p1_PDMS.pkl
#   2 dataset      PDMS-score every stored trajectory (weights EP:TTC:C = 5:5:2) -> <dir>/dataset.pkl
#   3 context      the frozen encoder's scene features the scorer conditions on
#                  mf: 64 BEV tokens + 1 status token, [N, 65, 128] -> plain_dataset_<split>/context_features.pkl
#                  dd: the decoder's memory, [N, 65, 256]           -> plain_dataset_dd_<split>/context_features.pkl
# A stage whose output exists is skipped, so re-running resumes.
#
# Usage: bash scripts/dump_candidates.sh mf|dd navtrain|navtest
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
use_navsim_v1

USAGE="usage: bash scripts/dump_candidates.sh mf|dd navtrain|navtest"
planner=${1:?${USAGE}}; split=${2:?${USAGE}}
case "${split}" in navtrain|navtest) ;; *) echo "unknown split '${split}'. ${USAGE}" >&2; exit 1 ;; esac

SCRIPT=${NAVSIM_DEVKIT_ROOT}/navsim/planning/script
V1_CACHE=${NAVSIM_CACHE_ROOT}/${split}_v1_metric_cache
case "${planner}" in
    mf)
        dir=${NAVSIM_EXP_ROOT}/meanfuser_checkpoints/plain_dataset_${split}
        ckpt=${NAVSIM_EXP_ROOT}/meanfuser_checkpoints/meanfuser_pdms_89.0.ckpt
        predictions=${dir}/meanfuser_pdms_89.0.pkl
        AGENT=(agent=meanfuser_agent)
        MODEL=(agent.config.num_proposals=8 agent.config.noise_type=multi_gaussian)
        feature_cache=${NAVSIM_CACHE_ROOT}/trainval_v1_cache
        ctx_dir=${dir}
        ctx_out=${dir}/context_features.pkl ;;
    dd)
        dir=${NAVSIM_EXP_ROOT}/diffusiondrive_checkpoints/plain_dataset_dd_${split}
        ckpt=${NAVSIM_EXP_ROOT}/diffusiondrive_checkpoints/diffusiondrive_navsim_88p1_PDMS.ckpt
        predictions=${dir}/diffusiondrive_navsim_88p1_PDMS.pkl
        # the DD agent's default anchor path points at the upstream authors' machine; the checkpoint
        # load overwrites the anchor values, so only the (20, 8, 2) shape matters
        AGENT=(agent=diffusiondrive_agent
               +agent.config.plan_anchor_path="${NAVSIM_EXP_ROOT}/diffusiondrive_checkpoints/plan_anchor_kmeans20.npy")
        MODEL=()
        feature_cache=${NAVSIM_CACHE_ROOT}/trainval_v1_cache_dd
        ctx_dir=${dir}
        ctx_out=${dir}/context_features.pkl ;;
    *) echo "unknown planner '${planner}'. ${USAGE}" >&2; exit 1 ;;
esac
mkdir -p "${dir}" "${ctx_dir}"
echo "planner / split  : ${planner} / ${split}"
echo "checkpoint       : ${ckpt}"
echo "candidate dir    : ${dir}"

# the dumpers stamp their outputs; rename the newest match to the stable name every tool reads
normalise() {  # normalise <glob> <target>
    local built; built=$(ls -t $1 2>/dev/null | head -1) || true   # no match: fall through to the error below
    if [ -z "${built}" ]; then echo "ERROR: no $1 was written" >&2; exit 1; fi
    mv "${built}" "$2" && echo "wrote            : $2"
}

if [ -f "${predictions}" ]; then echo "[skip] stage 1: ${predictions} exists"; else
    echo ">>> stage 1/3: predictions"
    python "${SCRIPT}/run_pdm_score_gpu_v1.py" "${AGENT[@]}" ${MODEL[@]+"${MODEL[@]}"} \
        dataloader.params.batch_size=32 trainer.params.precision=32 \
        agent.checkpoint_path="${ckpt}" output_dir="${dir}/" \
        cache_path="${feature_cache}" metric_cache_path="${V1_CACHE}" train_test_split="${split}"
    [ -f "${predictions}" ] || { echo "ERROR: stage 1 did not write ${predictions}" >&2; exit 1; }
fi
if [ -f "${dir}/dataset.pkl" ]; then echo "[skip] stage 2: ${dir}/dataset.pkl exists"; else
    echo ">>> stage 2/3: PDMS dataset"
    python "${SCRIPT}/build_plain_dataset.py" "${AGENT[@]}" \
        +predictions_pkl="${predictions}" output_dir="${dir}/" \
        metric_cache_path="${V1_CACHE}" train_test_split="${split}"
    normalise "${dir}/dataset_*.pkl" "${dir}/dataset.pkl"
fi
if [ -f "${ctx_out}" ]; then echo "[skip] stage 3: ${ctx_out} exists"; else
    echo ">>> stage 3/3: scene context"
    case "${planner}" in
        mf) python "${SCRIPT}/dump_context_features_mf.py" "${AGENT[@]}" "${MODEL[@]}" dataloader.params.batch_size=32 \
                agent.checkpoint_path="${ckpt}" output_dir="${ctx_dir}/" \
                cache_path="${feature_cache}" metric_cache_path="${V1_CACHE}" train_test_split="${split}"
            normalise "${ctx_dir}/context_features_*.pkl" "${ctx_out}" ;;
        dd) python "${SCRIPT}/dump_context_features_dd.py" "${AGENT[@]}" dataloader.params.batch_size=32 \
                agent.checkpoint_path="${ckpt}" output_dir="${ctx_dir}/" \
                cache_path="${feature_cache}" metric_cache_path="${V1_CACHE}" train_test_split="${split}"
            normalise "${ctx_dir}/context_features_*.pkl" "${ctx_out}" ;;
    esac
fi
