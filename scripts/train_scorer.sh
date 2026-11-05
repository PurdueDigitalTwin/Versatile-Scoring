#!/usr/bin/env bash
# Step 4: train the nine-head supervised scorer of one planner on one candidate set, one
# independent run per seed -> exp/<planner dir>/scorer_<planner>_<set>_seed<s>/.
#   plain    the planner's own proposals only (the baseline)
#   design   + the lateral and longitudinal rungs, each supervising the head of its own rule
#   random   + the volume-matched random trajectories, delivered to the same heads (the control)
# Every run logs navtest EPDMS each epoch for diagnostics only (--select navtest also writes the
# navtest-selected comp*_bestreal.pt); the reported checkpoint is comp*_bestval.pt, selected on the held-out
# validation logs. A seed whose train.log has the trainer's closing "done. best" line is skipped,
# so re-running resumes; failed seeds are listed at the end and the script exits 1.
# The GPU saturates at two concurrent runs. DD_EPOCHS overrides DiffusionDrive's 100 epochs.
#
# Usage: bash scripts/train_scorer.sh mf|dd plain|design|random [seeds...]   (default seeds 0 1 2)
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
use_navsim_v1

USAGE="usage: bash scripts/train_scorer.sh mf|dd plain|design|random [seeds...]"
planner=${1:?${USAGE}}; set_=${2:?${USAGE}}; shift 2
seeds=${@:-0 1 2}
case "${set_}" in plain|design|random) ;; *) echo "unknown set '${set_}'. ${USAGE}" >&2; exit 1 ;; esac
MF=exp/meanfuser_checkpoints
DD=exp/diffusiondrive_checkpoints
case "${planner}" in
    mf)
        if [ "${set_}" = plain ]; then
            CAND=${MF}/plain_dataset_navtrain/dataset.pkl; LB=${MF}/components_mf_navtrain.pkl; EXTRA=""
        else
            CAND=${MF}/versatile_dataset_navtrain_${set_}/dataset.pkl; LB=${MF}/versatile_dataset_navtrain_${set_}/components.pkl; EXTRA="--val_pool 8"
        fi
        ARGS=(--candidates "${CAND}" --ctx "${MF}/plain_dataset_navtrain/context_features.pkl" --labels "${LB}" --human
              --logs "${MF}/navtrain_token_to_log.pkl" ${EXTRA} --select navtest)
        TAG=${MF}/scorer_mf_${set_} ;;
    dd)
        ARGS=(--candidates "${DD}/versatile_dataset_dd_navtrain_${set_}/dataset.pkl"
              --ctx "${DD}/plain_dataset_dd_navtrain/context_features.pkl"
              --labels "${DD}/versatile_dataset_dd_navtrain_${set_}/components.pkl" --human
              --logs "${MF}/navtrain_token_to_log.pkl"
              --d_model 256 --val_pool 20 --epochs "${DD_EPOCHS:-100}" --select navtest
              --navtest_candidates "${DD}/versatile_dataset_dd_navtest_plain/dataset.pkl"
              --navtest_ctx "${DD}/plain_dataset_dd_navtest/context_features.pkl"
              --navtest_labels "${DD}/versatile_dataset_dd_navtest_plain/components.pkl")
        TAG=${DD}/scorer_dd_${set_} ;;
    *) echo "unknown planner '${planner}'. ${USAGE}" >&2; exit 1 ;;
esac
failed=""
for s in ${seeds}; do
    out=${TAG}_seed${s}
    if grep -q "done. best" "${out}/train.log" 2>/dev/null; then echo ">>> ${planner} ${set_} seed${s} already done, skipping"; continue; fi
    mkdir -p "${out}"
    echo ">>> ${planner} ${set_} seed${s} -> ${out}  $(date +%H:%M)"
    python navsim/planning/script/run_scorer_training.py "${ARGS[@]}" --seed "${s}" --out "${out}" 2>&1 | tee "${out}/train.log" \
        || { echo ">>> ${planner} ${set_} seed${s} FAILED (see ${out}/train.log)"; failed="${failed} ${s}"; }
done
if [ -n "${failed}" ]; then echo "########## ${planner} ${set_} SWEEP FAILED for seed(s):${failed} ##########"; exit 1; fi
echo "########## ${planner} ${set_} SWEEP DONE ##########"
