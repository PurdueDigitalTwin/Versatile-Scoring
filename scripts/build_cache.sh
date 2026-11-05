#!/usr/bin/env bash
# Step 1: the caches of one split. Each kind is one Hydra job on CPU (ray workers); extra
# arguments are passed through as overrides, e.g. train_test_split.scene_filter.max_scenes=10
# for a quick check. Expect 103,288 navtrain and 12,146 navtest scenes.
#
#   v1_metric    PDMS (NAVSIM-v1) metric cache     -> cache/<split>_v1_metric_cache   root navsim
#   v2_metric    EPDMS (NAVSIM-v2) metric cache    -> cache/<split>_v2_metric_cache   navsim_epdms: its
#                metric_cache_processor computes the traffic-light / lane data the v1 one does not
#   mf_features  MeanFuser camera feature cache    -> cache/trainval_v1_cache         both splits share it
#   dd_features  DiffusionDrive camera + LiDAR BEV -> cache/trainval_v1_cache_dd
#   all          the four above, in that order
#
# Usage: bash scripts/build_cache.sh v1_metric|v2_metric|mf_features|dd_features|all navtrain|navtest [hydra overrides...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

USAGE="usage: bash scripts/build_cache.sh v1_metric|v2_metric|mf_features|dd_features|all navtrain|navtest [hydra overrides...]"
kind=${1:?${USAGE}}; split=${2:?${USAGE}}; shift 2
case "${split}" in navtrain|navtest) ;; *) echo "unknown split '${split}'. ${USAGE}" >&2; exit 1 ;; esac

if [ "${kind}" = all ]; then
    for k in v1_metric v2_metric mf_features dd_features; do bash "${BASH_SOURCE[0]}" "${k}" "${split}" "$@"; done
    exit 0
fi
echo ">>> ${kind} cache (${split})  extra overrides: $*"
case "${kind}" in
    v1_metric)
        use_navsim_v1
        python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" \
            train_test_split="${split}" \
            cache.cache_path="${NAVSIM_CACHE_ROOT}/${split}_v1_metric_cache" "$@" ;;
    v2_metric)
        use_navsim_epdms
        python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" \
            train_test_split="${split}" \
            metric_cache_path="${NAVSIM_CACHE_ROOT}/${split}_v2_metric_cache" "$@" ;;
    mf_features)
        use_navsim_v1
        python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching.py" \
            train_test_split="${split}" agent=meanfuser_agent experiment_name=dataset_caching \
            cache_path="${NAVSIM_CACHE_ROOT}/trainval_v1_cache" \
            output_dir="${NAVSIM_EXP_ROOT}/dataset_caching" "$@" ;;
    dd_features)
        use_navsim_v1
        python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching.py" \
            train_test_split="${split}" agent=diffusiondrive_agent experiment_name=dataset_caching_dd \
            +agent.config.plan_anchor_path="${NAVSIM_EXP_ROOT}/diffusiondrive_checkpoints/plan_anchor_kmeans20.npy" \
            cache_path="${NAVSIM_CACHE_ROOT}/trainval_v1_cache_dd" \
            output_dir="${NAVSIM_EXP_ROOT}/dataset_caching_dd" "$@" ;;
    *) echo "unknown kind '${kind}'. ${USAGE}" >&2; exit 1 ;;
esac
