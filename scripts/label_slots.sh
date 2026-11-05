#!/usr/bin/env bash
# Steps 2 and 3: simulator labels. Every stored trajectory slot k of a candidate pool is deployed
# once over the whole split under the official one-stage EPDMS scorer (navsim_epdms,
# deploy_index_agent) -> one CSV of the nine EPDMS terms per slot. A slot whose CSV exists is
# skipped, so re-running resumes. ~28 min per slot on navtrain with 24 CPU cores, ~3 min on navtest.
#
#   pool          candidates                              slots         CSVs                         labels pickle
#   mf            plain_dataset_<split>/dataset.pkl       0..7 arm gt   epdms_proposals_<split>/     components_mf_<split>.pkl
#   dd            plain_dataset_dd_<split>/dataset.pkl    0..19 arm     epdms_proposals_<split>/     components_dd_<split>.pkl
#   lateral       lateral_samples_<split>/samples.pkl       8..11         epdms_lateral_<split>/
#   longitudinal  longitudinal_samples_<split>/samples.pkl  8..11         epdms_longitudinal_<split>/
#   random        random_samples_<split>/samples.pkl        8..12         epdms_random_<split>/
#
# For mf and dd, once every slot has its CSV, tools/dump_components.py stitches them into the
# labels pickle {token: [N (+arm, +human), 9]}: arm (the planner's own pick) is the eval-only
# baseline, gt (the logged human trajectory) the extra training target. The arm slot doubles as
# a base-planner check: navtest EPDMS ~89.61 for MeanFuser, ~88.22 for DiffusionDrive.
# Slots 0..7 of the sample banks are the MeanFuser proposals, already labelled by `mf`; the
# trainset builder reads the sample CSVs directly.
#
# Usage: bash scripts/label_slots.sh mf|dd|lateral|longitudinal|random navtrain|navtest [k ...]   (default: every slot)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
use_navsim_epdms   # the root navsim's PDMScorerConfig lacks the EPDMS terms

USAGE="usage: bash scripts/label_slots.sh mf|dd|lateral|longitudinal|random navtrain|navtest [k ...]"
pool=${1:?${USAGE}}; split=${2:?${USAGE}}; shift 2
case "${split}" in navtrain|navtest) ;; *) echo "unknown split '${split}'. ${USAGE}" >&2; exit 1 ;; esac
MF=${NAVSIM_EXP_ROOT}/meanfuser_checkpoints
DD=${NAVSIM_EXP_ROOT}/diffusiondrive_checkpoints
stitch=()
case "${pool}" in
    mf)           RL=${MF}/plain_dataset_${split}/dataset.pkl;          OUT_ROOT=${MF}/epdms_proposals_${split};    SLOTS="0 1 2 3 4 5 6 7 arm gt"
                  stitch=(--n 8 --include_arm --include_human --out "${MF}/components_mf_${split}.pkl") ;;
    dd)           RL=${DD}/plain_dataset_dd_${split}/dataset.pkl;       OUT_ROOT=${DD}/epdms_proposals_${split};    SLOTS="$(seq -s ' ' 0 19) arm"
                  stitch=(--n 20 --include_arm --out "${DD}/components_dd_${split}.pkl") ;;
    lateral)      RL=${MF}/lateral_samples_${split}/samples.pkl;        OUT_ROOT=${MF}/epdms_lateral_${split};      SLOTS="8 9 10 11" ;;
    longitudinal) RL=${MF}/longitudinal_samples_${split}/samples.pkl;   OUT_ROOT=${MF}/epdms_longitudinal_${split}; SLOTS="8 9 10 11" ;;
    random)       RL=${MF}/random_samples_${split}/samples.pkl;         OUT_ROOT=${MF}/epdms_random_${split};       SLOTS="8 9 10 11 12" ;;
    *) echo "unknown pool '${pool}'. ${USAGE}" >&2; exit 1 ;;
esac
slots=${*:-${SLOTS}}
METRIC_CACHE=${NAVSIM_CACHE_ROOT}/${split}_v2_metric_cache
if ! ls "${METRIC_CACHE}"/metadata/*_metadata_node_*.csv >/dev/null 2>&1; then
    echo "ERROR: no EPDMS metric cache at ${METRIC_CACHE}; run 'bash scripts/build_cache.sh v2_metric ${split}' first" >&2
    exit 1
fi
[ -f "${RL}" ] || { echo "ERROR: candidate pool not found: ${RL}" >&2; exit 1; }

for k in ${slots}; do
    out=${OUT_ROOT}/k${k}_non_reactive/
    if ls "${out}"*.csv >/dev/null 2>&1; then echo "[skip] ${pool} ${split} k=${k}: CSV exists"; continue; fi
    echo ">>> ${pool} ${split}: deploy slot k=${k} -> ${out}"
    python "${EPDMS_ROOT}/navsim/planning/script/run_pdm_score_one_stage.py" \
        agent=deploy_index_agent traffic_agents=non_reactive \
        agent.candidate_dataset_path="${RL}" agent.deploy_index="${k}" \
        output_dir="${out}" metric_cache_path="${METRIC_CACHE}" train_test_split="${split}"
done

if [ ${#stitch[@]} -gt 0 ]; then
    for k in ${SLOTS}; do
        ls "${OUT_ROOT}/k${k}_non_reactive/"*.csv >/dev/null 2>&1 \
            || { echo "slot k=${k} has no CSV yet; run the remaining slots before the labels can be stitched"; exit 0; }
    done
    echo ">>> stitching ${stitch[-1]}"
    python "${REPO_ROOT}/tools/dump_components.py" --deploy_dir "${OUT_ROOT}" --dataset "${RL}" "${stitch[@]}"
fi
