#!/usr/bin/env bash
# OFFICIAL one-stage navtest EPDMS for the paper's MeanFuser scorers.
#
# Why this exists: the navtest numbers in the paper come from the deploy-index
# sweep (k0..k7 + arm + gt, each an official run_pdm_score_one_stage.py run) read
# offline at the scorer's argmax. Eight of the nine EPDMS components are pure
# functions of the single deployed trajectory, so that lookup is exact for them.
# `two_frame_extended_comfort` is not: SceneAggregator pairs each frame with the
# previous frame of the same log (<=0.55 s) and compares ego states from the SAME
# run, i.e. the label assumes slot k was deployed at both frames. A re-ranker that
# switches slots between adjacent frames gets a different TFC (weight 2/16).
# Running the re-ranking agent through the official evaluator removes that
# approximation. PDMS is unaffected (no cross-frame term).
#
# Arms (CONFIGS) are the step-4 scorers, exp/meanfuser_checkpoints/scorer_mf_<arm>_seed<S>:
# plain, design, random. TAG picks the checkpoint comp*_<TAG>.pt; the default bestval
# is the validation-selected checkpoint the paper reports (bestreal is the navtest-selected
# diagnostic).
#
# ~40 min per run. Resumable: a config already reporting a final score is skipped.
#
#   SEEDS="0"         bash scripts/eval_official.sh   # ~2 h
#   SEEDS="0 1 2"     bash scripts/eval_official.sh   # ~6 h
#   CONFIGS="design" SEEDS="0" bash scripts/eval_official.sh
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# EPDMS needs the navsim_epdms package (the root navsim is PDMS / v1).
use_navsim_epdms

MF=${NAVSIM_EXP_ROOT}/meanfuser_checkpoints
split=navtest
traffic_agents=non_reactive
# the config default is cwd-relative and navsim_epdms/ has no tools/
mean_std_path=${REPO_ROOT}/tools/gaussian_mixed_noise/navtrain_8_mean_std.pkl

SEEDS=${SEEDS:-"0 1 2"}
CONFIGS=${CONFIGS:-"plain design random"}
TAG=${TAG:-bestval}

scorer_dir_for() {
  case "$1" in
    plain|design|random) echo "scorer_mf_$1" ;;
    *) echo "unknown config: $1 (expected plain, design or random)" >&2; return 1 ;;
  esac
}
for cfg in ${CONFIGS}; do scorer_dir_for "${cfg}" >/dev/null || exit 1; done

for cfg in ${CONFIGS}; do
  for s in ${SEEDS}; do
    name="navtest_mf_${cfg}_seed${s}"
    out="${NAVSIM_EXP_ROOT}/${name}"
    sdir="${MF}/$(scorer_dir_for "${cfg}")_seed${s}"

    if [ ! -f "${sdir}/comp8_${TAG}.pt" ]; then
      echo "[skip] ${name}: missing ${sdir}/comp8_${TAG}.pt"; continue
    fi
    # NB: the one-stage script prints "Final average score of valid results";
    # only the two-stage (navhard) script says "Final extended pdm score".
    if grep -qs "Final average score of valid results" "${out}/run_pdm_score.log"; then
      echo "[done] ${name}"; continue
    fi

    echo "=== ${name}  (scorer dir=${sdir}, tag=${TAG}) ==="
    mkdir -p "${out}"
    python "${EPDMS_ROOT}/navsim/planning/script/run_pdm_score_one_stage.py" \
        agent=meanfuser_scorer_agent \
        agent.checkpoint_path="${MF}/meanfuser_pdms_89.0.ckpt" \
        agent.config.navtrain_mean_std_path="${mean_std_path}" \
        agent.scorer_dir="${sdir}" \
        agent.scorer_tag="${TAG}" \
        agent.scorer_d_model=128 \
        traffic_agents=${traffic_agents} \
        experiment_name="${name}" output_dir="${out}" \
        metric_cache_path="${NAVSIM_CACHE_ROOT}/${split}_v2_metric_cache" \
        train_test_split=${split} 2>&1 | tee "${out}/run_pdm_score.log"
  done
done

echo
echo "=== official navtest EPDMS ==="
final_score() {
  grep -oE 'Final average score of valid results: [0-9]+\.[0-9]+' "$1" 2>/dev/null \
    | tail -1 | grep -oE '[0-9]+\.[0-9]+'
}
for cfg in ${CONFIGS}; do
  for s in ${SEEDS}; do
    f="${NAVSIM_EXP_ROOT}/navtest_mf_${cfg}_seed${s}/run_pdm_score.log"
    [ -f "$f" ] || continue
    printf '  %-24s %s\n' "${cfg} seed${s}" "$(final_score "$f")"
  done
done
