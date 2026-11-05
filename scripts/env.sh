# Shared environment for every pipeline script. Source it, do not run it:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/env.sh"          # from scripts/*.sh
#
# Every path derives from one variable, NAVSIM_WORKSPACE. It defaults to the
# directory that contains this repo; export it to keep dataset/ and cache/ elsewhere.
#
#     $NAVSIM_WORKSPACE/dataset/{maps,navsim_logs,sensor_blobs}
#     $NAVSIM_WORKSPACE/cache/
#     <this repo, any name>/exp/       checkpoints, candidate pools, labels, scorers
#
# Then call exactly one of the two package selectors. Each one cds into the package
# root, puts it first on PYTHONPATH and sets NAVSIM_DEVKIT_ROOT:
#
#     use_navsim_v1      root navsim/        PDMS (v1), planners, feature dumps, scorer training
#     use_navsim_epdms   navsim_epdms/       official one-stage EPDMS (v2) caching and labelling

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPDMS_ROOT="${REPO_ROOT}/navsim_epdms"

NAVSIM_WORKSPACE="${NAVSIM_WORKSPACE:-$(dirname "${REPO_ROOT}")}"
if [ ! -d "${NAVSIM_WORKSPACE}" ]; then
    echo "[env.sh] NAVSIM_WORKSPACE=${NAVSIM_WORKSPACE} is not a directory" >&2
    exit 1
fi
export NAVSIM_WORKSPACE="$(cd "${NAVSIM_WORKSPACE}" && pwd)"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="${NAVSIM_WORKSPACE}/dataset"
export NUPLAN_MAPS_ROOT="${OPENSCENE_DATA_ROOT}/maps"
export NAVSIM_CACHE_ROOT="${NAVSIM_WORKSPACE}/cache"
export NAVSIM_EXP_ROOT="${REPO_ROOT}/exp"
export HYDRA_FULL_ERROR=1

use_navsim_v1() {
    export NAVSIM_DEVKIT_ROOT="${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    cd "${REPO_ROOT}" || exit 1
}

use_navsim_epdms() {
    export NAVSIM_DEVKIT_ROOT="${EPDMS_ROOT}"
    export PYTHONPATH="${EPDMS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    cd "${EPDMS_ROOT}" || exit 1   # sys.path[0] must not resolve `navsim` to the root package
}

if ! python -c "import hydra" >/dev/null 2>&1; then
    echo "[env.sh] 'python' cannot import hydra. Activate the environment from the README Setup section first." >&2
    exit 1
fi
