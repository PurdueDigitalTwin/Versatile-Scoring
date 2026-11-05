# Designing Versatile Samples for Learned Trajectory Scoring

Paper: [arXiv 2609.01799](https://arxiv.org/abs/2609.01799) (WACV 2027)

A plug-and-play trajectory scorer trained on **designed** candidates rather than a planner's
own proposals. Nine small transformer heads predict the nine EPDMS components separately and
compose them at inference; the planner, its backbone and its candidate generator are frozen
throughout. Everything in the paper is reproduced from this directory.

The problem the design addresses: a planner's own proposals violate the two multiplicative
gates in only 8.4% (MeanFuser, N=8) to 16.6% (DiffusionDrive, N=20) of candidate rows, so a
scorer trained on them sees almost no examples of the boundary it has to learn. We add rungs
that straddle that boundary by construction: perturbed copies of the logged human trajectory,
displaced laterally toward the drivable-area boundary and longitudinally toward the lead
vehicle, at graded margins on both sides of the rule.

## Layout

```
versatile_training/            <- the entrance; run everything from here
├── navsim/                    NAVSIM package: planners, proposals, PDMS, scorer training
├── navsim_epdms/              second NAVSIM release, package only — the official
│                              one-stage EPDMS scorer used to label candidates
├── tools/                     THIS WORK: sample generators, label stitching, trainset builder
├── scripts/                   one script per pipeline stage; env.sh holds the shared paths
├── download/                  dataset download scripts (from NAVSIM)
└── exp/                       created by the pipeline: checkpoints, datasets and labels
```

`navsim_epdms/` holds a **second NAVSIM release**, not a variant of the first. The two differ
in dozens of shared files — including `pdm_scorer.py`, `pdm_score.py` and `dataclasses.py` —
and `run_pdm_score_one_stage.py` plus `scene_aggregator.py` exist only in it. Collapsing
them into one package would change either the PDMS labels or the EPDMS ones, so the package
tree is kept apart while its scripts and artifacts live in the normal places
(`scripts/`, `exp/`). Every script selects its package through `scripts/env.sh`
(`use_navsim_v1` or `use_navsim_epdms`), and `tools/make_longitudinal_samples.py` pins `sys.path` to
`navsim_epdms/` and stops with an error if the wrong package is imported. Install only the root package:
`navsim_epdms/` is also named `navsim`, and `pip install`-ing it would replace the root one.


## Setup

### Environment

Tested with Python 3.9, PyTorch 2.8.0 (CUDA 12.8), PyTorch Lightning 2.2.1 and Hydra 1.2.0.
Install a PyTorch build that matches your CUDA first, then the root package and its
requirements:

```bash
conda create -n versatile python=3.9 -y
conda activate versatile
# install torch + torchvision for your CUDA from pytorch.org, then:
pip install -e .                      # the root navsim package only (see the note above)
```

`pip install -e .` pulls `requirements.txt`, including nuplan-devkit from GitHub, so `git`
must be on the PATH; the download scripts additionally use `wget`, `tar`, `unzip` and `rsync`.

Activate this environment before running any script. `scripts/env.sh` stops with a message
if `python` cannot import Hydra. MeanFuser and DiffusionDrive build an ImageNet ResNet-34
through `timm` when they are constructed, so the first run needs network access to the
Hugging Face hub, or a pre-filled hub cache on an offline machine.

### Workspace and data

All paths derive from one variable, `NAVSIM_WORKSPACE`. It defaults to the directory that
contains this repo, so cloning the repo into the workspace is enough; export it to keep the
data somewhere else.

```bash
export NAVSIM_WORKSPACE=/path/to/navsim_workspace
git clone https://github.com/mrbyflyg/Versatile-Scoring.git $NAVSIM_WORKSPACE/versatile_training
mkdir -p $NAVSIM_WORKSPACE/dataset
```

**NOTE: review and agree to the [nuPlan LICENSE](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE)
before downloading the data.**

#### 1. Download the dataset

navtrain and navtest need the nuPlan maps, the `navtrain` subset of the OpenScene `trainval`
split and the whole `test` split, each with camera and LiDAR: 130 archives, 677 GB. Either
option below runs from `$NAVSIM_WORKSPACE/dataset` and downloads into it.

**a. With the NAVSIM download scripts.** `download/` holds the scripts from the
[NAVSIM installation guide](https://github.com/autonomousvision/navsim/blob/main/docs/install.md#2-download-the-dataset);
each one downloads, extracts and deletes its archives in turn.

```bash
cd $NAVSIM_WORKSPACE/dataset
bash ../versatile_training/download/download_maps.sh          # -> maps/
bash ../versatile_training/download/download_navtrain_hf.sh   # navtrain logs and sensors
bash ../versatile_training/download/download_test.sh          # test logs and sensors
mkdir -p navsim_logs sensor_blobs
mv trainval_navsim_logs/trainval navsim_logs/   && mv test_navsim_logs/test navsim_logs/
mv trainval_sensor_blobs/trainval sensor_blobs/ && mv test_sensor_blobs/test sensor_blobs/
rmdir trainval_navsim_logs test_navsim_logs trainval_sensor_blobs test_sensor_blobs
```

**b. With the Hugging Face CLI.** Fetches the same archives from
[OpenDriveLab/OpenScene](https://huggingface.co/datasets/OpenDriveLab/OpenScene) in parallel
and resumes after an interruption; the archives are then extracted straight into place.

```bash
pip install -U huggingface_hub
# export HF_ENDPOINT=https://hf-mirror.com   # if huggingface.co is slow or unreachable
cd $NAVSIM_WORKSPACE/dataset
bash ../versatile_training/download/download_maps.sh          # -> maps/
hf download --repo-type dataset OpenDriveLab/OpenScene --local-dir . \
    --include "openscene-v1.1/openscene_metadata_trainval.tgz" \
    --include "openscene-v1.1/openscene_metadata_test.tgz" \
    --include "openscene-v1.1/openscene_sensor_test_*/*.tgz" \
    --include "navsim/navtrain_*.tgz"
mkdir -p navsim_logs sensor_blobs/trainval
for f in openscene-v1.1/openscene_metadata_*.tgz;      do tar -xzf "$f" -C navsim_logs           --strip-components=2; done
for f in openscene-v1.1/openscene_sensor_test_*/*.tgz; do tar -xzf "$f" -C sensor_blobs          --strip-components=2; done
for f in navsim/navtrain_*.tgz;                        do tar -xzf "$f" -C sensor_blobs/trainval --strip-components=1; done
rm -r openscene-v1.1 navsim .cache                            # the archives and the CLI's bookkeeping
```

#### 2. Check the layout

Both options end in this layout. There is nothing to configure: `scripts/env.sh` derives
`OPENSCENE_DATA_ROOT`, `NUPLAN_MAPS_ROOT` and `NAVSIM_CACHE_ROOT` from `NAVSIM_WORKSPACE`.

```
$NAVSIM_WORKSPACE/
├── versatile_training/        this repo
├── dataset/
│   ├── maps/                  nuplan-maps-v1.0.json and the four map folders
│   ├── navsim_logs/
│   │   ├── trainval/          1,310 log pickles
│   │   └── test/              147 log pickles
│   └── sensor_blobs/
│       ├── trainval/          one folder per log (navtrain frames only): CAM_*/ and MergedPointCloud/
│       └── test/
└── cache/                     written by step 1 of Reproducing
```

### Planner checkpoints

Both planners are frozen released checkpoints. Place them under `exp/` with these exact names:

```bash
mkdir -p exp/meanfuser_checkpoints exp/diffusiondrive_checkpoints
# MeanFuser, ResNet-34, 89.0 PDMS (upstream MeanFuser release on Google Drive; pip install gdown)
gdown 16989kIYhM3wQgxjSKvRFfK9cdKZfuU2P -O exp/meanfuser_checkpoints/meanfuser_pdms_89.0.ckpt
# DiffusionDrive, 88.1 PDMS (hustvl/DiffusionDrive on Hugging Face) and its 20 k-means plan anchors
wget -O exp/diffusiondrive_checkpoints/diffusiondrive_navsim_88p1_PDMS.ckpt \
    https://huggingface.co/hustvl/DiffusionDrive/resolve/main/diffusiondrive_navsim_88p1_PDMS
wget -O exp/diffusiondrive_checkpoints/plan_anchor_kmeans20.npy \
    https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy
md5sum exp/meanfuser_checkpoints/meanfuser_pdms_89.0.ckpt exp/diffusiondrive_checkpoints/diffusiondrive_navsim_88p1_PDMS.ckpt
# b323f65a1c13fdc7fe8df9d14e961531  meanfuser_pdms_89.0.ckpt
# f82c3dddbfaa7d84445867299634d817  diffusiondrive_navsim_88p1_PDMS.ckpt
```

## Reproducing

Three scorers are compared, 3 seeds each (0 1 2) on both planners: `plain` (trained on the
planner's own proposals, the paper's "plain scorer"), `design` (proposals + the designed lateral
and longitudinal rungs, the "augmented scorer") and `random` (proposals + the volume-matched
random samples of Table 4). Run everything from the repo root with the environment active. The
scripts take the planner (`mf` MeanFuser, `dd` DiffusionDrive) and the split (`navtrain`,
`navtest`) as arguments and resume where they stopped: a labelling run skips a slot whose CSV
exists and a training sweep skips a seed whose `train.log` records a finished run (an interrupted
slot or seed starts over). Both planners are frozen released checkpoints (Setup); retraining them
is not part of the pipeline.

### 1 · Caches

```bash
for s in navtrain navtest; do bash scripts/build_cache.sh all $s; done
```

`all` builds, in order, the PDMS (v1) metric cache `cache/<split>_v1_metric_cache`, the EPDMS
(v2) metric cache `cache/<split>_v2_metric_cache` (built by `navsim_epdms`, whose cache processor
computes the traffic-light and lane data the v1 one does not), the MeanFuser camera feature cache
`cache/trainval_v1_cache` and the DiffusionDrive camera + LiDAR BEV feature cache
`cache/trainval_v1_cache_dd`. Any one of them can be built alone
(`bash scripts/build_cache.sh v2_metric navtest`). Extra arguments are passed to Hydra, e.g.
`train_test_split.scene_filter.max_scenes=10` for a quick check. Expect 103,288 navtrain and
12,146 navtest scenes.

### 2 · The planners' proposals and their labels

```bash
for s in navtrain navtest; do
  for p in mf dd; do
    bash scripts/dump_candidates.sh $p $s   # run the frozen planner once: proposals, their PDMS, scene context
    bash scripts/label_slots.sh $p $s       # EPDMS of every proposal slot under the official scorer
  done
  python tools/make_token_to_log.py --split $s   # token -> log map: the generators open logs, the trainer holds out whole logs
done
```

`dump_candidates.sh` writes `exp/*/plain_dataset[_dd]_<split>/`: the planner's trajectories
(8 proposals + its own pick for MeanFuser, 20 modes + the cls pick for DiffusionDrive), their
PDMS in `dataset.pkl` and the frozen encoder's scene features in `context_features.pkl`.
`label_slots.sh` deploys each proposal slot once over the whole split under the official
one-stage EPDMS scorer and stitches the per-slot CSVs into `components_<planner>_<split>.pkl`,
`{token: [N (+arm, +human), 9]}`: the human (logged) trajectory is the extra training target
(`--human` in step 4), the planner's own pick is the eval-only baseline. DiffusionDrive takes its
human row from the MeanFuser labels. Navtrain labelling takes about 28 minutes per slot on 24 CPU
cores. The `arm` slot doubles as a base-planner check: its navtest EPDMS should be about 89.61
for MeanFuser and 88.22 for DiffusionDrive.

### 3 · Designed samples

```bash
for s in navtrain navtest; do
  python tools/make_lateral_samples.py      --split $s   # toward the drivable-area boundary (Sec. 4; Eq. 5 and 7)
  python tools/make_longitudinal_samples.py --split $s   # toward the lead vehicle, simulator in the loop (Eq. 6, Algorithm 1)
  python tools/make_random_samples.py       --split $s   # the volume-matched random control (Table 4)
  for d in lateral longitudinal random; do bash scripts/label_slots.sh $d $s; done
done
```

Each generator writes `exp/meanfuser_checkpoints/<design>_samples_<split>/samples.pkl`: the 8
MeanFuser proposals in slots 0..7 (already labelled in step 2) followed by its new rungs, which
`label_slots.sh` then scores in the simulator, the expensive stage (slots 8..11 for the two
designs, 8..12 for the random control). The two designs accept `--shard I N` / `--merge N` (on
navtrain: 8 shards in parallel, then `--merge 8`).

### 4 · Training sets and scorers

```bash
for s in navtrain navtest; do
  for p in mf dd; do python tools/make_trainsets.py --planner $p --split $s; done
done
for p in mf dd; do
  for d in plain design random; do bash scripts/train_scorer.sh $p $d 0 1 2; done
done
```

Each set is a directory holding `dataset.pkl` (candidates) and `components.pkl` (labels):
`exp/meanfuser_checkpoints/versatile_dataset_<split>_{design,random}` and
`exp/diffusiondrive_checkpoints/versatile_dataset_dd_<split>_{plain,design,random}`. The rungs
are anchored on the logged human trajectory, so they and their labels are planner-independent
and need no new simulator pass; behind DiffusionDrive's 20 proposals they sit at shifted slots.
A rung supervises only the head of the rule it was designed against (lateral: dac,
longitudinal: nc), and the random samples get the same head assignment in the same numbers.
The trainer reads the navtrain sets, plus DiffusionDrive's navtest `plain` set for its per-epoch
navtest report.

18 runs (3 sets x 3 seeds x 2 planners), about 50 minutes each on one GPU; the GPU saturates at
two concurrent runs, so launch at most two streams. Each run writes `comp*_bestval.pt`, selected
on the held-out validation logs (59 of 1192 logs, ~5% of scenes; the split RNG is independent of
the initialisation seed, so every run shares one held-out set), and `comp*_bestreal.pt`, selected
on navtest and logged for diagnostics only. The reported checkpoint is `bestval`.

## Citation

```bibtex
@article{li2026versatile,
  title   = {Designing Versatile Samples for Learned Trajectory Scoring},
  author  = {Li, Yaguang and Zhang, Jiaru and Wei, Chuheng and Cui, Can and Wang, Ziran},
  journal = {arXiv preprint arXiv:2609.01799},
  year    = {2026}
}
```

## Acknowledgements and license

`navsim/` and `navsim_epdms/` are the two [NAVSIM](https://github.com/autonomousvision/navsim)
releases (Apache 2.0). They carry the [MeanFuser](https://github.com/wjl2244/MeanFuser) planner
(Apache 2.0) and the [DiffusionDrive](https://github.com/hustvl/DiffusionDrive) planner (MIT).
`tools/`, `scripts/` and the scorer code added to `navsim/` are this work, released under the
Apache 2.0 license in `LICENSE`.
