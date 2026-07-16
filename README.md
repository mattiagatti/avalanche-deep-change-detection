# 🏔️ Large-Scale Avalanche Mapping from SAR Images with Deep Learning-based Change Detection

This repository provides the training, evaluation, and inference pipelines for large-scale snow-avalanche mapping via bi-temporal change detection on Sentinel-1 SAR imagery, together with baseline change-detection models and a Swin-UNet architecture.

> 📄 **Related paper:** *Large-Scale Avalanche Mapping from SAR Images with Deep Learning-based Change Detection.*
> Available on [arXiv](https://arxiv.org/abs/2603.22658). See [Citation](#-citation) below.

---

## ⚙️ Environment Setup

### 1. Install Python 3.11 and create a virtual environment

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Dataset paths:** the commands below use `--dataset-root` and `--event-path` values
> pointing to the authors' environment (e.g. `/home/jovyan/nfs/mgatti/...`). Replace them
> with the path where you extracted the dataset (see [Dataset and Code Release](#-dataset-and-code-release)).

---

## 🚀 Running the Experiments

Run the full pipeline (all patch sizes):
```bash
./preprocessing/patchify_all.sh                # Create patches at multiple scales
./train_all_patch_sizes.sh --model=swinunet    # Train Swin-UNet models on all scales
./test_all_patch_sizes.sh --model=swinunet     # Evaluate all trained models
```

Run the pipeline for a single patch size:
```bash
# 1. Patchify (128×128 patches, stride 64)
python preprocessing/patchify.py --patch-size 128 --stride 64

# 2. Train
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset-root "/path/to/Avalanches/patches/" \
    --model "swinunet" \
    --patch-size 128 \
    --description "training patch_size 128"

# 2.1 Train specifying all parameters
CUDA_VISIBLE_DEVICES=0 python train.py \
    --description "FBeta=1.5, precision>=0.60, BCE" \
    --model swinunet --model-size tiny --fusion-type diff \
    --patch-size 128 --batch-size 32 --lr 1e-4 \
    --beta 1.5 --precision-floor 0.60 \
    --loss "bce" --pos-weight 3.0
```

Available models: `swinunet` plus the baselines `bit`, `changeformer`, `siamunet_conc`,
`siamunet_diff`, `snunet`, `stanet`, `stnet`, and `tinycd`.

---

## 🧪 Test

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataset-root "/path/to/Avalanches/patches/" \
    --model "swinunet" \
    --patch-size 128
```

By default the checkpoint is loaded from `exp/<model>_<patch-size>[_aux]/best_model.pth`
(the location `train.py` writes to). Override the lookup with `--exp-root <dir>` or point
directly at a file with `--model-ckpt <path>`.

---

## 🔍 Inference

```bash
python infer.py \
  --event-path /path/to/Avalanches/AvalCD/Tromso_20241220 \
  --model-ckpt exp/swinunet_128/best_model.pth \
  --output-dir outputs/inference \
  --patch-size 128 \
  --stride 64 \
  --blending center_crop
```

If your base directory contains subfolders organized by acquisition date, you can process
them all at once by running:
```bash
./infer_timeseries.sh /path/to/Avalanches/sar_avalanche_timeseries/Livigno_ron15
./infer_timeseries.sh /path/to/Avalanches/sar_avalanche_timeseries/Livigno_ron168
./infer_timeseries.sh /path/to/Avalanches/sar_avalanche_timeseries/Marche_A01
./infer_timeseries.sh /path/to/Avalanches/sar_avalanche_timeseries/Marche_A02
./infer_timeseries.sh /path/to/Avalanches/sar_avalanche_timeseries/Marche_A03
./infer_timeseries.sh /path/to/Avalanches/sar_avalanche_timeseries/Marche_A04
./infer_timeseries.sh /path/to/Avalanches/sar_avalanche_timeseries/Nuuk
```

---

## 📦 Dataset and Code Release

The annotated avalanche inventory is available at:

[Zenodo DOI: 10.5281/zenodo.15863589](https://doi.org/10.5281/zenodo.15863589)

> **Note:** Use the `patchify.py` script provided within the Zenodo dataset.
> The version in this repository performs additional resampling, which is unnecessary since
> the Zenodo images are already pre-resampled.

---

## 📖 Citation

If you use this work, please cite the paper:

```bibtex
@article{gatti2026avalanche,
  title   = {Large-Scale Avalanche Mapping from SAR Images with Deep Learning-based Change Detection},
  author  = {Gatti, Mattia and Mariani, Alberto and Gallo, Ignazio and Monti, Fabiano},
  journal = {arXiv preprint arXiv:2603.22658},
  year    = {2026}
}
```

If you use the dataset, please cite:

```bibtex
@dataset{avalcd_dataset,
  author       = {Gatti, Mattia},
  title        = {AvalCD},
  month        = jul,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15863589},
  url          = {https://doi.org/10.5281/zenodo.15863589},
}
```