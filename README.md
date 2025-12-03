# Mapping of Avalanches from SAR Satellite Images with Deep Learning Change Detection

This repository accompanies the paper:

> **Mapping of Avalanches from SAR Satellite Images with Deep Learning Change Detection**

---

## Environment Setup

### 1. Install Python 3.11 and create a virtual environment:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Experiments

Run the full pipeline (all patch sizes):
```bash
./patchify_all.sh                          # Create patches at multiple scales
./train_all_patch_sizes.sh --model=swinunet    # Train Swin-UNet models on all scales
./test_all_patch_sizes.sh --model=swinunet     # Evaluate all trained models
```

Run the pipeline for a single patch size:
```bash
# 1. Patchify (128×128 patches, stride 64)
python patchify.py --patch-size 128 --stride 64

# 2. Train
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset-root "/home/jovyan/nfs/mgatti/datasets/Avalanches/patches/" \
    --model "swinunet" \
    --patch-size 128 \
    --description "training patch_size 128"

# 2.1 Train with best parameters
CUDA_VISIBLE_DEVICES=0 python train.py \
  --description "FBeta=1.5, precision>=0.60, BCE" \
  --model swinunet --model-size tiny --fusion-type diff \
  --patch-size 128 --batch-size 32 --lr 1e-4 \
  --beta 1.5 --precision-floor 0.60 \
  --loss "bce" --pos-weight 3.0
```

# 3. Test
CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataset-root "/home/jovyan/nfs/mgatti/datasets/Avalanches/patches/" \
    --model "swinunet" \
    --patch-size 128

Run for a single patch size with forced resolution (e.g. 10 m):
```bash
# 1. Patchify with enforced 10 m resolution
python patchify.py --patch-size 128 --stride 64 --force-resolution 10

# 2. Train
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset-root "/home/jovyan/nfs/mgatti/datasets/Avalanches/patches_10m/" \
    --patch-size 128 \
    --description "training patch_size 128"

# 3. Test
CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataset-root "/home/jovyan/nfs/mgatti/datasets/Avalanches/patches_10m/" \
    --patch-size 128
```

## Inference

```bash
python infer.py \
  --event-path /home/jovyan/nfs/mgatti/datasets/Avalanches/AvalCD/Tromso_20241220 \
  --model-ckpt exp/swinunet_128/best_model.pth \
  --output-dir outputs/inference \
  --patch-size 128 \
  --stride 64 \
  --blending center_crop
```

If your BASE_DIR contains subfolders organized by acquisition date, you can process them all at once by running:
```bash
./run_infer.sh /home/jovyan/nfs/mgatti/datasets/Avalanches/sar_avalanche_timeseries/Livigno_ron15
./run_infer.sh /home/jovyan/nfs/mgatti/datasets/Avalanches/sar_avalanche_timeseries/Livigno_ron168
./run_infer.sh /home/jovyan/nfs/mgatti/datasets/Avalanches/sar_avalanche_timeseries/Marche_A01
./run_infer.sh /home/jovyan/nfs/mgatti/datasets/Avalanches/sar_avalanche_timeseries/Marche_A02
./run_infer.sh /home/jovyan/nfs/mgatti/datasets/Avalanches/sar_avalanche_timeseries/Marche_A03
./run_infer.sh /home/jovyan/nfs/mgatti/datasets/Avalanches/sar_avalanche_timeseries/Marche_A04
./run_infer.sh /home/jovyan/nfs/mgatti/datasets/Avalanches/sar_avalanche_timeseries/Nuuk
```

---

## Dataset and Code Release

The annotated avalanche inventory is available at:

[Zenodo DOI: 10.5281/zenodo.15863589](https://doi.org/10.5281/zenodo.15863589)

> **Note:** Use the `patchify.py` script provided within the Zenodo dataset.  
> The version in this repository performs additional resampling, which is unnecessary since the Zenodo images are already pre-resampled.

---

## Citation

If you use this work, please cite:

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