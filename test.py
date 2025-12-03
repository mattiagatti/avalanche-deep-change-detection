import argparse
import json
import logging
import numpy as np
import torch

from dataset.avalanches import AvalancheDataset
from models.baselines.adapter import CDModelAdapter
from models.baselines.factory import BuildArgs as BaselineBuildArgs, build_baseline, available_models
from models.swinunet import ChangeDetectionSwinUNet
from pathlib import Path
from PIL import Image
from utils.morph import morph_close
from torch.utils.data import DataLoader
from torchmetrics import F1Score, Recall, Precision, JaccardIndex
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# CLI + paths
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

parser = argparse.ArgumentParser(description="Avalanche change-detection test")
parser.add_argument("--patch-size", default=128, type=int)
parser.add_argument("--dataset-root", type=str, default="/home/jovyan/nfs/mgatti/datasets/Avalanches/patches/",
                    help="Root directory of the dataset (without patch size).")
parser.add_argument("--model", type=str, choices=available_models() + ["swinunet"],
                    help="Choose an architecture")
parser.add_argument("--model-size", type=str, choices=["tiny", "small", "base"], default="tiny",
                    help="Size to use: tiny, small, base")
parser.add_argument("--use-aux", action="store_true",
                    help="Enable Auxiliary data usage.")
parser.add_argument("--plot-every", type=int, default=1,
                    help="Plot every Nth sample that has any change in GT")
parser.add_argument("--no-morph", dest="use_morph", action="store_false",
                    help="Disable morphological closing (dilate -> erode) ...")
parser.add_argument("--kernel-size", type=int, default=3,
                    help="Kernel size for morphological operations (must be odd)")
parser.add_argument("--iterations", type=int, default=1,
                    help="Number of dilation / erosion iterations")

args = parser.parse_args()

MODEL_SIZE = args.model_size
PATCH_SIZE = args.patch_size
USE_AUX = args.use_aux
PLOT_EVERY = args.plot_every

USE_MORPH = args.use_morph
KERNEL_SIZE = args.kernel_size
NUM_ITERS = args.iterations

DATASET_DIR = Path(args.dataset_root) / f"{PATCH_SIZE}"
MODEL_CKPT = Path(f"/home/jovyan/nfs/mgatti/python/avalanches/exp/{args.model}_{PATCH_SIZE}"
                   f"{'_aux' if USE_AUX else ''}/best_model.pth")

OUTPUT_DIR = Path(f"output/test/{PATCH_SIZE}")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# --------------------------------------------------------------------------- #
# data + model
# --------------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_ds = AvalancheDataset(DATASET_DIR, ["Tromso_20241220"], apply_transform=False)
test_dl = DataLoader(test_ds, batch_size=32, num_workers=4)

TOTAL_TEST_PATCHES = len(test_ds)
logging.info(f"Total test patches (dataset): {TOTAL_TEST_PATCHES}")

if args.model == "swinunet":
    model = ChangeDetectionSwinUNet(model_size=MODEL_SIZE, img_size=PATCH_SIZE, use_aux=USE_AUX).to(device)
else:
    core = build_baseline(args.model, BaselineBuildArgs(device=device, patch_size=PATCH_SIZE, in_ch=2, out_ch=1))
    model = CDModelAdapter(core, model_name=args.model).to(device)


def count_parameters(model, trainable_only: bool = True) -> int:
    params = (
        p for p in model.parameters()
        if (p.requires_grad or not trainable_only)
    )
    return sum(p.numel() for p in params)


total_params = count_parameters(model, trainable_only=False)
trainable_params = count_parameters(model, trainable_only=True)

logging.info(
    f"Model size: {total_params:,d} parameters "
    f"({total_params/1e6:.2f} M total / {trainable_params/1e6:.2f} M trainable)"
)

ckpt = torch.load(MODEL_CKPT, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
best_thr = ckpt.get("best_threshold", 0.5)
logging.info(f"Decision threshold (best_threshold from ckpt): {best_thr:.4f}")

metrics = {
    "recall": Recall(task="binary", num_classes=1, threshold=best_thr).to(device),
    "precision": Precision(task="binary", num_classes=1, threshold=best_thr).to(device),
    "f1": F1Score(task="binary",  num_classes=1, threshold=best_thr).to(device),
    "iou": JaccardIndex(task="binary", num_classes=1, threshold=best_thr).to(device),
}

# --------------------------------------------------------------------------- #
# plotting utils
# --------------------------------------------------------------------------- #
STATS_PATH = DATASET_DIR / "stats.json"
stats = json.loads(Path(STATS_PATH).read_text())
IMG_MEAN_DB, IMG_STD_DB = map(torch.tensor, (stats["img_mean"], stats["img_std"]))
SENTINEL_Z_IMG = torch.tensor(stats["sentinel_z_img"])  # shape: (2,)


def _save_images(idx: int,
                 pre:  torch.Tensor,
                 post: torch.Tensor,
                 aux:  torch.Tensor,
                 gt:   torch.Tensor,
                 pred: torch.Tensor,
                 bin_mask: torch.Tensor,
                 out_dir: Path = OUTPUT_DIR):

    # Check if any value in pre or post equals its corresponding sentinel_z
    if any(torch.any(pre[c] == SENTINEL_Z_IMG[c]) or torch.any(post[c] == SENTINEL_Z_IMG[c]) for c in range(2)):
        logging.info(f"Patch {idx} skipped – contains extreme values in VV/VH")
        return

    # -------------------- helper: log-power-scale back to 0-1 --------------------
    def z_to_db_band(z, c: int):
        # Convert normalized input back to dB scale
        return z * IMG_STD_DB[c].to(z.device) + IMG_MEAN_DB[c].to(z.device)

    def joint_minmax_norm(pre: torch.Tensor, post: torch.Tensor, c: int):
        pre_db = z_to_db_band(pre, c)
        post_db = z_to_db_band(post, c)

        joint_min = torch.min(pre_db.min(), post_db.min())
        joint_max = torch.max(pre_db.max(), post_db.max())

        pre_norm = (pre_db - joint_min) / (joint_max - joint_min + 1e-6)
        post_norm = (post_db - joint_min) / (joint_max - joint_min + 1e-6)
        return pre_norm.clamp(0, 1).cpu().numpy(), post_norm.clamp(0, 1).cpu().numpy()

    # Generate RGB quick-looks using joint min-max normalization
    vv_pre, vv_post = joint_minmax_norm(pre[0], post[0], c=0)
    vh_pre, vh_post = joint_minmax_norm(pre[1], post[1], c=1)

    vv_rgb = np.stack([vv_pre, vv_post, vv_pre], axis=-1)
    vh_rgb = np.stack([vh_pre, vh_post, vh_pre], axis=-1)

    vv_rgb = (vv_rgb * 255).astype(np.uint8)
    vh_rgb = (vh_rgb * 255).astype(np.uint8)

    # -------------------- auxiliary rasters --------------------
    lia = aux[0].cpu().numpy().astype(np.float32)
    slope = aux[1].cpu().numpy().astype(np.float32)
    lia_img = ((lia - lia.min()) / (np.ptp(lia) + 1e-9) * 255).astype(np.uint8)
    slope_img = ((slope - slope.min()) / (np.ptp(slope) + 1e-9) * 255).astype(np.uint8)

    # -------------------- masks --------------------
    g = gt.squeeze().cpu().numpy().astype(bool)
    prb = pred.squeeze().cpu().numpy()  # raw probability ∈ [0,1]
    bin_mask = bin_mask.squeeze().cpu().numpy().astype(bool)

    # coloured confusion map
    err = np.zeros((*g.shape, 3), dtype=np.uint8)
    err[~g & ~bin_mask] = (0, 0, 0)   # TN – black
    err[g & bin_mask] = (0, 255, 0)   # TP – green
    err[~g & bin_mask] = (255, 255, 0)   # FP – yellow
    err[g & ~bin_mask] = (255, 0, 0)   # FN – red

    # -------------------- save all artefacts --------------------
    sample_dir = out_dir / str(idx)
    sample_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(vv_rgb).save(sample_dir / "vv_rgb.png")
    Image.fromarray(vh_rgb).save(sample_dir / "vh_rgb.png")
    Image.fromarray(lia_img).save(sample_dir / "lia.png")
    Image.fromarray(slope_img).save(sample_dir / "slope.png")
    Image.fromarray(err).save(sample_dir / "pred_confusion.png")

    # new: GT, raw prob, binary pred
    Image.fromarray((g * 255).astype(np.uint8)).save(sample_dir / "gt.png")
    Image.fromarray((prb * 255).astype(np.uint8)).save(sample_dir / "pred_prob.png")
    Image.fromarray((bin_mask * 255).astype(np.uint8)).save(sample_dir / "pred_bin.png")


if __name__ == "__main__":
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Testing", ncols=100):
            pre, post, aux = batch["pre"].to(device), batch["post"].to(device), batch["aux"].to(device)
            gt = batch["mask"].to(device)

            # ---- forward (with/without AUX) ----
            if not USE_AUX:
                logits = model(pre, post)
            else:
                logits = model(pre, post, aux)

            # ---- probabilities + threshold ----
            prb = torch.sigmoid(logits)           # (B,1,H,W)
            bin_mask = (prb > best_thr).float()

            # -------- optional morphology -------- #
            if USE_MORPH:
                bin_mask = morph_close(bin_mask, kernel_size=KERNEL_SIZE, iterations=NUM_ITERS)

            # -------- metric update -------- #
            for m in metrics.values():
                # torchmetrics will internally threshold again, but that's fine
                m.update(bin_mask, gt)

            for b in range(pre.size(0)):
                if PLOT_EVERY > 0 and (global_idx % PLOT_EVERY == 0) and torch.any(gt[b]):
                    _save_images(global_idx, pre[b], post[b], aux[b], gt[b], prb[b], bin_mask[b])
                global_idx += 1

    for n, m in metrics.items():
        logging.info(f"{n.capitalize()}: {m.compute().cpu().numpy():.4f}")

    logging.info(f"Total number of patches: {global_idx}")