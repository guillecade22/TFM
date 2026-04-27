"""
Multi-Experiment Reconstruction Evaluation Script
==================================================
Evaluates multiple reconstruction folders against a single ground-truth directory
in one pass. Models and GT features are loaded ONCE and reused across all experiments,
making it much faster than running separate scripts.

Metrics
-------
  PixCorr      - Pixel-level Pearson correlation              (higher is better)
  SSIM         - Structural Similarity Index                  (higher is better)
  AlexNet(2)   - 2-way ID accuracy, AlexNet layer 2          (higher is better)
  AlexNet(5)   - 2-way ID accuracy, AlexNet layer 5          (higher is better)
  InceptionV3  - 2-way ID accuracy, InceptionV3 avgpool      (higher is better)
  CLIP         - 2-way ID accuracy, CLIP ViT-L/14            (higher is better)
  EffNet-B     - EfficientNet-B1 feature distance             (lower  is better)
  SwAV         - SwAV ResNet-50 feature distance              (lower  is better)

Usage
-----
  # Run with all defaults (evaluates all 7 experiments vs required/test):
  python evaluate_reconstructions.py

  # Override GT or experiments:
  python evaluate_reconstructions.py --gt_path other/test --recon_dirs other/exp_a other/exp_b

  # Skip heavy models while iterating:
  python evaluate_reconstructions.py --skip clip swav inception

  # Show a comparison grid and limit to 50 samples:
  python evaluate_reconstructions.py --show_grid --n_samples 50

Defaults
--------
  gt_path    : required/test
  recon_dirs : required/balanced, required/exp_img2img, required/exp_img2img_caption,
               required/exp_notext, required/exp_simple_caption,
               required/generated_gt_caption, required/more_image
  output_csv : results/all_experiments.csv
  imsize     : 425
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _sorted_image_files(directory: Path) -> list:
    files = sorted(
        [f for f in directory.iterdir() if f.suffix.lower() in IMG_EXTENSIONS],
        key=lambda p: p.name,
    )
    if not files:
        raise FileNotFoundError(f"No image files found in: {directory}")
    return files


def load_images_from_dir(directory, n_samples=None) -> torch.Tensor:
    """Load a flat directory of images into a float (N, C, H, W) tensor in [0, 1]."""
    directory = Path(directory)
    files = _sorted_image_files(directory)
    if n_samples is not None:
        files = files[:n_samples]

    to_tensor = transforms.ToTensor()
    tensors = [
        to_tensor(Image.open(f).convert("RGB"))
        for f in tqdm(files, desc=f"  Loading {directory.name}", leave=False)
    ]
    return torch.stack(tensors)   # (N, 3, H, W)


def load_images(path: str, n_samples=None) -> torch.Tensor:
    p = Path(path)
    if p.is_dir():
        return load_images_from_dir(p, n_samples=n_samples)
    elif p.suffix == ".pt":
        t = torch.load(p, map_location="cpu")
        if t.ndim == 4 and t.shape[-1] in (1, 3, 4):
            t = t.permute(0, 3, 1, 2)
        t = t.float()
        if t.max() > 1.0:
            t /= 255.0
        t = t.clamp(0.0, 1.0)
        return t[:n_samples] if n_samples else t
    else:
        raise ValueError(f"Expected a directory or .pt file, got: {path}")


def report_pairing(recon_dir: str, gt_dir: str, n: int):
    rp, gp = Path(recon_dir), Path(gt_dir)
    if rp.is_dir() and gp.is_dir():
        r5 = _sorted_image_files(rp)[:3]
        g5 = _sorted_image_files(gp)[:3]
        pairs = "  |  ".join(f"{r.name} <-> {g.name}" for r, g in zip(r5, g5))
        print(f"    Pairing (first 3): {pairs}  ... ({n} total)")


# ---------------------------------------------------------------------------
# Feature extraction — shared across experiments
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(images: torch.Tensor, model, preprocess, feature_layer=None,
                     batch_size: int = 32) -> np.ndarray:
    """Run a model on a tensor of images and return flattened features as numpy."""
    feats = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack(
            [preprocess(img) for img in images[start : start + batch_size]]
        ).to(DEVICE)
        out  = model(batch)
        feat = out if feature_layer is None else out[feature_layer]
        feats.append(feat.float().flatten(1).cpu())
    return torch.cat(feats, dim=0).numpy()


def two_way_identification(gt_feats: np.ndarray, recon_feats: np.ndarray) -> float:
    """
    2-way identification accuracy given pre-extracted feature matrices.

    For each image i, counts how many distractors j!=i satisfy
    corr(gt_i, recon_j) < corr(gt_i, recon_i).
    Returns mean accuracy in [0, 1].
    """
    n = len(gt_feats)
    r = np.corrcoef(gt_feats, recon_feats)   # (2N, 2N)
    r = r[:n, n:]                             # (N, N): r[i,j] = corr(gt_i, recon_j)
    congruents = np.diag(r)                   # correct-pair correlations

    success = r < congruents[:, None]        # distractors that scored lower
    np.fill_diagonal(success, False)
    return float(success.sum(axis=1).mean() / (n - 1))


# ---------------------------------------------------------------------------
# Model builders  (called once, then reused)
# ---------------------------------------------------------------------------

def build_alexnet():
    from torchvision.models import alexnet, AlexNet_Weights
    model = create_feature_extractor(
        alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
        return_nodes={"features.4": "l2", "features.11": "l5"},
    ).to(DEVICE).eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


def build_inception():
    from torchvision.models import inception_v3, Inception_V3_Weights
    model = create_feature_extractor(
        inception_v3(weights=Inception_V3_Weights.DEFAULT),
        return_nodes={"avgpool": "avgpool"},
    ).to(DEVICE).eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


def build_clip():
    try:
        import clip as openai_clip
    except ImportError:
        raise ImportError("pip install git+https://github.com/openai/CLIP.git")
    model, _ = openai_clip.load("ViT-L/14", device=DEVICE)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std= [0.26862954, 0.26130258, 0.27577711]),
    ])
    return model.encode_image, preprocess


def build_effnet():
    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    model = create_feature_extractor(
        efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT),
        return_nodes={"avgpool": "avgpool"},
    ).to(DEVICE).eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


def build_swav():
    backbone = torch.hub.load("facebookresearch/swav:main", "resnet50")
    model = create_feature_extractor(
        backbone, return_nodes={"avgpool": "avgpool"}
    ).to(DEVICE).eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


# ---------------------------------------------------------------------------
# Per-experiment metrics  (GT features passed in, already computed)
# ---------------------------------------------------------------------------

def eval_pixcorr(gt: torch.Tensor, recons: torch.Tensor, imsize: int) -> float:
    resize = transforms.Resize(imsize, interpolation=transforms.InterpolationMode.BILINEAR)
    g = resize(gt).reshape(len(gt), -1).numpy()
    r = resize(recons).reshape(len(recons), -1).numpy()
    return float(np.mean([np.corrcoef(g[i], r[i])[0, 1] for i in range(len(g))]))


def eval_ssim(gt: torch.Tensor, recons: torch.Tensor, imsize: int) -> float:
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim_fn
    resize = transforms.Resize(imsize, interpolation=transforms.InterpolationMode.BILINEAR)
    g = rgb2gray(resize(gt).permute(0, 2, 3, 1).numpy())
    r = rgb2gray(resize(recons).permute(0, 2, 3, 1).numpy())
    scores = [
        ssim_fn(r[i], g[i], gaussian_weights=True, sigma=1.5,
                use_sample_covariance=False, data_range=1.0)
        for i in range(len(g))
    ]
    return float(np.mean(scores))


def eval_two_way(gt_feats: np.ndarray, recon_feats: np.ndarray) -> float:
    return two_way_identification(gt_feats, recon_feats)


def eval_distance(gt_feats: np.ndarray, recon_feats: np.ndarray) -> float:
    return float(np.mean([
        sp.spatial.distance.correlation(gt_feats[i], recon_feats[i])
        for i in range(len(gt_feats))
    ]))


# ---------------------------------------------------------------------------
# Optional visualisation
# ---------------------------------------------------------------------------

def show_grid(gt: torch.Tensor, recons: torch.Tensor, title: str,
              n_pairs: int = 8, imsize: int = 224):
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    resize  = transforms.Resize((imsize, imsize))
    indices = np.random.choice(len(gt), size=min(n_pairs, len(gt)), replace=False)

    pairs = []
    for idx in sorted(indices):
        pairs.append(resize(gt[idx]))
        pairs.append(resize(recons[idx]))

    grid = make_grid(torch.stack(pairs), nrow=n_pairs, padding=3)
    plt.figure(figsize=(24, 4))
    plt.imshow(grid.permute(1, 2, 0).clamp(0, 1).numpy())
    plt.axis("off")
    plt.title(f"{title}  —  top row: GT  |  bottom row: Reconstruction", fontsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_GT_PATH = "required/test"

DEFAULT_RECON_DIRS = [
    "required/balanced",
    "required/exp_img2img",
    "required/exp_img2img_caption",
    "required/exp_notext",
    "required/exp_simple_caption",
    "required/generated_gt_caption",
    "required/more_image",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate multiple reconstruction folders against one ground-truth dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--gt_path",
        default=DEFAULT_GT_PATH,
        help=f"Ground-truth image directory (default: {DEFAULT_GT_PATH})",
    )
    p.add_argument(
        "--recon_dirs",
        nargs="+",
        default=DEFAULT_RECON_DIRS,
        help="One or more reconstruction directories to evaluate (default: all 7 experiments)",
    )
    p.add_argument(
        "--output_csv",
        default="results/all_experiments.csv",
        help="Path for the combined CSV output",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Use only the first N image pairs per experiment (default: all)",
    )
    p.add_argument(
        "--imsize",
        type=int,
        default=425,
        help="Resize target for PixCorr and SSIM",
    )
    p.add_argument(
        "--show_grid",
        action="store_true",
        help="Display a visual GT vs recon grid for each experiment",
    )
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["pixcorr", "ssim", "alexnet", "inception", "clip", "effnet", "swav"],
        help="Metrics to skip (e.g. --skip clip swav)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    skip  = set(args.skip or [])
    exps  = [Path(d) for d in args.recon_dirs]

    print(f"\n{'='*65}")
    print("  Multi-Experiment Reconstruction Evaluation")
    print(f"{'='*65}")
    print(f"  Device       : {DEVICE}")
    print(f"  Ground truth : {args.gt_path}")
    print(f"  Experiments  : {len(exps)}")
    for e in exps:
        print(f"    - {e}")
    print(f"  Skipping     : {', '.join(skip) or 'none'}")
    print(f"{'='*65}\n")

    # ------------------------------------------------------------------ GT --
    print("[ GT ] Loading ground-truth images...")
    gt_full = load_images(args.gt_path, n_samples=args.n_samples)
    print(f"       {gt_full.shape}\n")

    # -------------------------------------------------- Load models once ----
    models = {}

    if "alexnet" not in skip:
        print("[ MODEL ] Loading AlexNet...")
        models["alexnet"] = build_alexnet()

    if "inception" not in skip:
        print("[ MODEL ] Loading InceptionV3...")
        models["inception"] = build_inception()

    if "clip" not in skip:
        print("[ MODEL ] Loading CLIP ViT-L/14...")
        models["clip"] = build_clip()

    if "effnet" not in skip:
        print("[ MODEL ] Loading EfficientNet-B1...")
        models["effnet"] = build_effnet()

    if "swav" not in skip:
        print("[ MODEL ] Loading SwAV ResNet-50...")
        models["swav"] = build_swav()

    print()

    # ---------------------------------------- Pre-extract GT features once --
    gt_feats = {}

    if "alexnet" not in skip:
        m, pre = models["alexnet"]
        print("[ GT feats ] AlexNet...")
        gt_feats["alex_l2"] = extract_features(gt_full, m, pre, "l2")
        gt_feats["alex_l5"] = extract_features(gt_full, m, pre, "l5")

    if "inception" not in skip:
        m, pre = models["inception"]
        print("[ GT feats ] InceptionV3...")
        gt_feats["inception"] = extract_features(gt_full, m, pre, "avgpool")

    if "clip" not in skip:
        m, pre = models["clip"]
        print("[ GT feats ] CLIP...")
        gt_feats["clip"] = extract_features(gt_full, m, pre, None)

    if "effnet" not in skip:
        m, pre = models["effnet"]
        print("[ GT feats ] EfficientNet-B1...")
        gt_feats["effnet"] = extract_features(gt_full, m, pre, "avgpool")

    if "swav" not in skip:
        m, pre = models["swav"]
        print("[ GT feats ] SwAV...")
        gt_feats["swav"] = extract_features(gt_full, m, pre, "avgpool")

    print()

    # ------------------------------------------------ Per-experiment loop ---
    all_rows = []

    for exp_dir in exps:
        exp_name = exp_dir.name
        sep = f"{'─'*65}"
        print(f"\n{sep}")
        print(f"  Experiment: {exp_name}")
        print(sep)

        # Load reconstructions
        recons = load_images(str(exp_dir), n_samples=args.n_samples)
        n = min(len(gt_full), len(recons))
        gt     = gt_full[:n]
        recons = recons[:n]
        report_pairing(str(exp_dir), args.gt_path, n)

        if args.show_grid:
            show_grid(gt, recons, title=exp_name)

        row = {"Experiment": exp_name}

        # PixCorr
        if "pixcorr" not in skip:
            print("  [1/7] PixCorr", end="  ", flush=True)
            row["PixCorr"] = eval_pixcorr(gt, recons, args.imsize)
            print(f"-> {row['PixCorr']:.4f}")

        # SSIM
        if "ssim" not in skip:
            print("  [2/7] SSIM", end="  ", flush=True)
            row["SSIM"] = eval_ssim(gt, recons, args.imsize)
            print(f"-> {row['SSIM']:.4f}")

        # AlexNet
        if "alexnet" not in skip:
            m, pre = models["alexnet"]
            print("  [3/7] AlexNet", end="  ", flush=True)
            rf_l2 = extract_features(recons, m, pre, "l2")
            rf_l5 = extract_features(recons, m, pre, "l5")
            row["AlexNet(2)"] = eval_two_way(gt_feats["alex_l2"][:n], rf_l2)
            row["AlexNet(5)"] = eval_two_way(gt_feats["alex_l5"][:n], rf_l5)
            print(f"-> L2={row['AlexNet(2)']:.4f}  L5={row['AlexNet(5)']:.4f}")

        # InceptionV3
        if "inception" not in skip:
            m, pre = models["inception"]
            print("  [4/7] InceptionV3", end="  ", flush=True)
            rf = extract_features(recons, m, pre, "avgpool")
            row["InceptionV3"] = eval_two_way(gt_feats["inception"][:n], rf)
            print(f"-> {row['InceptionV3']:.4f}")

        # CLIP
        if "clip" not in skip:
            m, pre = models["clip"]
            print("  [5/7] CLIP", end="  ", flush=True)
            rf = extract_features(recons, m, pre, None)
            row["CLIP"] = eval_two_way(gt_feats["clip"][:n], rf)
            print(f"-> {row['CLIP']:.4f}")

        # EfficientNet
        if "effnet" not in skip:
            m, pre = models["effnet"]
            print("  [6/7] EffNet-B", end="  ", flush=True)
            rf = extract_features(recons, m, pre, "avgpool")
            row["EffNet-B"] = eval_distance(gt_feats["effnet"][:n], rf)
            print(f"-> {row['EffNet-B']:.4f}")

        # SwAV
        if "swav" not in skip:
            m, pre = models["swav"]
            print("  [7/7] SwAV", end="  ", flush=True)
            rf = extract_features(recons, m, pre, "avgpool")
            row["SwAV"] = eval_distance(gt_feats["swav"][:n], rf)
            print(f"-> {row['SwAV']:.4f}")

        all_rows.append(row)

    # -------------------------------------------------- Summary table -------
    df = pd.DataFrame(all_rows).set_index("Experiment")

    print(f"\n\n{'='*65}")
    print("  Final Results")
    print(f"{'='*65}")
    print(df.to_string())
    print(f"{'='*65}\n")

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"Results saved -> {out.resolve()}")


if __name__ == "__main__":
    main()