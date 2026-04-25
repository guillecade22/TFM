import os
import glob
import argparse
import warnings
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import alexnet, inception_v3
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim_fn

warnings.filterwarnings("ignore")

# How many random distractors to draw per image for 2AFC scoring.
# 50 gives stable estimates; lower = faster but noisier.
NUM_DISTRACTORS = 50


# ──────────────────────────────────────────────────────────────────────────────
# Folder helpers
# ──────────────────────────────────────────────────────────────────────────────

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def collect_test_images(test_dir):
    """
    Flat folder: test_dir/00001_aircraft_carrier.jpg ...
    Returns sorted list of (class_name, path).
    """
    found = []
    for ext in IMG_EXTS:
        found.extend(glob.glob(os.path.join(test_dir, ext)))
    if not found:
        raise RuntimeError(f"No images found in test_dir: {test_dir}")
    found = sorted(found)
    result = []
    for path in found:
        fname = os.path.splitext(os.path.basename(path))[0]
        parts = fname.split("_", 1)
        cls = parts[1] if (len(parts) == 2 and parts[0].isdigit()) else fname
        result.append((cls, path))
    return result


def collect_gen_images(gen_dir):
    """
    Returns sorted list of reconstructed_XXXX.png paths.
    Falls back to all images if naming convention not found.
    """
    found = []
    for ext in IMG_EXTS:
        found.extend(glob.glob(os.path.join(gen_dir, ext)))
    recon = sorted([f for f in found if os.path.basename(f).startswith("reconstructed_")])
    return recon if recon else sorted(found)


def pair_images(test_dir, gen_dir):
    test_imgs = collect_test_images(test_dir)
    gen_imgs  = collect_gen_images(gen_dir)
    n = min(len(test_imgs), len(gen_imgs))
    if len(test_imgs) != len(gen_imgs):
        print(f"  [warn] test={len(test_imgs)}, gen={len(gen_imgs)}, using first {n}.")
    return [(test_imgs[i][0], test_imgs[i][1], gen_imgs[i]) for i in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# Low-level metrics  (direct pairwise comparison)
# ──────────────────────────────────────────────────────────────────────────────

def compute_pixcorr(a_np, b_np):
    r, _ = pearsonr(a_np.flatten(), b_np.flatten())
    return float(r)


def compute_ssim(a_np, b_np):
    return float(ssim_fn(a_np, b_np, data_range=1.0, channel_axis=-1))


# ──────────────────────────────────────────────────────────────────────────────
# LPIPS  (direct pairwise, lower = more similar)
# ──────────────────────────────────────────────────────────────────────────────

def load_lpips(device):
    try:
        import lpips
        fn = lpips.LPIPS(net="vgg").to(device).eval()
        return fn
    except ImportError:
        print("[LPIPS] Not installed. Run: pip install lpips")
        return None


def compute_lpips(fn, pil_a, pil_b, device):
    if fn is None:
        return float("nan")
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    a = tf(pil_a).unsqueeze(0).to(device)
    b = tf(pil_b).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(fn(a, b).item())


# ──────────────────────────────────────────────────────────────────────────────
# DreamSim  (direct pairwise, lower = more similar)
# ──────────────────────────────────────────────────────────────────────────────

def load_dreamsim(device):
    try:
        from dreamsim import dreamsim
        model, preprocess = dreamsim(pretrained=True, device=device)
        return model, preprocess
    except ImportError:
        print("[DreamSim] Not installed. Run: pip install dreamsim")
        return None, None


def compute_dreamsim(model, preprocess, pil_a, pil_b, device):
    if model is None:
        return float("nan")
    a = preprocess(pil_a).to(device)
    b = preprocess(pil_b).to(device)
    with torch.no_grad():
        return float(model(a, b).item())


# ──────────────────────────────────────────────────────────────────────────────
# Feature extractors for 2AFC metrics
# ──────────────────────────────────────────────────────────────────────────────

class AlexNetFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base  = alexnet(weights="DEFAULT")
        feats = base.features
        self.layer2 = torch.nn.Sequential(*list(feats.children())[:5])
        self.layer5 = torch.nn.Sequential(*list(feats.children())[:12])

    def forward(self, x):
        return self.layer2(x).flatten(1), self.layer5(x).flatten(1)


class InceptionFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = inception_v3(weights="DEFAULT", aux_logits=True)
        self.backbone = torch.nn.Sequential(
            base.Conv2d_1a_3x3, base.Conv2d_2a_3x3, base.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(3, 2), base.Conv2d_3b_1x1, base.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(3, 2), base.Mixed_5b, base.Mixed_5c, base.Mixed_5d,
            base.Mixed_6a, base.Mixed_6b, base.Mixed_6c, base.Mixed_6d,
            base.Mixed_6e, base.Mixed_7a, base.Mixed_7b, base.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.backbone(x).flatten(1)


def load_clip(device):
    try:
        import clip
        model, preprocess = clip.load("ViT-L/14", device=device)
        return model, preprocess, "openai"
    except ImportError:
        pass
    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        return model, proc, "hf"
    except ImportError:
        pass
    raise ImportError("CLIP not found. pip install openai-clip  OR  pip install transformers")


def clip_embed(model, preprocess, pil_img, device, backend):
    if backend == "openai":
        import clip
        x = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            return model.encode_image(x)
    else:
        inputs = preprocess(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            return model.get_image_features(**inputs)


def load_swav(device):
    try:
        model = torch.hub.load("facebookresearch/swav:main", "resnet50", verbose=False)
    except Exception:
        import torchvision.models as tvm
        print("[SwAV] torch.hub failed, falling back to supervised ResNet-50")
        model = tvm.resnet50(weights="DEFAULT")
    model.fc = torch.nn.Identity()
    return model.to(device).eval()


# Shared transforms
imagenet_tf = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

inception_tf = T.Compose([
    T.Resize(342, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

LOW_RES = (425, 425)


# ──────────────────────────────────────────────────────────────────────────────
# 2-way forced choice (2AFC) accuracy  — matches paper exactly
# ──────────────────────────────────────────────────────────────────────────────

def twoafc_score(feat_real, feat_gen, n_distractors=NUM_DISTRACTORS, rng=None):
    """
    Compute 2AFC identification accuracy.

    feat_real : Tensor [N, D]  features of all N real test images
    feat_gen  : Tensor [N, D]  features of all N generated images
                               feat_gen[i] is the generation for feat_real[i]
    n_distractors : int        random distractors per image (default 50)

    Returns float in [0.5, 1.0]:
        0.5 = chance level (model cannot distinguish), 1.0 = perfect

    This matches the evaluation protocol in the original paper (Tab. 1).
    For each generated image gen_i:
        - cos_sim(gen_i, real_i)        <- correct pair similarity
        - cos_sim(gen_i, real_j) j!=i   <- distractor similarity
        - score = mean(correct > distractor) over n_distractors draws
    Final score = mean over all i.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = feat_real.shape[0]
    feat_real = F.normalize(feat_real.float(), dim=-1)  # [N, D]
    feat_gen  = F.normalize(feat_gen.float(),  dim=-1)  # [N, D]

    # Full similarity matrix: sim[i, j] = cosine_sim(gen_i, real_j)
    sim_matrix = (feat_gen @ feat_real.T).cpu().numpy()   # [N, N]

    scores = []
    for i in range(N):
        distractor_pool = [j for j in range(N) if j != i]
        sampled = rng.choice(distractor_pool,
                             size=min(n_distractors, len(distractor_pool)),
                             replace=False)
        correct_sim    = sim_matrix[i, i]
        distractor_sim = sim_matrix[i, sampled]
        scores.append(float((correct_sim > distractor_sim).mean()))

    return float(np.mean(scores))


# ──────────────────────────────────────────────────────────────────────────────
# Precompute all high-level features for a list of image paths
# ──────────────────────────────────────────────────────────────────────────────

def extract_all_features(image_paths, models, device, desc=""):
    """
    Extract AlexNet2, AlexNet5, Inception, CLIP, SwAV features.
    Returns dict of {feat_name: Tensor[N, D]}
    """
    feats = {k: [] for k in ["alex2", "alex5", "incep", "clip", "swav"]}
    n = len(image_paths)

    for idx, path in enumerate(image_paths):
        if idx % 50 == 0:
            print(f"    Extracting {desc} features [{idx}/{n}]...")

        pil = Image.open(path).convert("RGB")

        x_alex = imagenet_tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            f2, f5 = models["alex"](x_alex)
        feats["alex2"].append(f2.cpu())
        feats["alex5"].append(f5.cpu())

        x_inc = inception_tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            fi = models["incep"](x_inc)
        feats["incep"].append(fi.cpu())

        fc = clip_embed(models["clip"], models["clip_prep"],
                        pil, device, models["clip_backend"])
        feats["clip"].append(fc.cpu())

        x_sw = imagenet_tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            fs = models["swav"](x_sw)
        feats["swav"].append(fs.cpu())

    return {k: torch.cat(v, dim=0) for k, v in feats.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Metric metadata
# ──────────────────────────────────────────────────────────────────────────────

METRIC_KEYS = [
    "PixCorr", "SSIM", "LPIPS", "DreamSim",
    "AlexNet2", "AlexNet5", "Inception", "CLIP", "SwAV"
]

METRIC_META = {
    "PixCorr":   ("↑", "Pearson pixel correlation (pairwise)"),
    "SSIM":      ("↑", "Structural similarity (pairwise)"),
    "LPIPS":     ("↓", "VGG perceptual patch distance (pairwise)"),
    "DreamSim":  ("↓", "Holistic perceptual sim CLIP+OpenCLIP+DINO (pairwise)"),
    "AlexNet2":  ("↑", "2AFC accuracy — AlexNet layer 2"),
    "AlexNet5":  ("↑", "2AFC accuracy — AlexNet layer 5"),
    "Inception": ("↑", "2AFC accuracy — Inception-v3 pool"),
    "CLIP":      ("↑", "2AFC accuracy — CLIP ViT-L/14"),
    "SwAV":      ("↓", "SwAV ResNet-50 avg correlation distance (pairwise)"),
}

# Paper reference values — THINGS-EEG Subject 8 (Tab. 1)
# AlexNet/Inception/CLIP/SwAV are 2AFC scores in the paper
PAPER_REF = {
    "PixCorr":   0.160,
    "SSIM":      0.345,
    "LPIPS":     float("nan"),
    "DreamSim":  float("nan"),
    "AlexNet2":  0.776,
    "AlexNet5":  0.866,
    "Inception": 0.734,
    "CLIP":      0.786,
    "SwAV":      0.582,
}


# ──────────────────────────────────────────────────────────────────────────────
# Single experiment evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_experiment(name, test_dir, gen_dir, models, device,
                        real_feats=None, verbose=True):
    """
    Evaluate one experiment folder.

    real_feats: pass precomputed test image features to avoid recomputing
                across experiments (test images are always the same).
                If None, computed here and returned for reuse.

    Returns (means_dict, stds_dict, n_pairs, real_feats)
    """
    pairs = pair_images(test_dir, gen_dir)
    if not pairs:
        raise RuntimeError(f"No image pairs found for experiment '{name}'.")

    n          = len(pairs)
    test_paths = [p[1] for p in pairs]
    gen_paths  = [p[2] for p in pairs]
    classes    = [p[0] for p in pairs]

    if verbose:
        print(f"\n{'─'*62}")
        print(f"  Experiment : {name}")
        print(f"  Gen dir    : {gen_dir}")
        print(f"  Pairs      : {n}")
        print(f"{'─'*62}")

    # ── Feature extraction ────────────────────────────────────────────────────
    if real_feats is None:
        print("  Extracting real image features (shared across all experiments)...")
        real_feats = extract_all_features(test_paths, models, device, desc="real")

    print("  Extracting generated image features...")
    gen_feats = extract_all_features(gen_paths, models, device, desc="gen")

    # ── 2AFC scores ───────────────────────────────────────────────────────────
    print(f"  Computing 2AFC scores ({NUM_DISTRACTORS} distractors per image)...")
    rng = np.random.default_rng(42)

    alex2_2afc  = twoafc_score(real_feats["alex2"], gen_feats["alex2"], rng=rng)
    alex5_2afc  = twoafc_score(real_feats["alex5"], gen_feats["alex5"], rng=rng)
    incep_2afc  = twoafc_score(real_feats["incep"], gen_feats["incep"], rng=rng)
    clip_2afc   = twoafc_score(real_feats["clip"],  gen_feats["clip"],  rng=rng)

    # SwAV: pairwise average correlation distance (NOT 2AFC) — matches paper
    # Lower is better ↓
    rf_sw = F.normalize(real_feats["swav"].float(), dim=-1)
    gf_sw = F.normalize(gen_feats["swav"].float(),  dim=-1)
    swav_dist_list = (1.0 - (rf_sw * gf_sw).sum(dim=-1)).tolist()
    swav_mean = float(np.mean(swav_dist_list))
    swav_std  = float(np.std(swav_dist_list))

    print(f"  2AFC  AlexNet2={alex2_2afc:.4f}  AlexNet5={alex5_2afc:.4f}  "
          f"Inception={incep_2afc:.4f}  CLIP={clip_2afc:.4f}")
    print(f"  SwAV dist (pairwise ↓) = {swav_mean:.4f}")

    # ── Pairwise low-level + LPIPS + DreamSim ─────────────────────────────────
    print("  Computing pairwise low-level metrics...")
    pixcorr_list  = []
    ssim_list     = []
    lpips_list    = []
    dreamsim_list = []

    for i, (cls, tpath, gpath) in enumerate(pairs):
        real_pil = Image.open(tpath).convert("RGB")
        gen_pil  = Image.open(gpath).convert("RGB")

        real_np = np.array(real_pil.resize(LOW_RES, Image.BICUBIC)).astype(np.float32) / 255.
        gen_np  = np.array(gen_pil.resize(LOW_RES,  Image.BICUBIC)).astype(np.float32) / 255.
        pixcorr_list.append(compute_pixcorr(real_np, gen_np))
        ssim_list.append(compute_ssim(real_np, gen_np))
        lpips_list.append(compute_lpips(models["lpips_fn"], real_pil, gen_pil, device))
        dreamsim_list.append(
            compute_dreamsim(models["dreamsim_model"], models["dreamsim_prep"],
                             real_pil, gen_pil, device)
        )

        if verbose and i % 20 == 0:
            print(f"  [{i:03d}/{n}] {cls:<22} "
                  f"PixCorr={pixcorr_list[-1]:.4f}  "
                  f"SSIM={ssim_list[-1]:.4f}  "
                  f"LPIPS={lpips_list[-1]:.4f}  "
                  f"DreamSim={dreamsim_list[-1]:.4f}")

    means = {
        "PixCorr":   float(np.nanmean(pixcorr_list)),
        "SSIM":      float(np.nanmean(ssim_list)),
        "LPIPS":     float(np.nanmean(lpips_list)),
        "DreamSim":  float(np.nanmean(dreamsim_list)),
        "AlexNet2":  alex2_2afc,
        "AlexNet5":  alex5_2afc,
        "Inception": incep_2afc,
        "CLIP":      clip_2afc,
        "SwAV":      swav_mean,
    }
    # std only meaningful for pairwise metrics; 2AFC is one scalar per experiment
    stds = {
        "PixCorr":   float(np.nanstd(pixcorr_list)),
        "SSIM":      float(np.nanstd(ssim_list)),
        "LPIPS":     float(np.nanstd(lpips_list)),
        "DreamSim":  float(np.nanstd(dreamsim_list)),
        "AlexNet2":  float("nan"),
        "AlexNet5":  float("nan"),
        "Inception": float("nan"),
        "CLIP":      float("nan"),
        "SwAV":      swav_std,
    }

    return means, stds, n, real_feats


# ──────────────────────────────────────────────────────────────────────────────
# Summary printing
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(all_results, paper_ref=PAPER_REF):
    exp_names = list(all_results.keys())
    col_w = max(14, max(len(n) for n in exp_names) + 2)
    sep = "═" * (22 + col_w * (len(exp_names) + 1))

    print(f"\n{sep}")
    print("RESULTS SUMMARY")
    print("High-level metrics = 2AFC accuracy [0.5=chance, 1.0=perfect] — comparable to paper")
    print("Low-level / perceptual metrics = pairwise vs ground truth")
    print(sep)

    header = f"  {'Metric':<13}  {'Dir'}"
    for name in exp_names:
        header += f"  {name:>{col_w}}"
    header += f"  {'Paper':>{col_w}}"
    print(header)
    print("─" * len(header))

    for key in METRIC_KEYS:
        direction, _ = METRIC_META[key]
        row = f"  {key:<13}  {direction} "
        for name in exp_names:
            means, _, _ = all_results[name]
            v = means[key]
            row += f"  {v:>{col_w}.4f}" if not np.isnan(v) else f"  {'nan':>{col_w}}"
        ref = paper_ref.get(key, float("nan"))
        row += f"  {'—':>{col_w}}" if np.isnan(ref) else f"  {ref:>{col_w}.4f}"
        print(row)

    print(sep)
    print("  ↑ higher is better   ↓ lower is better   — not reported in paper")

    print("\nBest experiment per metric:")
    for key in METRIC_KEYS:
        direction, _ = METRIC_META[key]
        scores = {n: all_results[n][0][key] for n in exp_names
                  if not np.isnan(all_results[n][0][key])}
        if not scores:
            continue
        best = max(scores, key=scores.get) if direction == "↑" else min(scores, key=scores.get)
        ref  = paper_ref.get(key, float("nan"))
        delta_str = ""
        if not np.isnan(ref):
            delta = scores[best] - ref
            delta_str = f"  (vs paper: {'+' if delta >= 0 else ''}{delta:.4f})"
        print(f"  {key:<13}  {direction}  →  {best:<20}  {scores[best]:.4f}{delta_str}")


# ──────────────────────────────────────────────────────────────────────────────
# CSV export
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(all_results, out_path):
    import csv
    exp_names = list(all_results.keys())
    rows = []
    for key in METRIC_KEYS:
        direction, desc = METRIC_META[key]
        row = {"metric": key, "direction": direction, "description": desc,
               "paper": f"{PAPER_REF.get(key, float('nan')):.4f}"}
        for name in exp_names:
            means, stds, _ = all_results[name]
            row[f"{name}_mean"] = f"{means[key]:.4f}" if not np.isnan(means[key]) else "nan"
            row[f"{name}_std"]  = f"{stds[key]:.4f}"  if not np.isnan(stds[key])  else "nan"
        rows.append(row)

    fieldnames = ["metric", "direction", "description"]
    for name in exp_names:
        fieldnames += [f"{name}_mean", f"{name}_std"]
    fieldnames += ["paper"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EEG reconstruction — 2AFC + pairwise metrics."
    )
    parser.add_argument(
        "--test_dir",
        default="/hhome/ricse01/TFM/TFM/required/test",
        help="Flat folder of test images."
    )
    parser.add_argument(
        "--gen_dirs", nargs="+",
        default=[
            "/hhome/ricse01/TFM/TFM/required/exp_notext",
            "/hhome/ricse01/TFM/TFM/required/exp_simple_caption",
            "/hhome/ricse01/TFM/TFM/required/exp_img2img",
            "/hhome/ricse01/TFM/TFM/required/exp_img2img_caption",
            "/hhome/ricse01/TFM/TFM/required/generated_gt_caption",
            "/hhome/ricse01/TFM/TFM/required/balanced",
            "/hhome/ricse01/TFM/TFM/required/more_Image",
        ],
        help="Experiment output folders, each with reconstructed_XXXX.png"
    )
    parser.add_argument(
        "--names", nargs="+",
        default=[
            "notext",
            "simple_caption",
            "img2img",
            "img2img_caption",
            "gt_caption",
            "balanced",
            "more_image",
        ],
        help="Experiment names (same order as --gen_dirs)."
    )
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument(
        "--out",
        default="/hhome/ricse01/TFM/TFM/eval_results.csv",
        help="CSV output path."
    )
    args = parser.parse_args()

    gen_dirs = args.gen_dirs
    names    = args.names if args.names else [os.path.basename(d.rstrip("/")) for d in gen_dirs]
    if len(names) != len(gen_dirs):
        parser.error("--names must have same length as --gen_dirs")

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    print(f"Device: {device}")

    # ── Load all models once ──────────────────────────────────────────────────
    print("\nLoading models...")
    alex  = AlexNetFeatures().to(device).eval()
    incep = InceptionFeatures().to(device).eval()
    swav  = load_swav(device)
    clip_model, clip_prep, clip_backend = load_clip(device)
    lpips_fn                            = load_lpips(device)
    dreamsim_model, dreamsim_prep       = load_dreamsim(device)

    models = {
        "alex":          alex,
        "incep":         incep,
        "swav":          swav,
        "clip":          clip_model,
        "clip_prep":     clip_prep,
        "clip_backend":  clip_backend,
        "lpips_fn":      lpips_fn,
        "dreamsim_model":dreamsim_model,
        "dreamsim_prep": dreamsim_prep,
    }
    print("All models loaded.")

    # ── Evaluate all experiments ──────────────────────────────────────────────
    # real_feats is computed once and reused — test images don't change
    all_results = {}
    real_feats  = None

    for name, gen_dir in zip(names, gen_dirs):
        means, stds, n, real_feats = evaluate_experiment(
            name, args.test_dir, gen_dir, models, device,
            real_feats=real_feats, verbose=True
        )
        all_results[name] = (means, stds, n)
        print(f"\n  ✓ {name} done ({n} pairs).")

    print_summary(all_results)

    if args.out:
        save_csv(all_results, args.out)


if __name__ == "__main__":
    main()