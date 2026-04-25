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

# ──────────────────────────────────────────────────────────────────────────────
# Folder helpers
# ──────────────────────────────────────────────────────────────────────────────

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def collect_test_images(test_dir):
    """
    Collect all images directly inside test_dir (flat folder).
    Returns sorted list of (class_name, path).

    Naming: "00001_aircraft_carrier.jpg" → class = "aircraft_carrier"
            "wok.jpg"                    → class = "wok"
    """
    found = []
    for ext in IMG_EXTS:
        found.extend(glob.glob(os.path.join(test_dir, ext)))

    if not found:
        raise RuntimeError(
            f"No images found in test_dir: {test_dir}\n"
            "Expected flat folder: test_dir/00001_aircraft_carrier.jpg ..."
        )

    found = sorted(found)
    result = []
    for path in found:
        fname = os.path.splitext(os.path.basename(path))[0]
        parts = fname.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            cls = parts[1]
        else:
            cls = fname
        result.append((cls, path))

    return result  # [(class_name, path), ...]


def collect_gen_images(gen_dir):
    """
    Collect all reconstructed_XXXX.png images from gen_dir.
    Returns sorted list of paths.
    """
    found = []
    for ext in IMG_EXTS:
        found.extend(glob.glob(os.path.join(gen_dir, ext)))

    # filter to reconstructed_XXXX pattern if present, else take all
    recon = sorted([f for f in found if os.path.basename(f).startswith("reconstructed_")])
    if recon:
        return recon

    # fallback: all images sorted
    return sorted(found)


def pair_images(test_dir, gen_dir):
    """
    Pair test and generated images by sorted index.
    Returns list of (class_name, test_path, gen_path).
    """
    test_imgs = collect_test_images(test_dir)
    gen_imgs  = collect_gen_images(gen_dir)

    n = min(len(test_imgs), len(gen_imgs))
    if len(test_imgs) != len(gen_imgs):
        print(f"  [warn] test has {len(test_imgs)} images, gen has {len(gen_imgs)}."
              f" Using first {n}.")

    pairs = []
    for i in range(n):
        cls, tpath = test_imgs[i]
        gpath = gen_imgs[i]
        pairs.append((cls, tpath, gpath))

    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Low-level metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_pixcorr(a_np, b_np):
    r, _ = pearsonr(a_np.flatten(), b_np.flatten())
    return float(r)


def compute_ssim(a_np, b_np):
    return float(ssim_fn(a_np, b_np, data_range=1.0, channel_axis=-1))


# ──────────────────────────────────────────────────────────────────────────────
# LPIPS
# ──────────────────────────────────────────────────────────────────────────────

def load_lpips(device):
    try:
        import lpips
        loss_fn = lpips.LPIPS(net="vgg").to(device)
        loss_fn.eval()
        return loss_fn
    except ImportError:
        print("[LPIPS] Not installed. Run: pip install lpips")
        return None


def compute_lpips(loss_fn, pil_a, pil_b, device):
    """Returns LPIPS distance ↓ (lower = more similar)."""
    if loss_fn is None:
        return float("nan")
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # LPIPS expects [-1, 1]
    ])
    a = tf(pil_a).unsqueeze(0).to(device)
    b = tf(pil_b).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = loss_fn(a, b)
    return float(dist.item())


# ──────────────────────────────────────────────────────────────────────────────
# DreamSim
# ──────────────────────────────────────────────────────────────────────────────

def load_dreamsim(device):
    """
    DreamSim: NeurIPS 2023 Spotlight.
    Bridges low-level (SSIM/LPIPS) and high-level (CLIP) metrics.
    Trained on 20k human perceptual triplets using CLIP+OpenCLIP+DINO ensemble.
    Returns lower values for more similar images ↓.
    """
    try:
        from dreamsim import dreamsim
        model, preprocess = dreamsim(pretrained=True, device=device)
        return model, preprocess
    except ImportError:
        print("[DreamSim] Not installed. Run: pip install dreamsim")
        return None, None


def compute_dreamsim(model, preprocess, pil_a, pil_b, device):
    """Returns DreamSim distance ↓ (lower = more similar)."""
    if model is None:
        return float("nan")
    a = preprocess(pil_a).to(device)
    b = preprocess(pil_b).to(device)
    with torch.no_grad():
        dist = model(a, b)
    return float(dist.item())


# ──────────────────────────────────────────────────────────────────────────────
# AlexNet
# ──────────────────────────────────────────────────────────────────────────────

class AlexNetFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base  = alexnet(weights="DEFAULT")
        feats = base.features
        self.layer2 = torch.nn.Sequential(*list(feats.children())[:5])
        self.layer5 = torch.nn.Sequential(*list(feats.children())[:12])

    def forward(self, x):
        f2 = self.layer2(x)
        f5 = self.layer5(x)
        return f2.flatten(1), f5.flatten(1)


# ──────────────────────────────────────────────────────────────────────────────
# Inception
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# CLIP
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# SwAV
# ──────────────────────────────────────────────────────────────────────────────

def load_swav(device):
    try:
        model = torch.hub.load("facebookresearch/swav:main", "resnet50", verbose=False)
    except Exception:
        import torchvision.models as tvm
        print("[SwAV] torch.hub failed, falling back to supervised ResNet-50")
        model = tvm.resnet50(weights="DEFAULT")
    model.fc = torch.nn.Identity()
    return model.to(device).eval()


# ──────────────────────────────────────────────────────────────────────────────
# Shared transforms
# ──────────────────────────────────────────────────────────────────────────────

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


def cosine_sim(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1).mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Single experiment evaluation
# ──────────────────────────────────────────────────────────────────────────────

METRIC_KEYS = ["PixCorr", "SSIM", "LPIPS", "DreamSim",
               "AlexNet2", "AlexNet5", "Inception", "CLIP", "SwAV"]

METRIC_META = {
    # key: (direction, description)
    "PixCorr":  ("↑", "Pearson pixel correlation"),
    "SSIM":     ("↑", "Structural similarity"),
    "LPIPS":    ("↓", "Perceptual patch similarity (VGG)"),
    "DreamSim": ("↓", "Holistic perceptual similarity (CLIP+OpenCLIP+DINO)"),
    "AlexNet2": ("↑", "AlexNet layer-2 cosine sim"),
    "AlexNet5": ("↑", "AlexNet layer-5 cosine sim"),
    "Inception":("↑", "Inception-v3 pool cosine sim"),
    "CLIP":     ("↑", "CLIP ViT-L/14 cosine sim"),
    "SwAV":     ("↓", "SwAV ResNet-50 cosine distance"),
}

# Paper reference values (THINGS-EEG, Subject 8)
PAPER_REF = {
    "PixCorr":   0.160,
    "SSIM":      0.345,
    "LPIPS":     float("nan"),   # not reported in original paper
    "DreamSim":  float("nan"),   # not reported in original paper
    "AlexNet2":  0.776,
    "AlexNet5":  0.866,
    "Inception": 0.734,
    "CLIP":      0.786,
    "SwAV":      0.582,
}


def evaluate_experiment(name, test_dir, gen_dir, models, device, verbose=True):
    """
    Evaluate one (test_dir, gen_dir) pair.
    models dict must contain keys: alex, incep, swav, clip, clip_prep,
                                   clip_backend, lpips_fn, dreamsim_model,
                                   dreamsim_prep
    Returns dict {metric_key: mean_value}
    """
    pairs = pair_images(test_dir, gen_dir)
    if not pairs:
        raise RuntimeError(f"No image pairs found for experiment '{name}'.")

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Experiment : {name}")
        print(f"  Gen dir    : {gen_dir}")
        print(f"  Pairs      : {len(pairs)}")
        print(f"{'─'*60}")

    accum = {k: [] for k in METRIC_KEYS}

    for i, (cls, tpath, gpath) in enumerate(pairs):
        real_pil = Image.open(tpath).convert("RGB")
        gen_pil  = Image.open(gpath).convert("RGB")

        # ── Low-level ──────────────────────────────────────────────────────
        real_np = np.array(real_pil.resize(LOW_RES, Image.BICUBIC)).astype(np.float32) / 255.
        gen_np  = np.array(gen_pil.resize(LOW_RES,  Image.BICUBIC)).astype(np.float32) / 255.
        accum["PixCorr"].append(compute_pixcorr(real_np, gen_np))
        accum["SSIM"].append(compute_ssim(real_np, gen_np))

        # ── LPIPS ──────────────────────────────────────────────────────────
        accum["LPIPS"].append(
            compute_lpips(models["lpips_fn"], real_pil, gen_pil, device)
        )

        # ── DreamSim ───────────────────────────────────────────────────────
        accum["DreamSim"].append(
            compute_dreamsim(models["dreamsim_model"], models["dreamsim_prep"],
                             real_pil, gen_pil, device)
        )

        # ── AlexNet ────────────────────────────────────────────────────────
        r_alex = imagenet_tf(real_pil).unsqueeze(0).to(device)
        g_alex = imagenet_tf(gen_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            rf2, rf5 = models["alex"](r_alex)
            gf2, gf5 = models["alex"](g_alex)
        accum["AlexNet2"].append(cosine_sim(rf2, gf2))
        accum["AlexNet5"].append(cosine_sim(rf5, gf5))

        # ── Inception ──────────────────────────────────────────────────────
        r_inc = inception_tf(real_pil).unsqueeze(0).to(device)
        g_inc = inception_tf(gen_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            ri = models["incep"](r_inc)
            gi = models["incep"](g_inc)
        accum["Inception"].append(cosine_sim(ri, gi))

        # ── CLIP ───────────────────────────────────────────────────────────
        rf_clip = clip_embed(models["clip"], models["clip_prep"],
                             real_pil, device, models["clip_backend"])
        gf_clip = clip_embed(models["clip"], models["clip_prep"],
                             gen_pil,  device, models["clip_backend"])
        accum["CLIP"].append(cosine_sim(rf_clip, gf_clip))

        # ── SwAV ───────────────────────────────────────────────────────────
        r_sw = imagenet_tf(real_pil).unsqueeze(0).to(device)
        g_sw = imagenet_tf(gen_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            rs = models["swav"](r_sw)
            gs = models["swav"](g_sw)
        accum["SwAV"].append(1.0 - cosine_sim(rs, gs))

        if verbose and i % 20 == 0:
            print(f"  [{i:03d}/{len(pairs)}] {cls:<22} "
                  f"PixCorr={accum['PixCorr'][-1]:.4f}  "
                  f"SSIM={accum['SSIM'][-1]:.4f}  "
                  f"LPIPS={accum['LPIPS'][-1]:.4f}  "
                  f"DreamSim={accum['DreamSim'][-1]:.4f}  "
                  f"CLIP={accum['CLIP'][-1]:.4f}")

    means = {k: float(np.nanmean(accum[k])) for k in METRIC_KEYS}
    stds  = {k: float(np.nanstd(accum[k]))  for k in METRIC_KEYS}
    return means, stds, len(pairs)


# ──────────────────────────────────────────────────────────────────────────────
# Summary printing
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(all_results, paper_ref=PAPER_REF):
    """
    all_results: dict { exp_name: (means_dict, stds_dict, n_pairs) }
    """
    exp_names = list(all_results.keys())
    col_w = max(14, max(len(n) for n in exp_names) + 2)

    print("\n" + "═" * (20 + col_w * len(exp_names)))
    print("RESULTS SUMMARY")
    print("═" * (20 + col_w * len(exp_names)))

    # Header
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
            means, stds, _ = all_results[name]
            row += f"  {means[key]:>{col_w}.4f}"
        ref = paper_ref.get(key, float("nan"))
        if np.isnan(ref):
            row += f"  {'—':>{col_w}}"
        else:
            row += f"  {ref:>{col_w}.4f}"
        print(row)

    print("═" * (20 + col_w * len(exp_names)))
    print("  ↑ = higher is better   ↓ = lower is better   — = not in paper")

    # Best experiment per metric
    print("\nBest experiment per metric:")
    for key in METRIC_KEYS:
        direction, _ = METRIC_META[key]
        scores = {n: all_results[n][0][key] for n in exp_names
                  if not np.isnan(all_results[n][0][key])}
        if not scores:
            continue
        if direction == "↑":
            best = max(scores, key=scores.get)
        else:
            best = min(scores, key=scores.get)
        print(f"  {key:<13}  {direction}  →  {best}  ({scores[best]:.4f})")


# ──────────────────────────────────────────────────────────────────────────────
# CSV export
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(all_results, out_path):
    import csv
    exp_names = list(all_results.keys())
    rows = []
    for key in METRIC_KEYS:
        direction, desc = METRIC_META[key]
        row = {"metric": key, "direction": direction, "description": desc}
        for name in exp_names:
            means, stds, _ = all_results[name]
            row[f"{name}_mean"] = f"{means[key]:.4f}"
            row[f"{name}_std"]  = f"{stds[key]:.4f}"
        rows.append(row)

    fieldnames = ["metric", "direction", "description"]
    for name in exp_names:
        fieldnames += [f"{name}_mean", f"{name}_std"]

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
        description="Evaluate EEG reconstruction experiments with extended metrics."
    )
    parser.add_argument(
        "--test_dir", 
        default="/hhome/ricse01/TFM/TFM/required/test",
        help="Flat folder of test images: test_dir/00001_aircraft_carrier.jpg ..."
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
            "/hhome/ricse01/TFM/TFM/required/more_image",
        ],
        help="One or more experiment output folders, each containing reconstructed_XXXX.png"
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
    parser.add_argument(
        "--device", default="auto",
        help="auto | cuda | cpu"
    )
    parser.add_argument(
        "--out",
        default="/hhome/ricse01/TFM/TFM/eval_results.csv",
        help="CSV output path for results table."
    )
    args = parser.parse_args()

    # ── resolve names ─────────────────────────────────────────────────────────
    gen_dirs = args.gen_dirs
    names    = args.names if args.names else [os.path.basename(d.rstrip("/")) for d in gen_dirs]
    if len(names) != len(gen_dirs):
        parser.error("--names must have same length as --gen_dirs")

    # ── device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── load models (once, shared across all experiments) ─────────────────────
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
    print("All models loaded.\n")

    # ── evaluate each experiment ───────────────────────────────────────────────
    all_results = {}
    for name, gen_dir in zip(names, gen_dirs):
        means, stds, n = evaluate_experiment(
            name, args.test_dir, gen_dir, models, device, verbose=True
        )
        all_results[name] = (means, stds, n)
        print(f"\n  ✓ {name} done ({n} pairs).")

    # ── summary ───────────────────────────────────────────────────────────────
    print_summary(all_results)

    # ── CSV ───────────────────────────────────────────────────────────────────
    if args.out:
        save_csv(all_results, args.out)


if __name__ == "__main__":
    main()