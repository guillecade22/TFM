import os
import argparse
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import alexnet, inception_v3
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim_fn

# -- optional: suppress noisy warnings ----------------------------------------
import warnings
warnings.filterwarnings("ignore")


#------------------------------------------------------------
# Helpers
#------------------------------------------------------------

def load_image_pil(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.BICUBIC)
    return img


def pil_to_np(img):
    return np.array(img).astype(np.float32) / 255.0


def extract_class_name(subfolder_name):
    """
    Strip leading numeric prefix from a test subfolder name to get class name.
      "00001_aircraft_carrier" -> "aircraft_carrier"
      "wok"                    -> "wok"
    """
    parts = subfolder_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return subfolder_name


def collect_pairs(test_dir, gen_dir):
    """
    Match test images to generated images by class name.
    Returns list of (class_name, test_path, gen_path) sorted by class name.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

    # test side: strip numeric prefix to get class name
    test_map = {}
    for sf in os.listdir(test_dir):
        if not os.path.isdir(os.path.join(test_dir, sf)):
            continue
        cls = extract_class_name(sf)
        found = []
        for ext in exts:
            found.extend(glob.glob(os.path.join(test_dir, sf, ext)))
        if found:
            test_map[cls] = sorted(found)[0]

    # gen side: subfolder is already the plain class name, collect ALL images (0.jpg..9.jpg)
    gen_map = {}
    for sf in os.listdir(gen_dir):
        if not os.path.isdir(os.path.join(gen_dir, sf)):
            continue
        cls = sf
        found = []
        for ext in exts:
            found.extend(glob.glob(os.path.join(gen_dir, sf, ext)))
        if found:
            gen_map[cls] = sorted(found)  # list of all images

    common    = sorted(set(test_map) & set(gen_map))
    only_test = sorted(set(test_map) - set(gen_map))
    only_gen  = sorted(set(gen_map)  - set(test_map))

    if only_test:
        print(f"[warn] {len(only_test)} test class(es) skipped (no matching gen): "
              f"{only_test[:5]}{'...' if len(only_test) > 5 else ''}")
    if only_gen:
        print(f"[warn] {len(only_gen)} gen class(es) skipped (no matching test): "
              f"{only_gen[:5]}{'...' if len(only_gen) > 5 else ''}")

    # returns (class_name, test_image_path, [gen_image_path_0, ..., gen_image_path_9])
    return [(cls, test_map[cls], gen_map[cls]) for cls in common]


#------------------------------------------------------------
# Low-level metrics
#------------------------------------------------------------

def compute_pixcorr(img_real_np, img_gen_np):
    """Pearson correlation between flattened pixel arrays."""
    r, _ = pearsonr(img_real_np.flatten(), img_gen_np.flatten())
    return float(r)


def compute_ssim(img_real_np, img_gen_np):
    """SSIM across RGB channels (channel_axis=-1)."""
    score = ssim_fn(
        img_real_np, img_gen_np,
        data_range=1.0,
        channel_axis=-1
    )
    return float(score)


#------------------------------------------------------------
# High-level metrics - AlexNet layer features
#------------------------------------------------------------

class AlexNetFeatures(torch.nn.Module):
    """Extract features after ReLU of conv layers 2 and 5 (0-indexed in features)."""
    # AlexNet .features indices:
    #   0  Conv1  1  ReLU1  2  MaxPool1
    #   3  Conv2  4  ReLU2  5  MaxPool2   <-- layer 2 in paper (index 4 relu)
    #   6  Conv3  7  ReLU3
    #   8  Conv4  9  ReLU4
    #  10  Conv5 11  ReLU5 12  MaxPool2   <-- layer 5 in paper (index 11 relu)
    def __init__(self):
        super().__init__()
        base = alexnet(weights="DEFAULT")
        feats = base.features
        self.layer2 = torch.nn.Sequential(*list(feats.children())[:5])   # up to ReLU2
        self.layer5 = torch.nn.Sequential(*list(feats.children())[:12])  # up to ReLU5

    def forward(self, x):
        f2 = self.layer2(x)
        f5 = self.layer5(x)
        return f2.flatten(1), f5.flatten(1)


def cosine_sim(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1).mean().item()


#------------------------------------------------------------
# High-level metrics - Inception
#------------------------------------------------------------

class InceptionFeatures(torch.nn.Module):
    """Pool features before the final FC (same as FID feature extraction)."""
    def __init__(self):
        super().__init__()
        base = inception_v3(weights="DEFAULT", aux_logits=True)
        # strip classifier, keep up to adaptive pool
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


#------------------------------------------------------------
# High-level metrics - CLIP
#------------------------------------------------------------

def load_clip_model(device):
    try:
        import clip
        model, preprocess = clip.load("ViT-L/14", device=device)
        return model, preprocess, "openai_clip"
    except ImportError:
        pass
    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        return model, processor, "hf_clip"
    except ImportError:
        pass
    raise ImportError(
        "CLIP not found. Install with:\n"
        "  pip install openai-clip\n"
        "or:\n"
        "  pip install transformers"
    )


def clip_features(model, preprocess, pil_img, device, backend):
    if backend == "openai_clip":
        import clip
        x = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model.encode_image(x)
        return feats
    else:  # hf_clip
        inputs = preprocess(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        return feats


#------------------------------------------------------------
# High-level metrics - SwAV
#------------------------------------------------------------

def load_swav(device):
    try:
        model = torch.hub.load("facebookresearch/swav:main", "resnet50", verbose=False)
    except Exception:
        import torchvision.models as tvm
        print("[SwAV] torch.hub failed, falling back to supervised ResNet-50")
        model = tvm.resnet50(weights="DEFAULT")
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    return model


#------------------------------------------------------------
# Main evaluation
#------------------------------------------------------------

def evaluate(test_dir, gen_dir, device_str="auto"):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )
    print(f"Device: {device}")

    # -- collect & match pairs by class name ----------------------------------
    pairs = collect_pairs(test_dir, gen_dir)
    if not pairs:
        raise RuntimeError(
            "No matching class names found between test_dir and gen_dir. "
            "Check that gen subfolder names match the class part of test subfolder names."
        )
    n_imgs = len(pairs)
    print(f"\nMatched {n_imgs} image pairs:")
    for cls, tp, gps in pairs:
        print(f"  {cls}  ({len(gps)} generated images)")
        print(f"    test: {tp}")
        print(f"     gen: {gps[0]} ... {gps[-1]}")

    # -- build models ---------------------------------------------------------
    print("\nLoading models...")

    alex = AlexNetFeatures().to(device).eval()
    incep = InceptionFeatures().to(device).eval()
    swav  = load_swav(device)
    clip_model, clip_prep, clip_backend = load_clip_model(device)

    # shared ImageNet normalisation for AlexNet / Inception / SwAV
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

    # -- per-image metrics ----------------------------------------------------
    results = {
        "PixCorr": [], "SSIM": [],
        "AlexNet2": [], "AlexNet5": [],
        "Inception": [], "CLIP": [], "SwAV": []
    }

    for i, (cls, tp, gps) in enumerate(pairs):
        real_pil = load_image_pil(tp)

        # Pre-compute real image features once per class (reused across all gen images)
        sz = (425, 425)
        real_np = pil_to_np(real_pil.resize(sz, Image.BICUBIC))

        r_alex = imagenet_tf(real_pil).unsqueeze(0).to(device)
        r_inc  = inception_tf(real_pil).unsqueeze(0).to(device)
        r_sw   = imagenet_tf(real_pil).unsqueeze(0).to(device)
        rf_clip = clip_features(clip_model, clip_prep, real_pil, device, clip_backend)

        with torch.no_grad():
            rf2, rf5 = alex(r_alex)
            ri = incep(r_inc)
            rs = swav(r_sw)

        # Accumulate metrics over all generated images for this class
        class_scores = {k: [] for k in results}

        for gp in gps:
            gen_pil = load_image_pil(gp)

            # -- low-level ----------------------------------------------------
            gen_np = pil_to_np(gen_pil.resize(sz, Image.BICUBIC))
            class_scores["PixCorr"].append(compute_pixcorr(real_np, gen_np))
            class_scores["SSIM"].append(compute_ssim(real_np, gen_np))

            # -- AlexNet features ---------------------------------------------
            g_alex = imagenet_tf(gen_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                gf2, gf5 = alex(g_alex)
            class_scores["AlexNet2"].append(cosine_sim(rf2, gf2))
            class_scores["AlexNet5"].append(cosine_sim(rf5, gf5))

            # -- Inception features -------------------------------------------
            g_inc = inception_tf(gen_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                gi = incep(g_inc)
            class_scores["Inception"].append(cosine_sim(ri, gi))

            # -- CLIP features ------------------------------------------------
            gf_clip = clip_features(clip_model, clip_prep, gen_pil, device, clip_backend)
            class_scores["CLIP"].append(cosine_sim(rf_clip, gf_clip))

            # -- SwAV features ------------------------------------------------
            g_sw = imagenet_tf(gen_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                gs = swav(g_sw)
            class_scores["SwAV"].append(1.0 - cosine_sim(rs, gs))

        # Average across all generated images for this class
        for k in results:
            results[k].append(float(np.mean(class_scores[k])))

        print(f"  [{i:02d}] {cls:<25} "
              f"PixCorr={results['PixCorr'][-1]:.4f}  "
              f"SSIM={results['SSIM'][-1]:.4f}  "
              f"AlexNet2={results['AlexNet2'][-1]:.4f}  "
              f"CLIP={results['CLIP'][-1]:.4f}  "
              f"(avg over {len(gps)} images)")

    # -- summary --------------------------------------------------------------
    print("\n" + "="*62)
    print(f"{'Metric':<15}  {'Mean':>8}  {'Std':>8}  {'Note'}")
    print("="*62)
    meta = [
        ("PixCorr",   "(higher better) higher better"),
        ("SSIM",      "(higher better) higher better"),
        ("AlexNet2",  "(higher better) higher better"),
        ("AlexNet5",  "(higher better) higher better"),
        ("Inception", "(higher better) higher better"),
        ("CLIP",      "(higher better) higher better"),
        ("SwAV",      "(lower better) lower better  (distance)"),
    ]
    means = {}
    for key, note in meta:
        vals = results[key]
        m, s = np.mean(vals), np.std(vals)
        means[key] = m
        print(f"  {key:<13}  {m:>8.4f}  {s:>8.4f}  {note}")
    print("="*62)

    # -- comparison with paper Table 1 (THINGS-EEG row) -----------------------
    paper = {
        "PixCorr": 0.160, "SSIM": 0.345,
        "AlexNet2": 0.776, "AlexNet5": 0.866,
        "Inception": 0.734, "CLIP": 0.786, "SwAV": 0.582
    }
    print("\nComparison with paper THINGS-EEG (Subject 8):")
    print(f"{'Metric':<15}  {'Yours':>8}  {'Paper':>8}  {'Delta':>8}")
    print("-"*50)
    for key, note in meta:
        delta = means[key] - paper[key]
        arrow = "^" if delta > 0 else "v"
        print(f"  {key:<13}  {means[key]:>8.4f}  {paper[key]:>8.4f}  {arrow}{abs(delta):.4f}")

    return results, means


#------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute reconstruction quality metrics.")
    parser.add_argument("--test_dir", default="/hhome/ricse01/TFM/required/test_images/",
                        help="Root folder containing one subfolder per test image.")
    parser.add_argument("--gen_dir",  default="/hhome/ricse01/TFM/EEG_Image_decode/Generation/generated_sub-08_only_diff/",
                        help="Folder with class subfolders each containing 0.jpg.")
    parser.add_argument("--device",   default="auto",
                        help="auto, cuda, or cpu.")
    args = parser.parse_args()

    evaluate(args.test_dir, args.gen_dir, args.device)