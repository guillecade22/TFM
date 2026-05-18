# =============================================================================
# EEG-to-Image Reconstruction Pipeline
# =============================================================================
# Stages:
#   1. Class Retrieval       — eeg_embed @ img_features.T -> top-N classes
#   2. Diffusion Prior       — EEG embedding -> CLIP prior h
#   3. Candidate Generation  — SDXL + IP-Adapter, one image per top-N class
#   4. Re-Ranking            — weighted vote:
#
#       score_retrieval = (cosine_sim(eeg_embed, img_features[class_idx]) + 1) / 2
#       score_fidelity  = 1 - min LPIPS(candidate, class_reference_images)
#                         lower LPIPS = more perceptually similar = higher score
#       final_score     = W_RETRIEVAL * score_retrieval + W_FIDELITY * score_fidelity
#
# Scores saved to candidate_K_scores.json for use with weight_search.py.

# --- CONFIG -------------------------------------------------------------------

VIT_H_14_FEATURES_TEST    = "/hhome/ricse01/TFM/required/ViT-H-14_features_test.pt"
ATM_S_EEG_FEATURES_SUB_08 = "/hhome/ricse01/TFM/required/ATM_S_eeg_features_sub-08_test.pt"
DIFFUSION_PRIOR_PATH       = "/hhome/ricse01/TFM/required/sub-08/diffusion_prior.pt"
TEST_IMAGES_DIR            = "/hhome/ricse01/TFM/required/test_images/"
OUTPUT_DIR                 = "/hhome/ricse01/TFM/TFM/pipeline_output"

# --- HYPERPARAMETERS ----------------------------------------------------------

TOP_N                = 5
IP_ADAPTER_SCALE     = 0.75
GUIDANCE_SCALE       = 3.0
NUM_INFERENCE_STEPS  = 15
PRIOR_STEPS          = 10
PRIOR_GUIDANCE_SCALE = 2.0
NEGATIVE_PROMPT      = "cartoon, illustration, painting, drawing, render, cgi, blurry, low quality, artificial"
SEED                 = 42

W_RETRIEVAL = 0.5
W_FIDELITY  = 0.5

# --- IMPORTS ------------------------------------------------------------------

import os
import sys
import json
import shutil
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.append("../")
from shared.diffusion_prior import DiffusionPriorUNet, Pipe
from shared.custom_pipeline_low_level import Generator4Embeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# --- HELPERS ------------------------------------------------------------------

def extract_class_name(subfolder_name):
    parts = subfolder_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return subfolder_name


def collect_class_names(test_images_dir):
    subfolders = sorted([
        d for d in os.listdir(test_images_dir)
        if os.path.isdir(os.path.join(test_images_dir, d))
    ])
    return [extract_class_name(sf) for sf in subfolders]


def collect_subfolders(test_images_dir):
    return sorted([
        os.path.join(test_images_dir, d)
        for d in os.listdir(test_images_dir)
        if os.path.isdir(os.path.join(test_images_dir, d))
    ])


def make_prompt(class_name):
    return class_name.replace("_", " ").strip()


def load_features():
    img_features = torch.load(VIT_H_14_FEATURES_TEST,
                               map_location=device)["img_features"]  # [N, dim]
    eeg_embeds   = torch.load(ATM_S_EEG_FEATURES_SUB_08,
                               map_location=device).unsqueeze(1)      # [N, 1, dim]
    print(f"img_features : {img_features.shape}")
    print(f"eeg_embeds   : {eeg_embeds.shape}")
    return img_features, eeg_embeds


def load_diffusion_prior():
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    pipe = Pipe(diffusion_prior, device=device)
    pipe.diffusion_prior.load_state_dict(
        torch.load(DIFFUSION_PRIOR_PATH, map_location=device)
    )
    print("Diffusion prior loaded.")
    return pipe


# --- LPIPS MODEL --------------------------------------------------------------

_lpips_model = None

def get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").to(device)
        _lpips_model.eval()
        print("  [LPIPS] AlexNet-based LPIPS loaded.")
    return _lpips_model


# Image -> normalised tensor [-1, 1] for LPIPS
_lpips_preproc = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),                        # [0, 1]
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]),        # [-1, 1]
])

def pil_to_lpips_tensor(pil_image):
    return _lpips_preproc(pil_image).unsqueeze(0).to(device)


def compute_lpips(img_a, img_b):
    """
    Compute LPIPS distance between two PIL images.
    Returns a float in [0, ~1]. Lower = more perceptually similar.
    """
    model = get_lpips_model()
    ta    = pil_to_lpips_tensor(img_a)
    tb    = pil_to_lpips_tensor(img_b)
    with torch.no_grad():
        dist = model(ta, tb)
    return dist.item()


def load_class_reference_images(test_images_dir):
    """
    Load the first image from each class subfolder (sorted order).
    Returns a list of PIL Images, one per class.
    """
    subfolders = collect_subfolders(test_images_dir)
    ref_images = []
    print(f"  Loading {len(subfolders)} class reference images...")
    for sf in subfolders:
        img_file = sorted(os.listdir(sf))[0]
        img      = Image.open(os.path.join(sf, img_file)).convert("RGB")
        ref_images.append(img)
    return ref_images


# --- STAGE 1: CLASS RETRIEVAL -------------------------------------------------

def retrieve_top_n_classes(eeg_embed, img_features_norm, class_names, top_n):
    """
    Retrieve top-N classes: eeg_embed @ img_features_norm.T
    Returns list of dicts: rank, class, class_idx, raw_cosine
    """
    eeg_norm = F.normalize(eeg_embed.squeeze().unsqueeze(0), dim=-1)
    sims     = (eeg_norm @ img_features_norm.T).squeeze(0)
    top_vals, top_idx = torch.topk(sims, k=top_n)

    return [
        {
            "rank":       rank,
            "class":      class_names[idx],
            "class_idx":  idx,
            "raw_cosine": round(top_vals[rank].item(), 6),
        }
        for rank, idx in enumerate(top_idx.tolist())
    ]


# --- STAGE 2: DIFFUSION PRIOR -------------------------------------------------

def run_diffusion_prior(prior_pipe, eeg_embed):
    return prior_pipe.generate(
        c_embeds=eeg_embed,
        num_inference_steps=PRIOR_STEPS,
        guidance_scale=PRIOR_GUIDANCE_SCALE,
    )


# --- PATCHED GENERATOR --------------------------------------------------------

class Generator4EmbedsPatched(Generator4Embeds):
    def __init__(self, num_inference_steps=15, device="cuda",
                 ip_adapter_scale=0.75, guidance_scale=3.0):
        super().__init__(num_inference_steps=num_inference_steps, device=device)
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        self.guidance_scale = guidance_scale
        print(f"  [Generator] ip_adapter_scale={ip_adapter_scale}, "
              f"guidance_scale={guidance_scale}, steps={num_inference_steps}")

    def generate(self, image_embeds, text_prompt="", negative_prompt="", generator=None):
        image_embeds = image_embeds.to(device=self.device, dtype=self.dtype)
        return self.pipe.generate_ip_adapter_embeds(
            prompt=text_prompt,
            negative_prompt=negative_prompt if self.guidance_scale > 1.0 else None,
            ip_adapter_embeds=image_embeds,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            img2img_strength=self.img2img_strength,
            low_level_image=self.low_level_image,
            low_level_latent=self.low_level_latent,
        ).images[0]


# --- STAGE 3: CANDIDATE GENERATION --------------------------------------------

def generate_candidates(h, retrieved_classes, generator_sdxl, gen,
                        image_dir, class_ref_images):
    """
    Generate one SDXL image per retrieved class.
    Computes and saves scores per candidate to candidate_K_scores.json.

    Scores:
        raw_retrieval — cosine_sim(eeg_embed, img_features[class_idx])
        raw_fidelity  — 1 - min_LPIPS(candidate, all class reference images)
                        minimum LPIPS across all 200 classes, then inverted to [0,1]
                        higher = more perceptually similar to some class reference

    Returns list of dicts: rank, class, class_idx, raw_cosine, raw_fidelity,
                           image, path.
    """
    candidates = []
    for item in retrieved_classes:
        rank   = item["rank"]
        prompt = make_prompt(item["class"])

        image = generator_sdxl.generate(
            h,
            text_prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            generator=gen,
        )

        path = os.path.join(image_dir, f"candidate_{rank}.png")
        image.save(path)

        # Fidelity: 1 - min LPIPS across all class reference images
        # Lower LPIPS = more perceptually similar = higher fidelity score
        lpips_scores = [compute_lpips(image, ref) for ref in class_ref_images]
        min_lpips    = min(lpips_scores)
        raw_fidelity = 1.0 - min_lpips   # [0, 1], higher is better

        # Save scores for weight_search.py
        with open(os.path.join(image_dir, f"candidate_{rank}_scores.json"), "w") as f:
            json.dump({
                "raw_retrieval": item["raw_cosine"],
                "raw_fidelity":  round(raw_fidelity, 6),
                "min_lpips":     round(min_lpips,     6),
            }, f)

        candidates.append({
            "rank":         rank,
            "class":        item["class"],
            "class_idx":    item["class_idx"],
            "raw_cosine":   item["raw_cosine"],
            "raw_fidelity": raw_fidelity,
            "image":        image,
            "path":         path,
        })
        print(f"    candidate_{rank}: '{prompt}'  "
              f"retrieval={item['raw_cosine']:.3f}  "
              f"lpips={min_lpips:.3f}  fidelity={raw_fidelity:.3f}")

    return candidates


# --- STAGE 4: RE-RANKING ------------------------------------------------------

def rerank_candidates(candidates, w_retrieval, w_fidelity):
    """
    Score each candidate and select the best one.

    score_retrieval is already in [0,1] after the (x+1)/2 mapping.
    score_fidelity  is already in [0,1] as 1 - min_lpips.
    """
    scored = []
    for cand in candidates:
        # raw_cosine is in [-1,1], map to [0,1]
        score_retrieval = (cand["raw_cosine"] + 1.0) / 2.0
        # raw_fidelity is already in [0,1]
        score_fidelity  = cand["raw_fidelity"]
        final_score     = w_retrieval * score_retrieval + w_fidelity * score_fidelity

        scored.append({
            "old_rank":       cand["rank"],
            "class":          cand["class"],
            "candidate_path": cand["path"],
            "scores": {
                "raw_retrieval":   round(cand["raw_cosine"],   6),
                "raw_fidelity":    round(cand["raw_fidelity"], 6),
                "score_retrieval": round(score_retrieval,      6),
                "score_fidelity":  round(score_fidelity,       6),
                "w_retrieval":     round(w_retrieval,          6),
                "w_fidelity":      round(w_fidelity,           6),
                "final_score":     round(final_score,          6),
            },
        })

    scored.sort(key=lambda x: x["scores"]["final_score"], reverse=True)
    for new_rank, item in enumerate(scored):
        item["new_rank"] = new_rank

    return scored, scored[0]


# --- MAIN PIPELINE ------------------------------------------------------------

def run_pipeline(eeg_embeds, img_features, class_names,
                 prior_pipe, generator_sdxl, output_dir, seed=SEED):

    os.makedirs(output_dir, exist_ok=True)
    n = len(eeg_embeds)

    img_features_norm = F.normalize(img_features.float(), dim=-1)

    assert len(class_names) == img_features_norm.shape[0], (
        f"class_names ({len(class_names)}) must match img_features rows "
        f"({img_features_norm.shape[0]})."
    )

    # Load class reference images for LPIPS
    class_ref_images = load_class_reference_images(TEST_IMAGES_DIR)

    with open(os.path.join(output_dir, "pipeline_config.json"), "w") as f:
        json.dump({
            "top_n":                TOP_N,
            "ip_adapter_scale":     IP_ADAPTER_SCALE,
            "guidance_scale":       GUIDANCE_SCALE,
            "num_inference_steps":  NUM_INFERENCE_STEPS,
            "prior_steps":          PRIOR_STEPS,
            "prior_guidance_scale": PRIOR_GUIDANCE_SCALE,
            "negative_prompt":      NEGATIVE_PROMPT,
            "w_retrieval":          W_RETRIEVAL,
            "w_fidelity":           W_FIDELITY,
            "fidelity_metric":      "1 - min_LPIPS(candidate, all_class_refs)",
            "seed":                 seed,
        }, f, indent=2)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    correct = 0
    print(f"\nRunning pipeline on {n} EEG embeddings...")
    print(f"Output: {output_dir}\n")

    for i in range(n):
        print(f"\n{'='*60}")
        print(f"Image {i:04d} / {n-1}")
        print(f"{'='*60}")

        image_dir = os.path.join(output_dir, f"image_{i:04d}")
        os.makedirs(image_dir, exist_ok=True)

        # Stage 1: Retrieval
        print("  [Stage 1] Retrieving top-N classes...")
        retrieved = retrieve_top_n_classes(
            eeg_embeds[i], img_features_norm, class_names, TOP_N
        )
        print("    " + ", ".join(
            f"{r['class']}({r['raw_cosine']:.3f})" for r in retrieved
        ))
        with open(os.path.join(image_dir, "retrieved_classes.json"), "w") as f:
            json.dump(retrieved, f, indent=2)

        # Stage 2: Diffusion Prior
        print("  [Stage 2] Running Diffusion Prior...")
        h = run_diffusion_prior(prior_pipe, eeg_embeds[i])

        # Stage 3: Candidate Generation
        print(f"  [Stage 3] Generating {TOP_N} candidates...")
        candidates = generate_candidates(
            h, retrieved, generator_sdxl, gen, image_dir, class_ref_images
        )

        # Stage 4: Re-Ranking
        print("  [Stage 4] Re-ranking candidates...")
        scored, best = rerank_candidates(candidates, W_RETRIEVAL, W_FIDELITY)

        gt_class   = class_names[i]
        is_correct = best["class"] == gt_class
        correct   += int(is_correct)

        print(f"    Selected: '{best['class']}'  gt='{gt_class}'  "
              f"correct={is_correct}  "
              f"score={best['scores']['final_score']:.4f}")

        with open(os.path.join(image_dir, "rerank_scores.json"), "w") as f:
            json.dump({
                "gt_class":       gt_class,
                "selected_class": best["class"],
                "is_correct":     is_correct,
                "candidates":     scored,
            }, f, indent=2)

        shutil.copy2(best["candidate_path"],
                     os.path.join(image_dir, "selected.png"))

    print(f"\n{'='*60}")
    print(f"Done. {n} images processed.")
    print(f"Accuracy: {correct}/{n} = {correct/n:.4f}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


# --- ENTRY POINT --------------------------------------------------------------

def main():
    global TOP_N, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="EEG-to-Image pipeline: retrieve -> prior -> generate -> rerank"
    )
    parser.add_argument("--top_n",  type=int, default=TOP_N)
    parser.add_argument("--seed",   type=int, default=SEED)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--limit",  type=int, default=None,
                        help="Process only the first N images (for debugging)")
    args = parser.parse_args()

    TOP_N      = args.top_n
    OUTPUT_DIR = args.output

    img_features, eeg_embeds = load_features()
    class_names = collect_class_names(TEST_IMAGES_DIR)

    print(f"\n{len(class_names)} classes found in {TEST_IMAGES_DIR}")
    print(f"First 5: {class_names[:5]}")

    if args.limit:
        eeg_embeds = eeg_embeds[:args.limit]
        print(f"  [Debug] Limiting to first {args.limit} images.")

    prior_pipe     = load_diffusion_prior()
    generator_sdxl = Generator4EmbedsPatched(
        num_inference_steps=NUM_INFERENCE_STEPS,
        device=device,
        ip_adapter_scale=IP_ADAPTER_SCALE,
        guidance_scale=GUIDANCE_SCALE,
    )

    run_pipeline(
        eeg_embeds, img_features, class_names,
        prior_pipe, generator_sdxl,
        output_dir=OUTPUT_DIR,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()