# =============================================================================
# EEG-to-Image Reconstruction Pipeline
# =============================================================================
# Stages:
#   1. EEG Encoding & Class Retrieval  — eeg_embed @ img_features.T (cosine sim)
#                                        mapped to class names via folder structure
#   2. Low-Level Image Generation      — Diffusion Prior UNet -> CLIP embedding h
#   3. Multi-Candidate Generation      — SDXL + IP-Adapter, one image per top-N class
#   4. Re-Ranking                      — joint score: cosine_sim(candidate, h) + retrieval_conf
#
# Inputs required (same as your existing scripts):
#   VIT_H_14_FEATURES_TEST    — per-image CLIP embeddings, shape [200, dim]
#   ATM_S_EEG_FEATURES_SUB_08 — pre-computed EEG embeddings, shape [200, dim]
#   DIFFUSION_PRIOR_PATH      — trained diffusion prior weights
#   TEST_IMAGES_DIR           — folder with one subfolder per class (sorted = class index)
#
# Intermediate outputs saved per image:
#   image_XXXX/
#     retrieved_classes.json   — top-N classes + confidence scores
#     candidate_K.png          — generated image for rank-K class
#     scores.json              — per-candidate decomposed & merged scores
#     selected.png             — copy of best candidate
#   pipeline_config.json       — all hyperparameters
#   pipeline_summary.json      — one entry per image with selected class + scores

# --- CONFIG -------------------------------------------------------------------

VIT_H_14_FEATURES_TEST     = "/hhome/ricse01/TFM/required/ViT-H-14_features_test.pt"
ATM_S_EEG_FEATURES_SUB_08  = "/hhome/ricse01/TFM/required/ATM_S_eeg_features_sub-08_test.pt"
DIFFUSION_PRIOR_PATH        = "/hhome/ricse01/TFM/required/sub-08/diffusion_prior.pt"
TEST_IMAGES_DIR             = "/hhome/ricse01/TFM/required/test_images/"
OUTPUT_DIR                  = "/hhome/ricse01/TFM/TFM/pipeline_output"

# --- PIPELINE HYPERPARAMETERS -------------------------------------------------

TOP_N                  = 5      # number of retrieved classes per image
IP_ADAPTER_SCALE       = 0.75   # 1.0 = fully image-driven, 0.0 = fully text-driven
GUIDANCE_SCALE         = 3.0    # 0.0 = turbo/no-text, 3-7.5 = text-guided
NUM_INFERENCE_STEPS    = 15     # SDXL diffusion steps
PRIOR_STEPS            = 10     # Diffusion prior steps
PRIOR_GUIDANCE_SCALE   = 2.0    # Diffusion prior guidance scale

# Re-ranking weights — must sum to 1.0 (equal weighting)
W_COSINE               = 0.5   # cosine similarity: generated candidate vs h
W_CONFIDENCE           = 0.5   # retrieval confidence of the class label

NEGATIVE_PROMPT        = "cartoon, illustration, painting, drawing, render, cgi, blurry, low quality, artificial"
SEED                   = 42

# --- IMPORTS ------------------------------------------------------------------

import os
import sys
import json
import shutil
import argparse
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append("../")
from shared.diffusion_prior import DiffusionPriorUNet, Pipe
from shared.custom_pipeline_low_level import Generator4Embeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# --- HELPERS ------------------------------------------------------------------

def extract_class_name(subfolder_name):
    """'00001_aircraft_carrier' -> 'aircraft_carrier'"""
    parts = subfolder_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return subfolder_name


def collect_class_names(test_images_dir):
    """
    Returns a list of class names in sorted subfolder order.
    The sort order matches the index used in img_features (one entry per class).
    """
    subfolders = sorted([
        d for d in os.listdir(test_images_dir)
        if os.path.isdir(os.path.join(test_images_dir, d))
    ])
    return [extract_class_name(sf) for sf in subfolders]


def make_prompt(class_name):
    """'aircraft_carrier' -> 'aircraft carrier'"""
    return class_name.replace("_", " ").strip()


def load_features():
    """
    Load pre-computed CLIP image embeddings and EEG embeddings.

    img_features : [num_classes, dim]   — one CLIP embedding per test class image
    eeg_embeds   : [num_images, 1, dim] — one EEG embedding per test trial
    """
    img_features = torch.load(VIT_H_14_FEATURES_TEST,
                               map_location=device)["img_features"]   # [N, dim]
    eeg_embeds   = torch.load(ATM_S_EEG_FEATURES_SUB_08,
                               map_location=device).unsqueeze(1)       # [N, 1, dim]

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


# --- STAGE 1: EEG ENCODING & CLASS RETRIEVAL ----------------------------------

def retrieve_top_n_classes(eeg_embed, img_features_norm, class_names, top_n):
    """
    Retrieve top-N classes by comparing the EEG embedding against all
    per-image CLIP embeddings — same as the ATMS retrieval script:

        similarities = eeg_emb @ img_features_all.T

    Confidence scores are softmax-normalised over all classes -> [0, 1].

    Args:
        eeg_embed         : Tensor [1, dim] or [dim]
        img_features_norm : Tensor [num_classes, dim], L2-normalised
        class_names       : list[str], len == num_classes
        top_n             : int

    Returns:
        list of top_n dicts sorted by rank (best first), each with:
            rank, class, class_idx, confidence, raw_cosine
    """
    eeg_flat = F.normalize(eeg_embed.squeeze().unsqueeze(0), dim=-1)  # [1, dim]

    # Cosine similarities: [num_classes]
    sims = (eeg_flat @ img_features_norm.T).squeeze(0)

    # Softmax -> confidence in [0, 1]
    confidences = F.softmax(sims, dim=0)

    top_vals, top_idx = torch.topk(confidences, k=top_n)

    retrieved = []
    for rank, (idx, conf) in enumerate(zip(top_idx.tolist(), top_vals.tolist())):
        retrieved.append({
            "rank":       rank,
            "class":      class_names[idx],
            "class_idx":  idx,
            "confidence": round(conf, 6),
            "raw_cosine": round(sims[idx].item(), 6),
        })

    return retrieved


# --- STAGE 2: LOW-LEVEL IMAGE GENERATION (DIFFUSION PRIOR) --------------------

def run_diffusion_prior(prior_pipe, eeg_embed):
    """
    EEG embedding -> CLIP-space prior h via the Diffusion Prior UNet.
    Returns h: Tensor [1, 1, dim]
    """
    h = prior_pipe.generate(
        c_embeds=eeg_embed,
        num_inference_steps=PRIOR_STEPS,
        guidance_scale=PRIOR_GUIDANCE_SCALE,
    )
    return h


# --- PATCHED GENERATOR --------------------------------------------------------

class Generator4EmbedsPatched(Generator4Embeds):
    """
    Extends Generator4Embeds to expose ip_adapter_scale, guidance_scale,
    and negative_prompt as runtime parameters.
    """
    def __init__(self, num_inference_steps=15, device="cuda",
                 ip_adapter_scale=0.75, guidance_scale=3.0):
        super().__init__(num_inference_steps=num_inference_steps, device=device)
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        self.guidance_scale = guidance_scale
        print(f"  [Generator] ip_adapter_scale={ip_adapter_scale}, "
              f"guidance_scale={guidance_scale}, steps={num_inference_steps}")

    def generate(self, image_embeds, text_prompt="", negative_prompt="", generator=None):
        image_embeds = image_embeds.to(device=self.device, dtype=self.dtype)
        image = self.pipe.generate_ip_adapter_embeds(
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
        return image


# --- CLIP FEATURE EXTRACTOR (for re-ranking) ----------------------------------

_clip_model   = None
_clip_preproc = None

def get_clip_model():
    """Lazy-load open_clip for candidate embedding extraction."""
    global _clip_model, _clip_preproc
    if _clip_model is None:
        try:
            import open_clip
            _clip_model, _, _clip_preproc = open_clip.create_model_and_transforms(
                "ViT-H-14", pretrained="laion2b_s32b_b79k"
            )
            _clip_model = _clip_model.to(device).eval()
            print("  [CLIP] open_clip ViT-H-14 loaded for re-ranking.")
        except ImportError:
            raise ImportError(
                "open_clip is required for candidate re-ranking. "
                "Install with: pip install open-clip-torch"
            )
    return _clip_model, _clip_preproc


def extract_clip_embedding(pil_image):
    """
    Extract a CLIP image embedding from a PIL image.
    Returns: Tensor [1, dim], L2-normalised, on `device`.
    """
    model, preproc = get_clip_model()
    img_tensor = preproc(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor)
    return F.normalize(emb, dim=-1)


# --- STAGE 3: MULTI-CANDIDATE GENERATION --------------------------------------

def generate_candidates(h, retrieved_classes, generator_sdxl, gen, image_dir):
    """
    Generate one SDXL image per retrieved class, conditioned on:
      - h via IP-Adapter: structural prior from the EEG diffusion prior
      - class name as text prompt: semantic guidance

    Saves candidate_K.png to image_dir.

    Returns:
        list of dicts: rank, class, confidence, raw_cosine, image (PIL), path
    """
    candidates = []
    for item in retrieved_classes:
        rank       = item["rank"]
        class_name = item["class"]
        prompt     = make_prompt(class_name)

        image = generator_sdxl.generate(
            h,
            text_prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            generator=gen,
        )

        candidate_path = os.path.join(image_dir, f"candidate_{rank}.png")
        image.save(candidate_path)

        candidates.append({
            "rank":       rank,
            "class":      class_name,
            "confidence": item["confidence"],
            "raw_cosine": item["raw_cosine"],
            "image":      image,
            "path":       candidate_path,
        })
        print(f"    candidate_{rank}: '{prompt}' -> saved")

    return candidates


# --- STAGE 4: RE-RANKING ------------------------------------------------------

def rerank_candidates(candidates, h):
    """
    Score each candidate and select the best one.

    Both components are in [0, 1] with equal weights:

        cosine_score = (cosine_sim(candidate_clip_emb, h_norm) + 1) / 2
                       maps raw cosine [-1, 1] -> [0, 1]

        conf_score   = softmax retrieval confidence (already in [0, 1])

        final_score  = W_COSINE * cosine_score + W_CONFIDENCE * conf_score

    Args:
        candidates : list of dicts from generate_candidates()
        h          : Tensor [1, 1, dim] — diffusion prior output

    Returns:
        scored : list of dicts sorted best-first (original fields + score fields)
        best   : dict — the top-ranked candidate
    """
    h_norm = F.normalize(h.squeeze().unsqueeze(0), dim=-1)  # [1, dim]

    scored = []
    for cand in candidates:
        cand_emb    = extract_clip_embedding(cand["image"])   # [1, dim]

        raw_cos     = (cand_emb @ h_norm.T).item()            # [-1, 1]
        cos_score   = (raw_cos + 1.0) / 2.0                  # [0, 1]
        conf_score  = cand["confidence"]                      # [0, 1]
        final_score = W_COSINE * cos_score + W_CONFIDENCE * conf_score

        scored.append({
            **{k: v for k, v in cand.items() if k != "image"},
            "image":          cand["image"],
            "cosine_sim_raw": round(raw_cos,     6),
            "cosine_score":   round(cos_score,   6),
            "conf_score":     round(conf_score,  6),
            "final_score":    round(final_score, 6),
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored, scored[0]


# --- SAVE INTERMEDIATES -------------------------------------------------------

def save_intermediates(image_dir, retrieved_classes, scored_candidates, best):
    """
    Saves:
        retrieved_classes.json  — top-N classes with cosine and confidence
        scores.json             — per-candidate decomposed + merged scores, best-first
        selected.png            — copy of the winning candidate
    """
    with open(os.path.join(image_dir, "retrieved_classes.json"), "w") as f:
        json.dump(retrieved_classes, f, indent=2)

    scores_out = []
    for idx, c in enumerate(scored_candidates):
        scores_out.append({
            "rerank_position":   idx,
            "original_rank":     c["rank"],
            "class":             c["class"],
            "candidate_path":    c["path"],
            "scores": {
                "cosine_sim_raw":    c["cosine_sim_raw"],
                "cosine_score_0_1":  c["cosine_score"],
                "conf_score_0_1":    c["conf_score"],
                "weight_cosine":     W_COSINE,
                "weight_confidence": W_CONFIDENCE,
                "final_score":       c["final_score"],
            },
            "is_selected": idx == 0,
        })

    with open(os.path.join(image_dir, "scores.json"), "w") as f:
        json.dump(scores_out, f, indent=2)

    selected_path = os.path.join(image_dir, "selected.png")
    shutil.copy2(best["path"], selected_path)

    return selected_path


# --- MAIN PIPELINE ------------------------------------------------------------

def run_pipeline(eeg_embeds, img_features, class_names,
                 prior_pipe, generator_sdxl, output_dir, seed=SEED):

    os.makedirs(output_dir, exist_ok=True)
    n = len(eeg_embeds)

    # L2-normalise image features once (reused in every retrieval call)
    img_features_norm = F.normalize(img_features.float(), dim=-1)  # [num_classes, dim]

    assert len(class_names) == img_features_norm.shape[0], (
        f"class_names ({len(class_names)}) must match img_features rows "
        f"({img_features_norm.shape[0]}). "
        f"Check TEST_IMAGES_DIR has exactly one subfolder per test image."
    )

    # Save config for reproducibility
    config = {
        "top_n":                TOP_N,
        "ip_adapter_scale":     IP_ADAPTER_SCALE,
        "guidance_scale":       GUIDANCE_SCALE,
        "num_inference_steps":  NUM_INFERENCE_STEPS,
        "prior_steps":          PRIOR_STEPS,
        "prior_guidance_scale": PRIOR_GUIDANCE_SCALE,
        "w_cosine":             W_COSINE,
        "w_confidence":         W_CONFIDENCE,
        "seed":                 seed,
        "negative_prompt":      NEGATIVE_PROMPT,
    }
    with open(os.path.join(output_dir, "pipeline_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    summary = []

    print(f"\nRunning pipeline on {n} EEG embeddings...")
    print(f"Output: {output_dir}\n")

    for i in range(n):
        print(f"\n{'='*60}")
        print(f"Image {i:04d} / {n-1}")
        print(f"{'='*60}")

        image_dir = os.path.join(output_dir, f"image_{i:04d}")
        os.makedirs(image_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # STAGE 1: Class Retrieval
        # ------------------------------------------------------------------
        print("  [Stage 1] Retrieving top-N classes via EEG @ img_features.T ...")
        retrieved = retrieve_top_n_classes(
            eeg_embeds[i], img_features_norm, class_names, TOP_N
        )
        print("    Top-{}: {}".format(
            TOP_N,
            ", ".join(f"{r['class']}({r['confidence']:.4f})" for r in retrieved)
        ))

        # ------------------------------------------------------------------
        # STAGE 2: Low-Level Image Generation
        # ------------------------------------------------------------------
        print("  [Stage 2] Running Diffusion Prior (EEG -> CLIP prior h)...")
        h = run_diffusion_prior(prior_pipe, eeg_embeds[i])

        # ------------------------------------------------------------------
        # STAGE 3: Multi-Candidate Generation
        # ------------------------------------------------------------------
        print(f"  [Stage 3] Generating {TOP_N} candidates with SDXL + IP-Adapter...")
        candidates = generate_candidates(h, retrieved, generator_sdxl, gen, image_dir)

        # ------------------------------------------------------------------
        # STAGE 4: Re-Ranking
        # ------------------------------------------------------------------
        print("  [Stage 4] Re-ranking candidates...")
        scored, best = rerank_candidates(candidates, h)
        print(f"    Selected: '{best['class']}' "
              f"(final={best['final_score']:.4f}, "
              f"cos={best['cosine_score']:.4f}, "
              f"conf={best['conf_score']:.4f})")

        # ------------------------------------------------------------------
        # Save intermediates
        # ------------------------------------------------------------------
        selected_path = save_intermediates(image_dir, retrieved, scored, best)
        print(f"  Saved: {image_dir}/")

        summary.append({
            "image_index":    i,
            "image_dir":      image_dir,
            "selected_class": best["class"],
            "selected_path":  selected_path,
            "top_retrieved":  [r["class"] for r in retrieved],
            "scores": {
                "cosine_score": best["cosine_score"],
                "conf_score":   best["conf_score"],
                "final_score":  best["final_score"],
            },
        })

    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Pipeline complete. {n} images processed.")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")


# --- ENTRY POINT --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EEG-to-Image pipeline: retrieve -> prior -> generate -> rerank"
    )
    parser.add_argument("--top_n",  type=int, default=TOP_N,
                        help=f"Number of retrieved classes per image (default: {TOP_N})")
    parser.add_argument("--seed",   type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Process only the first N images (for debugging)")
    args = parser.parse_args()

    global TOP_N, OUTPUT_DIR
    TOP_N      = args.top_n
    OUTPUT_DIR = args.output

    # Load data
    img_features, eeg_embeds = load_features()
    class_names = collect_class_names(TEST_IMAGES_DIR)

    print(f"\n{len(class_names)} classes found in {TEST_IMAGES_DIR}")
    print(f"First 5: {class_names[:5]}")

    if args.limit:
        eeg_embeds = eeg_embeds[:args.limit]
        print(f"  [Debug] Limiting to first {args.limit} images.")

    # Load models
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