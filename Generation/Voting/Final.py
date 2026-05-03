VIT_H_14_FEATURES_TEST    = "/hhome/ricse01/TFM/required/ViT-H-14_features_test.pt"
ATM_S_EEG_FEATURES_SUB_08 = "/hhome/ricse01/TFM/required/ATM_S_eeg_features_sub-08_test.pt"
DIFFUSION_PRIOR_PATH       = "/hhome/ricse01/TFM/required/sub-08/diffusion_prior.pt"
TEST_IMAGES_DIR            = "/hhome/ricse01/TFM/required/test_images/"
OUTPUT_DIR                 = "/hhome/ricse01/TFM/TFM/pipeline_output"

# --- HYPERPARAMETERS ----------------------------------------------------------

TOP_N                = 5      # number of retrieved classes (= number of candidates)
IP_ADAPTER_SCALE     = 0.75   # 1.0 = fully image-driven, 0.0 = fully text-driven
GUIDANCE_SCALE       = 3.0    # 0.0 = turbo/no-text, 3-7.5 = text-guided
NUM_INFERENCE_STEPS  = 15     # SDXL diffusion steps
PRIOR_STEPS          = 10     # diffusion prior steps
PRIOR_GUIDANCE_SCALE = 2.0    # diffusion prior guidance scale
NEGATIVE_PROMPT      = "cartoon, illustration, painting, drawing, render, cgi, blurry, low quality, artificial"
SEED                 = 42

# Re-ranking weights — must sum to 1.0
W_RETRIEVAL  = 0.0   # eeg_embed @ class_img_embed  (from retrieval)
W_CANDIDATE  = 1.0   # candidate_clip_embed @ h      (generated image vs prior)

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
    Returns class names in sorted subfolder order.
    This order matches the index of img_features (one entry per class).
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


# --- CLIP MODEL (for re-ranking) ----------------------------------------------

_clip_model   = None
_clip_preproc = None

def get_clip_model():
    global _clip_model, _clip_preproc
    if _clip_model is None:
        import open_clip
        _clip_model, _, _clip_preproc = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        _clip_model = _clip_model.to(device).eval()
        print("  [CLIP] ViT-H-14 loaded for re-ranking.")
    return _clip_model, _clip_preproc


def extract_clip_embedding(pil_image):
    """Returns L2-normalised CLIP embedding of a PIL image, shape [1, dim]."""
    model, preproc = get_clip_model()
    img_tensor = preproc(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor)
    return F.normalize(emb, dim=-1)


# --- STAGE 1: CLASS RETRIEVAL -------------------------------------------------

def retrieve_top_n_classes(eeg_embed, img_features_norm, class_names, top_n):
    """
    Retrieve top-N classes by comparing the EEG embedding against all
    per-image CLIP embeddings:

        similarities = eeg_embed @ img_features_norm.T

    Args:
        eeg_embed         : Tensor [1, dim] or [dim]
        img_features_norm : Tensor [num_classes, dim], L2-normalised
        class_names       : list[str], len == num_classes
        top_n             : int

    Returns:
        list of top_n dicts sorted by rank (best first):
            rank, class, class_idx, raw_cosine
    """
    eeg_norm = F.normalize(eeg_embed.squeeze().unsqueeze(0), dim=-1)  # [1, dim]
    sims     = (eeg_norm @ img_features_norm.T).squeeze(0)            # [num_classes]

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
    """EEG embedding -> CLIP prior h via the Diffusion Prior UNet."""
    return prior_pipe.generate(
        c_embeds=eeg_embed,
        num_inference_steps=PRIOR_STEPS,
        guidance_scale=PRIOR_GUIDANCE_SCALE,
    )


# --- PATCHED GENERATOR --------------------------------------------------------

class Generator4EmbedsPatched(Generator4Embeds):
    """Extends Generator4Embeds to expose ip_adapter_scale and guidance_scale."""

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

def generate_candidates(h, retrieved_classes, generator_sdxl, gen, image_dir):
    """
    Generate one SDXL image per retrieved class conditioned on h and class name.
    Saves candidate_K.png to image_dir.
    Returns list of dicts: rank, class, raw_cosine, image (PIL), path.
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
        candidates.append({
            "rank":       rank,
            "class":      item["class"],
            "raw_cosine": item["raw_cosine"],
            "image":      image,
            "path":       path,
        })
        print(f"    candidate_{rank}: '{prompt}' -> saved")

    return candidates


# --- STAGE 4: RE-RANKING ------------------------------------------------------

def rerank_candidates(candidates, h):
    """
    Score each candidate using a weighted vote between:

        score_retrieval = (raw_cosine_retrieval + 1) / 2
                          eeg_embed @ class_img_embed, already in retrieved_classes

        score_candidate = (cosine_sim(candidate_clip_embed, h) + 1) / 2
                          how similar the generated image is to the prior output

        final_score = W_RETRIEVAL * score_retrieval + W_CANDIDATE * score_candidate

    Returns scored list sorted best-first, and the best candidate.
    """
    h_norm = F.normalize(h.squeeze().unsqueeze(0), dim=-1)  # [1, dim]

    scored = []
    for cand in candidates:
        # Score 1: retrieval cosine mapped to [0, 1]
        score_retrieval = (cand["raw_cosine"] + 1.0) / 2.0

        # Score 2: candidate CLIP embedding vs h, mapped to [0, 1]
        cand_emb        = extract_clip_embedding(cand["image"])   # [1, dim]
        raw_candidate   = (cand_emb @ h_norm.T).item()
        score_candidate = (raw_candidate + 1.0) / 2.0

        final_score = W_RETRIEVAL * score_retrieval + W_CANDIDATE * score_candidate

        scored.append({
            "old_rank":       cand["rank"],
            "class":          cand["class"],
            "candidate_path": cand["path"],
            "scores": {
                "raw_retrieval":   round(cand["raw_cosine"], 6),
                "raw_candidate":   round(raw_candidate,      6),
                "score_retrieval": round(score_retrieval,    6),
                "score_candidate": round(score_candidate,    6),
                "w_retrieval":     W_RETRIEVAL,
                "w_candidate":     W_CANDIDATE,
                "final_score":     round(final_score,        6),
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

    # L2-normalise image features once, reused for every retrieval call
    img_features_norm = F.normalize(img_features.float(), dim=-1)

    assert len(class_names) == img_features_norm.shape[0], (
        f"class_names ({len(class_names)}) must match img_features rows "
        f"({img_features_norm.shape[0]}). "
        f"Check TEST_IMAGES_DIR has exactly one subfolder per test image."
    )

    config = {
        "top_n":                TOP_N,
        "ip_adapter_scale":     IP_ADAPTER_SCALE,
        "guidance_scale":       GUIDANCE_SCALE,
        "num_inference_steps":  NUM_INFERENCE_STEPS,
        "prior_steps":          PRIOR_STEPS,
        "prior_guidance_scale": PRIOR_GUIDANCE_SCALE,
        "negative_prompt":      NEGATIVE_PROMPT,
        "w_retrieval":          W_RETRIEVAL,
        "w_candidate":          W_CANDIDATE,
        "seed":                 seed,
    }
    with open(os.path.join(output_dir, "pipeline_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

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
        candidates = generate_candidates(h, retrieved, generator_sdxl, gen, image_dir)

        # Stage 4: Re-Ranking
        print("  [Stage 4] Re-ranking candidates...")
        scored, best = rerank_candidates(candidates, h)
        print(f"    Selected: '{best['class']}' "
              f"(final={best['scores']['final_score']:.4f}, "
              f"retrieval={best['scores']['score_retrieval']:.4f}, "
              f"candidate={best['scores']['score_candidate']:.4f})")
        
        gt_class = class_names[i]
        rerank_output = {
            "gt_class":      gt_class,
            "selected_class": best["class"],
            "is_correct":    best["class"] == gt_class,
            "candidates":    scored,
        }

        with open(os.path.join(image_dir, "rerank_scores.json"), "w") as f:
            json.dump(rerank_output, f, indent=2)

        shutil.copy2(best["candidate_path"],
                     os.path.join(image_dir, "selected.png"))

    print(f"\n{'='*60}")
    print(f"Done. {n} images processed. Output: {output_dir}")
    print(f"{'='*60}")


# --- ENTRY POINT --------------------------------------------------------------

def main():
    global TOP_N, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="EEG-to-Image pipeline: retrieve -> prior -> generate -> rerank"
    )
    parser.add_argument("--top_n",  type=int, default=TOP_N,
                        help=f"Number of candidates per image (default: {TOP_N})")
    parser.add_argument("--seed",   type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Output directory")
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