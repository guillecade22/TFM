# =============================================================================
# EEG-to-Image Reconstruction Pipeline with Weight Optimization
# =============================================================================
# Stages:
#   1. Class Retrieval       — eeg_embed @ img_features.T -> top-N classes
#   2. Diffusion Prior       — EEG embedding -> CLIP prior h
#   3. Candidate Generation  — SDXL + IP-Adapter, one image per top-N class
#   4. Weight Optimization   — gradient descent over:
#
#       score_retrieval = cosine_sim(eeg_embed,      img_features[class_idx])
#       score_fidelity  = cosine_sim(candidate_clip, img_features[class_idx])
#       final_score     = W_RETRIEVAL * score_retrieval + W_FIDELITY * score_fidelity
#
#   5. Re-Ranking            — apply optimized weights, select best candidate
#
# Generation (phases 1-3) runs once.
# Optimization (phase 4) can be re-run cheaply with --opt_only.

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

# Initial weights (will be overwritten by optimization)
W_RETRIEVAL = 0.5
W_FIDELITY  = 0.5

# Gradient descent hyperparameters
OPT_LR        = 0.01
OPT_STEPS     = 500
OPT_LOG_EVERY = 50

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


# --- CLIP MODEL ---------------------------------------------------------------

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
        print("  [CLIP] ViT-H-14 loaded.")
    return _clip_model, _clip_preproc


def extract_clip_embedding(pil_image):
    model, preproc = get_clip_model()
    img_tensor = preproc(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor)
    return F.normalize(emb, dim=-1)


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
                        image_dir, img_features_norm):
    """
    Generate one SDXL image per retrieved class.
    Also computes and saves both scores per candidate so optimization
    can run without reloading images.

    Saves per candidate:
        candidate_K.png
        candidate_K_scores.json  — raw_retrieval, raw_fidelity

    Returns list of dicts: rank, class, class_idx, raw_cosine,
                           raw_fidelity, image, path.
    """
    candidates = []
    for item in retrieved_classes:
        rank      = item["rank"]
        class_idx = item["class_idx"]
        prompt    = make_prompt(item["class"])

        image = generator_sdxl.generate(
            h,
            text_prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            generator=gen,
        )

        path = os.path.join(image_dir, f"candidate_{rank}.png")
        image.save(path)

        # Fidelity: cosine_sim(candidate_clip, img_features[class_idx])
        cand_emb     = extract_clip_embedding(image)                        # [1, dim]
        sims         = (cand_emb @ img_features_norm.T).squeeze(0)          # [200]
        raw_fidelity = sims.max().item()

        # Save scores to disk for later optimization
        scores_path = os.path.join(image_dir, f"candidate_{rank}_scores.json")
        with open(scores_path, "w") as f:
            json.dump({
                "raw_retrieval": item["raw_cosine"],
                "raw_fidelity":  round(raw_fidelity, 6),
            }, f)

        candidates.append({
            "rank":         rank,
            "class":        item["class"],
            "class_idx":    class_idx,
            "raw_cosine":   item["raw_cosine"],
            "raw_fidelity": raw_fidelity,
            "image":        image,
            "path":         path,
        })
        print(f"    candidate_{rank}: '{prompt}'  "
              f"retrieval={item['raw_cosine']:.3f}  "
              f"fidelity={raw_fidelity:.3f}")

    return candidates


# --- STAGE 4: WEIGHT OPTIMIZATION VIA GRADIENT DESCENT -----------------------

def load_all_scores(output_dir, class_names, top_n):
    """
    Load saved candidate scores for all images.

    Returns:
        score_retrieval : Tensor [N, top_n]  — (raw_retrieval + 1) / 2
        score_fidelity  : Tensor [N, top_n]  — (raw_fidelity  + 1) / 2
        gt_indices      : Tensor [N]          — index of gt in top-n (-1 if absent)
    """
    image_dirs = sorted([
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("image_") and os.path.isdir(os.path.join(output_dir, d))
    ])

    all_score_retrieval = []
    all_score_fidelity  = []
    all_gt_indices      = []

    for image_dir in image_dirs:
        image_idx = int(os.path.basename(image_dir).split("_")[1])
        gt_class  = class_names[image_idx]

        with open(os.path.join(image_dir, "retrieved_classes.json")) as f:
            retrieved = json.load(f)

        retrieved_classes = [r["class"] for r in retrieved]
        gt_idx = retrieved_classes.index(gt_class) if gt_class in retrieved_classes else -1

        row_retrieval = []
        row_fidelity  = []

        for r in retrieved:
            scores_path = os.path.join(image_dir, f"candidate_{r['rank']}_scores.json")
            with open(scores_path) as f:
                scores = json.load(f)

            row_retrieval.append((scores["raw_retrieval"] + 1.0) / 2.0)
            row_fidelity.append( (scores["raw_fidelity"]  + 1.0) / 2.0)

        all_score_retrieval.append(row_retrieval)
        all_score_fidelity.append(row_fidelity)
        all_gt_indices.append(gt_idx)

    return (
        torch.tensor(all_score_retrieval, dtype=torch.float32),
        torch.tensor(all_score_fidelity,  dtype=torch.float32),
        torch.tensor(all_gt_indices,      dtype=torch.long),
    )


def optimize_weights(score_retrieval, score_fidelity, gt_indices):
    """
    Find optimal W_RETRIEVAL and W_FIDELITY via gradient descent.

    Objective: maximise log-probability of the correct candidate (cross-entropy).

        final_score[i, k] = w_r * score_retrieval[i, k] + w_f * score_fidelity[i, k]
        loss = cross_entropy(final_score, gt_indices)

    Weights are parameterised via softmax to always sum to 1 and stay in [0, 1].
    Only images where gt is in the top-N contribute to the loss.

    Returns:
        w_retrieval, w_fidelity : optimised floats
        loss_history            : list of loss values
    """
    mask = gt_indices >= 0
    sr   = score_retrieval[mask].to(device)
    sf   = score_fidelity[mask].to(device)
    gt   = gt_indices[mask].to(device)

    print(f"\n  [Optimizer] {mask.sum().item()}/{len(gt_indices)} images "
          f"have gt in top-{TOP_N}")
    print(f"  [Optimizer] Running {OPT_STEPS} steps, lr={OPT_LR}")

    logits    = torch.zeros(2, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([logits], lr=OPT_LR)

    loss_history = []

    for step in range(OPT_STEPS):
        optimizer.zero_grad()

        w         = F.softmax(logits, dim=0)
        scores    = w[0] * sr + w[1] * sf              # [M, top_n]
        log_probs = F.log_softmax(scores, dim=1)        # [M, top_n]
        loss      = F.nll_loss(log_probs, gt)

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if step % OPT_LOG_EVERY == 0 or step == OPT_STEPS - 1:
            w_ = F.softmax(logits, dim=0)
            print(f"  step {step:4d} | loss={loss.item():.4f} | "
                  f"w_retrieval={w_[0].item():.4f}  "
                  f"w_fidelity={w_[1].item():.4f}")

    final       = F.softmax(logits, dim=0).detach()
    w_retrieval = final[0].item()
    w_fidelity  = final[1].item()
    print(f"\n  Optimized: w_retrieval={w_retrieval:.4f}  "
          f"w_fidelity={w_fidelity:.4f}")

    return w_retrieval, w_fidelity, loss_history


# --- STAGE 5: RE-RANKING WITH GIVEN WEIGHTS -----------------------------------

def rerank_candidates(candidates, w_retrieval, w_fidelity):
    scored = []
    for cand in candidates:
        score_retrieval = (cand["raw_cosine"]   + 1.0) / 2.0
        score_fidelity  = (cand["raw_fidelity"] + 1.0) / 2.0
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

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # ------------------------------------------------------------------
    # PHASE 1: Generate all candidates and compute scores
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 1: Generation")
    print(f"{'='*60}")

    all_candidates = []

    for i in range(n):
        print(f"\n[{i:04d}/{n-1}]")
        image_dir = os.path.join(output_dir, f"image_{i:04d}")
        os.makedirs(image_dir, exist_ok=True)

        # Stage 1: Retrieval
        retrieved = retrieve_top_n_classes(
            eeg_embeds[i], img_features_norm, class_names, TOP_N
        )
        print("  [Stage 1] " + ", ".join(
            f"{r['class']}({r['raw_cosine']:.3f})" for r in retrieved
        ))
        with open(os.path.join(image_dir, "retrieved_classes.json"), "w") as f:
            json.dump(retrieved, f, indent=2)

        # Stage 2: Diffusion Prior
        print("  [Stage 2] Running Diffusion Prior...")
        h = run_diffusion_prior(prior_pipe, eeg_embeds[i])

        # Stage 3: Generate + compute scores
        print(f"  [Stage 3] Generating {TOP_N} candidates...")
        candidates = generate_candidates(
            h, retrieved, generator_sdxl, gen, image_dir, img_features_norm
        )
        all_candidates.append(candidates)

    # ------------------------------------------------------------------
    # PHASE 2: Optimize weights
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 2: Weight Optimization")
    print(f"{'='*60}")

    score_retrieval, score_fidelity, gt_indices = load_all_scores(
        output_dir, class_names, TOP_N
    )
    w_retrieval, w_fidelity, loss_history = optimize_weights(
        score_retrieval, score_fidelity, gt_indices
    )

    with open(os.path.join(output_dir, "optimized_weights.json"), "w") as f:
        json.dump({
            "w_retrieval":  round(w_retrieval, 6),
            "w_fidelity":   round(w_fidelity,  6),
            "loss_history": loss_history,
        }, f, indent=2)

    # ------------------------------------------------------------------
    # PHASE 3: Re-rank with optimized weights
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 3: Re-Ranking with Optimized Weights")
    print(f"{'='*60}")

    for i in range(n):
        image_dir  = os.path.join(output_dir, f"image_{i:04d}")
        candidates = all_candidates[i]
        gt_class   = class_names[i]

        scored, best = rerank_candidates(candidates, w_retrieval, w_fidelity)

        with open(os.path.join(image_dir, "rerank_scores.json"), "w") as f:
            json.dump({
                "gt_class":       gt_class,
                "selected_class": best["class"],
                "is_correct":     best["class"] == gt_class,
                "w_retrieval":    round(w_retrieval, 6),
                "w_fidelity":     round(w_fidelity,  6),
                "candidates":     scored,
            }, f, indent=2)

        shutil.copy2(best["candidate_path"],
                     os.path.join(image_dir, "selected.png"))

        print(f"  [{i:04d}] selected='{best['class']}'  gt='{gt_class}'  "
              f"correct={best['class'] == gt_class}  "
              f"score={best['scores']['final_score']:.4f}")

    # Save full config
    with open(os.path.join(output_dir, "pipeline_config.json"), "w") as f:
        json.dump({
            "top_n":                 TOP_N,
            "ip_adapter_scale":      IP_ADAPTER_SCALE,
            "guidance_scale":        GUIDANCE_SCALE,
            "num_inference_steps":   NUM_INFERENCE_STEPS,
            "prior_steps":           PRIOR_STEPS,
            "prior_guidance_scale":  PRIOR_GUIDANCE_SCALE,
            "negative_prompt":       NEGATIVE_PROMPT,
            "seed":                  seed,
            "opt_lr":                OPT_LR,
            "opt_steps":             OPT_STEPS,
            "w_retrieval_optimized": round(w_retrieval, 6),
            "w_fidelity_optimized":  round(w_fidelity,  6),
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. {n} images processed.")
    print(f"Optimized weights: w_retrieval={w_retrieval:.4f}  "
          f"w_fidelity={w_fidelity:.4f}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


# --- ENTRY POINT --------------------------------------------------------------

def main():
    global TOP_N, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="EEG-to-Image pipeline with gradient descent weight optimization"
    )
    parser.add_argument("--top_n",    type=int, default=TOP_N)
    parser.add_argument("--seed",     type=int, default=SEED)
    parser.add_argument("--output",   type=str, default=OUTPUT_DIR)
    parser.add_argument("--limit",    type=int, default=None,
                        help="Process only the first N images (for debugging)")
    parser.add_argument("--opt_only", action="store_true",
                        help="Skip generation — re-run optimization + reranking "
                             "on an existing output directory")
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

    if args.opt_only:
        # Re-optimize and re-rank on existing output, no generation needed
        print("\n[--opt_only] Loading saved scores...")
        score_retrieval, score_fidelity, gt_indices = load_all_scores(
            OUTPUT_DIR, class_names, TOP_N
        )
        w_retrieval, w_fidelity, loss_history = optimize_weights(
            score_retrieval, score_fidelity, gt_indices
        )
        with open(os.path.join(OUTPUT_DIR, "optimized_weights.json"), "w") as f:
            json.dump({
                "w_retrieval":  round(w_retrieval, 6),
                "w_fidelity":   round(w_fidelity,  6),
                "loss_history": loss_history,
            }, f, indent=2)
        print(f"Weights saved to: {os.path.join(OUTPUT_DIR, 'optimized_weights.json')}")
        return

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