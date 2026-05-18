# =============================================================================
# CLIP Candidate Analysis
# =============================================================================
# For each generated candidate, computes CLIP cosine similarity against all
# 200 test images and finds the most similar class.
#
# Prints two metrics:
#
#   Metric 1 — ANY candidate correct (top-N oracle):
#       For each image, check if ANY of the 5 candidates' top CLIP match
#       is the correct class. This is the ceiling — if this is low, CLIP
#       similarity is not a useful signal at all.
#
#   Metric 2 — TOP-1 candidate correct (retrieval rank-0 only):
#       Only check the rank-0 candidate (generated with the top retrieved class).
#       This measures whether the best retrieval candidate is also the most
#       visually faithful one according to CLIP.
#
# Saves per candidate: candidate_K_clip_match.json
# Saves per image:     clip_analysis.json
#
# Usage:
#   python clip_analysis.py
#   python clip_analysis.py --output /path/to/pipeline_output
#   python clip_analysis.py --limit 5

# --- CONFIG -------------------------------------------------------------------

PIPELINE_OUTPUT_DIR = "/hhome/ricse01/TFM/TFM/pipeline_output"
TEST_IMAGES_DIR     = "/hhome/ricse01/TFM/required/test_images/"
VIT_H_14_FEATURES_TEST = "/hhome/ricse01/TFM/required/ViT-H-14_features_test.pt"

# --- IMPORTS ------------------------------------------------------------------

import os
import json
import argparse
import torch
import torch.nn.functional as F
from PIL import Image

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


# --- LOAD PRECOMPUTED TEST EMBEDDINGS -----------------------------------------

def load_test_embeddings(features_path, class_names):
    """
    Load pre-computed CLIP embeddings for all test images.
    Shape: [num_classes, dim], L2-normalised.
    """
    img_features = torch.load(features_path, map_location=device)["img_features"]
    img_features = F.normalize(img_features.float(), dim=-1)
    print(f"  Test CLIP embeddings: {img_features.shape}")
    assert img_features.shape[0] == len(class_names), (
        f"img_features rows ({img_features.shape[0]}) != "
        f"class_names ({len(class_names)})"
    )
    return img_features


# --- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=PIPELINE_OUTPUT_DIR)
    parser.add_argument("--limit",  type=int, default=None)
    args = parser.parse_args()

    class_names   = collect_class_names(TEST_IMAGES_DIR)
    img_features  = load_test_embeddings(VIT_H_14_FEATURES_TEST, class_names)

    image_dirs = sorted([
        os.path.join(args.output, d)
        for d in os.listdir(args.output)
        if d.startswith("image_") and os.path.isdir(
            os.path.join(args.output, d)
        )
    ])

    if args.limit:
        image_dirs = image_dirs[:args.limit]

    total = len(image_dirs)

    # Counters for the two metrics
    any_correct  = 0   # Metric 1: any candidate's top CLIP match == gt
    top1_correct = 0   # Metric 2: rank-0 candidate's top CLIP match == gt

    print(f"\nAnalysing {total} images...\n")
    print(f"  {'idx':<6} {'gt_class':<22} {'rank0_match':<22} "
          f"{'any_match':<22} {'m1':>4} {'m2':>4}")
    print("  " + "-"*85)

    for image_dir in image_dirs:
        image_idx = int(os.path.basename(image_dir).split("_")[1])
        gt_class  = class_names[image_idx]

        with open(os.path.join(image_dir, "retrieved_classes.json")) as f:
            retrieved = json.load(f)

        candidates_results = []
        any_match_class    = None
        rank0_match_class  = None

        for item in retrieved:
            rank           = item["rank"]
            candidate_path = os.path.join(image_dir, f"candidate_{rank}.png")

            if not os.path.exists(candidate_path):
                continue

            # CLIP embedding of the generated candidate
            cand_emb = extract_clip_embedding(
                Image.open(candidate_path).convert("RGB")
            )  # [1, dim]

            # Cosine similarity vs all 200 test images
            sims     = (cand_emb @ img_features.T).squeeze(0)  # [200]
            top_vals, top_idx = torch.topk(sims, k=5)

            top5 = [
                {
                    "class": class_names[int(idx)],
                    "cosine": round(float(top_vals[k]), 6),
                }
                for k, idx in enumerate(top_idx.tolist())
            ]

            top1_class = top5[0]["class"]

            # Save per-candidate result
            with open(os.path.join(image_dir,
                                   f"candidate_{rank}_clip_match.json"), "w") as f:
                json.dump({
                    "rank":            rank,
                    "generated_class": item["class"],
                    "top1_clip_match": top1_class,
                    "is_correct":      top1_class == gt_class,
                    "top5_clip":       top5,
                }, f, indent=2)

            candidates_results.append({
                "rank":            rank,
                "generated_class": item["class"],
                "top1_clip_match": top1_class,
                "top1_cosine":     top5[0]["cosine"],
                "is_correct":      top1_class == gt_class,
            })

            # Track rank-0 candidate
            if rank == 0:
                rank0_match_class = top1_class

            # Track if any candidate matches gt
            if top1_class == gt_class:
                any_match_class = top1_class

        # Metric 1: any candidate correct
        m1 = any_match_class is not None
        any_correct  += int(m1)

        # Metric 2: rank-0 candidate correct
        m2 = rank0_match_class == gt_class
        top1_correct += int(m2)

        print(f"  {image_idx:<6} {gt_class:<22} "
              f"{str(rank0_match_class):<22} "
              f"{str(any_match_class or '-'):<22} "
              f"{'OK' if m2 else 'X':>4} "
              f"{'OK' if m1 else 'X':>4}")

        # Save per-image summary
        with open(os.path.join(image_dir, "clip_analysis.json"), "w") as f:
            json.dump({
                "gt_class":         gt_class,
                "rank0_clip_match": rank0_match_class,
                "any_clip_match":   any_match_class,
                "metric1_any":      m1,
                "metric2_rank0":    m2,
                "candidates":       candidates_results,
            }, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print(f"RESULTS over {total} images:")
    print(f"  Metric 1 (any candidate correct) : "
          f"{any_correct}/{total} = {any_correct/total:.4f}")
    print(f"  Metric 2 (rank-0 correct)        : "
          f"{top1_correct}/{total} = {top1_correct/total:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()