# =============================================================================
# CLIP Candidate Analysis
# =============================================================================
# For each generated candidate, computes CLIP cosine similarity against all
# 200 test images and finds the most similar class.
#
# Prints two metrics:
#
#   Metric 1 — total candidates correct / total candidates (200 * top_n):
#       For each candidate, check if its top CLIP match == gt class.
#       e.g. if 3 out of 5 candidates for image 0 match -> contributes 3/5
#       Final: total_correct / (200 * 5)
#
#   Metric 2 — rank-0 candidate correct / total images:
#       Only the rank-0 candidate (top retrieved class) is checked.
#       Final: correct / 200
#
# Saves per candidate: candidate_K_clip_match.json
# Saves per image:     clip_analysis.json
#
# Usage:
#   python clip_analysis.py
#   python clip_analysis.py --output /path/to/pipeline_output
#   python clip_analysis.py --limit 5

# --- CONFIG -------------------------------------------------------------------

PIPELINE_OUTPUT_DIR    = "/hhome/ricse01/TFM/TFM/pipeline_output"
TEST_IMAGES_DIR        = "/hhome/ricse01/TFM/required/test_images/"
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

    class_names  = collect_class_names(TEST_IMAGES_DIR)
    img_features = load_test_embeddings(VIT_H_14_FEATURES_TEST, class_names)

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

    # Metric 1: total correct candidates / total candidates
    total_candidates_correct = 0
    total_candidates         = 0

    # Metric 2: rank-0 candidate correct / total images
    rank0_correct = 0

    print(f"\nAnalysing {total} images...\n")
    print(f"  {'idx':<6} {'gt_class':<22} {'rank0_match':<22} "
          f"{'cand_correct':<14} {'m2':>4}")
    print("  " + "-"*75)

    for image_dir in image_dirs:
        image_idx = int(os.path.basename(image_dir).split("_")[1])
        gt_class  = class_names[image_idx]

        with open(os.path.join(image_dir, "retrieved_classes.json")) as f:
            retrieved = json.load(f)

        candidates_results  = []
        rank0_match_class   = None
        image_correct_count = 0

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
            sims              = (cand_emb @ img_features.T).squeeze(0)  # [200]
            top_vals, top_idx = torch.topk(sims, k=5)

            top5 = [
                {
                    "class":  class_names[int(top_idx[k])],
                    "cosine": round(float(top_vals[k]), 6),
                }
                for k in range(5)
            ]

            top1_class = top5[0]["class"]
            is_correct = top1_class == gt_class

            total_candidates         += 1
            total_candidates_correct += int(is_correct)
            image_correct_count      += int(is_correct)

            # Save per-candidate result
            with open(os.path.join(image_dir,
                                   f"candidate_{rank}_clip_match.json"), "w") as f:
                json.dump({
                    "rank":            rank,
                    "generated_class": item["class"],
                    "top1_clip_match": top1_class,
                    "is_correct":      is_correct,
                    "top5_clip":       top5,
                }, f, indent=2)

            candidates_results.append({
                "rank":            rank,
                "generated_class": item["class"],
                "top1_clip_match": top1_class,
                "top1_cosine":     top5[0]["cosine"],
                "is_correct":      is_correct,
            })

            if rank == 0:
                rank0_match_class = top1_class

        # Metric 2: rank-0 correct
        m2 = rank0_match_class == gt_class
        rank0_correct += int(m2)

        print(f"  {image_idx:<6} {gt_class:<22} "
              f"{str(rank0_match_class):<22} "
              f"{image_correct_count}/{len(candidates_results):<12} "
              f"{'OK' if m2 else 'X':>4}")

        # Save per-image summary
        with open(os.path.join(image_dir, "clip_analysis.json"), "w") as f:
            json.dump({
                "gt_class":           gt_class,
                "rank0_clip_match":   rank0_match_class,
                "rank0_correct":      m2,
                "candidates_correct": image_correct_count,
                "candidates_total":   len(candidates_results),
                "candidates":         candidates_results,
            }, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print(f"RESULTS over {total} images:")
    print(f"  Metric 1 (all candidates): "
          f"{total_candidates_correct}/{total_candidates} = "
          f"{total_candidates_correct/total_candidates:.4f}")
    print(f"  Metric 2 (rank-0 only)  : "
          f"{rank0_correct}/{total} = {rank0_correct/total:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()