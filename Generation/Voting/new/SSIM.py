# =============================================================================
# Low-Level Reranking using SSIM
# =============================================================================
# For each generated candidate, computes SSIM against all 200 test images.
# Saves the top-5 most similar test classes and the SSIM-based fidelity score.
#
# Outputs per image (written into existing image_XXXX/ folders):
#   candidate_K_ssim.json  — top-5 similar classes + SSIM scores
#   rerank_scores.json     — all candidates ranked by SSIM fidelity
#   selected.png           — best candidate
#
# Usage:
#   python rerank_ssim.py
#   python rerank_ssim.py --output /path/to/pipeline_output
#   python rerank_ssim.py --limit 5   # debug on first 5 images

# --- CONFIG -------------------------------------------------------------------

PIPELINE_OUTPUT_DIR = "/hhome/ricse01/TFM/TFM/pipeline_output"
TEST_IMAGES_DIR     = "/hhome/ricse01/TFM/required/test_images/"

# Re-ranking weights
W_RETRIEVAL = 0.5   # cosine_sim(eeg_embed, img_features[class_idx])
W_FIDELITY  = 0.5   # SSIM-based fidelity score

# How many top similar classes to save per candidate
TOP_K_SIMILAR = 5

# --- IMPORTS ------------------------------------------------------------------

import os
import json
import shutil
import argparse
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize


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


def pil_to_array(pil_image, size=(256, 256)):
    """Resize PIL image and convert to float numpy array in [0, 1]."""
    img = pil_image.convert("RGB").resize(size, Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def compute_ssim(img_a, img_b):
    """
    Compute SSIM between two numpy arrays [H, W, 3] in [0, 1].
    Returns a float in [-1, 1]. Higher = more similar.
    """
    return ssim(img_a, img_b, channel_axis=2, data_range=1.0)


# --- LOAD TEST REFERENCE IMAGES -----------------------------------------------

def load_test_images(test_images_dir):
    """
    Load and preprocess all test reference images.
    Returns:
        class_names  : list[str]         — class name per image
        ref_arrays   : list[np.ndarray]  — preprocessed image arrays
    """
    subfolders  = collect_subfolders(test_images_dir)
    class_names = []
    ref_arrays  = []

    print(f"Loading {len(subfolders)} test reference images...")
    for sf in subfolders:
        cls      = extract_class_name(os.path.basename(sf))
        img_file = sorted(os.listdir(sf))[0]
        img      = Image.open(os.path.join(sf, img_file))
        class_names.append(cls)
        ref_arrays.append(pil_to_array(img))

    print(f"  Loaded {len(class_names)} reference images.")
    return class_names, ref_arrays


# --- SSIM SCORING -------------------------------------------------------------

def score_candidate_ssim(candidate_path, ref_arrays, class_names, top_k):
    """
    Compute SSIM between a candidate image and all test reference images.

    Returns:
        top_k_results : list of dicts {class, ssim_score} sorted best-first
        max_ssim      : float — highest SSIM score (used as fidelity signal)
    """
    candidate = pil_to_array(Image.open(candidate_path))

    scores = [
        {"class": cls, "ssim_score": round(compute_ssim(candidate, ref), 6)}
        for cls, ref in zip(class_names, ref_arrays)
    ]

    scores.sort(key=lambda x: x["ssim_score"], reverse=True)

    return scores[:top_k], scores[0]["ssim_score"]


# --- RERANKING ----------------------------------------------------------------

def rerank_image(image_dir, ref_arrays, class_names, w_retrieval, w_fidelity):
    """
    Score all candidates for one image and select the best one.

    Returns:
        scored : list of candidate dicts sorted best-first
        best   : dict — winning candidate
    """
    with open(os.path.join(image_dir, "retrieved_classes.json")) as f:
        retrieved = json.load(f)

    scored = []
    for item in retrieved:
        rank            = item["rank"]
        candidate_path  = os.path.join(image_dir, f"candidate_{rank}.png")

        if not os.path.exists(candidate_path):
            print(f"  [WARNING] Missing {candidate_path}, skipping.")
            continue

        # SSIM against all 200 test images
        top_k_similar, max_ssim = score_candidate_ssim(
            candidate_path, ref_arrays, class_names, TOP_K_SIMILAR
        )

        # Save per-candidate SSIM results
        with open(os.path.join(image_dir, f"candidate_{rank}_ssim.json"), "w") as f:
            json.dump({
                "rank":        rank,
                "class":       item["class"],
                "max_ssim":    round(max_ssim, 6),
                "top_similar": top_k_similar,
            }, f, indent=2)

        # Score retrieval: raw_cosine in [-1,1] -> [0,1]
        score_retrieval = (item["raw_cosine"] + 1.0) / 2.0
        # Score fidelity: SSIM in [-1,1] -> [0,1]
        score_fidelity  = (max_ssim + 1.0) / 2.0
        final_score     = w_retrieval * score_retrieval + w_fidelity * score_fidelity

        scored.append({
            "old_rank":       rank,
            "class":          item["class"],
            "candidate_path": candidate_path,
            "scores": {
                "raw_retrieval":   round(item["raw_cosine"], 6),
                "max_ssim":        round(max_ssim,           6),
                "score_retrieval": round(score_retrieval,    6),
                "score_fidelity":  round(score_fidelity,     6),
                "w_retrieval":     w_retrieval,
                "w_fidelity":      w_fidelity,
                "final_score":     round(final_score,        6),
            },
            "top_similar": top_k_similar,
        })

    scored.sort(key=lambda x: x["scores"]["final_score"], reverse=True)
    for new_rank, item in enumerate(scored):
        item["new_rank"] = new_rank

    return scored, scored[0]


# --- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rerank generated candidates using SSIM against full test set."
    )
    parser.add_argument("--output",      type=str,   default=PIPELINE_OUTPUT_DIR)
    parser.add_argument("--w_retrieval", type=float, default=W_RETRIEVAL)
    parser.add_argument("--w_fidelity",  type=float, default=W_FIDELITY)
    parser.add_argument("--limit",       type=int,   default=None,
                        help="Process only the first N images (for debugging)")
    args = parser.parse_args()

    # Load test reference images
    class_names, ref_arrays = load_test_images(TEST_IMAGES_DIR)

    # Collect image folders
    image_dirs = sorted([
        os.path.join(args.output, d)
        for d in os.listdir(args.output)
        if d.startswith("image_") and os.path.isdir(
            os.path.join(args.output, d)
        )
    ])

    if args.limit:
        image_dirs = image_dirs[:args.limit]

    print(f"\nRe-ranking {len(image_dirs)} images...")
    print(f"Weights: w_retrieval={args.w_retrieval}  w_fidelity={args.w_fidelity}\n")

    correct = 0
    total   = len(image_dirs)

    for image_dir in image_dirs:
        image_idx = int(os.path.basename(image_dir).split("_")[1])
        gt_class  = class_names[image_idx]

        print(f"  [{image_idx:04d}] gt='{gt_class}'")

        scored, best = rerank_image(
            image_dir, ref_arrays, class_names,
            args.w_retrieval, args.w_fidelity
        )

        is_correct = best["class"] == gt_class
        correct   += int(is_correct)

        print(f"    Selected: '{best['class']}'  "
              f"correct={is_correct}  "
              f"final={best['scores']['final_score']:.4f}  "
              f"ssim={best['scores']['max_ssim']:.4f}  "
              f"top_similar={[s['class'] for s in best['top_similar']]}")

        with open(os.path.join(image_dir, "rerank_scores.json"), "w") as f:
            json.dump({
                "gt_class":       gt_class,
                "selected_class": best["class"],
                "is_correct":     is_correct,
                "w_retrieval":    args.w_retrieval,
                "w_fidelity":     args.w_fidelity,
                "candidates":     scored,
            }, f, indent=2)

        shutil.copy2(best["candidate_path"],
                     os.path.join(image_dir, "selected.png"))

    print(f"\n{'='*60}")
    print(f"Done. {total} images processed.")
    print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()