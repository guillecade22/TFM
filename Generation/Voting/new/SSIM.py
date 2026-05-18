# =============================================================================
# SSIM Reranking
# =============================================================================
# For each image:
#   1. Compute SSIM between every candidate and all 200 test images
#   2. For each candidate, save the full SSIM ranking against the test set
#   3. Select the candidate with the highest SSIM score (vs any test image)
#   4. Report accuracy
#
# Usage:
#   python rerank_ssim.py
#   python rerank_ssim.py --output /path/to/pipeline_output
#   python rerank_ssim.py --limit 5

# --- CONFIG -------------------------------------------------------------------

PIPELINE_OUTPUT_DIR = "/hhome/ricse01/TFM/TFM/pipeline_output"
TEST_IMAGES_DIR     = "/hhome/ricse01/TFM/required/test_images/"

# --- IMPORTS ------------------------------------------------------------------

import os
import json
import shutil
import argparse
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


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


def pil_to_array(pil_image, size=(256, 256)):
    """Resize and convert PIL image to float numpy array in [0, 1]."""
    return np.array(
        pil_image.convert("RGB").resize(size, Image.BILINEAR)
    ).astype(np.float32) / 255.0


def compute_ssim(img_a, img_b):
    return ssim(img_a, img_b, channel_axis=2, data_range=1.0)


# --- LOAD TEST REFERENCE IMAGES -----------------------------------------------

def load_test_images(test_images_dir):
    subfolders = sorted([
        os.path.join(test_images_dir, d)
        for d in os.listdir(test_images_dir)
        if os.path.isdir(os.path.join(test_images_dir, d))
    ])
    class_names = []
    ref_arrays  = []
    print(f"Loading {len(subfolders)} test reference images...")
    for sf in subfolders:
        cls      = extract_class_name(os.path.basename(sf))
        img_file = sorted(os.listdir(sf))[0]
        img      = Image.open(os.path.join(sf, img_file))
        class_names.append(cls)
        ref_arrays.append(pil_to_array(img))
    print(f"  Done.\n")
    return class_names, ref_arrays


# --- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=PIPELINE_OUTPUT_DIR)
    parser.add_argument("--limit",  type=int, default=None)
    args = parser.parse_args()

    class_names, ref_arrays = load_test_images(TEST_IMAGES_DIR)

    image_dirs = sorted([
        os.path.join(args.output, d)
        for d in os.listdir(args.output)
        if d.startswith("image_") and os.path.isdir(
            os.path.join(args.output, d)
        )
    ])

    if args.limit:
        image_dirs = image_dirs[:args.limit]

    correct = 0
    total   = len(image_dirs)

    print(f"Processing {total} images...\n")

    for image_dir in image_dirs:
        image_idx = int(os.path.basename(image_dir).split("_")[1])
        gt_class  = class_names[image_idx]

        with open(os.path.join(image_dir, "retrieved_classes.json")) as f:
            retrieved = json.load(f)

        # ----------------------------------------------------------------
        # For every candidate compute SSIM against all 200 test images
        # ----------------------------------------------------------------
        best_candidate_class = None
        best_candidate_ssim  = -1.0
        all_candidates_info  = []

        for item in retrieved:
            rank           = item["rank"]
            candidate_path = os.path.join(image_dir, f"candidate_{rank}.png")

            if not os.path.exists(candidate_path):
                print(f"  [WARNING] Missing {candidate_path}, skipping.")
                continue

            candidate_arr = pil_to_array(Image.open(candidate_path))

            # SSIM vs all 200 test images, sorted best first
            scores = sorted(
                [
                    {"class": cls, "ssim": round(compute_ssim(candidate_arr, ref), 6)}
                    for cls, ref in zip(class_names, ref_arrays)
                ],
                key=lambda x: x["ssim"],
                reverse=True,
            )

            top_ssim = scores[0]["ssim"]

            # Save full SSIM ranking for this candidate
            with open(os.path.join(image_dir, f"candidate_{rank}_ssim.json"), "w") as f:
                json.dump({
                    "rank":        rank,
                    "class":       item["class"],
                    "top_ssim":    top_ssim,
                    "ssim_ranking": scores,
                }, f, indent=2)

            all_candidates_info.append({
                "rank":     rank,
                "class":    item["class"],
                "top_ssim": top_ssim,
            })

            # Track which candidate has the highest SSIM overall
            if top_ssim > best_candidate_ssim:
                best_candidate_ssim  = top_ssim
                best_candidate_class = item["class"]
                best_candidate_path  = candidate_path

        # ----------------------------------------------------------------
        # Select the candidate with the highest top SSIM
        # ----------------------------------------------------------------
        is_correct = best_candidate_class == gt_class
        correct   += int(is_correct)

        # Save selection result
        with open(os.path.join(image_dir, "ssim_selection.json"), "w") as f:
            json.dump({
                "gt_class":       gt_class,
                "selected_class": best_candidate_class,
                "is_correct":     is_correct,
                "best_ssim":      round(best_candidate_ssim, 6),
                "candidates":     sorted(
                    all_candidates_info,
                    key=lambda x: x["top_ssim"],
                    reverse=True,
                ),
            }, f, indent=2)

        shutil.copy2(best_candidate_path,
                     os.path.join(image_dir, "selected_ssim.png"))

        print(f"  [{image_idx:04d}] gt='{gt_class}'  "
              f"selected='{best_candidate_class}'  "
              f"correct={is_correct}  "
              f"ssim={best_candidate_ssim:.4f}")

    print(f"\n{'='*60}")
    print(f"Done. {total} images processed.")
    print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()