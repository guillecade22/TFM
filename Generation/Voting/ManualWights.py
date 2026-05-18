# =============================================================================
# Weight Search — find best W_RETRIEVAL / W_FIDELITY by brute force
# =============================================================================
# Loads saved candidate scores from the pipeline output and evaluates
# accuracy for every weight combination in a given list.
# Plots a heatmap and a curve of accuracy vs weight.
#
# Usage:
#   python weight_search.py
#   python weight_search.py --output /path/to/pipeline_output
#   python weight_search.py --steps 20   # test 20x20 grid instead of default

# --- CONFIG -------------------------------------------------------------------

PIPELINE_OUTPUT_DIR = "/hhome/ricse01/TFM/TFM/pipeline_output"
TEST_IMAGES_DIR     = "/hhome/ricse01/TFM/required/test_images/"
PDF_OUTPUT          = "weight_search.pdf"

# List of w_retrieval values to test.
# w_fidelity = 1 - w_retrieval (they must sum to 1).
# Edit this list directly or use --steps to auto-generate a uniform grid.
W_RETRIEVAL_LIST = [
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]

# --- IMPORTS ------------------------------------------------------------------

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend


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


def load_all_scores(output_dir, class_names):
    """
    Load saved candidate scores for all images.

    Returns:
        records : list of dicts per image:
            gt_class        — ground truth class name
            retrieved       — list of {class, raw_retrieval, raw_fidelity}
    """
    image_dirs = sorted([
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("image_") and os.path.isdir(os.path.join(output_dir, d))
    ])

    records = []
    missing = 0

    for image_dir in image_dirs:
        image_idx = int(os.path.basename(image_dir).split("_")[1])
        gt_class  = class_names[image_idx]

        retrieved_path = os.path.join(image_dir, "retrieved_classes.json")
        if not os.path.exists(retrieved_path):
            missing += 1
            continue

        with open(retrieved_path) as f:
            retrieved = json.load(f)

        candidates = []
        ok = True
        for r in retrieved:
            scores_path = os.path.join(image_dir, f"candidate_{r['rank']}_scores.json")
            if not os.path.exists(scores_path):
                ok = False
                break
            with open(scores_path) as f:
                scores = json.load(f)
            candidates.append({
                "class":         r["class"],
                "raw_retrieval": scores["raw_retrieval"],
                "raw_fidelity":  scores["raw_fidelity"],
            })

        if not ok:
            missing += 1
            continue

        records.append({
            "image_idx": image_idx,
            "gt_class":  gt_class,
            "candidates": candidates,
        })

    if missing:
        print(f"  [WARNING] Skipped {missing} images with missing score files.")

    return records


# --- ACCURACY COMPUTATION -----------------------------------------------------

def compute_accuracy(records, w_retrieval):
    """
    Given a weight w_retrieval (w_fidelity = 1 - w_retrieval),
    compute rerank accuracy over all records.

    score = w_retrieval * (raw_retrieval + 1) / 2
          + w_fidelity  * (raw_fidelity  + 1) / 2
    """
    w_fidelity = 1.0 - w_retrieval
    correct    = 0

    for rec in records:
        best_class = None
        best_score = -1.0

        for cand in rec["candidates"]:
            score = (w_retrieval * (cand["raw_retrieval"] + 1.0) / 2.0 +
                     w_fidelity  * (cand["raw_fidelity"]  + 1.0) / 2.0)
            if score > best_score:
                best_score = score
                best_class = cand["class"]

        if best_class == rec["gt_class"]:
            correct += 1

    return correct / len(records)


# --- PLOTTING -----------------------------------------------------------------

def make_pdf(w_list, accuracies, pdf_path):
    best_idx = int(np.argmax(accuracies))
    best_w   = w_list[best_idx]
    best_acc = accuracies[best_idx]

    with pdf_backend.PdfPages(pdf_path) as pdf:

        # ------------------------------------------------------------------
        # Page 1: Accuracy vs w_retrieval line plot
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(w_list, accuracies, marker="o", color="#4C72B0",
                linewidth=2, markersize=6, label="Rerank accuracy")

        # Mark best point
        ax.scatter([best_w], [best_acc], color="red", s=100, zorder=5,
                   label=f"Best: w_r={best_w:.2f}  acc={best_acc:.4f}")

        # Reference lines
        ax.axvline(x=best_w, color="red",   linestyle="--", alpha=0.4)
        ax.axvline(x=0.0,    color="gray",  linestyle=":",  alpha=0.4,
                   label="w_r=0 (fidelity only)")
        ax.axvline(x=1.0,    color="gray",  linestyle=":",  alpha=0.4,
                   label="w_r=1 (retrieval only)")

        ax.set_xlabel("w_retrieval  (w_fidelity = 1 - w_retrieval)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Rerank Accuracy vs Weight")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, min(1.0, max(accuracies) + 0.1))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Annotate each point with its accuracy
        for w, acc in zip(w_list, accuracies):
            ax.annotate(f"{acc:.3f}", (w, acc),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color="#333333")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page 2: Bar chart — easier to read exact values
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(max(8, len(w_list) * 0.9), 5))

        colors = ["#DD8452" if i == best_idx else "#4C72B0"
                  for i in range(len(w_list))]
        bars = ax.bar([f"{w:.2f}" for w in w_list], accuracies,
                      color=colors, alpha=0.85)

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("w_retrieval  (w_fidelity = 1 - w_retrieval)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Rerank Accuracy by Weight  —  Best: w_r={best_w:.2f}  acc={best_acc:.4f}")
        ax.set_ylim(0, min(1.0, max(accuracies) + 0.1))
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nPDF saved to: {pdf_path}")


# --- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test a list of weights and plot rerank accuracy."
    )
    parser.add_argument("--output",  type=str, default=PIPELINE_OUTPUT_DIR,
                        help="Pipeline output directory")
    parser.add_argument("--pdf",     type=str, default=PDF_OUTPUT,
                        help="Output PDF path")
    parser.add_argument("--steps",   type=int, default=None,
                        help="If set, override W_RETRIEVAL_LIST with a uniform "
                             "grid of this many steps from 0 to 1")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Override W_RETRIEVAL_LIST with a custom list, "
                             "e.g. --weights 0.0 0.3 0.5 0.7 1.0")
    args = parser.parse_args()

    # Build weight list
    if args.weights is not None:
        w_list = sorted(args.weights)
    elif args.steps is not None:
        w_list = [round(i / (args.steps - 1), 4) for i in range(args.steps)]
    else:
        w_list = sorted(W_RETRIEVAL_LIST)

    print(f"\nWeight list ({len(w_list)} values): {w_list}")

    # Load data
    class_names = collect_class_names(TEST_IMAGES_DIR)
    print(f"\nLoading scores from: {args.output}")
    records = load_all_scores(args.output, class_names)
    print(f"Loaded {len(records)} images.")

    # Evaluate each weight
    print(f"\n{'w_retrieval':>12}  {'w_fidelity':>10}  {'accuracy':>10}  {'correct':>8}")
    print("  " + "-" * 48)

    accuracies = []
    for w_r in w_list:
        acc = compute_accuracy(records, w_r)
        accuracies.append(acc)
        correct = int(acc * len(records))
        print(f"  {w_r:>10.4f}  {1-w_r:>10.4f}  {acc:>10.4f}  "
              f"{correct:>4}/{len(records)}")

    best_idx = int(np.argmax(accuracies))
    print(f"\n  Best: w_retrieval={w_list[best_idx]:.4f}  "
          f"w_fidelity={1-w_list[best_idx]:.4f}  "
          f"accuracy={accuracies[best_idx]:.4f}")

    # Plot
    make_pdf(w_list, accuracies, args.pdf)


if __name__ == "__main__":
    main()