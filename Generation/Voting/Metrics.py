# =============================================================================
# Pipeline Metrics
# =============================================================================
# Single experiment:
#   python metrics.py --output /path/to/pipeline_output
#
# Multiple experiments with different weights (generates comparison PDF):
#   python metrics.py \
#       --experiments \
#           "w=0.5/0.5:/path/to/exp1" \
#           "w=0.7/0.3:/path/to/exp2" \
#           "w=0.3/0.7:/path/to/exp3" \
#       --pdf comparison.pdf

# --- CONFIG -------------------------------------------------------------------

PIPELINE_OUTPUT_DIR = "/hhome/ricse01/TFM/TFM/pipeline_output"

# --- IMPORTS ------------------------------------------------------------------

import os
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np


# --- HELPERS ------------------------------------------------------------------

def load_image_results(pipeline_output_dir):
    """
    Load retrieved_classes.json and rerank_scores.json for every image_XXXX folder.
    Returns a list of dicts, one per image, sorted by image index.
    """
    image_dirs = sorted([
        os.path.join(pipeline_output_dir, d)
        for d in os.listdir(pipeline_output_dir)
        if d.startswith("image_") and os.path.isdir(
            os.path.join(pipeline_output_dir, d)
        )
    ])

    results = []
    for image_dir in image_dirs:
        image_idx = int(os.path.basename(image_dir).split("_")[1])

        retrieved_path = os.path.join(image_dir, "retrieved_classes.json")
        rerank_path    = os.path.join(image_dir, "rerank_scores.json")

        if not os.path.exists(retrieved_path):
            print(f"  [WARNING] Missing retrieved_classes.json in {image_dir}, skipping.")
            continue
        if not os.path.exists(rerank_path):
            print(f"  [WARNING] Missing rerank_scores.json in {image_dir}, skipping.")
            continue

        with open(retrieved_path) as f:
            retrieved = json.load(f)

        with open(rerank_path) as f:
            rerank = json.load(f)

        results.append({
            "image_idx":      image_idx,
            "image_dir":      image_dir,
            "retrieved":      retrieved,
            "gt_class":       rerank["gt_class"],
            "selected_class": rerank["selected_class"],
            "is_correct":     rerank["is_correct"],
        })

    return results


# --- METRICS ------------------------------------------------------------------

def compute_metrics(results):
    """
    Computes:
        top5_accuracy   — gt_class appears in the top-5 retrieved classes
        rerank_accuracy — re-ranking selected the correct class
    """
    total = len(results)

    top5_correct   = 0
    rerank_correct = 0
    per_image      = []

    for r in results:
        retrieved_classes   = [item["class"] for item in r["retrieved"]]
        in_top5             = r["gt_class"] in retrieved_classes
        rerank_correct_flag = r["is_correct"]

        top5_correct   += int(in_top5)
        rerank_correct += int(rerank_correct_flag)

        per_image.append({
            "image_idx":      r["image_idx"],
            "gt_class":       r["gt_class"],
            "selected_class": r["selected_class"],
            "retrieved_top5": retrieved_classes,
            "in_top5":        in_top5,
            "rerank_correct": rerank_correct_flag,
        })

    return {
        "total":           total,
        "top5_correct":    top5_correct,
        "rerank_correct":  rerank_correct,
        "top5_accuracy":   round(top5_correct   / total, 4),
        "rerank_accuracy": round(rerank_correct / total, 4),
    }, per_image


# --- PRINT --------------------------------------------------------------------

def print_metrics(label, metrics, per_image):
    print("\n" + "="*60)
    print(f"METRICS — {label}")
    print("="*60)
    print(f"  Total images     : {metrics['total']}")
    print(f"  Top-5 accuracy   : {metrics['top5_accuracy']:.4f}  "
          f"({metrics['top5_correct']}/{metrics['total']})")
    print(f"  Rerank accuracy  : {metrics['rerank_accuracy']:.4f}  "
          f"({metrics['rerank_correct']}/{metrics['total']})")
    print("="*60)

    missed = [r for r in per_image if r["in_top5"] and not r["rerank_correct"]]
    if missed:
        print(f"\nReranking failures (gt in top-5 but wrong selection): {len(missed)}")
        print(f"  {'idx':<6} {'gt_class':<22} {'selected':<22} top5")
        print("  " + "-"*90)
        for r in missed:
            print(f"  {r['image_idx']:<6} {r['gt_class']:<22} "
                  f"{r['selected_class']:<22} "
                  f"{', '.join(r['retrieved_top5'])}")

    not_retrieved = [r for r in per_image if not r["in_top5"]]
    if not_retrieved:
        print(f"\nRetrieval failures (gt not in top-5): {len(not_retrieved)}")
        print(f"  {'idx':<6} {'gt_class':<22} top5")
        print("  " + "-"*80)
        for r in not_retrieved:
            print(f"  {r['image_idx']:<6} {r['gt_class']:<22} "
                  f"{', '.join(r['retrieved_top5'])}")


# --- PDF GRAPHS ---------------------------------------------------------------

def make_comparison_pdf(experiments, pdf_path):
    """
    experiments : list of dicts with keys:
        label    — short name for this experiment
        metrics  — dict from compute_metrics()
        per_image — list from compute_metrics()
    pdf_path : output path for the PDF
    """
    labels         = [e["label"]                        for e in experiments]
    top5_accs      = [e["metrics"]["top5_accuracy"]     for e in experiments]
    rerank_accs    = [e["metrics"]["rerank_accuracy"]   for e in experiments]
    top5_counts    = [e["metrics"]["top5_correct"]      for e in experiments]
    rerank_counts  = [e["metrics"]["rerank_correct"]    for e in experiments]
    totals         = [e["metrics"]["total"]             for e in experiments]

    x      = np.arange(len(labels))
    width  = 0.35
    colors = {"top5": "#4C72B0", "rerank": "#DD8452"}

    with pdf_backend.PdfPages(pdf_path) as pdf:

        # ------------------------------------------------------------------
        # Page 1: Accuracy bar chart
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))

        bars1 = ax.bar(x - width/2, top5_accs,   width, label="Top-5 Retrieval",
                       color=colors["top5"],   alpha=0.85)
        bars2 = ax.bar(x + width/2, rerank_accs, width, label="Rerank Selection",
                       color=colors["rerank"], alpha=0.85)

        # Value labels on bars
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title("Top-5 Retrieval vs Rerank Accuracy by Experiment")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page 2: Absolute counts stacked bar
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 2), 5))

        for ax, counts, title, color in zip(
            axes,
            [top5_counts, rerank_counts],
            ["Top-5 Retrieval Correct", "Rerank Selection Correct"],
            [colors["top5"], colors["rerank"]],
        ):
            wrong_counts = [t - c for t, c in zip(totals, counts)]
            ax.bar(x, counts,       label="Correct", color=color,   alpha=0.85)
            ax.bar(x, wrong_counts, label="Wrong",   color="#cccccc", alpha=0.85,
                   bottom=counts)
            for xi, (c, t) in enumerate(zip(counts, totals)):
                ax.text(xi, t + 1, f"{c}/{t}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
            ax.set_ylabel("Number of images")
            ax.set_title(title)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page 3: Rerank accuracy vs Top-5 accuracy scatter
        #         (shows how much reranking recovers from retrieval)
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 5))

        for i, (label, t5, rr) in enumerate(zip(labels, top5_accs, rerank_accs)):
            ax.scatter(t5, rr, s=80, zorder=5)
            ax.annotate(label, (t5, rr),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)

        # Diagonal — rerank == top5 (no gain or loss)
        lim_min = min(min(top5_accs), min(rerank_accs)) - 0.05
        lim_max = max(max(top5_accs), max(rerank_accs)) + 0.05
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                "k--", alpha=0.3, label="rerank = top5")

        ax.set_xlabel("Top-5 Retrieval Accuracy")
        ax.set_ylabel("Rerank Accuracy")
        ax.set_title("Rerank vs Retrieval Accuracy\n(above diagonal = reranking helps)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page 4: Per-image correctness heatmap (rerank) for each experiment
        # ------------------------------------------------------------------
        n_images = max(e["metrics"]["total"] for e in experiments)
        n_exp    = len(experiments)

        grid = np.zeros((n_exp, n_images))
        for ei, exp in enumerate(experiments):
            for r in exp["per_image"]:
                grid[ei, r["image_idx"]] = 1 if r["rerank_correct"] else 0

        fig, ax = plt.subplots(figsize=(min(20, n_images / 5), max(3, n_exp * 0.6 + 1)))
        ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_yticks(range(n_exp))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Image index")
        ax.set_title("Per-image Rerank Correctness (green=correct, red=wrong)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nPDF saved to: {pdf_path}")


# --- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics for one or multiple EEG-to-Image pipeline experiments."
    )
    # Single experiment mode
    parser.add_argument("--output", type=str, default=None,
                        help="Single pipeline output directory")

    # Multi-experiment mode
    parser.add_argument(
        "--experiments", nargs="+",
        default=[
        "w=0.0/1.0:/hhome/ricse01/TFM/TFM/pipeline_output_0_1",
        "w=0.25/0.75:/hhome/ricse01/TFM/TFM/pipeline_output_25_75",
        "w=0.75/0.25:/hhome/ricse01/TFM/TFM/pipeline_output_75_25",
        "w=0.5/0.5:/hhome/ricse01/TFM/TFM/pipeline_output_50_50",
        "w=1.0/0.0:/hhome/ricse01/TFM/TFM/pipeline_output_1_0",
    ],
        metavar="LABEL:PATH",
        help=(
            'List of experiments as "label:path" pairs. '
            'Example: --experiments "w=0.5/0.5:/path/exp1" "w=0.7/0.3:/path/exp2"'
        ),
    )

    parser.add_argument("--pdf",  type=str, default="metrics_comparison.pdf",
                        help="Output PDF path for comparison graphs (multi-experiment mode)")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional path to save per-image results as JSON")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Single experiment
    # ------------------------------------------------------------------
    if args.experiments is None:
        output_dir = args.output or PIPELINE_OUTPUT_DIR
        print(f"\nLoading results from: {output_dir}")
        results = load_image_results(output_dir)
        print(f"Loaded {len(results)} image results.")
        metrics, per_image = compute_metrics(results)
        print_metrics(os.path.basename(output_dir), metrics, per_image)

        if args.save:
            with open(args.save, "w") as f:
                json.dump({"metrics": metrics, "per_image": per_image}, f, indent=2)
            print(f"\nPer-image results saved to: {args.save}")
        return

    # ------------------------------------------------------------------
    # Multi-experiment comparison
    # ------------------------------------------------------------------
    experiments = []
    for entry in args.experiments:
        # Expected format: "label:path"
        parts = entry.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid experiment format '{entry}'. "
                f"Expected 'label:path', e.g. 'w=0.5/0.5:/path/to/exp'"
            )
        label, path = parts[0].strip(), parts[1].strip()

        print(f"\nLoading [{label}] from: {path}")
        results = load_image_results(path)
        print(f"  Loaded {len(results)} image results.")
        metrics, per_image = compute_metrics(results)
        print_metrics(label, metrics, per_image)

        experiments.append({
            "label":     label,
            "path":      path,
            "metrics":   metrics,
            "per_image": per_image,
        })

    make_comparison_pdf(experiments, args.pdf)

    if args.save:
        out = [{"label": e["label"], "metrics": e["metrics"],
                "per_image": e["per_image"]} for e in experiments]
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Per-image results saved to: {args.save}")


if __name__ == "__main__":
    main()