# =============================================================================
# Pipeline Metrics
# =============================================================================
# Single experiment:
#   python metrics.py --output /path/to/pipeline_output
#
# Multiple experiments with different weights (generates comparison PDF):
#   python metrics.py --pdf comparison.pdf
#   python metrics.py --experiments "w=0.5/0.5:/path/exp1" "w=0.7/0.3:/path/exp2"

# --- CONFIG -------------------------------------------------------------------

PIPELINE_OUTPUT_DIR = "/export/hhome/ricse01/TFM/TFM/pipeline_output"

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
            "retrieved":      retrieved,
            "gt_class":       rerank["gt_class"],
            "selected_class": rerank["selected_class"],
            "is_correct":     rerank["is_correct"],
        })

    return results


# --- METRICS ------------------------------------------------------------------

def compute_metrics(results):
    total          = len(results)
    top5_correct   = 0
    rerank_correct = 0
    per_image      = []

    for r in results:
        retrieved_classes = [item["class"] for item in r["retrieved"]]
        in_top5           = r["gt_class"] in retrieved_classes
        is_correct        = r["is_correct"]

        top5_correct   += int(in_top5)
        rerank_correct += int(is_correct)

        per_image.append({
            "image_idx":      r["image_idx"],
            "gt_class":       r["gt_class"],
            "selected_class": r["selected_class"],
            "retrieved_top5": retrieved_classes,
            "in_top5":        in_top5,
            "rerank_correct": is_correct,
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


# --- PDF GRAPH ----------------------------------------------------------------

def make_comparison_pdf(experiments, pdf_path):
    labels      = [e["label"]                      for e in experiments]
    top5_accs   = [e["metrics"]["top5_accuracy"]   for e in experiments]
    rerank_accs = [e["metrics"]["rerank_accuracy"] for e in experiments]

    x     = np.arange(len(labels))
    width = 0.35

    with pdf_backend.PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))

        bars1 = ax.bar(x - width/2, top5_accs,   width, label="Top-5 Retrieval",
                       color="#4C72B0", alpha=0.85)
        bars2 = ax.bar(x + width/2, rerank_accs, width, label="Rerank Selection",
                       color="#DD8452", alpha=0.85)

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

    print(f"\nPDF saved to: {pdf_path}")


# --- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics for one or multiple EEG-to-Image pipeline experiments."
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Single pipeline output directory")
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
        help='List of experiments as "label:path" pairs.',
    )
    parser.add_argument("--pdf",  type=str, default="metrics_comparison.pdf",
                        help="Output PDF path for the comparison graph")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional path to save per-image results as JSON")
    args = parser.parse_args()

    # Single experiment mode (--output given, no --experiments override)
    if args.output is not None:
        print(f"\nLoading results from: {args.output}")
        results = load_image_results(args.output)
        print(f"Loaded {len(results)} image results.")
        metrics, per_image = compute_metrics(results)
        print_metrics(os.path.basename(args.output), metrics, per_image)
        if args.save:
            with open(args.save, "w") as f:
                json.dump({"metrics": metrics, "per_image": per_image}, f, indent=2)
            print(f"\nResults saved to: {args.save}")
        return

    # Multi-experiment mode
    experiments = []
    for entry in args.experiments:
        parts = entry.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid format '{entry}'. Expected 'label:path'.")
        label, path = parts[0].strip(), parts[1].strip()

        print(f"\nLoading [{label}] from: {path}")
        results = load_image_results(path)
        print(f"  Loaded {len(results)} image results.")
        metrics, per_image = compute_metrics(results)
        print_metrics(label, metrics, per_image)

        experiments.append({
            "label":     label,
            "metrics":   metrics,
            "per_image": per_image,
        })

    make_comparison_pdf(experiments, args.pdf)

    if args.save:
        out = [{"label": e["label"], "metrics": e["metrics"],
                "per_image": e["per_image"]} for e in experiments]
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to: {args.save}")


if __name__ == "__main__":
    main()