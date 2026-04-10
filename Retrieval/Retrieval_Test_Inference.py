"""
Retrieval Inference Script for ATMS EEG Model
----------------------------------------------
Loads a trained ATMS checkpoint and runs inference on the test set.
For each test EEG sample, computes cosine similarity against all class
image CLIP embeddings and reports:
  - Top-1 predicted class name
  - Top-5 predicted class names
  - Whether the prediction is correct
  - Confidence score (cosine similarity of top-1)

The output is a CSV file with one row per test sample, plus a summary
of Top-1 and Top-5 accuracy.

Usage:
  python retrieval_inference.py \
      --checkpoint /path/to/model.pth \
      --data_path  /hhome/ricse01/TFM/required/Preprocessed_data_250Hz/Preprocessed_data_250Hz/ \
      --subject    sub-08 \
      --output     retrieval_results.csv
"""

import os
import re
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

# -- model imports (same as training script) ----------------------------------
from einops.layers.torch import Rearrange
from shared.eegdatasets_leaveone import EEGDataset
from shared.subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from shared.subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from shared.subject_layers.Embed import DataEmbedding

import warnings
warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Model definition (copied exactly from training script)
# -----------------------------------------------------------------------------

class Config:
    def __init__(self):
        self.task_name = 'classification'
        self.seq_len = 250
        self.pred_len = 250
        self.output_attention = False
        self.d_model = 250
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.25
        self.factor = 1
        self.n_heads = 4
        self.e_layers = 1
        self.d_ff = 256
        self.activation = 'gelu'
        self.enc_in = 63


class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_embedding = DataEmbedding(
            configs.seq_len, configs.d_model, configs.embed, configs.freq,
            configs.dropout, joint_train=False, num_subjects=num_subjects
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :63, :]
        return enc_out


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(PatchEmbedding(emb_size), FlattenHead())


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class ATMS(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=2,
                 num_features=64, num_latents=1024, num_blocks=1):
        super(ATMS, self).__init__()
        default_config = Config()
        self.encoder = iTransformer(default_config)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)]
        )
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None


def load_model(checkpoint_path, device):
    model = ATMS()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

def run_inference(model, dataloader, img_features_all, text_features_all,
                  class_names, subject_id, device, top_k=5):
    """
    For each test EEG sample:
      - Get EEG embedding from model
      - Compare against all class image CLIP embeddings
      - Record top-k predictions and whether top-1 is correct

    Returns a list of dicts, one per sample.
    """
    model.eval()
    img_features_all = F.normalize(img_features_all.to(device).float(), dim=-1)
    logit_scale = model.logit_scale

    # class_names has one entry per class (200 classes)
    # img_features_all has one entry per class image (one per class for test set)
    # Take every 10th if features were stored with 10 images per class
    if img_features_all.shape[0] > len(class_names):
        step = img_features_all.shape[0] // len(class_names)
        img_features_all = img_features_all[::step]
        print(f"Subsampled img_features_all to {img_features_all.shape[0]} entries (one per class)")

    results = []
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            labels   = labels.to(device)

            batch_size  = eeg_data.size(0)
            subject_ids = torch.full((batch_size,), subject_id,
                                     dtype=torch.long).to(device)

            # Forward pass -> EEG embedding [batch x 1024]
            eeg_emb = model(eeg_data, subject_ids).float()
            eeg_emb = F.normalize(eeg_emb, dim=-1)

            # Cosine similarity against all class embeddings [batch x n_classes]
            similarities = logit_scale * eeg_emb @ img_features_all.T

            # Top-k predictions
            topk_vals, topk_idxs = torch.topk(similarities, top_k, dim=-1)

            for i in range(batch_size):
                true_label   = labels[i].item()
                top1_idx     = topk_idxs[i, 0].item()
                top1_score   = topk_vals[i, 0].item()
                topk_idx_list = topk_idxs[i].tolist()

                true_class  = class_names[true_label]
                top1_class  = class_names[top1_idx]
                topk_classes = [class_names[j] for j in topk_idx_list]

                is_top1_correct = (top1_idx == true_label)
                is_top5_correct = (true_label in topk_idx_list)

                correct_top1 += int(is_top1_correct)
                correct_top5 += int(is_top5_correct)
                total        += 1

                results.append({
                    "sample_idx":    total - 1,
                    "true_label":    true_label,
                    "true_class":    true_class,
                    "top1_class":    top1_class,
                    "top1_score":    round(top1_score, 4),
                    "top1_correct":  is_top1_correct,
                    "top5_correct":  is_top5_correct,
                    "top5_classes":  " | ".join(topk_classes),
                    "top5_scores":   " | ".join(
                        [str(round(topk_vals[i, j].item(), 4)) for j in range(top_k)]
                    ),
                })

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {total} samples... "
                      f"Top-1 so far: {correct_top1/total:.3f}  "
                      f"Top-5 so far: {correct_top5/total:.3f}")

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total

    print(f"\nFinal accuracy over {total} samples:")
    print(f"  Top-1: {top1_acc:.4f} ({correct_top1}/{total})")
    print(f"  Top-5: {top5_acc:.4f} ({correct_top5}/{total})")

    return results, top1_acc, top5_acc


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ATMS retrieval inference on test set.")
    parser.add_argument("--checkpoint", type=str,
                        default="./models/contrast/ATMS/sub-08/best.pth",
                        help="Path to trained .pth checkpoint.")
    parser.add_argument("--data_path", type=str,
                        default="/hhome/ricse01/TFM/required/Preprocessed_data_250Hz/Preprocessed_data_250Hz/",
                        help="Path to preprocessed EEG dataset.")
    parser.add_argument("--subject", type=str, default="sub-08",
                        help="Subject ID to evaluate.")
    parser.add_argument("--output", type=str, default="retrieval_results.csv",
                        help="Output CSV file path.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top predictions to record.")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto, cuda, or cpu.")
    parser.add_argument("--caption_dir", type=str,
                        default="/hhome/ricse01/TFM/EEG_Image_decode/Generation/captions/",
                        help="Folder where caption txt files will be saved.")
    args = parser.parse_args()

    # -- device ---------------------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # -- load dataset ---------------------------------------------------------
    print(f"\nLoading test dataset for {args.subject}...")
    test_dataset = EEGDataset(args.data_path, subjects=[args.subject], train=False)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              num_workers=0, drop_last=False)
    print(f"Test samples: {len(test_dataset)}")

    # -- inspect dataset attributes to find class names ----------------------
    print("\nDataset attributes:")
    for attr in sorted(vars(test_dataset).keys()):
        val = getattr(test_dataset, attr)
        if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], str):
            print(f"  {attr}: list of strings, len={len(val)}, e.g. {val[:3]}")
        elif isinstance(val, torch.Tensor):
            print(f"  {attr}: Tensor {tuple(val.shape)}")
        elif isinstance(val, (list, tuple)):
            print(f"  {attr}: {type(val).__name__} len={len(val)}")
        else:
            print(f"  {attr}: {type(val).__name__} = {str(val)[:60]}")

    # -- class names: try known attribute names, guard against None -----------
    class_names = None
    for attr in ["labels_text", "class_names", "classes", "label_names",
                 "text_labels", "idx_to_class", "label_list"]:
        if hasattr(test_dataset, attr):
            val = getattr(test_dataset, attr)
            if val is not None:
                class_names = list(val)
                print(f"\nFound class names via dataset.{attr} ({len(class_names)} classes)")
                break

    if class_names is None:
        # Fallback: walk dataloader once and read the text field per sample.
        # Your dataloader yields: (eeg_data, labels, text, text_features, img, img_features)
        # text is the class name string.
        print("\nNo named attribute found - extracting class names from dataloader text field...")
        class_names_dict = {}
        tmp_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                num_workers=0, drop_last=False)
        for eeg_data, labels, text, text_features, img, img_features in tmp_loader:
            label = labels[0].item()
            if label not in class_names_dict:
                name = text[0] if isinstance(text[0], str) else str(text[0])
                class_names_dict[label] = name
        class_names = [class_names_dict[i] for i in sorted(class_names_dict.keys())]
        print(f"Extracted {len(class_names)} class names from text field.")

    print(f"First 5 class names: {class_names[:5]}")
    print(f"Last  5 class names: {class_names[-5:]}")

    # -- image CLIP features (one per class for the test set) -----------------
    img_features_all = test_dataset.img_features   # [n_classes x 1024] or [n_classes*10 x 1024]
    print(f"img_features_all shape: {img_features_all.shape}")

    # -- text features (used only as reference, not for retrieval here) -------
    text_features_all = test_dataset.text_features

    # -- load model -----------------------------------------------------------
    model = load_model(args.checkpoint, device)
    subject_id = extract_id_from_string(args.subject)
    print(f"Subject numeric ID: {subject_id}")

    # -- run inference --------------------------------------------------------
    print(f"\nRunning retrieval inference...")
    results, top1_acc, top5_acc = run_inference(
        model, test_loader, img_features_all, text_features_all,
        class_names, subject_id, device, top_k=args.top_k
    )

    # -- save CSV -------------------------------------------------------------
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {args.output}")

    # -- save caption files ---------------------------------------------------
    # Caption format: one line per test sample (line N = caption for image N).
    # This matches exactly what the SDXL generation script expects when it does:
    #   captions = [line.strip() for line in f.readlines()]
    #
    # captions_top1.txt  : 200 lines, one predicted class per line
    #   e.g. "a photo of an aircraft carrier"
    #
    # captions_top5.txt  : 200 * 5 = 1000 lines, 5 consecutive lines per image
    #   images 0-4 = top-5 candidates for test image 0
    #   images 5-9 = top-5 candidates for test image 1  etc.
    #   This lets you run the generation loop with i*5 .. i*5+5 per test image.
    #
    # captions_top5_per_image.txt : same as top5 but with a header comment
    #   before each group so it is human-readable for debugging.

    os.makedirs(args.caption_dir, exist_ok=True)

    top1_path  = os.path.join(args.caption_dir, "captions_top1.txt")
    top5_path  = os.path.join(args.caption_dir, "captions_top5.txt")
    top5h_path = os.path.join(args.caption_dir, "captions_top5_readable.txt")

    with open(top1_path, "w") as f1,          open(top5_path, "w") as f5,          open(top5h_path, "w") as f5h:

        for r in results:
            # top-1: one caption per image, plain class name as prompt
            top1_caption = f"a photo of {r['top1_class']}"
            f1.write(top1_caption + "\n")

            # top-5: 5 captions per image (one per candidate class)
            top5_cls_list = [c.strip() for c in r["top5_classes"].split("|")]
            top5_scores   = [s.strip() for s in r["top5_scores"].split("|")]

            f5h.write(f"# sample {r['sample_idx']}  true={r['true_class']}  "
                      f"top1_correct={r['top1_correct']}\n")
            for cls, score in zip(top5_cls_list, top5_scores):
                caption = f"a photo of {cls}"
                f5.write(caption + "\n")
                f5h.write(f"  {caption}  (score={score})\n")

    print(f"\nCaption files saved to: {args.caption_dir}")
    print(f"  captions_top1.txt          : {len(results)} lines (1 per test image)")
    print(f"  captions_top5.txt          : {len(results)*5} lines (5 per test image, consecutive)")
    print(f"  captions_top5_readable.txt : same with headers for debugging")
    print(f"\nTo use top-1 in your generation script:")
    print(f"  CAPTIONS_TXT_PATH = '{top1_path}'")
    print(f"\nTo use top-5, loop with:  caption = captions[i*5 + k]  for k in range(5)")

    print(f"\nSummary:")
    print(f"  Subject:  {args.subject}")
    print(f"  Samples:  {len(results)}")
    print(f"  Top-1 accuracy: {top1_acc:.4f}")
    print(f"  Top-5 accuracy: {top5_acc:.4f}")

    # -- show a few examples --------------------------------------------------
    print("\nSample predictions (first 10):")
    print(f"  {'True class':<25} {'Top-1 predicted':<25} {'Score':>7}  {'Correct'}")
    print("  " + "-" * 70)
    for r in results[:10]:
        marker = "OK" if r["top1_correct"] else "X"
        print(f"  {r['true_class']:<25} {r['top1_class']:<25} {r['top1_score']:>7.4f}  {marker}")

    # -- show per-class accuracy ----------------------------------------------
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    for r in results:
        class_total[r["true_class"]]   += 1
        class_correct[r["true_class"]] += int(r["top1_correct"])

    print("\nPer-class Top-1 accuracy (classes with >1 sample):")
    multi = {c: class_correct[c]/class_total[c]
             for c in class_total if class_total[c] > 1}
    for cls, acc in sorted(multi.items(), key=lambda x: -x[1]):
        print(f"  {cls:<30} {acc:.2f} ({class_correct[cls]}/{class_total[cls]})")


if __name__ == "__main__":
    main()