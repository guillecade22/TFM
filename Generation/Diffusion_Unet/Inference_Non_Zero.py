import os
import sys
import re
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Project imports ────────────────────────────────────────────────────────────
from shared.diffusion_prior import DiffusionPriorUNet, Pipe
from shared.custom_pipeline import Generator4Embeds

# ── These mirror Train_Unet_NonZeroShot.py exactly ────────────────────────────
from einops.layers.torch import Rearrange
from shared.subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from shared.subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from shared.subject_layers.Embed import DataEmbedding
from shared.loss import ClipLoss

import json

# ─────────────────────────────────────────────────────────────────────────────
#  Path config (same source as the dataset)
# ─────────────────────────────────────────────────────────────────────────────
_CONFIG_PATH = "/hhome/ricse01/TFM/TFM/shared/data_config.json"
with open(_CONFIG_PATH, "r") as _f:
    _cfg = json.load(_f)

IMG_DIRECTORY_TRAINING = _cfg["img_directory_training"]


# ─────────────────────────────────────────────────────────────────────────────
#  ATMS model  (must match Train_Unet_NonZeroShot.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    def __init__(self):
        self.task_name        = 'classification'
        self.seq_len          = 250
        self.pred_len         = 250
        self.output_attention = False
        self.d_model          = 250
        self.embed            = 'timeF'
        self.freq             = 'h'
        self.dropout          = 0.25
        self.factor           = 1
        self.n_heads          = 4
        self.e_layers         = 1
        self.d_ff             = 256
        self.activation       = 'gelu'
        self.enc_in           = 63


class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10):
        super().__init__()
        self.enc_embedding = DataEmbedding(
            configs.seq_len, configs.d_model,
            configs.embed, configs.freq, configs.dropout,
            joint_train=False, num_subjects=num_subjects,
        )
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor,
                                  attention_dropout=configs.dropout,
                                  output_attention=configs.output_attention),
                    configs.d_model, configs.n_heads,
                ),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation,
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return enc_out[:, :63, :]


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
        return self.projection(self.tsconv(x.unsqueeze(1)))


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class FlattenHead(nn.Sequential):
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
    def __init__(self, num_channels=63, sequence_length=250,
                 num_subjects=2, num_features=64,
                 num_latents=1024, num_blocks=1):
        super().__init__()
        default_config   = Config()
        self.encoder     = iTransformer(default_config)
        self.subject_wise_linear = nn.ModuleList([
            nn.Linear(default_config.d_model, sequence_length)
            for _ in range(num_subjects)
        ])
        self.enc_eeg     = Enc_eeg()
        self.proj_eeg    = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func   = ClipLoss()

    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        return self.proj_eeg(self.enc_eeg(x))


# ─────────────────────────────────────────────────────────────────────────────
#  EEG loading helpers  (mirrors EEGDatasetNonZeroShot, no CLIP dependency)
# ─────────────────────────────────────────────────────────────────────────────
N_TRAIN_CLASSES   = 1654
SAMPLES_PER_CLASS = 10   # repetitions per class in training file
TIME_SLICE_OFFSET = 50   # drop first 50 time points (same as dataset)


def load_held_out_eeg(data_path: str, subject: str,
                      holdout_img_idx: int,
                      time_window: list,
                      classes: list = None) -> tuple:
    """
    Load and return the held-out EEG repetitions for every training class.

    Returns
    -------
    eeg_data   : Tensor (n_cls, n_ch, n_times_windowed)
    labels     : Tensor (n_cls,)   — local 0-based class indices
    class_indices : list[int]      — original global class indices
    image_paths   : list[str]      — path to the held-out training image
    texts         : list[str]      — text descriptions ("This picture is …")
    """
    if classes is None:
        classes = list(range(N_TRAIN_CLASSES))

    file_path = os.path.join(data_path, subject, 'preprocessed_eeg_training.npy')
    raw       = np.load(file_path, allow_pickle=True)
    # shape: (N_TRAIN_CLASSES * SAMPLES_PER_CLASS, n_repeats, n_ch, n_times)
    eeg_all   = torch.from_numpy(raw['preprocessed_eeg_data']).float()
    times     = torch.from_numpy(raw['times']).float()[TIME_SLICE_OFFSET:]

    # Time-window mask
    start, end  = time_window
    t_mask      = (times >= start) & (times <= end)

    all_folders = sorted([
        d for d in os.listdir(IMG_DIRECTORY_TRAINING)
        if os.path.isdir(os.path.join(IMG_DIRECTORY_TRAINING, d))
    ])

    eeg_list, label_list, image_paths, texts = [], [], [], []

    for local_idx, class_idx in enumerate(classes):
        # ── Text ──────────────────────────────────────────────────────────
        folder_name = all_folders[class_idx]
        try:
            sep         = folder_name.index('_')
            description = folder_name[sep + 1:]
        except ValueError:
            description = folder_name
        texts.append(f"This picture is {description}")

        # ── Held-out image path ────────────────────────────────────────────
        folder_path = os.path.join(IMG_DIRECTORY_TRAINING, folder_name)
        all_imgs    = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        img_file_idx = holdout_img_idx % len(all_imgs)
        image_paths.append(os.path.join(folder_path, all_imgs[img_file_idx]))

        # ── EEG: held-out repetition, averaged over temporal repeats ───────
        start_row = class_idx * SAMPLES_PER_CLASS
        held_eeg  = eeg_all[start_row + holdout_img_idx]   # (n_repeats, n_ch, n_times)
        held_avg  = torch.mean(held_eeg, dim=0)            # (n_ch, n_times)
        held_avg  = held_avg[..., t_mask]                  # apply time window

        eeg_list.append(held_avg)
        label_list.append(local_idx)

    eeg_data = torch.stack(eeg_list, dim=0)                # (n_cls, n_ch, n_times_w)
    labels   = torch.tensor(label_list, dtype=torch.long)  # (n_cls,)

    print(f"[EEG loader] subject={subject} | holdout={holdout_img_idx} | "
          f"eeg_data={eeg_data.shape} | n_classes={len(classes)}")
    return eeg_data, labels, classes, image_paths, texts


def extract_id_from_string(s: str) -> int:
    match = re.search(r'\d+$', s)
    return int(match.group()) if match else 0


# ─────────────────────────────────────────────────────────────────────────────
#  Main inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(args):
    device = (torch.device(args.gpu)
              if torch.cuda.is_available() and args.device == 'gpu'
              else torch.device('cpu'))
    print(f"Device: {device}")

    # ── 1. Load held-out EEG ────────────────────────────────────────────────
    print("\nLoading held-out EEG data …")
    eeg_data, labels, class_indices, image_paths, texts = load_held_out_eeg(
        data_path=args.data_path,
        subject=args.subject,
        holdout_img_idx=args.holdout_img_idx,
        time_window=args.time_window,
        classes=args.classes,           # None → all 1654 classes
    )
    n_samples  = eeg_data.shape[0]
    subject_id = extract_id_from_string(args.subject)

    # ── 2. Load ATMS encoder and compute EEG embeddings ────────────────────
    print(f"\nLoading ATMS checkpoint: {args.atms_ckpt}")
    atms_model = ATMS().to(device)
    atms_model.load_state_dict(
        torch.load(args.atms_ckpt, map_location=device)
    )
    atms_model.eval()

    print("Computing EEG embeddings …")
    eeg_embeddings = []
    batch_size = args.embed_batch_size
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_eeg = eeg_data[i:i + batch_size].to(device)        # (B, n_ch, n_t)
            B         = batch_eeg.size(0)
            subj_ids  = torch.full((B,), subject_id,
                                   dtype=torch.long, device=device)
            emb = atms_model(batch_eeg, subj_ids)                    # (B, 1024)
            eeg_embeddings.append(emb.cpu())
            print(f"  Embedded {min(i + batch_size, n_samples)}/{n_samples}")

    eeg_embeddings = torch.cat(eeg_embeddings, dim=0).to(device)     # (n_cls, 1024)
    print(f"EEG embeddings shape: {eeg_embeddings.shape}")

    # Optionally save embeddings for later reuse
    if args.save_embeddings:
        emb_path = os.path.join(
            args.output_dir,
            f"eeg_embeddings_{args.subject}_hold{args.holdout_img_idx}.pt"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(eeg_embeddings.cpu(), emb_path)
        print(f"Embeddings saved → {emb_path}")

    # ── 3. Load diffusion prior ─────────────────────────────────────────────
    print(f"\nLoading diffusion prior: {args.diffusion_ckpt}")
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1).to(device)
    diffusion_prior.load_state_dict(
        torch.load(args.diffusion_ckpt, map_location=device)
    )
    diffusion_prior.eval()

    pipe      = Pipe(diffusion_prior, device=device)
    generator = Generator4Embeds(
        num_inference_steps=args.num_inference_steps,
        device=device,
    )

    # ── 4. Generate images ──────────────────────────────────────────────────
    print(f"\nGenerating images → {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    for k in range(n_samples):
        eeg_embed   = eeg_embeddings[k:k + 1]          # (1, 1024)
        text_label  = texts[k]
        gt_img_path = image_paths[k]

        # Folder name: class index + sanitised description
        safe_name   = text_label.replace("This picture is ", "").replace("/", "-")
        sample_dir  = os.path.join(
            args.output_dir,
            f"{class_indices[k]:04d}_{safe_name}"
        )
        os.makedirs(sample_dir, exist_ok=True)

        # ── Copy / symlink ground-truth image for easy comparison ──────────
        gt_save_path = os.path.join(sample_dir, "ground_truth.jpg")
        if not os.path.exists(gt_save_path):
            try:
                gt_img = Image.open(gt_img_path).convert("RGB")
                gt_img.save(gt_save_path)
            except Exception as e:
                print(f"  [warn] Could not save ground truth for {text_label}: {e}")

        # ── Diffusion prior: EEG embedding → image latent ─────────────────
        with torch.no_grad():
            h = pipe.generate(
                c_embeds=eeg_embed,
                num_inference_steps=args.prior_steps,
                guidance_scale=args.guidance_scale,
            )

        # ── Decode latent → image(s) ───────────────────────────────────────
        for j in range(args.n_images_per_sample):
            image     = generator.generate(h.to(dtype=torch.float16))
            save_path = os.path.join(sample_dir, f"generated_{j:02d}.png")
            image.save(save_path)

        print(f"[{k+1:4d}/{n_samples}] {safe_name[:60]}  → {sample_dir}")

    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Non-zero-shot inference: generate images from held-out EEG")

    # Paths
    parser.add_argument('--data_path', type=str,
                        default="/hhome/ricse01/TFM/required/Preprocessed_data_250Hz/Preprocessed_data_250Hz/",
                        help="Root path to preprocessed EEG data")
    parser.add_argument('--atms_ckpt', type=str, required=True,
                        help="Path to the trained ATMS .pth checkpoint "
                             "(from Train_Unet_NonZeroShot.py)"
                             , default="/hhome/ricse01/TFM/TFM/models/nonzeroshot/ATMS/sub-08/04-15_00-20/20.pth")
    parser.add_argument('--diffusion_ckpt', type=str,
                        default="/hhome/ricse01/TFM/required/sub-08/diffusion_prior.pt",
                        help="Path to the diffusion prior checkpoint")
    parser.add_argument('--output_dir', type=str,
                        default="./generated_imgs/nonzeroshot",
                        help="Root directory for generated images")

    # Experiment settings
    parser.add_argument('--subject', type=str, default='sub-08')
    parser.add_argument('--holdout_img_idx', type=int, default=0,
                        help="Which EEG repetition was held out during training (0–9). "
                             "Must match the value used in Train_Unet_NonZeroShot.py")
    parser.add_argument('--time_window', type=float, nargs=2, default=[0.0, 1.0],
                        metavar=('START', 'END'),
                        help="EEG time window in seconds (default: 0.0 1.0)")
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help="Optional subset of class indices to run inference on. "
                             "Default: all 1654 training classes")

    # Generation settings
    parser.add_argument('--n_images_per_sample', type=int, default=10,
                        help="How many images to generate per EEG sample")
    parser.add_argument('--prior_steps', type=int, default=50,
                        help="Diffusion prior inference steps")
    parser.add_argument('--num_inference_steps', type=int, default=4,
                        help="Generator (SDXL) inference steps")
    parser.add_argument('--guidance_scale', type=float, default=5.0)

    # Misc
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--embed_batch_size', type=int, default=64,
                        help="Batch size for ATMS embedding pass")
    parser.add_argument('--save_embeddings', action='store_true',
                        help="Save computed EEG embeddings to output_dir")

    args = parser.parse_args()

    run_inference(args)


if __name__ == '__main__':
    main()