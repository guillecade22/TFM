import os
import re
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

os.environ["WANDB_MODE"] = "offline"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.diffusion_prior import DiffusionPriorUNet, EmbeddingDataset, Pipe
from shared.eegdatasets_leaveone_nonzeroshot import EEGDatasetNonZeroShot

# Re-use ATMS definition from the training script
from Train_ATMS_nonzeroshot import ATMS


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_id_from_string(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


@torch.no_grad()
def extract_eeg_features(sub, model, loader, device):
    """Forward-pass the whole dataloader and collect EEG embeddings."""
    model.eval()
    subject_id = extract_id_from_string(sub)
    feats = []
    for eeg, *_ in loader:
        eeg  = eeg.to(device)
        sids = torch.full((eeg.size(0),), subject_id,
                          dtype=torch.long, device=device)
        feats.append(model(eeg, sids).cpu())
    return torch.cat(feats, dim=0)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train DiffusionPrior (non-zero-shot LOO)')
    parser.add_argument('--data_path', type=str,
        default="/hhome/ricse01/TFM/required/Preprocessed_data_250Hz/Preprocessed_data_250Hz/")
    parser.add_argument('--atms_checkpoint', type=str, required=True,
        help='Path to trained ATMS .pth checkpoint')
    parser.add_argument('--vit_train_features', type=str,
        default="/hhome/ricse01/TFM/authors/ViT-H-14_features_train.pt",
        help='ViT-H-14 features for ALL training images '
             '(shape: 1654*10 × 1024 or dict with key img_features)')
    parser.add_argument('--subject', type=str, default='sub-08')
    parser.add_argument('--leave_one_out_pic', type=int, default=9)
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int,   default=1024)
    parser.add_argument('--gpu',        type=str,   default='cuda:0')
    parser.add_argument('--output_dir', type=str,
        default='./fintune_ckpts/ATMS_nonzeroshot')
    args = parser.parse_args()

    device = (torch.device(args.gpu)
              if torch.cuda.is_available() else torch.device('cpu'))
    sub = args.subject
    loo = args.leave_one_out_pic
    PICS_PER_CLASS = 10  # total images per training class

    # ── 1. Load ATMS model ────────────────────────────────────────────────
    print("Loading ATMS model …")
    atms_model = ATMS().to(device)
    state = torch.load(args.atms_checkpoint, map_location=device)
    incompatible = atms_model.load_state_dict(state, strict=False)
    print(f"  Missing:    {incompatible.missing_keys}")
    print(f"  Unexpected: {incompatible.unexpected_keys}")
    atms_model.eval()

    # ── 2. Build datasets (same LOO split used during ATMS training) ──────
    print("Building non-zero-shot datasets …")
    train_ds = EEGDatasetNonZeroShot(
        args.data_path, subjects=[sub], train=True,
        leave_one_out_pic=loo)
    test_ds  = EEGDatasetNonZeroShot(
        args.data_path, subjects=[sub], train=False,
        leave_one_out_pic=loo)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── 3. Extract EEG embeddings ─────────────────────────────────────────
    print("Extracting EEG embeddings (train split) …")
    eeg_feats_train = extract_eeg_features(sub, atms_model, train_loader, device)
    train_feat_path = f"ATM_S_eeg_features_{sub}_train_nzs_loo{loo}.pt"
    torch.save(eeg_feats_train, train_feat_path)
    print(f"  Saved → {train_feat_path}  shape: {eeg_feats_train.shape}")

    print("Extracting EEG embeddings (test / held-out split) …")
    eeg_feats_test  = extract_eeg_features(sub, atms_model, test_loader,  device)
    test_feat_path  = f"ATM_S_eeg_features_{sub}_test_nzs_loo{loo}.pt"
    torch.save(eeg_feats_test,  test_feat_path)
    print(f"  Saved → {test_feat_path}   shape: {eeg_feats_test.shape}")

    # ── 4. Load & align ViT image features ───────────────────────────────
    print("Loading ViT-H-14 image features …")
    vit_data = torch.load(args.vit_train_features, map_location='cpu')
    if isinstance(vit_data, dict):
        vit_all = vit_data['img_features']   # (1654*10, 1024)  or (1654, 10, 1024)
    else:
        vit_all = vit_data
    vit_all = vit_all.float()

    # Reshape to (1654, 10, 1024) regardless of original layout
    if vit_all.dim() == 2:
        # Assumes ordering: class0_img0, class0_img1, ..., class0_img9, class1_img0, ...
        vit_all = vit_all.view(1654, PICS_PER_CLASS, 1024)

    # Select the 9 training images (all except leave_one_out_pic)
    train_pic_idxs = [p for p in range(PICS_PER_CLASS) if p != loo]
    vit_train_9    = vit_all[:, train_pic_idxs, :]  # (1654, 9, 1024)

    # The EEG training data has 4 repetitions per (class, image) pair.
    # eeg_feats_train shape: (1654 * 9 * 4, 1024)
    n_cls      = 1654
    n_pics     = len(train_pic_idxs)   # 9
    n_repeats  = eeg_feats_train.shape[0] // (n_cls * n_pics)

    print(f"  n_cls={n_cls}, n_pics={n_pics}, n_repeats={n_repeats}")
    assert eeg_feats_train.shape[0] == n_cls * n_pics * n_repeats, (
        f"Shape mismatch: EEG has {eeg_feats_train.shape[0]} rows, "
        f"expected {n_cls * n_pics * n_repeats}")

    # Expand ViT features to match EEG: (1654, 9, 1024) → (1654*9*n_repeats, 1024)
    vit_expanded = (vit_train_9
                    .unsqueeze(2)                         # (1654, 9, 1, 1024)
                    .expand(n_cls, n_pics, n_repeats, 1024)
                    .reshape(-1, 1024))                   # (1654*9*n_repeats, 1024)

    print(f"  EEG features:  {eeg_feats_train.shape}")
    print(f"  ViT features:  {vit_expanded.shape}")
    assert eeg_feats_train.shape[0] == vit_expanded.shape[0]

    # ── 5. Train diffusion prior ──────────────────────────────────────────
    print("\nTraining DiffusionPriorUNet …")
    dataset = EmbeddingDataset(
        c_embeddings=eeg_feats_train,
        h_embeddings=vit_expanded,
    )
    dl = DataLoader(dataset, batch_size=args.batch_size,
                    shuffle=True, num_workers=0)

    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    print(f"  Parameters: {sum(p.numel() for p in diffusion_prior.parameters()):,}")

    pipe = Pipe(diffusion_prior, device=device)
    pipe.train(dl, num_epochs=args.epochs, learning_rate=args.lr)

    # ── 6. Save ───────────────────────────────────────────────────────────
    save_dir = os.path.join(args.output_dir, sub, f"loo{loo}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'diffusion_prior.pt')
    torch.save(pipe.diffusion_prior.state_dict(), save_path)
    print(f"\nDiffusion prior saved → {save_path}")


if __name__ == '__main__':
    main()
