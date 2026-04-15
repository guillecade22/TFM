import os
import re
import csv
import sys
import math
import random
import datetime
import itertools
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader
from einops.layers.torch import Rearrange

os.environ["WANDB_MODE"] = "offline"

# ── project imports ──────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.loss import ClipLoss
from shared.util import wandb_logger
from shared.subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from shared.subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from shared.subject_layers.Embed import DataEmbedding

# Non-zero-shot dataset
from shared.eegdatasets_leaveone_nonzeroshot import EEGDatasetNonZeroShot


# ── Model definitions (identical to Train_ATMS.py) ──────────────────────────

class Config:
    task_name       = 'classification'
    seq_len         = 250
    pred_len        = 250
    output_attention = False
    d_model         = 250
    embed           = 'timeF'
    freq            = 'h'
    dropout         = 0.25
    factor          = 1
    n_heads         = 4
    e_layers        = 1
    d_ff            = 256
    activation      = 'gelu'
    enc_in          = 63


class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10):
        super().__init__()
        self.task_name       = configs.task_name
        self.seq_len         = configs.seq_len
        self.pred_len        = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_embedding   = DataEmbedding(
            configs.seq_len, configs.d_model, configs.embed,
            configs.freq, configs.dropout,
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
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        enc_out, _ = self.encoder(
            self.enc_embedding(x_enc, x_mark_enc, subject_ids),
            attn_mask=None,
        )
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
            nn.Conv2d(40, emb_size, (1, 1)),
            Rearrange('b e h w -> b (h w) e'),
        )

    def forward(self, x):
        return self.projection(self.tsconv(x.unsqueeze(1)))


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kw):
        return x + self.fn(x, **kw)


class FlattenHead(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40):
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
        cfg = Config()
        self.encoder          = iTransformer(cfg)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(cfg.d_model, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg          = Enc_eeg()
        self.proj_eeg         = Proj_eeg()
        self.logit_scale      = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func        = ClipLoss()

    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        return self.proj_eeg(self.enc_eeg(x))


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_id_from_string(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def train_one_epoch(sub, model, loader, optimizer, device,
                    text_features_all, img_features_all):
    model.train()
    text_features_all = text_features_all.to(device).float()
    # One representative image feature per class (first of the train images)
    # img_features_all has shape (n_cls * pics_per_class, 1024)
    # We use stride=pics_per_class to get one per class for logit comparisons
    pics = img_features_all.shape[0] // text_features_all.shape[0]
    img_cls_features = img_features_all[::pics].to(device).float()  # (n_cls, 1024)

    mse_loss  = nn.MSELoss()
    alpha     = 0.90
    subject_id = extract_id_from_string(sub)

    total_loss, correct, total = 0.0, 0, 0
    for eeg, labels, _, text_feat, _, img_feat in loader:
        eeg       = eeg.to(device)
        text_feat = text_feat.to(device).float()
        img_feat  = img_feat.to(device).float()
        labels    = labels.to(device)

        sids = torch.full((eeg.size(0),), subject_id,
                          dtype=torch.long, device=device)
        optimizer.zero_grad()

        eeg_feat    = model(eeg, sids).float()
        logit_scale = model.logit_scale
        img_loss    = model.loss_func(eeg_feat, img_feat, logit_scale)
        reg_loss    = mse_loss(eeg_feat, img_feat)
        loss        = alpha * reg_loss * 10 + (1 - alpha) * img_loss * 10
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        logits      = logit_scale * eeg_feat @ img_cls_features.T
        predicted   = logits.argmax(dim=1)
        total      += eeg.size(0)
        correct    += (predicted == labels).sum().item()

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(sub, model, loader, device,
             text_features_all, img_features_all, k=10):
    model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all  = img_features_all.to(device).float()
    mse_loss   = nn.MSELoss()
    alpha      = 0.99
    subject_id = extract_id_from_string(sub)
    all_labels = set(range(text_features_all.size(0)))

    total_loss, correct, total = 0.0, 0, 0
    for eeg, labels, _, text_feat, _, img_feat in loader:
        eeg       = eeg.to(device)
        text_feat = text_feat.to(device).float()
        img_feat  = img_feat.to(device).float()
        labels    = labels.to(device)

        sids     = torch.full((eeg.size(0),), subject_id,
                              dtype=torch.long, device=device)
        eeg_feat = model(eeg, sids).float()

        logit_scale = model.logit_scale
        reg_loss    = mse_loss(eeg_feat, img_feat)
        img_loss    = model.loss_func(eeg_feat, img_feat, logit_scale)
        total_loss += (alpha * reg_loss * 10 + (1 - alpha) * img_loss * 10).item()

        for i, label in enumerate(labels):
            possible = list(all_labels - {label.item()})
            sel_cls  = random.sample(possible, k - 1) + [label.item()]
            sel_feats = img_features_all[sel_cls]
            logits    = logit_scale * eeg_feat[i] @ sel_feats.T
            pred      = sel_cls[logits.argmax().item()]
            correct  += int(pred == label.item())
            total    += 1

    return total_loss / len(loader), correct / total


# ── main training loop ───────────────────────────────────────────────────────

def main_train_loop(sub, current_time, model, train_loader, test_loader,
                    optimizer, device,
                    text_train_all, text_test_all,
                    img_train_all, img_test_all,
                    config, logger=None):

    logger = wandb_logger(config) if logger else None
    if logger:
        logger.watch(model, logger)

    results      = []
    best_acc     = 0.0
    best_info    = {}
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    v2_accs, v4_accs, v10_accs = [], [], []

    for epoch in range(config.epochs):
        tr_loss, tr_acc = train_one_epoch(
            sub, model, train_loader, optimizer, device,
            text_train_all, img_train_all,
        )

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_dir = (f"./models/contrast/ATMS_nonzeroshot/{sub}/"
                        f"loo{config.leave_one_out_pic}/{current_time}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       f"{ckpt_dir}/{epoch+1}.pth")
            print(f"  Checkpoint saved → {ckpt_dir}/{epoch+1}.pth")

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)

        te_loss, te_acc   = evaluate(sub, model, test_loader, device,
                                     text_test_all, img_test_all, k=10)
        _,       v2_acc   = evaluate(sub, model, test_loader, device,
                                     text_test_all, img_test_all, k=2)
        _,       v4_acc   = evaluate(sub, model, test_loader, device,
                                     text_test_all, img_test_all, k=4)

        test_losses.append(te_loss)
        test_accs.append(te_acc)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(te_acc)   # k=10 from te_acc above

        epoch_res = dict(epoch=epoch+1,
                         train_loss=tr_loss, train_acc=tr_acc,
                         test_loss=te_loss,  test_acc=te_acc,
                         v2_acc=v2_acc, v4_acc=v4_acc)
        results.append(epoch_res)

        if te_acc > best_acc:
            best_acc  = te_acc
            best_info = epoch_res.copy()

        if logger:
            logger.log({"Train Loss": tr_loss, "Train Acc": tr_acc,
                        "Test Loss":  te_loss,  "Test Acc":  te_acc,
                        "v2 Acc": v2_acc, "v4 Acc": v4_acc,
                        "Epoch": epoch})

        print(f"[{epoch+1:3d}/{config.epochs}] "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"te_loss={te_loss:.4f} te_acc={te_acc:.4f} | "
              f"v2={v2_acc:.4f} v4={v4_acc:.4f}")

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].plot(train_losses, label='Train'); axs[0, 0].plot(test_losses, label='Test')
    axs[0, 0].set_title('Loss'); axs[0, 0].legend()
    axs[0, 1].plot(train_accs,  label='Train'); axs[0, 1].plot(test_accs,  label='Test')
    axs[0, 1].set_title('Accuracy'); axs[0, 1].legend()
    axs[1, 0].plot(v2_accs, label='v2'); axs[1, 0].plot(v4_accs, label='v4')
    axs[1, 0].set_title('v2 / v4 Accuracy'); axs[1, 0].legend()
    info = (f"Best epoch {best_info.get('epoch','?')}\n"
            f"Test acc: {best_info.get('test_acc',0):.4f}\n"
            f"v2: {best_info.get('v2_acc',0):.4f}  "
            f"v4: {best_info.get('v4_acc',0):.4f}")
    axs[1, 1].axis('off')
    axs[1, 1].text(0.5, 0.5, info, ha='center', va='center',
                   transform=axs[1, 1].transAxes, fontsize=10)
    plt.tight_layout()
    plt.savefig(f'nonzeroshot_curves_{sub}_loo{config.leave_one_out_pic}.png')
    plt.close()

    if logger:
        logger.finish()

    return results


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Non-zero-shot ATMS training (leave-one-image-out)')
    parser.add_argument('--data_path', type=str,
        default="/hhome/ricse01/TFM/required/Preprocessed_data_250Hz/Preprocessed_data_250Hz/")
    parser.add_argument('--output_dir', type=str,
        default='./outputs/nonzeroshot')
    parser.add_argument('--project',  type=str, default='atms_nonzeroshot')
    parser.add_argument('--entity',   type=str, default='sustech_rethinkingbci')
    parser.add_argument('--name',     type=str, default='nonzeroshot_run')
    parser.add_argument('--lr',       type=float, default=3e-4)
    parser.add_argument('--epochs',   type=int,   default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--logger',   type=bool,  default=True)
    parser.add_argument('--gpu',      type=str,   default='cuda:0')
    parser.add_argument('--subjects', nargs='+',  default=['sub-08'])
    parser.add_argument('--leave_one_out_pic', type=int, default=9,
        help='Image index (0-9) to hold out as the non-zero-shot test image')
    parser.add_argument('--encoder_type', type=str, default='ATMS')
    args = parser.parse_args()

    device = (torch.device(args.gpu)
              if torch.cuda.is_available() else torch.device('cpu'))
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    for sub in args.subjects:
        print(f"\n{'='*60}")
        print(f"Subject: {sub}  |  LOO image index: {args.leave_one_out_pic}")
        print(f"{'='*60}")

        model = ATMS().to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr)

        train_ds = EEGDatasetNonZeroShot(
            args.data_path, subjects=[sub], train=True,
            leave_one_out_pic=args.leave_one_out_pic)
        test_ds  = EEGDatasetNonZeroShot(
            args.data_path, subjects=[sub], train=False,
            leave_one_out_pic=args.leave_one_out_pic)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=0, drop_last=True)
        test_loader  = DataLoader(test_ds,  batch_size=1,
                                  shuffle=False, num_workers=0, drop_last=False)

        results = main_train_loop(
            sub, current_time, model,
            train_loader, test_loader, optimizer, device,
            train_ds.text_features, test_ds.text_features,
            train_ds.img_features,  test_ds.img_features,
            config=args, logger=args.logger,
        )

        # Save CSV
        out_dir = os.path.join(args.output_dir, args.encoder_type, sub,
                               f"loo{args.leave_one_out_pic}", current_time)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"ATMS_{sub}_loo{args.leave_one_out_pic}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved → {csv_path}")


if __name__ == '__main__':
    main()
