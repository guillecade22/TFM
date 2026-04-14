import os, sys, csv, math, re, random, itertools, datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ── Path setup (adjust to match your project layout) ─────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.eeg_non_zeroshot import EEGDatasetNonZeroShot
from shared.util import wandb_logger
from shared.loss import ClipLoss
from shared.subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from shared.subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from shared.subject_layers.Embed import DataEmbedding

from einops.layers.torch import Rearrange


# ─────────────────────────────────────────────────────────────────────────────
#  Model definitions  (identical to original Train_Unet.py)
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    def __init__(self):
        self.task_name      = 'classification'
        self.seq_len        = 250
        self.pred_len       = 250
        self.output_attention = False
        self.d_model        = 250
        self.embed          = 'timeF'
        self.freq           = 'h'
        self.dropout        = 0.25
        self.factor         = 1
        self.n_heads        = 4
        self.e_layers       = 1
        self.d_ff           = 256
        self.activation     = 'gelu'
        self.enc_in         = 63


class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10):
        super().__init__()
        self.task_name        = configs.task_name
        self.seq_len          = configs.seq_len
        self.pred_len         = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_embedding    = DataEmbedding(
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
        x = self.tsconv(x.unsqueeze(1))
        return self.projection(x)


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
        x             = self.encoder(x, None, subject_ids)
        eeg_embedding = self.enc_eeg(x)
        return self.proj_eeg(eeg_embedding)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    return int(match.group()) if match else None


# ─────────────────────────────────────────────────────────────────────────────
#  Train / Evaluate
# ─────────────────────────────────────────────────────────────────────────────
def train_model(sub, eeg_model, dataloader, optimizer, device,
                text_features_all, img_features_all, config):
    eeg_model.train()
    text_features_all = text_features_all.to(device).float()
    # img_features_all has one feature per unique image; for computing logits
    # we use the first image per class (index 0 of every 9-image block).
    # A cleaner approach: take one representative per class (every 9th image).
    img_features_per_class = img_features_all[::9].to(device).float()  # (n_cls, d)

    total_loss, correct, total = 0.0, 0, 0
    mse_loss_fn = nn.MSELoss()
    alpha = 0.90
    subject_id = extract_id_from_string(sub)

    for batch_idx, (eeg_data, labels, _, text_features, _, img_features) in enumerate(dataloader):
        eeg_data      = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features  = img_features.to(device).float()
        labels        = labels.to(device)

        batch_size  = eeg_data.size(0)
        subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)

        optimizer.zero_grad()
        eeg_features = eeg_model(eeg_data, subject_ids).float()

        logit_scale = eeg_model.logit_scale
        img_loss    = eeg_model.loss_func(eeg_features, img_features, logit_scale)
        regress_loss = mse_loss_fn(eeg_features, img_features)
        loss        = alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Training accuracy: match against one-image-per-class gallery
        logits_img  = logit_scale * eeg_features @ img_features_per_class.T
        predicted   = torch.argmax(logits_img, dim=1)
        total      += batch_size
        correct    += (predicted == labels).sum().item()

        del eeg_data, eeg_features, img_features

    return total_loss / (batch_idx + 1), correct / total


def evaluate_model(sub, eeg_model, dataloader, device,
                   text_features_all, img_features_all, k, config):
    """
    k-way evaluation mirroring the original evaluate_model.
    img_features_all: (n_cls, d)  — one feature per class (held-out image).
    """
    eeg_model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all  = img_features_all.to(device).float()

    n_cls = img_features_all.size(0)
    all_labels = set(range(n_cls))

    total_loss, correct, total = 0.0, 0, 0
    top5_correct_count = 0
    mse_loss_fn        = nn.MSELoss()
    alpha              = 0.99
    subject_id         = extract_id_from_string(sub)

    with torch.no_grad():
        for batch_idx, (eeg_data, labels, _, text_features, _, img_features) in enumerate(dataloader):
            eeg_data      = eeg_data.to(device)
            text_features = text_features.to(device).float()
            img_features  = img_features.to(device).float()
            labels        = labels.to(device)

            batch_size  = eeg_data.size(0)
            subject_ids = torch.full((batch_size,), subject_id,
                                     dtype=torch.long).to(device)
            eeg_features = eeg_model(eeg_data, subject_ids)

            logit_scale  = eeg_model.logit_scale
            img_loss     = eeg_model.loss_func(eeg_features, img_features, logit_scale)
            regress_loss = mse_loss_fn(eeg_features, img_features)
            loss         = alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10
            total_loss  += loss.item()

            for idx, label in enumerate(labels):
                label_item       = label.item()
                possible_classes = list(all_labels - {label_item})

                if k == n_cls:
                    selected_classes      = list(range(n_cls))
                elif k in (2, 4, 10, 50, 100):
                    selected_classes      = random.sample(possible_classes, k - 1) + [label_item]
                else:
                    selected_classes      = random.sample(possible_classes, k - 1) + [label_item]

                selected_img_features = img_features_all[selected_classes]
                logits_img            = logit_scale * eeg_features[idx] @ selected_img_features.T
                predicted_label       = selected_classes[torch.argmax(logits_img).item()]

                if predicted_label == label_item:
                    correct += 1

                if k >= 5:
                    _, top5_idx = torch.topk(logits_img, min(5, len(selected_classes)), largest=True)
                    if label_item in [selected_classes[i] for i in top5_idx.tolist()]:
                        top5_correct_count += 1

                total += 1

            del eeg_data, eeg_features, img_features

    avg_loss  = total_loss / (batch_idx + 1)
    accuracy  = correct / total
    top5_acc  = top5_correct_count / total if total > 0 else 0.0
    return avg_loss, accuracy, top5_acc


# ─────────────────────────────────────────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────────────────────────────────────────
def main_train_loop(sub, current_time, eeg_model, train_loader, test_loader,
                    optimizer, device,
                    text_features_train_all, text_features_test_all,
                    img_features_train_all,  img_features_test_all,
                    config, logger=None):

    log = wandb_logger(config) if logger else None
    if log:
        log.watch(eeg_model, log)

    train_losses, train_accuracies = [], []
    test_losses,  test_accuracies  = [], []
    v2_accs, v4_accs, v10_accs     = [], [], []
    best_accuracy  = 0.0
    best_epoch_info = {}
    results         = []
    n_test_cls      = img_features_test_all.size(0)

    for epoch in range(config.epochs):
        # ── Train ──────────────────────────────────────────────────────────
        train_loss, train_accuracy = train_model(
            sub, eeg_model, train_loader, optimizer, device,
            text_features_train_all, img_features_train_all, config,
        )

        # ── Save checkpoint every 5 epochs ─────────────────────────────────
        if (epoch + 1) % 5 == 0:
            ckpt_dir = (f"./models/nonzeroshot/{config.encoder_type}/{sub}/{current_time}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = f"{ckpt_dir}/{epoch+1}.pth"
            torch.save(eeg_model.state_dict(), ckpt_path)
            print(f"Model saved → {ckpt_path}")

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # ── Evaluate ────────────────────────────────────────────────────────
        # Full n_cls-way (non-zero-shot analogue of the 200-way zero-shot eval)
        test_loss, test_accuracy, top5_acc = evaluate_model(
            sub, eeg_model, test_loader, device,
            text_features_test_all, img_features_test_all,
            k=n_test_cls, config=config,
        )
        _, v2_acc,  _ = evaluate_model(sub, eeg_model, test_loader, device,
                                        text_features_test_all, img_features_test_all,
                                        k=2, config=config)
        _, v4_acc,  _ = evaluate_model(sub, eeg_model, test_loader, device,
                                        text_features_test_all, img_features_test_all,
                                        k=4, config=config)
        _, v10_acc, _ = evaluate_model(sub, eeg_model, test_loader, device,
                                        text_features_test_all, img_features_test_all,
                                        k=10, config=config)
        _, v50_acc,  v50_top5_acc  = evaluate_model(sub, eeg_model, test_loader, device,
                                                     text_features_test_all, img_features_test_all,
                                                     k=50,  config=config)
        _, v100_acc, v100_top5_acc = evaluate_model(sub, eeg_model, test_loader, device,
                                                     text_features_test_all, img_features_test_all,
                                                     k=100, config=config)

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)

        epoch_results = {
            "epoch":          epoch + 1,
            "train_loss":     train_loss,
            "train_accuracy": train_accuracy,
            "test_loss":      test_loss,
            "test_accuracy":  test_accuracy,
            "v2_acc":         v2_acc,
            "v4_acc":         v4_acc,
            "v10_acc":        v10_acc,
            "top5_acc":       top5_acc,
            "v50_acc":        v50_acc,
            "v100_acc":       v100_acc,
            "v50_top5_acc":   v50_top5_acc,
            "v100_top5_acc":  v100_top5_acc,
        }
        results.append(epoch_results)

        if test_accuracy > best_accuracy:
            best_accuracy   = test_accuracy
            best_epoch_info = {
                "epoch":          epoch + 1,
                "train_loss":     train_loss,
                "train_accuracy": train_accuracy,
                "test_loss":      test_loss,
                "test_accuracy":  test_accuracy,
                "v2_acc":         v2_acc,
                "v4_acc":         v4_acc,
                "v10_acc":        v10_acc,
            }

        if log:
            log.log({
                "Train Loss":     train_loss,
                "Train Accuracy": train_accuracy,
                "Test Loss":      test_loss,
                "Test Accuracy":  test_accuracy,
                "v2 Accuracy":    v2_acc,
                "v4 Accuracy":    v4_acc,
                "v10 Accuracy":   v10_acc,
                "Epoch":          epoch,
            })

        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
              f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f} top5={top5_acc:.4f}")
        print(f"  v2={v2_acc:.4f}  v4={v4_acc:.4f}  v10={v10_acc:.4f}  "
              f"v50={v50_acc:.4f}  v100={v100_acc:.4f}")

    # ── Plots ───────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses,  label='Test Loss')
    axs[0, 0].legend(); axs[0, 0].set_title("Loss Curve")

    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies,  label='Test Accuracy')
    axs[0, 1].legend(); axs[0, 1].set_title("Accuracy Curve")

    axs[1, 0].plot(v2_accs,  label='2-class Accuracy')
    axs[1, 0].legend(); axs[1, 0].set_title("2-Class Accuracy Curve")

    axs[1, 1].plot(v4_accs,  label='4-class Accuracy')
    axs[1, 1].legend(); axs[1, 1].set_title("4-Class Accuracy Curve")

    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend(); axs[2, 0].set_title("10-Class Accuracy Curve")

    if best_epoch_info:
        info_text = (
            f"Best Epoch {best_epoch_info['epoch']}:\n"
            f"train_loss={best_epoch_info['train_loss']:.4f}\n"
            f"train_acc={best_epoch_info['train_accuracy']:.4f}\n"
            f"test_loss={best_epoch_info['test_loss']:.4f}\n"
            f"test_acc={best_epoch_info['test_accuracy']:.4f}\n"
            f"v2={best_epoch_info['v2_acc']:.4f}  "
            f"v4={best_epoch_info['v4_acc']:.4f}  "
            f"v10={best_epoch_info['v10_acc']:.4f}"
        )
        axs[2, 1].axis('off')
        axs[2, 1].text(0.5, 0.5, info_text, fontsize=9,
                       ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()
    plt.suptitle(f'NonZeroShot_{sub}', fontsize=16, y=1.02)
    plot_path = f'nonzeroshot_{sub}.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved → {plot_path}")

    if log:
        log.finish()

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Non-zero-shot EEG training: leave-one-image-out per class')

    parser.add_argument('--data_path', type=str,
                        default="/hhome/ricse01/TFM/required/Preprocessed_data_250Hz/Preprocessed_data_250Hz/",
                        help='Root path to preprocessed EEG data')
    parser.add_argument('--output_dir', type=str, default='./outputs/nonzeroshot',
                        help='Directory to save results')
    parser.add_argument('--project', type=str, default='nonzeroshot_eeg',
                        help='WandB project name')
    parser.add_argument('--entity', type=str, default='sustech_rethinkingbci',
                        help='WandB entity name')
    parser.add_argument('--name', type=str, default='nonzeroshot_sub08',
                        help='Experiment name')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--logger', type=bool, default=True,
                        help='Enable WandB logging')
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--encoder_type', type=str, default='ATMS')
    parser.add_argument('--subject', type=str, default='sub-08',
                        help='Single subject to train on')
    parser.add_argument('--holdout_img_idx', type=int, default=0,
                        help='Which of the 10 EEG repetitions to hold out (0–9)')
    parser.add_argument('--insubject', type=bool, default=True,
                        help='Kept for compatibility; always True here')

    args = parser.parse_args()

    device = (torch.device(args.gpu)
              if args.device == 'gpu' and torch.cuda.is_available()
              else torch.device('cpu'))

    sub          = args.subject
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    # ── Build datasets ───────────────────────────────────────────────────────
    print(f"\n=== Non-zero-shot experiment | subject={sub} | "
          f"holdout_idx={args.holdout_img_idx} ===\n")

    train_dataset = EEGDatasetNonZeroShot(
        data_path=args.data_path,
        subject=sub,
        train=True,
        holdout_img_idx=args.holdout_img_idx,
    )
    test_dataset  = EEGDatasetNonZeroShot(
        data_path=args.data_path,
        subject=sub,
        train=False,
        holdout_img_idx=args.holdout_img_idx,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=1,
                              shuffle=False, num_workers=0, drop_last=False)

    # Feature tensors for loss / evaluation
    text_features_train_all = train_dataset.text_features   # (n_cls, d)
    text_features_test_all  = test_dataset.text_features    # (n_cls, d)
    img_features_train_all  = train_dataset.img_features    # (n_cls*9, d)  one per kept rep
    img_features_test_all   = test_dataset.img_features     # (n_cls, d)    held-out images

    # ── Build model ──────────────────────────────────────────────────────────
    eeg_model = globals()[args.encoder_type]()
    eeg_model.to(device)
    optimizer = AdamW(eeg_model.parameters(), lr=args.lr)

    # ── Train ────────────────────────────────────────────────────────────────
    results = main_train_loop(
        sub, current_time, eeg_model,
        train_loader, test_loader, optimizer, device,
        text_features_train_all, text_features_test_all,
        img_features_train_all,  img_features_test_all,
        config=args, logger=args.logger,
    )

    # ── Save CSV results ─────────────────────────────────────────────────────
    results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir,
                                f"{args.encoder_type}_{sub}_hold{args.holdout_img_idx}.csv")
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved → {results_file}")


if __name__ == '__main__':
    main()