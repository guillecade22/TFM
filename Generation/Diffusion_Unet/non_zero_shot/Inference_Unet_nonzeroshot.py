import os
import argparse

import torch
from PIL import Image

from shared.diffusion_prior import DiffusionPriorUNet, Pipe
from shared.custom_pipeline import Generator4Embeds


def main():
    parser = argparse.ArgumentParser(
        description='Non-zero-shot UNet inference')
    parser.add_argument('--subject', type=str, default='sub-08')
    parser.add_argument('--leave_one_out_pic', type=int, default=9)
    parser.add_argument('--eeg_features', type=str,
        default=None,
        help='Path to EEG test features .pt file. '
             'Defaults to ATM_S_eeg_features_<sub>_test_nzs_loo<N>.pt')
    parser.add_argument('--diffusion_model', type=str,
        default=None,
        help='Path to diffusion prior .pt. '
             'Defaults to ./fintune_ckpts/ATMS_nonzeroshot/<sub>/loo<N>/diffusion_prior.pt')
    parser.add_argument('--training_images_dir', type=str,
        default="/hhome/ricse01/TFM/required/training_images",
        help='Root directory of training images (to extract class names)')
    parser.add_argument('--num_images_per_eeg', type=int, default=10,
        help='Number of images to generate per EEG sample')
    parser.add_argument('--num_diffusion_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    parser.add_argument('--gpu', type=str, default='cuda:0')
    args = parser.parse_args()

    sub = args.subject
    loo = args.leave_one_out_pic
    device = (torch.device(args.gpu)
              if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Device: {device}")

    # ── resolve default paths ─────────────────────────────────────────────
    eeg_feat_path = (args.eeg_features or
                     f"ATM_S_eeg_features_{sub}_test_nzs_loo{loo}.pt")
    diff_path     = (args.diffusion_model or
                     f"./fintune_ckpts/ATMS_nonzeroshot/{sub}/loo{loo}/diffusion_prior.pt")

    # ── load EEG embeddings ───────────────────────────────────────────────
    print(f"Loading EEG test embeddings from {eeg_feat_path} …")
    eeg_test = torch.load(eeg_feat_path, map_location='cpu').float().to(device)
    print(f"  EEG shape: {eeg_test.shape}")

    # ── load diffusion prior ──────────────────────────────────────────────
    print(f"Loading diffusion prior from {diff_path} …")
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1).to(device)
    diffusion_prior.load_state_dict(torch.load(diff_path, map_location=device))
    diffusion_prior.eval()

    pipe      = Pipe(diffusion_prior, device=device)
    generator = Generator4Embeds(num_inference_steps=4, device=device)

    # ── collect class names from training image directory ─────────────────
    all_folders = sorted([
        d for d in os.listdir(args.training_images_dir)
        if os.path.isdir(os.path.join(args.training_images_dir, d))
    ])
    # Extract text label after the first '_'
    class_names = []
    for folder in all_folders:
        try:
            idx = folder.index('_')
            class_names.append(folder[idx + 1:])
        except ValueError:
            class_names.append(folder)

    # ── generate images ───────────────────────────────────────────────────
    out_root = os.path.join(f"generated_imgs_nonzeroshot", sub, f"loo{loo}")
    os.makedirs(out_root, exist_ok=True)
    n_samples = eeg_test.shape[0]
    print(f"\nGenerating images for {n_samples} EEG samples …")

    for k in range(n_samples):
        eeg_embed = eeg_test[k: k + 1]  # (1, 1024)

        with torch.no_grad():
            h = pipe.generate(
                c_embeds=eeg_embed,
                num_inference_steps=args.num_diffusion_steps,
                guidance_scale=args.guidance_scale,
            )

        # Class index:  the test split has exactly 1 EEG sample per class.
        #   If there are multiple EEG repetitions per class, adjust accordingly.
        cls_idx    = k % len(class_names)
        class_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
        save_dir   = os.path.join(out_root, class_name)
        os.makedirs(save_dir, exist_ok=True)

        for j in range(args.num_images_per_eeg):
            img  = generator.generate(h.to(dtype=torch.float16))
            path = os.path.join(save_dir, f"{j}.png")
            img.save(path)

        print(f"  [{k+1:4d}/{n_samples}] {class_name}  → {save_dir}/")

    print(f"\nDone. Images saved under: {out_root}")


if __name__ == '__main__':
    main()
