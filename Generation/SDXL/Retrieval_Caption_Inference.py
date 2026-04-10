# --- CONFIG -------------------------------------------------------------------
VIT_H_14_FEATURES_TEST    = "/hhome/ricse01/TFM/required/ViT-H-14_features_test.pt"
ATM_S_EEG_FEATURES_SUB_08 = "/hhome/ricse01/TFM/required/ATM_S_eeg_features_sub-08_test.pt"
DIFFUSION_PRIOR_PATH       = "/hhome/ricse01/TFM/required/sub-08/diffusion_prior.pt"
CAPTIONS_TOP1_PATH         = "/hhome/ricse01/TFM/required/captions_retrieval/captions_top1.txt"
CAPTIONS_TOP5_PATH         = "/hhome/ricse01/TFM/required/captions_retrieval/captions_top5.txt"
OUTPUT_DIR_TOP1            = "/hhome/ricse01/TFM/TFM/generated_retrieval_top1"
OUTPUT_DIR_TOP5            = "/hhome/ricse01/TFM/TFM/generated_retrieval_top1"

# --- IMPORTS ------------------------------------------------------------------
import os
import sys
import argparse
import torch
from PIL import Image

sys.path.append("../")
from shared.diffusion_prior import DiffusionPriorUNet, Pipe
from shared.custom_pipeline_low_level import Generator4Embeds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- HELPERS ------------------------------------------------------------------

def load_captions(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_features():
    image_embeds = torch.load(VIT_H_14_FEATURES_TEST,
                              map_location=device)['img_features'].unsqueeze(1)
    eeg_embeds   = torch.load(ATM_S_EEG_FEATURES_SUB_08,
                              map_location=device).unsqueeze(1)
    print(f"image_embeds shape : {image_embeds.shape}")
    print(f"eeg_embeds shape   : {eeg_embeds.shape}")
    return image_embeds, eeg_embeds


def load_diffusion_prior():
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    pipe = Pipe(diffusion_prior, device=device)
    pipe.diffusion_prior.load_state_dict(
        torch.load(DIFFUSION_PRIOR_PATH, map_location=device)
    )
    print("Diffusion prior loaded.")
    return pipe


def run_stage1(pipe, eeg_embed):
    """EEG embedding -> CLIP-space prior via diffusion."""
    return pipe.generate(
        c_embeds=eeg_embed,
        num_inference_steps=10,
        guidance_scale=2.0
    )


def run_stage2(h, caption, gen):
    """CLIP prior + text caption -> image via SDXL + IP-Adapter."""
    generator = Generator4Embeds(num_inference_steps=10, device=device)
    return generator.generate(h, text_prompt=caption, generator=gen)


# --- TOP-1 MODE ---------------------------------------------------------------

def generate_top1(eeg_embeds, captions, output_dir, seed=42):
    assert len(captions) == len(eeg_embeds), (
        f"Expected {len(eeg_embeds)} captions, got {len(captions)}. "
        f"Check captions_top1.txt has one line per test image."
    )

    os.makedirs(output_dir, exist_ok=True)
    pipe = load_diffusion_prior()

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    print(f"\nGenerating {len(eeg_embeds)} images (top-1 mode)...")
    print(f"Output dir: {output_dir}\n")

    for i in range(len(eeg_embeds)):
        caption = captions[i]
        h       = run_stage1(pipe, eeg_embeds[i])
        image   = run_stage2(h, caption, gen)

        out_path = os.path.join(output_dir, f"reconstructed_{i:04d}.png")
        image.save(out_path)

        if i % 20 == 0:
            print(f"  [{i:03d}/{len(eeg_embeds)}] {caption[:70]}  ->  {out_path}")

    print(f"\nDone. {len(eeg_embeds)} images saved to: {output_dir}")


# --- TOP-5 MODE ---------------------------------------------------------------

def generate_top5(eeg_embeds, captions, output_dir, top_k=5, seed=42):
    n_images = len(eeg_embeds)
    assert len(captions) == n_images * top_k, (
        f"Expected {n_images * top_k} caption lines ({n_images} images x {top_k} candidates), "
        f"got {len(captions)}. Check captions_top5.txt."
    )

    os.makedirs(output_dir, exist_ok=True)
    pipe = load_diffusion_prior()

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    print(f"\nGenerating {n_images} x {top_k} = {n_images * top_k} images (top-5 mode)...")
    print(f"Output dir: {output_dir}\n")

    for i in range(n_images):
        image_dir = os.path.join(output_dir, f"image_{i:04d}")
        os.makedirs(image_dir, exist_ok=True)

        # Stage I is the same for all 5 candidates of this image
        h = run_stage1(pipe, eeg_embeds[i])

        for k in range(top_k):
            caption  = captions[i * top_k + k]
            image    = run_stage2(h, caption, gen)
            out_path = os.path.join(image_dir, f"candidate_{k}.png")
            image.save(out_path)

        if i % 20 == 0:
            print(f"  [{i:03d}/{n_images}] {top_k} candidates saved to {image_dir}/")
            for k in range(top_k):
                print(f"    candidate_{k}: {captions[i * top_k + k][:70]}")

    print(f"\nDone. {n_images * top_k} images saved under: {output_dir}")


# --- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate images from EEG using retrieval-based captions."
    )
    parser.add_argument("--mode", choices=["top1", "top5", "both"],
                        default="both",
                        help="Which caption mode to run (default: both).")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of candidates per image for top-5 mode (default: 5).")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Random seed for generation (default: 42).")
    args = parser.parse_args()

    _, eeg_embeds = load_features()

    if args.mode in ("top1", "both"):
        print("\n" + "="*60)
        print("MODE: TOP-1")
        print("="*60)
        captions_top1 = load_captions(CAPTIONS_TOP1_PATH)
        print(f"Loaded {len(captions_top1)} captions from {CAPTIONS_TOP1_PATH}")
        print(f"Example: {captions_top1[0]}")
        generate_top1(eeg_embeds, captions_top1, OUTPUT_DIR_TOP1, seed=args.seed)

    if args.mode in ("top5", "both"):
        print("\n" + "="*60)
        print("MODE: TOP-5")
        print("="*60)
        captions_top5 = load_captions(CAPTIONS_TOP5_PATH)
        print(f"Loaded {len(captions_top5)} captions from {CAPTIONS_TOP5_PATH}")
        print(f"Example (image 0): {captions_top5[:args.top_k]}")
        generate_top5(eeg_embeds, captions_top5, OUTPUT_DIR_TOP5,
                      top_k=args.top_k, seed=args.seed)

    print("\nAll done.")


if __name__ == "__main__":
    main()