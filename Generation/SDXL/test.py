# --- CONFIG -------------------------------------------------------------------
VIT_H_14_FEATURES_TEST    = "/hhome/ricse01/TFM/required/ViT-H-14_features_test.pt"
ATM_S_EEG_FEATURES_SUB_08 = "/hhome/ricse01/TFM/required/ATM_S_eeg_features_sub-08_test.pt"
DIFFUSION_PRIOR_PATH       = "/hhome/ricse01/TFM/required/sub-08/diffusion_prior.pt"
TEST_IMAGES_DIR            = "/hhome/ricse01/TFM/required/test_images/"

# Output dirs - one per experiment variant
OUTPUT_DIR_BASE            = "/hhome/ricse01/TFM/TFM/exp_base"
OUTPUT_DIR_SIMPLE          = "/hhome/ricse01/TFM/TFM/exp_simple_caption"
OUTPUT_DIR_NOTEXT          = "/hhome/ricse01/TFM/TFM/exp_notext"
OUTPUT_DIR_IMG2IMG         = "/hhome/ricse01/TFM/TFM/exp_img2img"
OUTPUT_DIR_IMG2IMG_CAPTION = "/hhome/ricse01/TFM/TFM/exp_img2img_caption"

# --- EXPERIMENT VARIANTS ------------------------------------------------------
# Each variant is a dict of parameters.
# Run them all with --mode all, or one at a time with e.g. --mode simple_caption
#
# PROMPT STYLES being tested:
#   ""              -> no text (IP-Adapter only)
#   simple          -> "aircraft carrier"  (bare class name, no extra words)
#   medium          -> "a photo of an aircraft carrier"
#   rich            -> current long prompt (photorealistic, natural lighting)
#
# IMG2IMG:
#   img2img_strength = 0.0  -> ignore Unet image entirely (current behaviour)
#   img2img_strength = 0.5  -> start denoising halfway through, Unet image anchors layout
#   img2img_strength = 0.7  -> Unet image strongly anchors layout

EXPERIMENTS = {
    "notext": {
        "prompt_style":       "none",
        "ip_adapter_scale":   0.9,
        "guidance_scale":     0.0,
        "num_steps":          4,
        "img2img_strength":   0.0,
        "output_dir":         OUTPUT_DIR_NOTEXT,
        "description": "IP-Adapter only, no text, turbo mode - closest to base",
    },
    "simple_caption": {
        "prompt_style":       "simple",
        "ip_adapter_scale":   0.7,
        "guidance_scale":     3.0,
        "num_steps":          15,
        "img2img_strength":   0.0,
        "output_dir":         OUTPUT_DIR_SIMPLE,
        "description": "Simple class name prompt, no extra words",
    },
    "img2img_only": {
        "prompt_style":       "none",
        "ip_adapter_scale":   0.9,
        "guidance_scale":     0.0,
        "num_steps":          4,
        "img2img_strength":   0.5,
        "output_dir":         OUTPUT_DIR_IMG2IMG,
        "description": "Unet image as img2img anchor, no caption - preserves Unet layout",
    },
    "img2img_caption": {
        "prompt_style":       "simple",
        "ip_adapter_scale":   0.7,
        "guidance_scale":     3.0,
        "num_steps":          15,
        "img2img_strength":   0.5,
        "output_dir":         OUTPUT_DIR_IMG2IMG_CAPTION,
        "description": "Unet image as anchor + simple caption - best of both",
    },
}

NEGATIVE_PROMPT = "cartoon, illustration, painting, blurry, low quality, cgi"

# --- IMPORTS ------------------------------------------------------------------
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.append("../")
from shared.diffusion_prior import DiffusionPriorUNet, Pipe
from shared.custom_pipeline_low_level import Generator4Embeds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# --- PROMPT STYLES ------------------------------------------------------------

def make_prompt(class_name, style):
    """
    style options:
      "none"   -> empty string (IP-Adapter drives everything)
      "simple" -> bare class name, no article, no extra words
                  e.g. "aircraft carrier"
                  This is the minimal hint to SDXL without strong priors
      "medium" -> "a photo of an aircraft carrier"
      "rich"   -> "a real photograph of an aircraft carrier, natural lighting, photorealistic"
    """
    label = class_name.replace("_", " ").strip()

    if style == "none":
        return ""
    elif style == "simple":
        # No article, no qualifiers - just the object name
        # SDXL understands bare class names well from CLIP training
        return label
    elif style == "medium":
        vowels = ("a", "e", "i", "o", "u")
        article = "an" if label[0].lower() in vowels else "a"
        return f"a photo of {article} {label}"
    elif style == "rich":
        vowels = ("a", "e", "i", "o", "u")
        article = "an" if label[0].lower() in vowels else "a"
        return f"a real photograph of {article} {label}, natural lighting, photorealistic"
    else:
        return label


# --- HELPERS ------------------------------------------------------------------

def extract_class_name(subfolder_name):
    parts = subfolder_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return subfolder_name


def collect_gt_classes(test_images_dir):
    subfolders = sorted([
        d for d in os.listdir(test_images_dir)
        if os.path.isdir(os.path.join(test_images_dir, d))
    ])
    return [extract_class_name(sf) for sf in subfolders]


def load_features():
    image_embeds = torch.load(VIT_H_14_FEATURES_TEST,
                              map_location=device)['img_features'].unsqueeze(1)
    eeg_embeds   = torch.load(ATM_S_EEG_FEATURES_SUB_08,
                              map_location=device).unsqueeze(1)
    print(f"image_embeds : {image_embeds.shape}")
    print(f"eeg_embeds   : {eeg_embeds.shape}")
    return image_embeds, eeg_embeds


def load_diffusion_prior():
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    pipe = Pipe(diffusion_prior, device=device)
    pipe.diffusion_prior.load_state_dict(
        torch.load(DIFFUSION_PRIOR_PATH, map_location=device)
    )
    print("Diffusion prior loaded.")
    return pipe


def decode_unet_to_image(pipe, h):
    """
    Decode the diffusion prior output h back to pixel space using the VAE.
    This gives the rough Unet image that can be used as img2img anchor.
    h shape: [1, 1024]
    Returns a PIL Image.
    """
    # h is a CLIP embedding, not a VAE latent directly.
    # We use the pipeline's VAE decoder via the IP-Adapter image path.
    # The simplest approach: generate at guidance_scale=0, ip_scale=1, no text,
    # turbo steps=4 - this effectively just decodes h through SDXL with
    # minimal denoising, giving us SDXL's interpretation of h as an image.
    gen_tmp = Generator4EmbedsPatched(
        num_inference_steps=4,
        device=device,
        ip_adapter_scale=1.0,
        guidance_scale=0.0,
    )
    return gen_tmp.generate(h, text_prompt="", generator=None)


# --- PATCHED GENERATOR --------------------------------------------------------

class Generator4EmbedsPatched(Generator4Embeds):
    """
    Extends Generator4Embeds to expose:
      - ip_adapter_scale    : CLIP embedding influence (1.0=full, 0.0=none)
      - guidance_scale      : text prompt strength (0.0=turbo/off, 3-7=active)
      - negative_prompt     : what to avoid
      - img2img_strength    : how much to anchor on low_level_image (0=ignore, 1=copy)
    """
    def __init__(self, num_inference_steps=15, device='cuda',
                 ip_adapter_scale=0.7, guidance_scale=3.0,
                 low_level_image=None):
        super().__init__(
            num_inference_steps=num_inference_steps,
            device=device,
            low_level_image=low_level_image,
        )
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        self.guidance_scale = guidance_scale

    def generate(self, image_embeds, text_prompt="", negative_prompt="",
                 generator=None, img2img_strength=0.0, low_level_image=None):
        image_embeds = image_embeds.to(device=self.device, dtype=self.dtype)

        # Temporarily override low_level_image if passed per-call
        orig_low = self.low_level_image
        if low_level_image is not None:
            self.low_level_image = low_level_image

        image = self.pipe.generate_ip_adapter_embeds(
            prompt=text_prompt,
            negative_prompt=negative_prompt if self.guidance_scale > 1.0 else None,
            ip_adapter_embeds=image_embeds,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            img2img_strength=img2img_strength,
            low_level_image=self.low_level_image,
            low_level_latent=self.low_level_latent,
        ).images[0]

        self.low_level_image = orig_low
        return image


# --- SINGLE EXPERIMENT RUNNER -------------------------------------------------

def run_experiment(name, cfg, eeg_embeds, gt_classes, prior_pipe, seed=42):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  {cfg['description']}")
    print(f"  prompt_style={cfg['prompt_style']}  "
          f"ip_adapter_scale={cfg['ip_adapter_scale']}  "
          f"guidance_scale={cfg['guidance_scale']}  "
          f"steps={cfg['num_steps']}  "
          f"img2img_strength={cfg['img2img_strength']}")
    print(f"{'='*60}")

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Log experiment config
    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")

    # Log captions used
    with open(os.path.join(out_dir, "captions_used.txt"), "w") as f:
        for cls in gt_classes:
            prompt = make_prompt(cls, cfg["prompt_style"])
            f.write(f"{cls}\t{prompt}\n")

    generator_sdxl = Generator4EmbedsPatched(
        num_inference_steps=cfg["num_steps"],
        device=device,
        ip_adapter_scale=cfg["ip_adapter_scale"],
        guidance_scale=cfg["guidance_scale"],
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    use_img2img = cfg["img2img_strength"] > 0.0

    for i in range(len(eeg_embeds)):
        cls    = gt_classes[i]
        prompt = make_prompt(cls, cfg["prompt_style"])

        # Stage I: EEG -> CLIP-space prior
        h = prior_pipe.generate(
            c_embeds=eeg_embeds[i],
            num_inference_steps=10,
            guidance_scale=2.0
        )

        # Optionally decode Unet output to use as img2img anchor
        unet_image = None
        if use_img2img:
            # Quick 4-step decode of h to get the rough Unet image
            unet_image = Generator4EmbedsPatched(
                num_inference_steps=4,
                device=device,
                ip_adapter_scale=1.0,
                guidance_scale=0.0,
            ).generate(h, text_prompt="", generator=None)

        # Stage II: SDXL guided by h + optional caption + optional Unet image
        image = generator_sdxl.generate(
            h,
            text_prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT if cfg["guidance_scale"] > 1.0 else "",
            generator=gen,
            img2img_strength=cfg["img2img_strength"],
            low_level_image=unet_image,
        )

        out_path = os.path.join(out_dir, f"reconstructed_{i:04d}.png")
        image.save(out_path)

        if use_img2img:
            unet_image.save(os.path.join(out_dir, f"unet_{i:04d}.png"))

        if i % 20 == 0:
            print(f"  [{i:03d}/{len(eeg_embeds)}]  {cls:<22}  prompt='{prompt[:50]}'")

    print(f"Done. Images saved to: {out_dir}")


# --- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["all"] + list(EXPERIMENTS.keys()),
                        help="Which experiment to run.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, eeg_embeds  = load_features()
    gt_classes     = collect_gt_classes(TEST_IMAGES_DIR)
    prior_pipe     = load_diffusion_prior()

    n = min(len(gt_classes), len(eeg_embeds))
    gt_classes = gt_classes[:n]
    eeg_embeds = eeg_embeds[:n]

    print(f"\nRunning on {n} images.")
    print(f"First 5 classes: {gt_classes[:5]}")

    to_run = EXPERIMENTS if args.mode == "all" else {args.mode: EXPERIMENTS[args.mode]}

    for name, cfg in to_run.items():
        run_experiment(name, cfg, eeg_embeds, gt_classes, prior_pipe, seed=args.seed)

    print("\nAll experiments done.")
    print("Run evaluate_reconstruction.py on each output dir to compare metrics.")


if __name__ == "__main__":
    main()