# --- CONFIG -------------------------------------------------------------------
VIT_H_14_FEATURES_TEST    = "/hhome/ricse01/TFM/required/ViT-H-14_features_test.pt"
ATM_S_EEG_FEATURES_SUB_08 = "/hhome/ricse01/TFM/required/ATM_S_eeg_features_sub-08_test.pt"
DIFFUSION_PRIOR_PATH       = "/hhome/ricse01/TFM/required/sub-08/diffusion_prior.pt"
TEST_IMAGES_DIR            = "/hhome/ricse01/TFM/required/test_images/"
OUTPUT_DIR                 = "/hhome/ricse01/TFM/TFM/generated_gt_caption"

# IP-Adapter scale: how much the CLIP image embedding (from EEG) drives the result.
# 1.0 = fully image-driven (text prompt ignored, current default)
# 0.5 = balanced between image embedding and text prompt
# 0.3 = text prompt has more influence
IP_ADAPTER_SCALE = 0.7

# SDXL guidance scale: how strongly the text prompt is followed.
# 0.0 = SDXL-turbo mode (text ignored, fastest)
# 5.0 to 7.5 = standard SDXL (text followed, slower, needs more steps)
GUIDANCE_SCALE = 3.0

# Inference steps: 4 for turbo (guidance_scale=0.0), 20-30 for standard
NUM_INFERENCE_STEPS = 15

# Negative prompt: steers SDXL away from unwanted styles
NEGATIVE_PROMPT = "cartoon, illustration, painting, drawing, render, cgi, blurry, low quality, artificial"

# --- IMPORTS ------------------------------------------------------------------
import os
import sys
import glob
import torch
from PIL import Image

sys.path.append("../")
from shared.diffusion_prior import DiffusionPriorUNet, Pipe
from shared.custom_pipeline_low_level import Generator4Embeds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# --- HELPERS ------------------------------------------------------------------

def extract_class_name(subfolder_name):
    """'00001_aircraft_carrier' -> 'aircraft_carrier'"""
    parts = subfolder_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return subfolder_name


def make_caption(class_name):
    """Convert class name to a natural SDXL prompt."""
    label = class_name.replace("_", " ").strip()
    vowels = ("a", "e", "i", "o", "u")
    article = "an" if label[0].lower() in vowels else "a"
    return f"a real photograph of {article} {label}, natural lighting, photorealistic"


def collect_gt_captions(test_images_dir):
    """
    Walk the test images directory and collect ground truth class names
    in sorted subfolder order. Returns a list of (class_name, caption) tuples,
    one per test image, sorted by subfolder name (matching EEG embedding order).
    """
    subfolders = sorted([
        d for d in os.listdir(test_images_dir)
        if os.path.isdir(os.path.join(test_images_dir, d))
    ])
    captions = []
    for sf in subfolders:
        cls = extract_class_name(sf)
        captions.append((cls, make_caption(cls)))
    return captions


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


# --- PATCHED Generator4Embeds ------------------------------------------------
# The original Generator4Embeds hardcodes guidance_scale=0.0 and ip_adapter_scale=1,
# which makes text prompts have no effect. We patch it here to expose those params.

class Generator4EmbedsPatched(Generator4Embeds):
    """
    Extends Generator4Embeds to allow:
      - ip_adapter_scale  : controls how much the CLIP embedding drives the image
                            (1.0 = fully image, 0.0 = fully text)
      - guidance_scale    : controls text prompt adherence
                            (0.0 = turbo/no text, 5-7.5 = standard SDXL)
      - negative_prompt   : steers generation away from unwanted content
      - num_inference_steps: set at init time
    """
    def __init__(self, num_inference_steps=20, device='cuda',
                 ip_adapter_scale=0.5, guidance_scale=5.0):
        super().__init__(num_inference_steps=num_inference_steps, device=device)
        # override the ip_adapter_scale set in parent __init__ (which sets it to 1)
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        self.guidance_scale = guidance_scale
        print(f"Generator patched: ip_adapter_scale={ip_adapter_scale}, "
              f"guidance_scale={guidance_scale}, steps={num_inference_steps}")

    def generate(self, image_embeds, text_prompt='', negative_prompt='', generator=None):
        image_embeds = image_embeds.to(device=self.device, dtype=self.dtype)
        image = self.pipe.generate_ip_adapter_embeds(
            prompt=text_prompt,
            negative_prompt=negative_prompt if self.guidance_scale > 1.0 else None,
            ip_adapter_embeds=image_embeds,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            img2img_strength=self.img2img_strength,
            low_level_image=self.low_level_image,
            low_level_latent=self.low_level_latent,
        ).images[0]
        return image


# --- GENERATION ---------------------------------------------------------------

def generate_gt_captions(eeg_embeds, gt_captions, output_dir, seed=42):
    assert len(gt_captions) == len(eeg_embeds), (
        f"Mismatch: {len(eeg_embeds)} EEG embeddings but {len(gt_captions)} captions. "
        f"Check TEST_IMAGES_DIR has exactly one subfolder per test image."
    )

    os.makedirs(output_dir, exist_ok=True)

    # Save the captions used so the experiment is reproducible
    captions_log = os.path.join(output_dir, "gt_captions_used.txt")
    with open(captions_log, "w") as f:
        for cls, caption in gt_captions:
            f.write(f"{cls}\t{caption}\n")
    print(f"Captions logged to: {captions_log}")

    # Load prior
    prior_pipe = load_diffusion_prior()

    # Load patched generator
    generator_sdxl = Generator4EmbedsPatched(
        num_inference_steps=NUM_INFERENCE_STEPS,
        device=device,
        ip_adapter_scale=IP_ADAPTER_SCALE,
        guidance_scale=GUIDANCE_SCALE,
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    print(f"\nGenerating {len(eeg_embeds)} images with ground truth captions...")
    print(f"Settings: ip_adapter_scale={IP_ADAPTER_SCALE}, "
          f"guidance_scale={GUIDANCE_SCALE}, steps={NUM_INFERENCE_STEPS}")
    print(f"Output dir: {output_dir}\n")

    for i in range(len(eeg_embeds)):
        cls, caption = gt_captions[i]

        # Stage I: EEG embedding -> CLIP-space prior
        h = prior_pipe.generate(
            c_embeds=eeg_embeds[i],
            num_inference_steps=10,
            guidance_scale=2.0
        )

        # Stage II: CLIP prior + GT caption -> image
        image = generator_sdxl.generate(
            h,
            text_prompt=caption,
            negative_prompt=NEGATIVE_PROMPT,
            generator=gen
        )

        out_path = os.path.join(output_dir, f"reconstructed_{i:04d}.png")
        image.save(out_path)

        if i % 20 == 0:
            print(f"  [{i:03d}/{len(eeg_embeds)}]  class={cls:<25}  caption='{caption[:60]}'")

    print(f"\nDone. {len(eeg_embeds)} images saved to: {output_dir}")


# --- MAIN ---------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ORACLE TEST: Ground truth captions")
    print("This uses the CORRECT class name for every image.")
    print("If this does not improve over no-caption results,")
    print("the issue is in ip_adapter_scale / guidance_scale,")
    print("not in caption quality.")
    print("="*60 + "\n")

    # Load EEG embeddings
    _, eeg_embeds = load_features()

    # Collect ground truth captions from test folder structure
    gt_captions = collect_gt_captions(TEST_IMAGES_DIR)
    print(f"\nCollected {len(gt_captions)} ground truth captions.")
    print("First 5:")
    for cls, cap in gt_captions[:5]:
        print(f"  {cls:<25} -> {cap}")

    if len(gt_captions) != len(eeg_embeds):
        print(f"\nWARNING: {len(gt_captions)} captions vs {len(eeg_embeds)} EEG embeddings.")
        print("Make sure TEST_IMAGES_DIR has one subfolder per test image in the same order.")
        print("Truncating to the smaller of the two.")
        n = min(len(gt_captions), len(eeg_embeds))
        gt_captions = gt_captions[:n]
        eeg_embeds  = eeg_embeds[:n]

    generate_gt_captions(eeg_embeds, gt_captions, OUTPUT_DIR)