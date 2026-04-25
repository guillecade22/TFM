# --- CONFIG -------------------------------------------------------------------
VIT_H_14_FEATURES_TEST    = "/hhome/ricse01/TFM/required/ViT-H-14_features_test.pt"
ATM_S_EEG_FEATURES_SUB_08 = "/hhome/ricse01/TFM/required/ATM_S_eeg_features_sub-08_test.pt"
DIFFUSION_PRIOR_PATH       = "/hhome/ricse01/TFM/required/sub-08/diffusion_prior.pt"
TEST_IMAGES_DIR            = "/hhome/ricse01/TFM/required/test_images/"
OUTPUT_DIR                 = "/hhome/ricse01/TFM/TFM/generated_antelope_captions"

IP_ADAPTER_SCALE    = 0.85
GUIDANCE_SCALE      = 1.5
NUM_INFERENCE_STEPS = 10
NEGATIVE_PROMPT     = "cartoon, illustration, painting, drawing, render, cgi, blurry, low quality, artificial"

# --- The three captions to test for the antelope image ---
CANDIDATE_CAPTIONS = ["antelope", "cheetah", "panther"]

# --- IMPORTS ------------------------------------------------------------------
import os
import sys
import torch

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


def find_antelope_index(test_images_dir):
    """
    Returns the index of the antelope subfolder in sorted order,
    which matches the position of its EEG embedding in the features file.
    """
    subfolders = sorted([
        d for d in os.listdir(test_images_dir)
        if os.path.isdir(os.path.join(test_images_dir, d))
    ])
    for idx, sf in enumerate(subfolders):
        if extract_class_name(sf) == "antelope":
            print(f"Found antelope at index {idx} (subfolder: '{sf}')")
            return idx
    raise ValueError("No 'antelope' subfolder found in TEST_IMAGES_DIR.")


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

class Generator4EmbedsPatched(Generator4Embeds):
    def __init__(self, num_inference_steps=20, device='cuda',
                 ip_adapter_scale=0.5, guidance_scale=5.0):
        super().__init__(num_inference_steps=num_inference_steps, device=device)
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

    def generate_from_prior_only(self, image_embeds, generator=None):
        """
        Decode the raw prior embedding into an image using SDXL with:
          - no text prompt
          - ip_adapter_scale = 1.0 (embedding drives everything)
          - guidance_scale   = 0.0 (turbo mode, no text influence)
        This shows what the UNet prior alone produces before any caption steering.
        """
        image_embeds = image_embeds.to(device=self.device, dtype=self.dtype)
        # Temporarily override scales for a clean prior-only decode
        self.pipe.set_ip_adapter_scale(1.0)
        image = self.pipe.generate_ip_adapter_embeds(
            prompt='',
            negative_prompt=None,
            ip_adapter_embeds=image_embeds,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=0.0,
            generator=generator,
            img2img_strength=self.img2img_strength,
            low_level_image=self.low_level_image,
            low_level_latent=self.low_level_latent,
        ).images[0]
        # Restore original scale for subsequent calls
        self.pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
        return image


# --- GENERATION ---------------------------------------------------------------

def generate_antelope_captions(eeg_embeds, antelope_idx, output_dir, seed=42):
    os.makedirs(output_dir, exist_ok=True)

    # Extract only the antelope EEG embedding
    antelope_eeg = eeg_embeds[antelope_idx]

    prior_pipe = load_diffusion_prior()

    generator_sdxl = Generator4EmbedsPatched(
        num_inference_steps=NUM_INFERENCE_STEPS,
        device=device,
        ip_adapter_scale=IP_ADAPTER_SCALE,
        guidance_scale=GUIDANCE_SCALE,
    )

    # Stage I: EEG embedding -> CLIP-space prior (done once, reused for all captions)
    print("\nRunning diffusion prior on antelope EEG embedding...")
    h = prior_pipe.generate(
        c_embeds=antelope_eeg,
        num_inference_steps=10,
        guidance_scale=2.0
    )

    # Save the raw prior output (no caption steering, pure UNet output)
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    print("Saving raw prior/UNet output (no caption)...")
    prior_image = generator_sdxl.generate_from_prior_only(h, generator=gen)
    prior_out_path = os.path.join(output_dir, "antelope_prior_only.png")
    prior_image.save(prior_out_path)
    print(f"  Saved: {prior_out_path}")

    gen = torch.Generator(device=device)

    print(f"\nGenerating {len(CANDIDATE_CAPTIONS)} images (one per caption candidate)...")
    print(f"Settings: ip_adapter_scale={IP_ADAPTER_SCALE}, "
          f"guidance_scale={GUIDANCE_SCALE}, steps={NUM_INFERENCE_STEPS}\n")

    for caption in CANDIDATE_CAPTIONS:
        # Use a fixed seed per caption so results are reproducible
        gen.manual_seed(seed)

        image = generator_sdxl.generate(
            h,
            text_prompt=caption,
            negative_prompt=NEGATIVE_PROMPT,
            generator=gen
        )

        out_path = os.path.join(output_dir, f"antelope_as_{caption}.png")
        image.save(out_path)
        print(f"  Saved: {out_path}")

    print(f"\nDone. 3 images saved to: {output_dir}")


# --- MAIN ---------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ANTELOPE CAPTION TEST")
    print("Generates one image per candidate caption:")
    for c in CANDIDATE_CAPTIONS:
        print(f"  - {c}")
    print("Using the antelope EEG embedding for all three.")
    print("="*60 + "\n")

    _, eeg_embeds = load_features()

    antelope_idx = find_antelope_index(TEST_IMAGES_DIR)

    generate_antelope_captions(eeg_embeds, antelope_idx, OUTPUT_DIR)