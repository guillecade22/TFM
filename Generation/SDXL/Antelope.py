# --- CONFIG -------------------------------------------------------------------
# Pre-generated image to use as input — skips the diffusion prior entirely
INPUT_IMAGE_PATH = "/hhome/ricse01/TFM/EEG_Image_decode/Generation/generated_sub-08_only_diff/antelope/0.png"
OUTPUT_DIR       = "/hhome/ricse01/TFM/TFM/generated_antelope_captions"

IP_ADAPTER_SCALE    = 0.85
GUIDANCE_SCALE      = 1.5
NUM_INFERENCE_STEPS = 10
NEGATIVE_PROMPT     = "cartoon, illustration, painting, drawing, render, cgi, blurry, low quality, artificial"

# --- The three captions to test ---
CANDIDATE_CAPTIONS = ["antelope", "cheetah", "panther"]

# --- IMPORTS ------------------------------------------------------------------
import os
import sys
import torch
import open_clip
from PIL import Image

sys.path.append("../")
from shared.custom_pipeline_low_level import Generator4Embeds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# --- CLIP ENCODER -------------------------------------------------------------

def load_clip_encoder():
    """Load ViT-H-14 CLIP encoder (same model used to produce the features file)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14', pretrained='laion2b_s32b_b79k'
    )
    model = model.to(device).eval()
    print("CLIP ViT-H-14 loaded.")
    return model, preprocess


def encode_image(image_path, clip_model, preprocess):
    """
    Encode a PIL image into a CLIP embedding using ViT-H-14.
    Returns tensor of shape [1, 1, 1024] to match the prior output shape.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        h = clip_model.encode_image(image_tensor)
        h = h / h.norm(dim=-1, keepdim=True)   # L2 normalise (same as prior output)
    h = h.unsqueeze(1)  # [1, 1024] -> [1, 1, 1024]
    print(f"Encoded image embedding shape: {h.shape}")
    return h


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


# --- GENERATION ---------------------------------------------------------------

def generate_from_image(input_image_path, output_dir, seed=42):
    os.makedirs(output_dir, exist_ok=True)

    # Encode the pre-generated image into a CLIP embedding (replaces the prior)
    print(f"\nEncoding input image: {input_image_path}")
    clip_model, preprocess = load_clip_encoder()
    h = encode_image(input_image_path, clip_model, preprocess)

    # Free CLIP from GPU memory before loading SDXL
    del clip_model
    torch.cuda.empty_cache()

    generator_sdxl = Generator4EmbedsPatched(
        num_inference_steps=NUM_INFERENCE_STEPS,
        device=device,
        ip_adapter_scale=IP_ADAPTER_SCALE,
        guidance_scale=GUIDANCE_SCALE,
    )

    gen = torch.Generator(device=device)

    print(f"\nGenerating {len(CANDIDATE_CAPTIONS)} images (one per caption)...")
    print(f"Settings: ip_adapter_scale={IP_ADAPTER_SCALE}, "
          f"guidance_scale={GUIDANCE_SCALE}, steps={NUM_INFERENCE_STEPS}\n")

    for caption in CANDIDATE_CAPTIONS:
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

    print(f"\nDone. {len(CANDIDATE_CAPTIONS)} images saved to: {output_dir}")


# --- MAIN ---------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ANTELOPE CAPTION TEST (from pre-generated image)")
    print(f"Input image : {INPUT_IMAGE_PATH}")
    print("Captions    :", ", ".join(CANDIDATE_CAPTIONS))
    print("Diffusion prior: SKIPPED")
    print("="*60 + "\n")

    generate_from_image(INPUT_IMAGE_PATH, OUTPUT_DIR)