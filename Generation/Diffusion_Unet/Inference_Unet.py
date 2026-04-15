#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import DataLoader
from PIL import Image

# ----------------- Paths -----------------
# TEST_DIR = "/hhome/ricse01/TFM/required/test_images"
# VIT_H_14_FEATURES_TEST_PATH = "/hhome/ricse01/TFM/TFM/ViT-H-14_features_test.pt"
# MODEL_ATMS_EEG_FEATURES_TEST_PATH = "/hhome/ricse01/TFM/TFM/ATM_S_eeg_features_sub-08_test.pt"
# DIFFUSION_MODEL_PATH = "/hhome/ricse01/TFM/TFM/fintune_ckpts/ATMS/sub-08/diffusion_prior.pt"
# SUBJECT = "sub-08"
TEST_DIR = "/hhome/ricse01/TFM/required/test_images"
VIT_H_14_FEATURES_TEST_PATH = "/hhome/ricse01/TFM/required/ViT-H-14_features_test.pt"
MODEL_ATMS_EEG_FEATURES_TEST_PATH = "/hhome/ricse01/TFM/required/ATM_S_eeg_features_sub-08_test.pt"
DIFFUSION_MODEL_PATH = "/hhome/ricse01/TFM/required/sub-08/diffusion_prior.pt"
SUBJECT = "sub-08"

# ----------------- Device -----------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ----------------- Load embeddings -----------------
print("Loading VIT image features...")
vit_test = torch.load(VIT_H_14_FEATURES_TEST_PATH, map_location='cpu')
if isinstance(vit_test, dict):
    vit_test = vit_test['img_features']
vit_test = vit_test.to(device).float()

print("Loading EEG features...")
eeg_test = torch.load(MODEL_ATMS_EEG_FEATURES_TEST_PATH, map_location='cpu')
eeg_test = eeg_test.to(device).float()

print(f"VIT test shape: {vit_test.shape}, EEG test shape: {eeg_test.shape}")

# ----------------- Load diffusion prior -----------------
from shared.diffusion_prior import *
from shared.custom_pipeline import *

print("Loading diffusion prior model...")
diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1).to(device)
diffusion_prior.load_state_dict(torch.load(DIFFUSION_MODEL_PATH, map_location=device))
diffusion_prior.eval()

pipe = Pipe(diffusion_prior, device=device)

# ----------------- Generator -----------------
generator = Generator4Embeds(num_inference_steps=4, device=device)

# ----------------- Load text descriptions -----------------
# Extract folder names after '_'
texts = []
for folder in sorted(os.listdir(TEST_DIR)):
    if os.path.isdir(os.path.join(TEST_DIR, folder)):
        try:
            idx = folder.index('_')
            texts.append(folder[idx+1:])
        except ValueError:
            texts.append(folder)
print(f"Loaded {len(texts)} text descriptions")

# ----------------- Generate images -----------------
output_dir = f"generated_imgs/{SUBJECT}"
os.makedirs(output_dir, exist_ok=True)

for k in range(eeg_test.shape[0]):  # Loop over EEG samples
    eeg_embed = eeg_test[k:k+1]  # Single EEG embedding
    # Generate diffusion latent
    with torch.no_grad():
        h = pipe.generate(c_embeds=eeg_embed, num_inference_steps=50, guidance_scale=5.0)

    # Generate 10 images per EEG embedding
    for j in range(10):
        image = generator.generate(h.to(dtype=torch.float16))
        text_name = texts[k] if k < len(texts) else f"sample_{k}"
        save_path = os.path.join(output_dir, text_name, f"{j}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)
        print(f"Saved: {save_path}")
