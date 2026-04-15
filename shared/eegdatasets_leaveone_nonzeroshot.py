"""
eegdatasets_leaveone_nonzeroshot.py

Drop-in replacement for eegdatasets_leaveone.py that supports a
**non-zero-shot** leave-one-image-out (LOO) protocol on the 1654
*training* classes.

For every class the dataset has 10 images (indices 0-9).
  - train=True  → uses the 9 images whose index != leave_one_out_pic
  - train=False → uses ONLY the 1 held-out image (index == leave_one_out_pic)

All other logic (subject filtering, EEG slicing, CLIP feature caching,
__getitem__ indexing) is preserved from the original.

Usage
-----
    from eegdatasets_leaveone_nonzeroshot import EEGDatasetNonZeroShot

    # Train on images 0-8, test on image 9
    train_ds = EEGDatasetNonZeroShot(data_path, subjects=['sub-08'],
                                     train=True,  leave_one_out_pic=9)
    test_ds  = EEGDatasetNonZeroShot(data_path, subjects=['sub-08'],
                                     train=False, leave_one_out_pic=9)
"""

import os
import json
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import numpy as np
from PIL import Image

import clip
import open_clip

# ── proxy (keep identical to original) ──────────────────────────────────────
proxy = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy

# ── CLIP model (keep identical to original) ──────────────────────────────────
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_type = 'ViT-H-14'
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type,
    pretrained='/hhome/ricse01/TFM/required/open_clip_model.safetensors',
    precision='fp32',
    device=device,
)

# ── data paths from config ───────────────────────────────────────────────────
config_path = "/hhome/ricse01/TFM/TFM/shared/data_config.json"
with open(config_path, "r") as f:
    _cfg = json.load(f)

data_path           = _cfg["data_path"]
img_directory_training = _cfg["img_directory_training"]
img_directory_test     = _cfg["img_directory_test"]

# ── constants ────────────────────────────────────────────────────────────────
N_TRAIN_CLASSES   = 1654
SAMPLES_PER_CLASS = 10   # EEG repetitions per class in training split


class EEGDatasetNonZeroShot(Dataset):
    """
    Leave-one-image-out non-zero-shot variant.

    Parameters
    ----------
    data_path        : str   – root of preprocessed EEG data
    subjects         : list  – e.g. ['sub-08']
    exclude_subject  : str   – subject to skip during multi-subject training
    train            : bool  – True → 9-image training split
                               False → 1-image held-out test split
    leave_one_out_pic: int   – image index (0-9) held out for testing
    time_window      : list  – [start, end] in seconds
    classes          : list  – subset of class indices (None = all 1654)
    """

    def __init__(
        self,
        data_path,
        subjects=None,
        exclude_subject=None,
        train=True,
        leave_one_out_pic=9,
        time_window=[0, 1.0],
        classes=None,
    ):
        self.data_path         = data_path
        self.train             = train
        self.leave_one_out_pic = leave_one_out_pic
        self.time_window       = time_window
        self.classes           = classes          # None → all 1654 classes
        self.exclude_subject   = exclude_subject

        self.subject_list = os.listdir(data_path)
        self.subjects     = self.subject_list if subjects is None else subjects
        self.n_sub        = len(self.subjects)

        # Number of classes in this split
        self.n_cls = N_TRAIN_CLASSES if classes is None else len(classes)

        # Images per class exposed by this split
        #   train → 9 (all except leave_one_out_pic)
        #   test  → 1 (only leave_one_out_pic)
        self.pics_per_class = (SAMPLES_PER_CLASS - 1) if train else 1

        assert any(s in self.subject_list for s in self.subjects), \
            "None of the requested subjects found in data_path."

        self.data, self.labels, self.text, self.img = self._load_data()
        self.data = self._extract_eeg(self.data, time_window)

        # ── CLIP features ────────────────────────────────────────────────────
        split_tag = "train" if train else "test_nonzeroshot"
        features_filename = f"{model_type}_features_nonzeroshot_{split_tag}_loo{leave_one_out_pic}.pt"

        if os.path.exists(features_filename):
            saved = torch.load(features_filename)
            self.text_features = saved['text_features']
            self.img_features  = saved['img_features']
        else:
            self.text_features = self._encode_text(self.text)
            self.img_features  = self._encode_images(self.img)
            torch.save(
                {'text_features': self.text_features.cpu(),
                 'img_features':  self.img_features.cpu()},
                features_filename,
            )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _image_indices_for_class(self):
        """
        Returns the list of picture indices (0-9) used by this split.
        Train  → [0,1,...,9] minus leave_one_out_pic   (9 images)
        Test   → [leave_one_out_pic]                   (1 image)
        """
        all_pics = list(range(SAMPLES_PER_CLASS))
        if self.train:
            return [p for p in all_pics if p != self.leave_one_out_pic]
        else:
            return [self.leave_one_out_pic]

    def _load_data(self):
        data_list  = []
        label_list = []
        texts      = []
        images     = []

        img_dir  = img_directory_training  # always training classes
        pic_idxs = self._image_indices_for_class()

        # ── text descriptions ─────────────────────────────────────────────
        dirnames = sorted([
            d for d in os.listdir(img_dir)
            if os.path.isdir(os.path.join(img_dir, d))
        ])
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        for d in dirnames:
            try:
                idx = d.index('_')
                texts.append(f"This picture is {d[idx+1:]}")
            except ValueError:
                print(f"Skipped folder '{d}': no '_' found.")

        # ── image paths ───────────────────────────────────────────────────
        all_folders = sorted([
            d for d in os.listdir(img_dir)
            if os.path.isdir(os.path.join(img_dir, d))
        ])
        if self.classes is not None:
            all_folders = [all_folders[i] for i in self.classes]

        for folder in all_folders:
            folder_path  = os.path.join(img_dir, folder)
            all_imgs     = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            for p in pic_idxs:
                if p < len(all_imgs):
                    images.append(os.path.join(folder_path, all_imgs[p]))
                else:
                    print(f"Warning: class '{folder}' has no image at index {p}.")

        # ── EEG data ──────────────────────────────────────────────────────
        print(f"Subjects: {self.subjects}  |  exclude: {self.exclude_subject}")
        for subject in self.subjects:
            if subject == self.exclude_subject:
                continue

            file_path = os.path.join(self.data_path, subject,
                                     'preprocessed_eeg_training.npy')
            data      = np.load(file_path, allow_pickle=True)
            eeg       = torch.from_numpy(
                            data['preprocessed_eeg_data']).float().detach()
            times     = torch.from_numpy(data['times']).detach()[50:]
            ch_names  = data['ch_names']

            # Resolve class indices
            class_indices = (list(range(N_TRAIN_CLASSES))
                             if self.classes is None else self.classes)

            for cls_i in class_indices:
                eeg_start = cls_i * SAMPLES_PER_CLASS
                for p in pic_idxs:
                    eeg_idx = eeg_start + p
                    if eeg_idx >= len(eeg):
                        print(f"Warning: EEG index {eeg_idx} out of range.")
                        continue
                    # shape: (n_repeats, n_channels, n_times)
                    eeg_sample = eeg[eeg_idx]          # (n_repeats, C, T)
                    data_list.append(eeg_sample)
                    label_list.append(torch.tensor(cls_i, dtype=torch.long))

        self.times    = times
        self.ch_names = ch_names

        # ── stack tensors ─────────────────────────────────────────────────
        # Each entry in data_list: (n_repeats, C, T)  – typically 4 repeats
        data_tensor  = torch.cat([d.unsqueeze(0) for d in data_list], dim=0)
        # Flatten repeats into batch dimension  → (N * n_repeats, C, T)
        data_tensor  = data_tensor.view(-1, *data_list[0].shape[1:])
        label_tensor = torch.stack(label_list)           # (N,)
        label_tensor = label_tensor.repeat_interleave(data_list[0].shape[0])  # match repeats

        # Re-map labels to contiguous 0-based indices when using a class subset
        if self.classes is not None:
            unique_vals = sorted(set(label_tensor.tolist()))
            mapping     = {v: i for i, v in enumerate(unique_vals)}
            label_tensor = torch.tensor(
                [mapping[v] for v in label_tensor.tolist()], dtype=torch.long)

        print(f"EEG tensor: {data_tensor.shape} | "
              f"Labels: {label_tensor.shape} | "
              f"Texts: {len(texts)} | Images: {len(images)}")

        return data_tensor, label_tensor, texts, images

    def _extract_eeg(self, eeg_data, time_window):
        start, end = time_window
        indices = (self.times >= start) & (self.times <= end)
        return eeg_data[..., indices]

    def _encode_text(self, text_list):
        tokens = torch.cat([clip.tokenize(t) for t in text_list]).to(device)
        with torch.no_grad():
            feats = vlmodel.encode_text(tokens)
        return F.normalize(feats, dim=-1).detach().cpu()

    def _encode_images(self, img_paths, batch_size=20):
        all_feats = []
        for i in range(0, len(img_paths), batch_size):
            batch = img_paths[i: i + batch_size]
            imgs  = torch.stack([
                preprocess_train(Image.open(p).convert("RGB")) for p in batch
            ]).to(device)
            with torch.no_grad():
                feats = vlmodel.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x     = self.data[index]
        label = self.labels[index]

        # Number of EEG entries per class (after repeat-interleave of repeats)
        n_repeats       = self.data.shape[0] // (self.n_cls * self.n_sub)
        entries_per_cls = self.pics_per_class * n_repeats  # e.g. 9*4 = 36

        # Which class does this index belong to?
        cls_within_sub = (index % (self.n_cls * entries_per_cls)) // entries_per_cls
        # Which picture within that class?
        pic_within_cls = (index % entries_per_cls) // n_repeats

        # Map to global image/text index
        img_index  = cls_within_sub * self.pics_per_class + pic_within_cls
        text_index = cls_within_sub

        text          = self.text[text_index]
        img           = self.img[img_index]
        text_features = self.text_features[text_index]
        img_features  = self.img_features[img_index]

        return x, label, text, text_features, img, img_features
