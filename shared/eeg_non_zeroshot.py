import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json

# ── Proxy / device setup (keep identical to original) ────────────────────────
proxy = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy

cuda_device_count = torch.cuda.device_count()
print(cuda_device_count)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_type = 'ViT-H-14'
import open_clip
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type,
    pretrained='/hhome/ricse01/TFM/required/open_clip_model.safetensors',
    precision='fp32',
    device=device,
)

# ── Path config ───────────────────────────────────────────────────────────────
config_path = "/hhome/ricse01/TFM/TFM/shared/data_config.json"
with open(config_path, "r") as config_file:
    config = json.load(config_file)

data_path           = config["data_path"]
img_directory_training = config["img_directory_training"]
img_directory_test     = config["img_directory_test"]


# ─────────────────────────────────────────────────────────────────────────────
#  Original zero-shot dataset  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class EEGDataset:
    """
    Original leave-one-subject-out / zero-shot dataset.
    Kept intact so existing imports still work.
    """

    def __init__(self, data_path, exclude_subject=None, subjects=None,
                 train=True, time_window=[0, 1.0], classes=None, pictures=None,
                 val_size=None):
        self.data_path       = data_path
        self.train           = train
        self.subject_list    = os.listdir(data_path)
        self.subjects        = self.subject_list if subjects is None else subjects
        self.n_sub           = len(self.subjects)
        self.time_window     = time_window
        self.n_cls           = 1654 if train else 200
        self.classes         = classes
        self.pictures        = pictures
        self.exclude_subject = exclude_subject
        self.val_size        = val_size

        assert any(sub in self.subject_list for sub in self.subjects)

        self.data, self.labels, self.text, self.img = self.load_data()
        self.data = self.extract_eeg(self.data, time_window)

        if self.classes is None and self.pictures is None:
            features_filename = (
                os.path.join(f'{model_type}_features_train.pt')
                if self.train else
                os.path.join(f'{model_type}_features_test.pt')
            )
            if os.path.exists(features_filename):
                saved_features       = torch.load(features_filename)
                self.text_features   = saved_features['text_features']
                self.img_features    = saved_features['img_features']
            else:
                self.text_features = self.Textencoder(self.text)
                self.img_features  = self.ImageEncoder(self.img)
                torch.save({'text_features': self.text_features.cpu(),
                            'img_features':  self.img_features.cpu()},
                           features_filename)
        else:
            self.text_features = self.Textencoder(self.text)
            self.img_features  = self.ImageEncoder(self.img)

    # ------------------------------------------------------------------
    def load_data(self):
        data_list, label_list, texts, images = [], [], [], []

        directory = img_directory_training if self.train else img_directory_test
        dirnames  = sorted([d for d in os.listdir(directory)
                            if os.path.isdir(os.path.join(directory, d))])
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        for d in dirnames:
            try:
                idx         = d.index('_')
                description = d[idx+1:]
            except ValueError:
                print(f"Skipped: {d} due to no '_' found.")
                continue
            texts.append(f"This picture is {description}")

        img_directory = img_directory_training if self.train else img_directory_test
        all_folders   = sorted([d for d in os.listdir(img_directory)
                                 if os.path.isdir(os.path.join(img_directory, d))])

        if self.classes is not None and self.pictures is not None:
            for class_idx, pic_idx in zip(self.classes, self.pictures):
                if class_idx < len(all_folders):
                    folder_path = os.path.join(img_directory, all_folders[class_idx])
                    all_images  = sorted([img for img in os.listdir(folder_path)
                                          if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    if pic_idx < len(all_images):
                        images.append(os.path.join(folder_path, all_images[pic_idx]))
        elif self.classes is not None and self.pictures is None:
            for class_idx in self.classes:
                if class_idx < len(all_folders):
                    folder_path = os.path.join(img_directory, all_folders[class_idx])
                    all_images  = sorted([img for img in os.listdir(folder_path)
                                          if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    images.extend(os.path.join(folder_path, img) for img in all_images)
        else:
            for folder in all_folders:
                folder_path = os.path.join(img_directory, folder)
                all_images  = sorted([img for img in os.listdir(folder_path)
                                       if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
                images.extend(os.path.join(folder_path, img) for img in all_images)

        print("self.subjects", self.subjects)
        print("exclude_subject", self.exclude_subject)

        for subject in self.subjects:
            if self.train:
                if subject == self.exclude_subject:
                    continue
                file_path = os.path.join(self.data_path, subject,
                                         'preprocessed_eeg_training.npy')
                data = np.load(file_path, allow_pickle=True)
                preprocessed_eeg_data = torch.from_numpy(
                    data['preprocessed_eeg_data']).float().detach()
                times    = torch.from_numpy(data['times']).detach()[50:]
                ch_names = data['ch_names']

                n_classes        = 1654
                samples_per_class = 10

                if self.classes is not None and self.pictures is not None:
                    for c, p in zip(self.classes, self.pictures):
                        start_index = c * 1 + p
                        if start_index < len(preprocessed_eeg_data):
                            data_list.append(preprocessed_eeg_data[start_index: start_index+1])
                            label_list.append(torch.full((1,), c, dtype=torch.long).detach())
                elif self.classes is not None and self.pictures is None:
                    for c in self.classes:
                        start_index = c * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[
                            start_index: start_index+samples_per_class]
                        label_list.append(
                            torch.full((samples_per_class,), c, dtype=torch.long).detach())
                        data_list.append(preprocessed_eeg_data_class)
                else:
                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[
                            start_index: start_index+samples_per_class]
                        label_list.append(
                            torch.full((samples_per_class,), i, dtype=torch.long).detach())
                        data_list.append(preprocessed_eeg_data_class)
            else:
                if subject == self.exclude_subject or self.exclude_subject is None:
                    file_path = os.path.join(self.data_path, subject,
                                             'preprocessed_eeg_test.npy')
                    data = np.load(file_path, allow_pickle=True)
                    preprocessed_eeg_data = torch.from_numpy(
                        data['preprocessed_eeg_data']).float().detach()
                    times    = torch.from_numpy(data['times']).detach()[50:]
                    ch_names = data['ch_names']
                    n_classes        = 200
                    samples_per_class = 1

                    for i in range(n_classes):
                        if self.classes is not None and i not in self.classes:
                            continue
                        start_index = i * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[
                            start_index: start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        preprocessed_eeg_data_class = torch.mean(
                            preprocessed_eeg_data_class.squeeze(0), 0)
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)
                else:
                    continue

        if self.train:
            data_tensor  = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])
            print("data_tensor", data_tensor.shape)
        else:
            data_tensor  = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)

        label_tensor = torch.cat(label_list, dim=0)

        if self.train:
            label_tensor = label_tensor.repeat_interleave(4)
            if self.classes is not None:
                unique_values = list(label_tensor.numpy())
                lis = []
                for i in unique_values:
                    if i not in lis:
                        lis.append(i)
                unique_values = torch.tensor(lis)
                mapping      = {val.item(): index for index, val in enumerate(unique_values)}
                label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor],
                                            dtype=torch.long)

        self.times    = times
        self.ch_names = ch_names
        print(f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, "
              f"text length: {len(texts)}, image length: {len(images)}")
        return data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):
        start, end = time_window
        indices = (self.times >= start) & (self.times <= end)
        return eeg_data[..., indices]

    def Textencoder(self, text):
        text_inputs    = torch.cat([clip.tokenize(t) for t in text]).to(device)
        with torch.no_grad():
            text_features = vlmodel.encode_text(text_inputs)
        return F.normalize(text_features, dim=-1).detach()

    def ImageEncoder(self, images):
        batch_size          = 20
        image_features_list = []
        for i in range(0, len(images), batch_size):
            batch_images  = images[i:i+batch_size]
            image_inputs  = torch.stack([
                preprocess_train(Image.open(img).convert("RGB"))
                for img in batch_images
            ]).to(device)
            with torch.no_grad():
                batch_image_features  = vlmodel.encode_image(image_inputs)
                batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(batch_image_features)
        return torch.cat(image_features_list, dim=0)

    def __getitem__(self, index):
        x     = self.data[index]
        label = self.labels[index]

        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 10 * 4
                index_n_sub_test  = self.n_cls * 1 * 80
            else:
                index_n_sub_test  = len(self.classes) * 1 * 80
                index_n_sub_train = len(self.classes) * 10 * 4
            if self.train:
                text_index = (index % index_n_sub_train) // (10 * 4)
                img_index  = (index % index_n_sub_train) // 4
            else:
                text_index = index % index_n_sub_test
                img_index  = index % index_n_sub_test
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 4
                index_n_sub_test  = self.n_cls * 1 * 80
            else:
                index_n_sub_test  = len(self.classes) * 1 * 80
                index_n_sub_train = len(self.classes) * 1 * 4
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
                img_index  = (index % index_n_sub_train) // 4
            else:
                text_index = index % index_n_sub_test
                img_index  = index % index_n_sub_test

        text          = self.text[text_index]
        img           = self.img[img_index]
        text_features = self.text_features[text_index]
        img_features  = self.img_features[img_index]
        return x, label, text, text_features, img, img_features

    def __len__(self):
        return self.data.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Non-zero-shot leave-one-out dataset
# ─────────────────────────────────────────────────────────────────────────────
class EEGDatasetNonZeroShot:
    """
    Leave-one-image-out dataset for non-zero-shot evaluation.

    For each of the 1654 training classes:
      - Train split: 9 EEG repetitions (all except `holdout_img_idx`)
      - Test  split: 1 EEG repetition  (the held-out one, averaged over repetitions
                     the same way the original test set is built)

    The image features / text features are drawn from the *training* image directory
    (not the 200-class zero-shot test directory).

    Parameters
    ----------
    data_path       : str   — path to preprocessed EEG data root
    subject         : str   — single subject ID, e.g. 'sub-08'
    train           : bool  — True → return 9-image training split
                              False → return 1-image held-out test split
    holdout_img_idx : int   — which of the 10 EEG repetitions to hold out (0–9)
    time_window     : list  — [start, end] in seconds
    classes         : list  — optional subset of class indices (0-based)
    """

    N_TRAIN_CLASSES   = 1654
    SAMPLES_PER_CLASS = 10   # EEG repetitions per class in training file
    EEG_REPEATS       = 4    # temporal augmentation repeats kept from original code

    def __init__(self, data_path, subject, train=True,
                 holdout_img_idx=0, time_window=[0, 1.0], classes=None):
        self.data_path       = data_path
        self.subject         = subject
        self.train           = train
        self.holdout_img_idx = holdout_img_idx
        self.time_window     = time_window
        self.classes         = list(range(self.N_TRAIN_CLASSES)) if classes is None else classes
        self.n_cls           = len(self.classes)

        assert subject in os.listdir(data_path), \
            f"Subject {subject} not found in {data_path}"
        assert 0 <= holdout_img_idx < self.SAMPLES_PER_CLASS, \
            f"holdout_img_idx must be in 0..{self.SAMPLES_PER_CLASS-1}"

        # Build EEG data, labels, image paths, text descriptions
        self.data, self.labels, self.text, self.img, self.times = self._load_data()
        self.data = self._extract_eeg(self.data, time_window)

        # Encode text & image features (cached per subject + split)
        cache_tag  = f"nonzeroshot_{subject}_{'train' if train else 'test'}_hold{holdout_img_idx}"
        cache_file = os.path.join(f'{model_type}_{cache_tag}.pt')
        if os.path.exists(cache_file):
            saved            = torch.load(cache_file)
            self.text_features = saved['text_features']
            self.img_features  = saved['img_features']
        else:
            self.text_features = self._encode_text(self.text)
            self.img_features  = self._encode_images(self.img)
            torch.save({'text_features': self.text_features.cpu(),
                        'img_features':  self.img_features.cpu()},
                       cache_file)

    # ------------------------------------------------------------------
    def _load_data(self):
        """
        Load EEG from the *training* file and split into 9-train / 1-test
        using holdout_img_idx.

        EEG training file layout (per subject):
          preprocessed_eeg_data: shape (N_TRAIN_CLASSES * SAMPLES_PER_CLASS, n_repeats, n_ch, n_times)
          i.e. row k  = class (k // 10), repetition (k % 10)

        Each EEG sample has 4 temporal repeats (dim 1), which the original
        code flattens with repeat_interleave(4) on the labels side. We replicate
        that behaviour here.
        """
        file_path = os.path.join(self.data_path, self.subject,
                                 'preprocessed_eeg_training.npy')
        raw = np.load(file_path, allow_pickle=True)

        # shape: (16540, n_repeats, n_ch, n_times)  for 1654 classes × 10 reps
        eeg_all  = torch.from_numpy(raw['preprocessed_eeg_data']).float().detach()
        times    = torch.from_numpy(raw['times']).detach()[50:]
        ch_names = raw['ch_names']

        # ── Collect image paths from the training image directory ──────────
        all_folders = sorted([d for d in os.listdir(img_directory_training)
                              if os.path.isdir(os.path.join(img_directory_training, d))])

        data_list, label_list, texts, images = [], [], [], []

        for local_idx, class_idx in enumerate(self.classes):
            # ── Text description ──────────────────────────────────────────
            folder_name = all_folders[class_idx]
            try:
                sep         = folder_name.index('_')
                description = folder_name[sep+1:]
            except ValueError:
                description = folder_name
            texts.append(f"This picture is {description}")

            # ── Image paths ───────────────────────────────────────────────
            folder_path = os.path.join(img_directory_training, folder_name)
            all_imgs    = sorted([f for f in os.listdir(folder_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            # ── EEG rows for this class ───────────────────────────────────
            # rows: class_idx*10 .. class_idx*10+9  (10 repetitions)
            start_row = class_idx * self.SAMPLES_PER_CLASS
            class_eeg = eeg_all[start_row: start_row + self.SAMPLES_PER_CLASS]
            # shape: (10, n_repeats, n_ch, n_times)

            if self.train:
                # ── 9 EEG repetitions (all except holdout) ────────────────
                keep_mask = [i for i in range(self.SAMPLES_PER_CLASS)
                             if i != self.holdout_img_idx]
                train_eeg = class_eeg[keep_mask]      # (9, n_repeats, n_ch, n_times)

                # Corresponding image paths (one per kept repetition)
                for rep_idx in keep_mask:
                    img_idx = rep_idx % len(all_imgs)
                    images.append(os.path.join(folder_path, all_imgs[img_idx]))

                # labels shape before repeat: (9,)
                labels = torch.full((len(keep_mask),), local_idx, dtype=torch.long)
                data_list.append(train_eeg)
                label_list.append(labels)

            else:
                # ── 1 held-out EEG repetition ─────────────────────────────
                held_eeg = class_eeg[self.holdout_img_idx]  # (n_repeats, n_ch, n_times)
                # Average over temporal repeats to match original test processing
                held_eeg_avg = torch.mean(held_eeg, dim=0)  # (n_ch, n_times)

                # Held-out image
                img_idx = self.holdout_img_idx % len(all_imgs)
                images.append(os.path.join(folder_path, all_imgs[img_idx]))

                labels = torch.full((1,), local_idx, dtype=torch.long)
                data_list.append(held_eeg_avg)
                label_list.append(labels)

        if self.train:
            # Each entry in data_list: (9, n_repeats, n_ch, n_times)
            # Flatten to (9 * n_classes, n_ch, n_times) by collapsing repeats
            data_tensor  = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])
            label_tensor = torch.cat(label_list, dim=0)
            # Replicate label for each of the 4 temporal repeats (matching original)
            label_tensor = label_tensor.repeat_interleave(self.EEG_REPEATS)
        else:
            # Each entry: (n_ch, n_times)  — already averaged
            data_tensor  = torch.stack(data_list, dim=0)   # (n_classes, n_ch, n_times)
            label_tensor = torch.cat(label_list, dim=0)    # (n_classes,)

        print(f"[NonZeroShot] {'Train' if self.train else 'Test'} | "
              f"subject={self.subject} | holdout={self.holdout_img_idx} | "
              f"data={data_tensor.shape} | labels={label_tensor.shape} | "
              f"images={len(images)}")

        return data_tensor, label_tensor, texts, images, times

    # ------------------------------------------------------------------
    def _extract_eeg(self, eeg_data, time_window):
        start, end = time_window
        indices    = (self.times >= start) & (self.times <= end)
        return eeg_data[..., indices]

    def _encode_text(self, text):
        text_inputs = torch.cat([clip.tokenize(t) for t in text]).to(device)
        with torch.no_grad():
            feats = vlmodel.encode_text(text_inputs)
        return F.normalize(feats, dim=-1).detach()

    def _encode_images(self, images):
        batch_size = 20
        feats_list = []
        for i in range(0, len(images), batch_size):
            batch  = images[i:i+batch_size]
            inputs = torch.stack([
                preprocess_train(Image.open(img).convert("RGB"))
                for img in batch
            ]).to(device)
            with torch.no_grad():
                f  = vlmodel.encode_image(inputs)
                f /= f.norm(dim=-1, keepdim=True)
            feats_list.append(f)
        return torch.cat(feats_list, dim=0)

    # ------------------------------------------------------------------
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x     = self.data[index]
        label = self.labels[index]

        if self.train:
            # index maps to (class_local, eeg_rep, temporal_repeat)
            # n_kept = 9 reps, EEG_REPEATS = 4 → 36 samples per class
            samples_per_class = (self.SAMPLES_PER_CLASS - 1) * self.EEG_REPEATS
            class_local = index // samples_per_class

            # img_index: one image per kept repetition (9 images × n_classes)
            rep_within_class = (index % samples_per_class) // self.EEG_REPEATS
            img_index  = class_local * (self.SAMPLES_PER_CLASS - 1) + rep_within_class
            text_index = class_local
        else:
            # One sample per class
            text_index = index
            img_index  = index

        text          = self.text[text_index]
        img           = self.img[img_index]
        text_features = self.text_features[text_index]
        img_features  = self.img_features[img_index]

        return x, label, text, text_features, img, img_features


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_ds = EEGDatasetNonZeroShot(
        data_path=data_path,
        subject='sub-08',
        train=True,
        holdout_img_idx=0,
    )
    test_ds = EEGDatasetNonZeroShot(
        data_path=data_path,
        subject='sub-08',
        train=False,
        holdout_img_idx=0,
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,  shuffle=False)

    x, lbl, txt, tf, img, imf = train_ds[0]
    print(f"Train sample 0 — x: {x.shape}, label: {lbl}, text: {txt}")
    x, lbl, txt, tf, img, imf = test_ds[0]
    print(f"Test  sample 0 — x: {x.shape}, label: {lbl}, text: {txt}")
