import os
import shutil

# -----------------------------
# CONFIG
# -----------------------------
txt_file = "/hhome/ricse01/TFM/TFM/generated_gt_caption/gt_captions_used.txt"              # your txt file
images_dir = "/hhome/ricse01/TFM/TFM/generated_gt_caption/"            # folder with reconstructed_XXXX.png
output_dir = "structured_output"      # where new structure will be created
images_per_class = 1                 # change if needed

# -----------------------------
# LOAD CLASS NAMES
# -----------------------------
classes = []
with open(txt_file, "r") as f:
    for line in f:
        cls = line.strip().split()[0]   # first word
        classes.append(cls)

print(f"Loaded {len(classes)} classes")

# -----------------------------
# LOAD & SORT IMAGES
# -----------------------------
images = sorted([
    f for f in os.listdir(images_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

print(f"Found {len(images)} images")

# sanity check
expected = len(classes) * images_per_class
if len(images) != expected:
    print(f"[WARNING] Expected {expected} images but found {len(images)}")

# -----------------------------
# CREATE STRUCTURE
# -----------------------------
os.makedirs(output_dir, exist_ok=True)

idx = 0
for cls in classes:
    cls_dir = os.path.join(output_dir, cls)
    os.makedirs(cls_dir, exist_ok=True)

    for i in range(images_per_class):
        if idx >= len(images):
            break

        src = os.path.join(images_dir, images[idx])
        dst = os.path.join(cls_dir, f"{i}.png")

        shutil.copy(src, dst)
        idx += 1

print("Done restructuring dataset.")