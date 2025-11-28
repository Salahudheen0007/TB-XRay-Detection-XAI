#!/usr/bin/env python3
"""
augment_and_split.py

- src_dir: source data folder with subfolders per class, e.g. data/Normal, data/Tuberculosis
- dest_dir: destination splits root, e.g. splits/
- m: max images to take per class (if fewer exist, uses all)
- augment_count: number of augmented images to produce per original image
- train/val/test ratios: should sum to 1.0

Example:
python augment_and_split.py --src data --dest splits --m 200 --augment-count 4 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""
import os
import argparse
import random
from glob import glob
from PIL import Image, ImageEnhance, ImageFilter
import shutil

random.seed(42)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def list_images(folder, exts=('jpg','jpeg','png','bmp','tif','tiff')):
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, f'*.{ext}')))
        files.extend(glob(os.path.join(folder, f'*.{ext.upper()}')))
    files = sorted(files)
    return files

def augment_image(img: Image.Image):
    """
    Apply a sequence of small random augmentations and return the augmented image.
    Augmentations applied randomly:
    - random horizontal flip
    - random rotation (-15..15)
    - random color jitter (brightness/contrast)
    - random slight affine via resize+crop
    - gaussian blur occasionally
    """
    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random rotation
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    # Random resize crop (scale 0.9-1.0)
    scale = random.uniform(0.9, 1.0)
    w, h = img.size
    new_w, new_h = int(w*scale), int(h*scale)
    if new_w < w and new_h < h:
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((w, h), resample=Image.BILINEAR)

    # Color jitter: brightness / contrast
    if random.random() < 0.7:
        factor = random.uniform(0.8, 1.2)
        img = ImageEnhance.Brightness(img).enhance(factor)
    if random.random() < 0.7:
        factor = random.uniform(0.85, 1.25)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # Slight sharpening or blur
    r = random.random()
    if r < 0.1:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,1.5)))
    elif r < 0.2:
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

    return img

def copy_and_augment_for_split(src_paths, dest_root, class_name, split_name, augment_count, start_index=0):
    """
    Copy original files and create augmentations.
    Returns number of images written.
    """
    wrote = 0
    dest_dir = os.path.join(dest_root, split_name, class_name)
    ensure_dir(dest_dir)
    for src in src_paths:
        base = os.path.splitext(os.path.basename(src))[0]
        # Copy original if not exists
        dest_orig = os.path.join(dest_dir, f"{base}.jpg")
        if not os.path.exists(dest_orig):
            try:
                # Open and convert, then save as JPG to standardize
                img = Image.open(src)
                img.convert('RGB').save(dest_orig, format='JPEG', quality=95)
            except Exception as e:
                print(f"Warning: failed to copy {src} -> {dest_orig}: {e}")
                continue
        wrote += 1

        # Augment and save augment_count variants
        for ai in range(augment_count):
            try:
                img = Image.open(src)
                aug = augment_image(img)
                aug_name = os.path.join(dest_dir, f"{base}_aug{ai+1}.jpg")
                # Avoid overwriting if file exists
                if os.path.exists(aug_name):
                    # find a safe name
                    k = 1
                    while os.path.exists(os.path.join(dest_dir, f"{base}_aug{ai+1}_{k}.jpg")):
                        k += 1
                    aug_name = os.path.join(dest_dir, f"{base}_aug{ai+1}_{k}.jpg")
                aug.save(aug_name, format='JPEG', quality=90)
            except Exception as e:
                print(f"Warning: failed to augment {src}: {e}")
                continue
    return wrote

def main(args):
    src_dir = args.src
    dest_dir = args.dest
    m = args.m
    augment_count = args.augment_count

    # Validate ratios sum to ~1.0
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory {src_dir} not found")

    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir,d))]
    if not classes:
        raise RuntimeError(f"No class subfolders found inside {src_dir}. Expected e.g. {src_dir}/Normal")

    print("Classes found:", classes)
    for cls in classes:
        cls_folder = os.path.join(src_dir, cls)
        images = list_images(cls_folder)
        if len(images) == 0:
            print(f"  -> WARNING: class '{cls}' has 0 images in {cls_folder}. Skipping.")
            continue

        # sample up to m images
        if m is None or m <= 0 or m >= len(images):
            selected = images[:]  # all
        else:
            selected = random.sample(images, min(m, len(images)))

        random.shuffle(selected)
        n = len(selected)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        # rest to test
        n_test = n - n_train - n_val

        train_sel = selected[:n_train]
        val_sel = selected[n_train:n_train + n_val]
        test_sel = selected[n_train + n_val:]

        print(f"\nClass '{cls}': total_found={len(images)}, selected_for_processing={n}, splits -> train:{len(train_sel)}, val:{len(val_sel)}, test:{len(test_sel)}")

        # Copy + augment
        if train_sel:
            w = copy_and_augment_for_split(train_sel, dest_dir, cls, 'train', augment_count)
            print(f"  -> wrote {w} originals + {w * augment_count} augmented to {os.path.join(dest_dir,'train',cls)}")
        if val_sel:
            w = copy_and_augment_for_split(val_sel, dest_dir, cls, 'val', augment_count)
            print(f"  -> wrote {w} originals + {w * augment_count} augmented to {os.path.join(dest_dir,'val',cls)}")
        if test_sel:
            w = copy_and_augment_for_split(test_sel, dest_dir, cls, 'test', augment_count)
            print(f"  -> wrote {w} originals + {w * augment_count} augmented to {os.path.join(dest_dir,'test',cls)}")

    print("\nDone. Check your splits/ folders.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample m images per class and augment into splits")
    parser.add_argument('--src', default='data', help='Source data dir (class subfolders inside)')
    parser.add_argument('--dest', default='splits', help='Destination splits root dir')
    parser.add_argument('--m', type=int, default=200, help='Max images to take per class (use all if <=0)')
    parser.add_argument('--augment-count', type=int, default=3, help='How many augmented images to create per original')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    args = parser.parse_args()
    main(args)
