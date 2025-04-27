import os
import glob
from tqdm import tqdm

from utils.slicing import (
    readImageVolume,
    readMaskVolume,
    sliceAndSaveVolumeImage
)

# ------------------ CONFIG ------------------

TARGET = "test"  # "train" or "test"
DATA_INPUT_PATH = f'train_data/{TARGET}/'
IMAGE_INPUT_PATH = os.path.join(DATA_INPUT_PATH, 'imgDT/')
MASK_INPUT_PATH = os.path.join(DATA_INPUT_PATH, 'mask/')

SLICE_OUTPUT_DIR = f'slices/{TARGET}'
IMAGE_SLICE_OUTPUT = os.path.join(SLICE_OUTPUT_DIR, 'img/')
MASK_SLICE_OUTPUT = os.path.join(SLICE_OUTPUT_DIR, 'mask/')

SUPPORTED_IMAGE_EXT = '*.tif'
SUPPORTED_MASK_EXT = '*.fda'


# ------------------ CORE FUNCTIONS ------------------

def ensure_directory(path):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def process_file(filename, index, is_image=True):
    """
    Reads and slices a volume (image or mask) and saves the slices.
    
    Args:
        filename (str): Path to the file.
        index (int): Index used for naming.
        is_image (bool): If True, treat as image; otherwise as mask.
    """
    try:
        if is_image:
            vol = readImageVolume(filename, normalize=False)
            out_path = IMAGE_SLICE_OUTPUT
        else:
            vol = readMaskVolume(filename, normalize=False)
            out_path = MASK_SLICE_OUTPUT

        sliceAndSaveVolumeImage(vol, str(index), out_path)

    except Exception as e:
        print(f"[ERROR] Failed to process {'image' if is_image else 'mask'}: {filename}")
        print(e)


def process_all_files():
    """Main slicing loop for both images and masks."""
    
    ensure_directory(IMAGE_SLICE_OUTPUT)
    ensure_directory(MASK_SLICE_OUTPUT)

    image_files = sorted(glob.glob(os.path.join(IMAGE_INPUT_PATH, SUPPORTED_IMAGE_EXT)))
    mask_files = sorted(glob.glob(os.path.join(MASK_INPUT_PATH, SUPPORTED_MASK_EXT)))

    print(f"[INFO] Found {len(image_files)} image(s), {len(mask_files)} mask(s)")

    for index, filename in tqdm(enumerate(image_files), total=len(image_files), desc="Slicing Images"):
        process_file(filename, index, is_image=True)

    for index, filename in tqdm(enumerate(mask_files), total=len(mask_files), desc="Slicing Masks"):
        process_file(filename, index, is_image=False)


# ------------------ MAIN ------------------

def main():
    print("[INFO] Starting volume slicing...")
    process_all_files()
    print("[INFO] Slicing complete. Output saved to:", SLICE_OUTPUT_DIR)


if __name__ == '__main__':
    main()