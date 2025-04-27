import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datetime import datetime

from keras.utils.image_utils import img_to_array, load_img 
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from skimage.transform import resize

from model_architecture import model_ADSNET, ssim_loss

IMAGE_HEIGHT, IMAGE_WIDTH = 512, 512
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH, 1)
BATCH_SIZE = 4
NUM_EPOCHS = 10
CHECKPOINT_DIR = 'checkpoints'

data_paths = {
    "train": {
        "img": "slices/train/img/",
        "mask": "slices/train/mask/"
    },
    "test": {
        "img": "slices/test/img/",
        "mask": "slices/test/mask/"
    }
}


# ------------------ UTILS ------------------

def load_data(image_dir, mask_dir, image_size):
    ids = [f for f in next(os.walk(image_dir))[2] if f.lower().endswith(".bmp")]
    X = np.zeros((len(ids), *image_size), dtype=np.float32)
    Y = np.zeros((len(ids), *image_size), dtype=np.float32)
    
    for i, file_id in tqdm(enumerate(ids), total=len(ids), desc=f"Loading BMP data from {image_dir}"):
        try:
            img_path = os.path.join(image_dir, file_id)
            mask_path = os.path.join(mask_dir, file_id)

            img = img_to_array(load_img(img_path, color_mode="grayscale"))
            mask = img_to_array(load_img(mask_path, color_mode="grayscale"))

            X[i] = resize(img, image_size, mode='constant', preserve_range=True) / 255.
            Y[i] = resize(mask, image_size, mode='constant', preserve_range=True) / 255.

        except Exception as e:
            print(f"[ERROR] Could not process {file_id}: {e}")
            continue

    return X, Y


def plot_training(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Val Loss')
    if 'ssim_loss' in history.history:
        plt.plot(history.history.get('ssim_loss', []), label='Train SSIM Loss')
        plt.plot(history.history.get('val_ssim_loss', []), label='Val SSIM Loss')
    
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# ------------------ MAIN ------------------

X_train, Y_train = load_data(data_paths['train']['img'], data_paths['train']['mask'], IMAGE_SIZE)
X_test, Y_test = load_data(data_paths['test']['img'], data_paths['test']['mask'], IMAGE_SIZE)

input_img = Input(IMAGE_SIZE, name='img')
model = model_ADSNET(input_img, 1)
model.compile(optimizer='adam', loss=ssim_loss, metrics=["accuracy"])
model.summary()

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

timestamp = datetime.now().strftime("%Y%m%d-%H%M")
checkpoint_path = os.path.join(CHECKPOINT_DIR, f"unetpp_{timestamp}_epoch{{epoch:02d}}.h5")

model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_best_only=False
)

history = model.fit(
    X_train, Y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(X_test, Y_test),
    callbacks=[model_checkpoint]
)

plot_training(history)