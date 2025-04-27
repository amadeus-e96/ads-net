import os
import numpy as np
import matplotlib.pyplot as plt
from imageio import volread
from tifffile import imwrite
from keras.layers import Input
from time import time

from model_architecture import model_UNETPP
from utils.prediction import (
    normalizeImageIntensityRange,
    scaleImg,
    process_img,
    scale_predicted_image,
    plot_and_save_image
)

# ------------------ CONFIG ------------------

TARGET_NAME = '18N480'
IMAGE_HEIGHT, IMAGE_WIDTH = 512, 512
ADD_BLACK_PIXELS = False
SCALE_TO_512 = False

SLICE_X, SLICE_Y, SLICE_Z = True, False, False

MODEL_PATH = 'checkpoints/unetpp_20250421-1801_epoch10.h5'
TARGET_IMAGE_PATH = f'predict/img/{TARGET_NAME}.tif'
PREDICTION_OUTPUT_PATH = f'predict/result/{TARGET_NAME}.tif'
PDF_OUTPUT_PATH = f'predict/result/{TARGET_NAME}.pdf'


# ------------------ SLICE-WISE PREDICTION ------------------

def predict_volume(volume, model, add_black_pixels=False, scale=False):
    """Predict the full volume slice-wise using the selected axis."""
    x_max, y_max, z_max = volume.shape
    output = np.zeros((x_max, y_max, z_max), dtype=np.float32)
    
    if SLICE_X:
        print("[INFO] Predicting along X-axis...")
        for i in range(x_max):
            img = process_img(volume[i, :, :], IMAGE_HEIGHT, IMAGE_WIDTH, add_black_pixels, scale)
            pred = model.predict(img[np.newaxis, ..., np.newaxis])[0, ..., 0]
            output[i, :, :] = scaleImg(pred, y_max, z_max)
    
    if SLICE_Y:
        print("[INFO] Predicting along Y-axis...")
        for i in range(y_max):
            img = process_img(volume[:, i, :], IMAGE_HEIGHT, IMAGE_WIDTH, add_black_pixels, scale)
            pred = model.predict(img[np.newaxis, ..., np.newaxis])[0, ..., 0]
            output[:, i, :] = scaleImg(pred, x_max, z_max)
    
    if SLICE_Z:
        print("[INFO] Predicting along Z-axis...")
        for i in range(z_max):
            img = process_img(volume[:, :, i], IMAGE_HEIGHT, IMAGE_WIDTH, add_black_pixels, scale)
            pred = model.predict(img[np.newaxis, ..., np.newaxis])[0, ..., 0]
            output[:, :, i] = scaleImg(pred, x_max, y_max)

    return output


# ------------------ MAIN ------------------

def main():
    print("[INFO] Loading image volume...")
    try:
        raw_volume = volread(TARGET_IMAGE_PATH)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load {TARGET_IMAGE_PATH}: {e}")

    print("[INFO] Normalizing image intensity...")
    normalized_volume = normalizeImageIntensityRange(raw_volume)

    print("[INFO] Visualizing a slice...")
    slice_index = normalized_volume.shape[2] // 2
    plt.imshow(normalized_volume[:, :, slice_index], cmap='gray')
    plt.title(f'Axial Slice at Z={slice_index}')
    plt.axis('off')
    plt.show()

    print("[INFO] Loading model...")
    input_img = Input((IMAGE_HEIGHT, IMAGE_WIDTH, 1), name='img')
    model = model_UNETPP(input_img, 1)
    model.load_weights(MODEL_PATH)

    print("[INFO] Starting volume prediction...")
    start_time = time()
    predicted_volume = predict_volume(normalized_volume, model, ADD_BLACK_PIXELS, SCALE_TO_512)
    print(f"[INFO] Prediction completed in {time() - start_time:.2f} seconds.")

    print("[INFO] Scaling and saving prediction...")
    final_output = scale_predicted_image(predicted_volume)

    os.makedirs(os.path.dirname(PREDICTION_OUTPUT_PATH), exist_ok=True)
    imwrite(PREDICTION_OUTPUT_PATH, final_output.astype(np.float32))
    plot_and_save_image(final_output, filename=PDF_OUTPUT_PATH)

    print(f"[INFO] Prediction saved to {PREDICTION_OUTPUT_PATH} and {PDF_OUTPUT_PATH}")

    
if __name__ == '__main__':
    main()