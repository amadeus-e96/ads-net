import numpy as np
import cv2
import matplotlib.pyplot as plt

def normalizeImageIntensityRange(img):
    """
    Normalizes the image intensity to the range [0, 1].

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The normalized image.
    """
    img[img < 0] = 0
    img[img > 255] = 255
    return img / 255

def process_img(img, target_height, target_width, black_pixels, scale):
    """
    Processes an image by either adding black pixels or scaling it to the target dimensions.

    Args:
        img (numpy.ndarray): The input image.
        target_height (int): The target height for scaling.
        target_width (int): The target width for scaling.
        black_pixels (bool): Whether to add black padding to the image.
        scale (bool): Whether to scale the image.

    Returns:
        numpy.ndarray: The processed image.
    """
    if black_pixels:
        return add_black_pixels(img, target_height, target_width)
    elif scale:
        return scaleImg(img, target_height, target_width)
    else:
        return img


def scaleImg(img, target_height, target_width):
    """
    Scales the input image to the target dimensions.

    Args:
        img (numpy.ndarray): The input image.
        target_height (int): The target height for scaling.
        target_width (int): The target width for scaling.

    Returns:
        numpy.ndarray: The scaled image.
    """
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def add_black_pixels(img, target_height, target_width):
    """
    Adds black padding to the image to match the target dimensions.

    Args:
        img (numpy.ndarray): The input image.
        target_height (int): The target height for padding.
        target_width (int): The target width for padding.

    Returns:
        numpy.ndarray: The padded image.
    """
    height, width = img.shape
    top_pad = (target_height - height) // 2
    bottom_pad = target_height - height - top_pad
    left_pad = (target_width - width) // 2
    right_pad = target_width - width - left_pad
    return np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=0)


def scale_predicted_image(predImg):
    """
    Processes the predicted image for visualization by adjusting values for display.

    Args:
        predImg (numpy.ndarray): The predicted image to process.

    Returns:
        numpy.ndarray: The processed image ready for visualization.
    """
    Final_img = np.array(predImg).astype(np.float32)
    Final_img[Final_img < 0] = 0  
    Final_img *= 255  
    return Final_img


def plot_and_save_image(Final_img, filename='18N480.pdf'):
    """
    Plots the processed image and saves it as a PDF file.

    Args:
        Final_img (numpy.ndarray): The image to plot and save.
        filename (str): The filename for saving the plot (default is '18N480.pdf').

    """
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'font.serif': ['Helvetica'],
        'axes.linewidth': 0.5,
        'xtick.major.size': 3,
        'xtick.major.width': 0.5,
        'ytick.major.size': 3,
        'ytick.major.width': 0.5,
        'legend.frameon': True,
        'legend.framealpha': 1,
        'legend.fancybox': False,
        'legend.edgecolor': 'black'
    })
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.imshow(Final_img[0, :, :], cmap='turbo')
    plt.colorbar()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()