import numpy as np
import cv2
import os 
from imageio import volread

MIN = 0
MAX = 255
RANGE = MAX - MIN

SLICE_X = True
SLICE_Y = False
SLICE_Z = False

SLICE_DECIMATE_IDENTIFIER = 0

incr_plan = 20

def read_fda(filename):
    """
    Reads a binary file in FDA format and returns the corresponding image data as a NumPy array.

    Args:
        filename (str): The path to the FDA file.

    Returns:
        numpy.ndarray: The image data as a 3D or 2D array, depending on the dimensions.
    """
    with open(filename, "rb") as f:
        hdr_buf = f.read(4)
        [code] = np.frombuffer(hdr_buf, dtype=np.int32, count=1)
        if code >= 300:
            hdr_buf = f.read(12)
            [nx, ny, nz] = np.frombuffer(hdr_buf, dtype=np.int32, count=3)
        else:
            hdr_buf = f.read(8)
            [nx, ny] = np.frombuffer(hdr_buf, dtype=np.int32, count=2)
            nz = 1
        dtype = {
            300: np.uint8,
            301: np.uint32,
            302: np.float32,
            303: np.float64,
            200: np.uint8,
            201: np.uint32,
            202: np.float32,
            203: np.float64,
        }[code]
        bytes_data = f.read()
        array_data = np.frombuffer(bytes_data, dtype=dtype, count=nx * ny * nz)
        if code >= 300:  
            return np.reshape(array_data, (nz, nx, ny))
        else:
            return np.reshape(array_data, (nx, ny, 1))


def normalizeImageIntensityRange(img):
    """
    Normalizes the image intensity to the range [0, 1].

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The normalized image.
    """
    img[img < MIN] = MIN
    img[img > MAX] = MAX
    return img / 255


def readImageVolume(imgPath, normalize=False):
    """
    Reads an image volume from a file and optionally normalizes the intensity.

    Args:
        imgPath (str): The path to the image file.
        normalize (bool): Whether to normalize the intensity.

    Returns:
        numpy.ndarray: The image volume.
    """
    img = volread(imgPath)
    if normalize:
        return normalizeImageIntensityRange(img)
    else:
        return img


def readMaskVolume(imgPath, normalize=False):
    """
    Reads a mask volume from a file and optionally normalizes the intensity.

    Args:
        imgPath (str): The path to the mask file.
        normalize (bool): Whether to normalize the intensity.

    Returns:
        numpy.ndarray: The mask volume.
    """
    img = read_fda(imgPath)
    if normalize:
        return normalizeImageIntensityRange(img)
    else:
        return img


def saveSlice(img, fname, path):
    """
    Saves a 2D slice of the image as a BMP file.

    Args:
        img (numpy.ndarray): The 2D slice to save.
        fname (str): The filename for saving the slice.
        path (str): The directory path where the slice should be saved.
    """
    fout = os.path.join(path, f'{fname}.bmp')
    cv2.imwrite(fout, img)


def sliceAndSaveVolumeImage(vol, fname, path):
    """
    Slices a 3D volume and saves the slices as BMP files.

    Args:
        vol (numpy.ndarray): The 3D image volume.
        fname (str): The base filename for the slices.
        path (str): The directory path where the slices should be saved.

    Returns:
        int: The number of slices saved.
    """
    (dimx, dimy, dimz) = vol.shape
    counter = 0
    
    if SLICE_X:
        counter += dimx
        for i in range (0, dimx, incr_plan):
            saveSlice(vol[i, :, :], fname + f'{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path)
        
    if SLICE_Y:
        counter += dimy
        for i in range(0, dimy, incr_plan):
            saveSlice(vol[:, i, :], fname + f'{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y', path)

    if SLICE_Z:
        counter += dimz
        for i in range(0, dimz, incr_plan):
            saveSlice(vol[:, :, i], fname + f'{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z', path)
            
    return counter

