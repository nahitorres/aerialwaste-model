import numpy as np
from PIL import Image

def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)

def get_strided_up_size(orig_size, stride):
    strided_size = get_strided_size(orig_size, stride)
    return strided_size[0]*stride, strided_size[1]*stride

def hwc_to_chw(image):
    """Covert the input image from HWC format (Height, Width, Channel) to CHW
    format (Channel, Height, Width).

    Parameters
    ----------
    image: numpy.ndarray
        Input image.

    Returns
    -------
    image: numpy.ndarray
        Converted image.
    """

    image = np.transpose(image, (2, 0, 1))
    return image


def normalize_image(image,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)):
    """Normalize the input image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    mean : tuple, optional
        Mean values ​​for each of the three RGB channels with which the
        normalized image is calculated, by default (0.485, 0.456, 0.406).
    std : tuple, optional
        Standard deviation values ​​for each of the three RGB channels with
        which the normalized image is calculated, by default
        (0.229, 0.224, 0.225).

    Returns
    -------
    numpy.ndarray
        Normalized image.
    """
    assert len(mean) == len(std) == 3, "mean and std must be both of length 3"

    normalized_image = np.empty_like(image, np.float32)
    for i in range(3):
        normalized_image[..., i] = (image[..., i] / 255. - mean[i]) / std[i]
    return normalized_image


def process_image_for_cams(image):
    """Process the image for CAMs.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        Processed image.
    """
    processed_image = np.stack([image, np.flip(image, axis=-1)], axis=0)
    return processed_image

def pre_process_image(image):
    """Pre-process the image for classification and for CAMs.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        Pre-processed image.
    """
    pre_processed_image = normalize_image(image)
    if pre_processed_image.shape[2] == 3:
        pre_processed_image = hwc_to_chw(pre_processed_image)
    return pre_processed_image

def rescale_image(image, scale, order):
    height, width = image.shape[:2]
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return resize_image(image, target_size, order)

def resize_image(image, size, order):
    if size[0] == image.shape[0] and size[1] == image.shape[1]:
        return image
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return np.asarray(Image.fromarray(image).resize(size[::-1], resample))