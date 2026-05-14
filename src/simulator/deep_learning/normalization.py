"""
Common normalization methods used by deep learning models.

Copyright © 2022 QuantrolOx Ltd
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import zoom  # type: ignore


def scale_image_array(x: np.ndarray, target_shape: List[int]) -> np.ndarray:
    """
    Resizes an image array to the provided shape
    Args:
        x: image to scale
        target_shape: the shape of the returned array.
    """
    # Compute scaling factors as ratios of target to current shape
    factors = [float(t) / float(c) if c != 0 else 1 for t, c in zip(target_shape, x.shape)]

    x_resized = zoom(x, factors)

    if list(x_resized.shape) == target_shape:
        return x_resized

    raise ValueError("Target shape could not be attained.")


def image_preprocess(
    x: np.ndarray, norm: Optional[float] = None, scale: float = 1.25, image_slice: int = 1
) -> Tuple[np.ndarray, float]:
    """
    Normalizes image-like array with norm factor if provided, else computes according to default
    scale.

    Args:
        x: array to normalize
        norm: normalization factor
        scale: factor used to compute normalization term if norm is not provided
    Returns:
        x: normalized array
        norm: normalization factor computed
    """
    x = x.astype(np.float32)
    if len(x.shape) == 2:
        x = x[np.newaxis, ...]
    x = x[:, ::image_slice, ::image_slice]

    if norm is None:
        norm = scale * np.abs(x).max(axis=(1, 2))[:, np.newaxis, np.newaxis]

    x = x / norm

    mask = np.sign(x)
    x = np.sqrt(np.abs(x)) * mask

    return x, norm


def normalize_image_input_output_pair(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jointly normalizes an input-output pair of images.
    The normalization factor is computed from the input image, and used for the output image.
    Works both with individual images and with batches of images.

    Args:
        x: input image
        y: output image
    Returns:
        Tuple of normalized images.
    """
    if not x.shape == y.shape:
        raise ValueError("Cannot normalize batches of different shapes.")
    normalized_x, norm_factor = image_preprocess(x)
    normalized_y, _ = image_preprocess(y, norm=norm_factor)
    return normalized_x, normalized_y


def normalize_image_input(x: np.ndarray) -> np.ndarray:
    """
    Returns the normalized image x, discarding the normalization factor.
    """
    x_normalized, _ = image_preprocess(x)
    return x_normalized
