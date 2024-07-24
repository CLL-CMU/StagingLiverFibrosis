import random
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from scipy import ndimage
from timm.models.layers import to_3tuple
import cv2

def load_nii_file(nii_image):
    """
    Load a NIfTI (.nii) file and convert it to a NumPy array.
    
    Args:
        nii_image (str): Path to the NIfTI file.
    
    Returns:
        np.ndarray: 3D array representing the image.
    """
    image = sitk.ReadImage(nii_image)
    image_array = sitk.GetArrayFromImage(image)
    return image_array

def image_normalization(image, win=None, adaptive=True):
    """
    Normalize an image either using a fixed window or adaptively.
    
    Args:
        image (np.ndarray): Input image array.
        win (tuple, optional): Window for fixed normalization (min, max).
        adaptive (bool): Whether to use adaptive normalization.

    Returns:
        np.ndarray: Normalized image array.
    """
    if win is not None:
        image = (image - win[0]) / (win[1] - win[0])
        np.clip(image, 0, 1, out=image) # Ensuring values are between 0 and 1
        return image
    elif adaptive:
        min_val, max_val = np.min(image), np.max(image)
        image = (image - min_val) / (max_val - min_val)
        return image
    else:
        return image

def resize3D(image, size):
    """
    Resize a 3D image to the specified size using trilinear interpolation.
    
    Args:
        image (np.ndarray): 3D image array.
        size (tuple): Target size (depth, height, width).

    Returns:
        np.ndarray: Resized 3D image.
    """
    size = to_3tuple(size)
    image = image.astype(np.float32)
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    resized_image = F.interpolate(image, size=size, mode='trilinear', align_corners=True).squeeze(0).squeeze(0)
    return resized_image.cpu().numpy()

def read_one_mode_proimg(nii_image, size):
    """
    Load, normalize, and resize a 3D NIfTI image.
    
    Args:
        nii_image (str): Path to the NIfTI file.
        size (tuple): Target size (depth, height, width).

    Returns:
        np.ndarray: Preprocessed image.
    """
    image = load_nii_file(nii_image)
    image = image_normalization(image)
    resized_image = resize3D(image, size)
    return np.expand_dims(resized_image, axis=0)

def random_crop(image, crop_shape):
    """
    Perform a random crop on a 3D image.
    
    Args:
        image (np.ndarray): 3D image array of shape (channels, depth, height, width).
        crop_shape (tuple): Target crop size (depth, height, width).

    Returns:
        np.ndarray: Cropped image.
    """
    crop_shape = to_3tuple(crop_shape)
    _, z_shape, y_shape, x_shape = image.shape
    z_min = np.random.randint(0, z_shape - crop_shape[0])
    y_min = np.random.randint(0, y_shape - crop_shape[1])
    x_min = np.random.randint(0, x_shape - crop_shape[2])
    return image[..., z_min:z_min+crop_shape[0], y_min:y_min+crop_shape[1], x_min:x_min+crop_shape[2]]

def center_crop(image, target_shape=(10, 80, 80)):
    """
    Perform a center crop on a 3D image.
    
    Args:
        image (np.ndarray): 3D image array of shape (channels, depth, height, width).
        target_shape (tuple): Target crop size (depth, height, width).

    Returns:
        np.ndarray: Center-cropped image.
    """
    target_shape = to_3tuple(target_shape)
    b, z_shape, y_shape, x_shape = image.shape
    z_min = (z_shape - target_shape[0]) // 2
    y_min = (y_shape - target_shape[1]) // 2
    x_min = (x_shape - target_shape[2]) // 2
    return image[:, z_min:z_min+target_shape[0], y_min:y_min+target_shape[1], x_min:x_min+target_shape[2]]

def random_flip(image, mode='z', p=0.5):
    """
    Perform a random flip on a 3D image along the specified axis.
    
    Args:
        image (np.ndarray): 3D image array.
        mode (str): Axis to flip ('x', 'y', 'z').
        p (float): Probability of performing the flip.

    Returns:
        np.ndarray: Flipped image.
    """
    if random.random() > p:
        return image
    
    if mode == 'x':
        return image[..., ::-1]
    elif mode == 'y':
        return image[:, :, ::-1, ...]
    elif mode == 'z':
        return image[:, ::-1, ...]
    else:
        raise NotImplementedError(f'Unknown flip mode ({mode})')

def rotate(image, angle=10):
    """
    Rotate a 3D image by a random angle.
    
    Args:
        image (np.ndarray): 3D image array.
        angle (int): Maximum rotation angle.

    Returns:
        np.ndarray: Rotated image.
    """
    angle = random.randint(-angle, angle)
    rotated_image = ndimage.rotate(image, angle=angle, axes=(-2, -1), reshape=True)
    
    if rotated_image.shape != image.shape:
        rotated_image = center_crop(rotated_image, target_shape=image.shape[1:])
    
    return rotated_image

def random_intensity(image, factor, p=0.5):
    """
    Randomly adjust the intensity of a 3D image.
    
    Args:
        image (np.ndarray): 3D image array.
        factor (float): Adjustment factor.
        p (float): Probability of performing the adjustment.

    Returns:
        np.ndarray: Intensity-adjusted image.
    """
    if random.random() > p:
        return image
    
    shift_factor = np.random.uniform(-factor, factor, size=[image.shape[0], 1, 1, 1]).astype('float32')
    scale_factor = np.random.uniform(1.0 - factor, 1.0 + factor, size=[image.shape[0], 1, 1, 1]).astype('float32')
    return image * scale_factor + shift_factor

def transform_string(s,mode):
    # 检查输入字符串是否符合预期的形式
    if len(s) > 1 and s[0] == 'S' and s[1].isdigit():
        # 插入下划线和'T1'到指定的位置
        new_s = s[:2] + '_' + mode + '_' + s[2:]
        return new_s
    else:
        # 如果不符合预期形式，返回原字符串或者进行错误处理
        return s