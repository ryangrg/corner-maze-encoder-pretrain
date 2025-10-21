import numpy as np
import torch, torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from skimage.filters import sobel
import cv2
from PIL import Image
import imagehash

def preprocess(u8):  # u8: HxW uint8
    x = u8.copy()
    x = cv2.medianBlur(x, 3)                  # impulse denoise
    return x

def phash_u8(u8):
    return imagehash.phash(Image.fromarray(u8))

def edge_binary(u8, q=0.90):
    e = sobel(u8.astype(np.float32)/255.0)
    thr = np.quantile(e, q)
    return (e >= thr).astype(np.uint8)

def masked_psnr(a, b, mask=None):
    a = a.astype(np.float32)/255.0
    b = b.astype(np.float32)/255.0
    if mask is None:
        mask = np.ones_like(a, dtype=bool)
    diff = (a-b)[mask]
    mse = np.mean(diff**2) if diff.size else 0.0
    if mse == 0: return 99.0
    return 10*np.log10(1.0/mse)

def dropout_mask(a, b):
    # mark isolated big diffs as impulse; dilate by 1 px
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)) > 64  # > ~25% intensity jump
    d = d.astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    d = cv2.dilate(d, np.ones((3,3), np.uint8), iterations=1)
    return d.astype(bool)

def verify(u8_a, u8_b):
    a = preprocess(u8_a); b = preprocess(u8_b)

    # SSIM (Gaussian weights)
    ssim_val = ssim(a, b, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

    # Edge IoU
    ea, eb = edge_binary(a), edge_binary(b)
    inter = np.logical_and(ea, eb).sum()
    union = np.logical_or(ea, eb).sum()
    iou = inter / max(union, 1)

    # Masked PSNR (ignore impulse pixels)
    mask = ~dropout_mask(u8_a, u8_b)
    psnr_val = masked_psnr(u8_a, u8_b, mask)

    return (ssim_val, iou, psnr_val)

# Example decision
# s, iou, p = verify(img1, img2)
# is_dup = (s >= 0.98) and (iou >= 0.95) and (p >= 40.0)