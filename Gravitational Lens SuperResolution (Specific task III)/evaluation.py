import torch
import torch.nn.functional as F
import math


def psnr(sr_img, hr_img, max_pixel_value=1.0):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between super-resolved and high-resolution images.
    
    Args:
        sr_img (torch.Tensor): Super-resolved image (C, H, W) or (H, W)
        hr_img (torch.Tensor): High-resolution (ground truth) image (C, H, W) or (H, W)
        max_pixel_value (float): Maximum possible pixel value (1.0 for normalized images, 255 for unnormalized)
    
    Returns:
        float: PSNR value
    """
    sr_img = sr_img * max_pixel_value
    hr_img = hr_img * max_pixel_value
    
    mse = F.mse_loss(sr_img, hr_img)
    if mse == 0:
        return float("inf")
    
    return 20 * math.log10(max_pixel_value) - 10 * math.log10(mse.item())


def gaussian_kernel(size=11, sigma=1.5):
    """Creates a 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = g / g.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(1, 1, size, size)  # Shape: (1,1,H,W)


def ssim(sr_img, hr_img, window_size=11, sigma=1.5):
    """Computes SSIM between two images using PyTorch."""
    C1 = 0.01**2
    C2 = 0.03**2

    # Create Gaussian kernel
    kernel = gaussian_kernel(window_size, sigma).to(sr_img.device)

    # Compute means
    mu_sr = F.conv2d(sr_img, kernel, padding=window_size//2, groups=sr_img.shape[1])
    mu_hr = F.conv2d(hr_img, kernel, padding=window_size//2, groups=hr_img.shape[1])

    mu_sr_sq = mu_sr ** 2
    mu_hr_sq = mu_hr ** 2
    mu_sr_hr = mu_sr * mu_hr

    # Compute variances & covariance
    sigma_sr_sq = F.conv2d(sr_img * sr_img, kernel, padding=window_size//2, groups=sr_img.shape[1]) - mu_sr_sq
    sigma_hr_sq = F.conv2d(hr_img * hr_img, kernel, padding=window_size//2, groups=hr_img.shape[1]) - mu_hr_sq
    sigma_sr_hr = F.conv2d(sr_img * hr_img, kernel, padding=window_size//2, groups=sr_img.shape[1]) - mu_sr_hr

    # Compute SSIM map
    ssim_map = ((2 * mu_sr_hr + C1) * (2 * sigma_sr_hr + C2)) / ((mu_sr_sq + mu_hr_sq + C1) * (sigma_sr_sq + sigma_hr_sq + C2))

    return ssim_map.mean().item()  # Return average SSIM
    
def mse(sr_img, hr_img):
    return F.mse_loss(sr_img, hr_img).item()