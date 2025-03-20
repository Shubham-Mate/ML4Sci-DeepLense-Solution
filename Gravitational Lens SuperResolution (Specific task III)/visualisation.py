import matplotlib.pyplot as plt
import torch

def visualize_samples(dataloader, num_samples=5):
    """Visualize LR and HR image pairs from the dataloader."""
    batch = next(iter(dataloader))  # Get a batch
    lr_images, hr_images = batch  # Unpack LR and HR images

    fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 3))

    for i in range(num_samples):
        lr_img = lr_images[i].permute(1, 2, 0).numpy()  # Convert to (H, W, C) for display
        hr_img = hr_images[i].permute(1, 2, 0).numpy()

        axes[i, 0].imshow(lr_img)
        axes[i, 0].set_title("Low-Resolution")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(hr_img)
        axes[i, 1].set_title("High-Resolution")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

def visualize_results(model, dataloader, num_samples=5, device="cuda"):
    model.eval()
    model.to(device)

    # Get a batch of test images
    lr_imgs, hr_imgs = next(iter(dataloader))
    lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

    # Generate super-resolved images
    with torch.no_grad():
        sr_imgs = model(lr_imgs)

    # Convert tensors to numpy arrays for visualization
    lr_imgs = lr_imgs.cpu().permute(0, 2, 3, 1).numpy()  # (B, C, H, W) → (B, H, W, C)
    sr_imgs = sr_imgs.cpu().permute(0, 2, 3, 1).numpy()
    hr_imgs = hr_imgs.cpu().permute(0, 2, 3, 1).numpy()

    # Plot images
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))

    for i in range(num_samples):
        axes[i, 0].imshow(lr_imgs[i])
        axes[i, 0].set_title("Low-Resolution")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(sr_imgs[i])
        axes[i, 1].set_title("Super-Resolved")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(hr_imgs[i])
        axes[i, 2].set_title("High-Resolution (Ground Truth)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()
