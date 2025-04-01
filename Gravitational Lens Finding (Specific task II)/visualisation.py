import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def visualize_dataset(dataset, num_images=4):
    """
    Visualizes an equal number of lens and non-lens images from the dataset.

    Args:
        dataset (LensDataset): The dataset object.
        num_images (int): Number of images to display per class.
    """
    lens_images = []
    nonlens_images = []

    # Collect an equal number of lens and non-lens images
    for img, label in dataset:
        if label == 1 and len(lens_images) < num_images:
            lens_images.append((img, "Lens"))
        elif label == 0 and len(nonlens_images) < num_images:
            nonlens_images.append((img, "Non-Lens"))
        
        if len(lens_images) >= num_images and len(nonlens_images) >= num_images:
            break

    # Combine both classes
    images = lens_images + nonlens_images

    # Plot images
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))

    for i, (img, label) in enumerate(images):
        ax = axes[i // num_images, i % num_images]
        img = img.numpy().transpose(1, 2, 0)  # Convert (3, 64, 64) → (64, 64, 3)
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")

    plt.show()



def plot_roc_curve(model, test_loader, device):
    """
    Plots the ROC curve for a binary classification model.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device (CPU/GPU) for computation.

    Returns:
        None (Displays the ROC curve)
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1]  # Probabilities for class 1

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()