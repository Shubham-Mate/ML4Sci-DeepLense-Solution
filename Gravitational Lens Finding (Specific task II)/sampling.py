## Python File which contains sampling based methods to deal with class imbalance
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np

def downsample_dataset(dataset, target_class_counts):
    """
    Downsamples the majority class in the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to downsample.
        target_class_counts (dict): Dictionary with {class_label: count} specifying the target count for each class.

    Returns:
        torch.utils.data.Subset: Downsampled dataset.
    """
    class_indices = {label: [] for label in target_class_counts.keys()}
    
    # Collect indices for each class
    for idx, (_, label) in enumerate(dataset):
        label = int(label)
        if label in class_indices:
            class_indices[label].append(idx)

    
    # Downsample by selecting a subset
    downsampled_indices = []
    for label, indices in class_indices.items():
        downsampled_indices.extend(np.random.choice(indices, target_class_counts[label], replace=False))
    
    return Subset(dataset, downsampled_indices)


# This is basically upsampling by making sure that each batch has equal number of samples from each class
def get_upsampled_sampler(dataset):
    """
    Returns a sampler that oversamples the minority class.
    """
    labels = np.array([label for _, label in dataset])
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
