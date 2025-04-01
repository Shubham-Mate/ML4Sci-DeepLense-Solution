import torch

def save_model(model, path):
    """Save the model to the given path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """Load the model from the given path."""
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")