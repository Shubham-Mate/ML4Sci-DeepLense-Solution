import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch.nn.functional as F
from save_and_load import save_model

def train_model(model, train_loader, val_loader, optimizer, criterion, device, save_path, epochs=10, patience=5):
    """
    Train the CNN model with early stopping based on validation AUC.
    """
    best_val_auc = 0  # Track best validation AUC
    epochs_no_improve = 0  # Track epochs since last improvement
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc, val_auc = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch+1}: Loss = {total_loss}, Train Acc = {train_acc}, Val Acc = {val_acc}, AUC = {val_auc}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_model(model, save_path)  # Save the best model
            epochs_no_improve = 0  # Reset patience counter
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break


def evaluate_model(model, val_loader, device):
    """
    Evaluate the model on validation/test dataset.

    Args:
        model (nn.Module): CNN model.
        val_loader (DataLoader): DataLoader for validation/test set.

    Returns:
        accuracy (float): Validation accuracy.
        auc_score (float): Area Under the ROC Curve.
    """
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():  # No gradient computation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Forward pass
            preds = outputs.argmax(dim=1)  # Predicted class
            probs = F.softmax(outputs, dim=1)[:, 1]  # Probability of Lens class
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / total
    auc_score = roc_auc_score(all_labels, all_probs)

    return accuracy, auc_score