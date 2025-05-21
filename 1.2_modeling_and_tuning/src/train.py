import torch
import torch.nn as nn
import torch.optim as optim
import os

def train_and_eval(
    model, train_loader, val_loader, lr, weight_decay, optimizer_type,
    epochs=10, weights_tensor=None, device=None, save_path=None
):
    # Gewichteter Loss
    if weights_tensor is not None:
        criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer definieren
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer")

    model.to(device)
    best_acc = 0
    val_loss_final = 0.0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += y_val.size(0)
                val_correct += (predicted == y_val).sum().item()

        val_acc = val_correct / val_total
        val_loss_final = val_loss / len(val_loader)

        if save_path is not None and val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Epoch: {epoch + 1} Modell gespeichert: {save_path} (Val-Acc: {val_acc:.4f})")

    return best_acc, val_acc, val_loss_final
