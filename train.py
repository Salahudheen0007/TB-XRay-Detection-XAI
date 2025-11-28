import torch
import torch.nn as nn
import torch.optim as optim
from models.densenet_model import get_densenet121
from utils import get_data_loaders

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model setup
model = get_densenet121().to(device)

# Data loaders
train_loader, val_loader, _ = get_data_loaders("splits")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Track best validation accuracy
best_val_acc = 0.0
save_path = "best_model.pth"

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/10], Train Loss: {avg_loss:.4f}")

    # --- Validation step ---
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    # --- Save best model ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss
        }, save_path)
        print(f"✅ Best model saved at epoch {epoch+1} with val_acc: {val_acc:.2f}%")

print("Training complete.")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
