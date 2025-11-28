# evaluate_full.py
import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from models.densenet_model import get_densenet121
from utils import get_data_loaders  # assumes this returns (train_loader, val_loader, test_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# create model instance (must match training architecture)
model = get_densenet121()
model = model.to(device)

# Load checkpoint (handles dict with 'model_state_dict' or direct state_dict)
ckpt_path = "best_model.pth"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"{ckpt_path} not found")

ckpt = torch.load(ckpt_path, map_location=device)
if isinstance(ckpt, dict) and ('model_state_dict' in ckpt or 'state_dict' in ckpt):
    state = ckpt.get('model_state_dict', ckpt.get('state_dict'))
else:
    state = ckpt
model.load_state_dict(state)
model.eval()
print("Checkpoint loaded.")

# load data
_, _, test_loader = get_data_loaders("splits")  # change path if needed

all_probs = []
all_preds = []
all_labels = []

criterion = torch.nn.CrossEntropyLoss()
test_loss = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)

        probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # prob of class 1 (TB)
        preds = outputs.argmax(1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

n_samples = len(all_labels)
if n_samples == 0:
    raise RuntimeError("No test samples found. Check splits/test has images in class subfolders.")

test_loss = test_loss / n_samples
print(f"Test loss: {test_loss:.4f}")

# classification report
# Determine class names from test_loader.dataset if available
try:
    classes = test_loader.dataset.classes
except Exception:
    classes = [str(i) for i in range(max(all_labels)+1)]

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion matrix:\n", cm)

# save confusion matrix figure
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")

# ROC AUC (optional)
try:
    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC = {auc:.4f})")
    plt.grid(True)
    plt.savefig("roc_curve.png")
    print(f"Saved roc_curve.png (AUC={auc:.4f})")
except Exception as e:
    print("Could not compute ROC AUC:", e)

print("Evaluation done.")
