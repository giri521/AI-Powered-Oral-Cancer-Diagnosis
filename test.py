import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from flask import Flask, render_template

# -------------------- Config --------------------
MODEL_DIR = "model"  # path to your saved model weights
TEST_DIR = r"C:\Users\giriv\Desktop\working Projects\oral-cancer-detector\test"  # path to test dataset
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # Cancer / Normal
CLASSES = ["Cancer", "Normal"]

# -------------------- Data Loader --------------------
transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform_eval)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------- Load Model --------------------
def load_model(model_name="EfficientNetV2", img_type="clinical"):
    if model_name == "EfficientNetV2":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
        weight_path = os.path.join(MODEL_DIR, f"efficientnet_{img_type}.pt")
    else:
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
        weight_path = os.path.join(MODEL_DIR, f"convnext_{img_type}.pt")
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weights not found at {weight_path}")
    
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -------------------- Evaluate --------------------
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# -------------------- Metrics --------------------
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")
precision = precision_score(all_labels, all_preds, average="weighted")
recall = recall_score(all_labels, all_preds, average="weighted")
class_report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
conf_mat = confusion_matrix(all_labels, all_preds)

# -------------------- Confusion Matrix Plot --------------------
plt.figure(figsize=(5, 4))
sns.heatmap(conf_mat, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")

buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)
conf_img_base64 = base64.b64encode(buf.read()).decode("utf-8")
buf.close()
plt.close()

# -------------------- Flask App --------------------
app = Flask(__name__)

@app.route("/")
def result():
    return render_template("result.html",
                           accuracy=accuracy,
                           f1=f1,
                           precision=precision,
                           recall=recall,
                           class_report=class_report,
                           conf_img=conf_img_base64)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
