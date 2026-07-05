import os
import gc
import torch
import cv2
import numpy as np

from flask import Flask, render_template, request
from PIL import Image
from torchvision import transforms

from models.backbone import DRModel
from utils.gradcam import GradCAM

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Always use CPU on Render Free
DEVICE = torch.device("cpu")

# ----------------------------
# Load Model
# ----------------------------

model = DRModel(num_classes=5).to(DEVICE)

weight_path = "models/dr_model_final.pth"

if os.path.exists(weight_path):
    state_dict = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    del state_dict
    print("✅ Model Loaded Successfully")
else:
    print("⚠️ WARNING: Weight file not found")

model.eval()

target_layer = model.backbone.layer4[-1]
cam_engine = GradCAM(model, target_layer)

classes = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ----------------------------
# Prediction
# ----------------------------

def process_and_predict(img_path):

    img = Image.open(img_path).convert("RGB")

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, 1)

    pred = prediction.item()
    conf = confidence.item() * 100

    # Grad-CAM requires gradients
    with torch.enable_grad():
        model.train()
        heatmap = cam_engine.generate(tensor, pred)
        model.eval()

    original = np.array(img.resize((224, 224)))

    heatmap_img = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        cv2.cvtColor(original, cv2.COLOR_RGB2BGR),
        0.6,
        heatmap_img,
        0.4,
        0
    )

    result_name = os.path.basename(img_path)
    result_path = os.path.join(RESULT_FOLDER, result_name)

    cv2.imwrite(result_path, overlay)

    # Free memory
    del tensor
    del output
    del probs
    gc.collect()

    return pred, conf, result_name


# ----------------------------
# Flask Routes
# ----------------------------

@app.route("/", methods=["GET", "POST"])
def dashboard():

    if request.method == "POST":

        if "file" not in request.files:
            return render_template("index.html")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html")

        upload_path = os.path.join(UPLOAD_FOLDER, file.filename)

        file.save(upload_path)

        prediction, confidence, result = process_and_predict(upload_path)

        return render_template(
            "index.html",
            label=classes[prediction],
            confidence=f"{confidence:.2f}%",
            result=result
        )

    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)