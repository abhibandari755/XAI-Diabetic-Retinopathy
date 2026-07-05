import os
from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from models.backbone import DRModel
from utils.gradcam import GradCAM
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch

# Define DEVICE globally
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model (Assuming weights are trained via SimCLR + Pseudo-labeling)
model = DRModel(num_classes=5)
# model.load_state_dict(torch.load('best_model.pth')) 
weight_path = 'models/dr_model_final.pth'

if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    print("✅ Successfully loaded trained weights.")
else:
    print("⚠️ WARNING: No weights found. Model is using random initialization (expect incorrect results).")
model.eval()

target_layer = model.backbone.layer4[-1]
cam_engine = GradCAM(model, target_layer)

def process_and_predict(img_path):
    # 1. Image Preprocessing (Ensure normalization matches training)
    img = Image.open(img_path).convert('RGB')
    prep = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = prep(img).unsqueeze(0).to(DEVICE)
    
    # 2. Inference
    with torch.no_grad(): # Critical to prevent memory leaks and static results
        model.eval()      # Ensure Dropout/BatchNorm are in eval mode
        logits = model(tensor)
        
        # 3. Probability Extraction
        probs = torch.softmax(logits, dim=1)
        conf_score, pred_idx = torch.max(probs, dim=1)
    
    # Convert to standard Python types
    confidence = conf_score.item() * 100
    prediction = pred_idx.item()
    
    # 4. Generate Grad-CAM (Requires gradients, so we enable them briefly)
    model.train() # Grad-CAM needs to backpropagate through the model
    heatmap = cam_engine.generate(tensor, prediction)
    model.eval()

    # 5. Overlay Logic
    orig = np.array(img.resize((224, 224)))
    heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(orig, cv2.COLOR_RGB2BGR), 0.6, heatmap_img, 0.4, 0)
    
    res_path = img_path.replace('uploads', 'results')
    cv2.imwrite(res_path, overlay)
    
    return prediction, confidence, os.path.basename(res_path)

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            idx, confidence, res_name = process_and_predict(path)
            # In app.py dashboard route
            classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
            label_text = classes[idx] # This ensures it pulls the correct name
            return render_template('index.html', 
                                   label=classes[idx], 
                                   confidence=f"{confidence:.2f}%", 
                                   result=res_name)
    return render_template('index.html')

def generate_training_report(history):
    """
    history: A dictionary containing 'simclr_loss' and 'val_acc'
    """
    plt.figure(figsize=(10, 5))
    
    # Plot SimCLR Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['simclr_loss'], label='Contrastive Loss')
    plt.title('Stage 1: SimCLR Pre-training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Plot Fine-tuning Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Validation Accuracy', color='green')
    plt.title('Stage 2: Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('static/results/training_report.png')
    print("Report saved to static/results/training_report.png")

def save_classification_report(y_true, y_pred):
    target_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # Create Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix: DR Classification')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('static/results/confusion_matrix.png')
    
    return report
if __name__ == '__main__':
    app.run(port=5000, debug=True)