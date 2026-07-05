import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.backbone import DRModel
from utils.dataset import DiabeticRetinopathyDataset
from utils.augment import SimCLRAugmentation
from torchvision import transforms
import torch.nn.functional as F
import os

# --- Configuration ---
BATCH_SIZE = 16
LR = 1e-4
EPOCHS_SIMCLR = 10  # Set higher for better results
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss for SimCLR"""
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    N = z_i.shape[0]
    mask = ~torch.eye(2*N, device=DEVICE).bool()
    pos_sim = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)])
    loss = -torch.log(torch.exp(pos_sim) / (torch.exp(sim) * mask).sum(dim=1))
    return loss.mean()

def run_pipeline():
    print(f"Using device: {DEVICE}")
    
    # 1. Initialize Model & Optimizer
    model = DRModel(num_classes=5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 2. Setup DataLoaders (FIX: Defined before use)
    simclr_transform = SimCLRAugmentation(size=224)
    
    unlabeled_path = 'data/unlabeled'
    os.makedirs(unlabeled_path, exist_ok=True)

    # Check if directory is empty to avoid further errors
    if not os.listdir(unlabeled_path):
        print(f"Error: No images found in {unlabeled_path}. Please add retinal images.")
        return

    unlabeled_ds = DiabeticRetinopathyDataset(
        root_dir=unlabeled_path, 
        transform=simclr_transform, 
        mode='unlabeled'
    )
    
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=BATCH_SIZE, shuffle=True)

    # --- STAGE 1: SimCLR Pre-training ---
    print(">>> Starting Stage 1: SimCLR Pre-training...")
    model.train()
    for epoch in range(EPOCHS_SIMCLR):
        total_loss = 0
        for batch_idx, (views, _) in enumerate(unlabeled_loader):
            # views contains (view1, view2) from SimCLRAugmentation
            view1, view2 = views[0].to(DEVICE), views[1].to(DEVICE)
            
            z1 = model(view1, mode='contrastive')
            z2 = model(view2, mode='contrastive')
            
            loss = nt_xent_loss(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"SimCLR Epoch [{epoch+1}/{EPOCHS_SIMCLR}] - Avg Loss: {total_loss/len(unlabeled_loader):.4f}")

    # --- STAGE 2: Saving the Backbone ---
    # After pre-training, we save the weights for the Flask App to use
    print(">>> Stage 1 Complete. Saving pre-trained model...")
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/dr_model_final.pth')
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    run_pipeline()