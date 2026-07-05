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
EPOCHS_SIMCLR = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nt_xent_loss(z_i, z_j, temperature=0.5):
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    N = z_i.shape[0]
    mask = ~torch.eye(2*N, device=DEVICE).bool()
    pos_sim = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)])
    loss = -torch.log(torch.exp(pos_sim) / (torch.exp(sim) * mask).sum(dim=1))
    return loss.mean()

def run_pipeline():
    # 1. Model & Optimizer
    model = DRModel(num_classes=5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 2. Setup DataLoaders (The fix for your NameError)
    # SimCLR needs the double-view augmentation
    simclr_transform = SimCLRAugmentation(size=224)
    
    # Ensure these directories exist or point to your actual data folders
    unlabeled_path = 'data/unlabeled'
    labeled_path = 'data/labeled'
    os.makedirs(unlabeled_path, exist_ok=True)
    os.makedirs(labeled_path, exist_ok=True)

    unlabeled_ds = DiabeticRetinopathyDataset(
        root_dir=unlabeled_path, 
        transform=simclr_transform, 
        mode='unlabeled'
    )
    
    # This defines the variable that was missing
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=BATCH_SIZE, shuffle=True)

    # --- STAGE 1: SimCLR Pre-training ---
    print(">>> Starting Stage 1: SimCLR Pre-training...")
    model.train()
    for epoch in range(EPOCHS_SIMCLR):
        total_loss = 0
        for batch_idx, ((view1, view2), _) in enumerate(unlabeled_loader):
            z1 = model(view1.to(DEVICE), mode='contrastive')
            z2 = model(view2.to(DEVICE), mode='contrastive')
            
            loss = nt_xent_loss(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"SimCLR Epoch [{epoch+1}/{EPOCHS_SIMCLR}] - Loss: {total_loss/len(unlabeled_loader):.4f}")

    # --- STAGE 2: Pseudo-labeling / Fine-tuning ---
    # (After SimCLR, you'd typically switch to the labeled_loader here)
    print(">>> SimCLR Complete. Saving Backbone...")
    torch.save(model.state_dict(), 'models/dr_model_final.pth')

if __name__ == "__main__":
    run_pipeline()