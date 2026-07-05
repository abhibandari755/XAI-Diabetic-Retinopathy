import os
from PIL import Image
from torch.utils.data import Dataset

class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='labeled', labels_dict=None):
        """
        Args:
            root_dir: Directory with all the images.
            mode: 'labeled' (returns img, label) or 'unlabeled' (returns img, placeholder)
            labels_dict: Dictionary mapping filename to DR grade (0-4)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.image_files = os.listdir(root_dir)
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.mode == 'labeled' and self.labels_dict:
            label = self.labels_dict.get(img_name, 0)
            return image, label
        
        # For SimCLR/Unlabeled, we just return the image
        return image, -1