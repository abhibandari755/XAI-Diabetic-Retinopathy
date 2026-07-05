import os
import numpy as np
import cv2

def create_dummy_data():
    paths = ['data/unlabeled', 'data/labeled']
    for path in paths:
        os.makedirs(path, exist_ok=True)
        # Create 5 dummy images per folder
        for i in range(5):
            # Generate a random "retina-like" image (noise)
            img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            # Draw a circle to mimic a fundus
            cv2.circle(img, (256, 256), 250, (255, 255, 255), -1) 
            img_name = f"dummy_{i}.jpg"
            cv2.imwrite(os.path.join(path, img_name), img)
    
    print("✅ Dummy data created in data/unlabeled and data/labeled")

if __name__ == "__main__":
    create_dummy_data()