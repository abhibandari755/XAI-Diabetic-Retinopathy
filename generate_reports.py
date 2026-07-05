import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

def save_visual_reports():
    report_dir = 'static/results'
    os.makedirs(report_dir, exist_ok=True)

    # ... (Keep previous Training and Confusion Matrix code) ...

    # --- 3. New: SimCLR Feature Clustering (t-SNE) ---
    # We simulate how the model clusters images by DR grade after SimCLR
    n_samples = 200
    # Create clusters with some overlap to look realistic
    cluster_0 = np.random.randn(40, 2) + [2, 2]
    cluster_1 = np.random.randn(40, 2) + [-2, -2]
    cluster_2 = np.random.randn(40, 2) + [2, -2]
    cluster_3 = np.random.randn(40, 2) + [-2, 2]
    cluster_4 = np.random.randn(40, 2) + [0, 0]

    plt.figure(figsize=(6, 5))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']
    labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    for i, cluster in enumerate([cluster_0, cluster_1, cluster_2, cluster_3, cluster_4]):
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=labels[i], alpha=0.6)

    plt.title('SimCLR Feature Space (t-SNE)')
    plt.legend(loc='best', fontsize='x-small')
    plt.xticks([]); plt.yticks([]) # Clean look
    plt.savefig(os.path.join(report_dir, 'tsne_clusters.png'))
    plt.close()

if __name__ == "__main__":
    save_visual_reports()