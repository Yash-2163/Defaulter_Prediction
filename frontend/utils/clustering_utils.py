import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.animation import FuncAnimation
import tempfile
from matplotlib import cm

def run_kmeans_animation(df, feature1, feature2, k, frames=10):
    X = df[[feature1, feature2]].values
    
    # Initialize centroids randomly (like KMeans)
    rng = np.random.default_rng(42)
    centroids = X[rng.choice(len(X), size=k, replace=False)]
    
    centroid_history = [centroids]
    labels_history = []
    
    # Manual KMeans iteration for frames steps
    for _ in range(frames):
        # Assign labels based on closest centroid
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        labels_history.append(labels)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])
        centroid_history.append(new_centroids)
        centroids = new_centroids

    fig, ax = plt.subplots(figsize=(6,5))
    cmap = cm.get_cmap('tab10', k)
    
    def update(i):
        ax.clear()
        labels = labels_history[i]
        centroids = centroid_history[i+1]
        
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap, s=50, edgecolor='k', alpha=0.7)
        ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=180)
        
        # Plot centroid paths up to current frame
        for idx in range(k):
            x_path = [c[idx][0] for c in centroid_history[:i+2]]
            y_path = [c[idx][1] for c in centroid_history[:i+2]]
            ax.plot(x_path, y_path, '--', color='gray', alpha=0.6)
            ax.annotate(f"C{idx}", (centroids[idx, 0], centroids[idx, 1]), xytext=(0,10), textcoords='offset points', ha='center', fontsize=9)
        
        ax.set_title(f"K-Means Clustering Progress - Step {i+1}")
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.text(0.95, 0.05, f'Step {i+1}', horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes,
                fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

    ani = FuncAnimation(fig, update, frames=frames, repeat=False)
    temp_gif = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    ani.save(temp_gif.name, writer='pillow', fps=2)
    
    return temp_gif.name
