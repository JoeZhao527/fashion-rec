from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def cluster_items(items, n_clusters=2000):
    # Step 1: Extract the arrays from the dictionary
    item_ids = list(items.keys())
    item_vectors = list(items.values())

    # Step 2: Stack the arrays to create a 2D NumPy array
    X = np.stack(item_vectors)

    # Step 3: Apply the K-means clustering algorithm
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, verbose=1)
    kmeans.fit(X)

    # Step 4: Assign each item to a cluster
    labels = kmeans.labels_

    # Create a dictionary to map item IDs to their respective clusters
    clustered_items = {item_id: labels[idx] for idx, item_id in enumerate(item_ids)}

    return clustered_items

if __name__ == '__main__':
    feat_dir = "./output"
    image_feature = {}
    missing = 0

    for item_id in tqdm(os.listdir(feat_dir)):
        feature_file = os.path.join(feat_dir, f"{item_id}")

        # 0 for class token, 1 for average of all patch tokens
        image_feature[item_id] = np.load(feature_file)[0]

    clustered_items = cluster_items(image_feature, n_clusters=2000)
    clustered_items = pd.DataFrame(clustered_items, index=[0]).transpose().reset_index()
    clustered_items.to_csv("./img_cluster_2000.csv", index=False)