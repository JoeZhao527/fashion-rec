from tqdm import tqdm
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
    # Load text embeddings
    text_emb = np.load("./glove_text_emb.npy", allow_pickle=True).item()
    
    # Perform clustering on text embeddings
    clustered_items = cluster_items(text_emb, n_clusters=2000)
    
    # Convert clustered items to DataFrame and save to CSV with the expected format
    clustered_items_df = pd.DataFrame(list(clustered_items.items()), columns=['article_id', 'cluster'])
    clustered_items_df.to_csv("./text_cluster_2000.csv", index=False)
