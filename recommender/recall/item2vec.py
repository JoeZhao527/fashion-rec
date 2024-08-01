import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
import torch
import seaborn as sns

import random
import gensim
assert gensim.models.word2vec.FAST_VERSION > -1

from gensim.models import Word2Vec
import math

from sklearn.preprocessing import normalize
import faiss
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")


def aggregate_recommendations(user_profiles, indices, recent_transactions, k=12):
    # Group transactions by customer_id and aggregate article counts
    grouped_transactions = recent_transactions.groupby('customer_id')['article_id'].agg(lambda x: x.value_counts().to_dict()).to_dict()
    
    user_id_list = user_profiles.index.tolist()
    recommendations = {}
    
    # Use tqdm to show the progress bar
    for i in tqdm(range(len(user_id_list)), desc="Aggregating Recommendations"):
        user_id = user_id_list[i]
        article_counts = {}
        for idx in indices[i]:
            neighbor_id = user_id_list[idx]
            if neighbor_id in grouped_transactions:
                for article_id, count in grouped_transactions[neighbor_id].items():
                    article_counts[article_id] = article_counts.get(article_id, 0) + count
        
        # Sort articles by their aggregated counts and select the top_k articles
        sorted_items = sorted(article_counts.items(), key=lambda item: item[1], reverse=True)
        recommendations[user_id] = [item[0] for item in sorted_items[:k]]

    return recommendations

def calculate_user_profiles(user_items: pd.Series, item_vectors: Dict[str, np.ndarray], vector_size: int) -> pd.Series:
    """
    Calculate user profiles based on item embeddings.

    Parameters:
    user_items (pd.Series): Series where index is user_id and value is list of article_ids.
    item_vectors (Dict[str, np.ndarray]): Dictionary of item embeddings.
    vector_size (int): Size of the embedding vectors.

    Returns:
    pd.Series: Series where index is user_id and value is the average embedding vector.
    """
    def calculate_user_profile(item_ids):
        vectors = []
        for item_id in item_ids:
            if item_id in item_vectors:
                vectors.append(item_vectors[item_id])
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(vector_size)
    
    user_profiles = user_items.apply(calculate_user_profile)
    return user_profiles

def calculate_item_similarities(user_profiles: pd.Series, item_vectors: list, item_ids: List[str], top_N: int, device) -> Dict[str, List[str]]:
    """
    Calculate item similarities and get top recommendations for each user.

    Parameters:
    user_profiles (pd.Series): Series where index is user_id and value is the average embedding vector.
    item_vectors (np.ndarray): item vectors.
    item_ids (List[str]): List of item IDs corresponding to the item vectors.
    top_N (int): Number of top recommendations to return for each user.
    device (str): Device to use for computation ('cpu' or 'cuda').

    Returns:
    Dict[str, List[str]]: Dictionary where key is user_id and value is list of recommended article_ids.
    """
    if str(device) == "cpu":
        # Using NumPy instead of PyTorch tensors
        norms = []
        item_vectors = np.stack(item_vectors)
        
        # Loop through each row in the tensor
        for i in tqdm(range(item_vectors.shape[0]), desc="item2vec similarity computation (cpu)"):
            sum_of_squares = 0
            # Loop through each element in the row
            for j in range(item_vectors.shape[1]):
                sum_of_squares += item_vectors[i, j] ** 2
            # Compute the norm (square root of the sum of squares)
            norm = math.sqrt(sum_of_squares)
            norms.append(norm)

        # Convert the list of norms to a NumPy array and reshape for broadcasting
        norms_arr = np.array(norms).reshape(-1, 1)

        # Normalize item vectors
        item_vectors_norm = item_vectors / norms_arr
        
        # Convert user profiles to a NumPy array
        user_ids = list(user_profiles.keys())
        user_vectors = np.vstack(user_profiles.values)  # Ensure user_profiles is converted to 2D array

        # Normalize user vectors
        user_norms = np.linalg.norm(user_vectors, axis=1).reshape(-1, 1)
        user_vectors_norm = user_vectors / user_norms

        # Compute cosine similarity
        similarities = np.dot(user_vectors_norm, item_vectors_norm.T)

        # Get top 12 recommendations for each user
        top_indices = np.argsort(-similarities, axis=1)[:, :top_N]
        
        # Map indices to item IDs
        user_recommendations = {
            user_ids[i]: [item_ids[idx] for idx in top_indices[i]]
            for i in range(len(user_ids))
        }
    else:
        item_vectors = torch.tensor(item_vectors, dtype=torch.float, device=device)
        item_vectors_norm = item_vectors / item_vectors.norm(dim=1, keepdim=True)

        # Convert user profiles to a tensor
        user_ids = list(user_profiles.keys())
        user_vectors = torch.tensor(list(user_profiles.values), dtype=torch.float, device=device)

        # Define batch size
        batch_size = 32

        # Function to process batches and get recommendations
        def process_batch(start_idx, end_idx):
            # Slice the batch
            batch_user_vectors = user_vectors[start_idx:end_idx]
            batch_user_vectors_norm = batch_user_vectors / batch_user_vectors.norm(dim=1, keepdim=True)
            
            # Compute cosine similarity
            similarities = torch.mm(batch_user_vectors_norm, item_vectors_norm.t())
            
            # Get top N recommendations for each user in the batch
            top_indices = torch.topk(similarities, top_N, dim=1).indices
            
            # Map indices to item IDs
            return {user_ids[i]: [item_ids[idx] for idx in top_indices[row_index].cpu().tolist()]
                    for row_index, i in enumerate(range(start_idx, end_idx))}

        # Process all batches and collect recommendations
        user_recommendations = {}
        for start_idx in tqdm(range(0, len(user_vectors), batch_size), desc="item2vec similarity computation (cuda)"):
            end_idx = min(start_idx + batch_size, len(user_vectors))
            user_recommendations.update(process_batch(start_idx, end_idx))

    return user_recommendations


def create_faiss_index(user_profiles, use_gpu=False, k=200):
    user_profiles_array = np.stack(user_profiles.values).astype('float32')
    user_profiles_array = normalize(user_profiles_array)
    
    # Create the index based on L2 distance
    index = faiss.IndexFlatL2(user_profiles_array.shape[1])

    if use_gpu:
        # Transfer the index to GPU
        gpu_resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

        # Add the user profiles to the index
        index.add(user_profiles_array)

        # Perform the search for all profiles at once
        distances, indices = index.search(user_profiles_array, k)
    else:
        # Add the user profiles to the index one by one
        for profile in user_profiles_array:
            index.add(profile.reshape(1, -1))

        # Perform the search one by one
        distances = []
        indices = []
        for profile in user_profiles_array:
            D, I = index.search(profile.reshape(1, -1), k)
            distances.append(D)
            indices.append(I)

        # Convert the results to numpy arrays
        distances = np.vstack(distances)
        indices = np.vstack(indices)
    
    return distances, indices


class Item2VecModel:
    def __init__(self, train: pd.DataFrame, articles: pd.DataFrame, top_N: int, *args, **kwargs):
        positive_samples = train.groupby('customer_id')['article_id'].agg(list).reset_index()
        all_articles = set(articles['article_id'].astype(str))

        # Ensure 't_dat' is a datetime object
        train['t_dat'] = pd.to_datetime(train['t_dat'])

        # Compute the current date and last week's start date
        current_date = train['t_dat'].max()
        last_week_start = current_date - timedelta(days=14)
        recent_transactions = train[(train['t_dat'] > last_week_start)]

        training_data = []
        for _, row in tqdm(positive_samples.iterrows(), total=len(positive_samples), desc="item2vec data prepare"):
            training_data.append(row['article_id'])

        for purchase in training_data:
            random.shuffle(purchase)
            
        model = Word2Vec(sentences=training_data,
                        epochs=10,
                        min_count=10,
                        vector_size=128,
                        workers=6,
                        sg=1,
                        hs=0,
                        negative=5,
                        window=9999)

        item_vectors = {item: model.wv[item] for item in model.wv.index_to_key}
        vector_size = model.vector_size

        user_items = train.groupby('customer_id')['article_id'].apply(list)
        user_profiles = calculate_user_profiles(user_items, item_vectors, vector_size)
        
        item_list = list(model.wv.index_to_key)
        item_vectors = [item_vectors[item] for item in item_list]

        self.user_profiles = user_profiles
        self.item_vectors = item_vectors
        self.item_list = item_list
        self.top_N = top_N
        self.recent_transactions = recent_transactions

    def similarity_recall(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        user_recommendations = calculate_item_similarities(
            self.user_profiles,
            self.item_vectors,
            self.item_list,
            self.top_N,
            device
        )

        return user_recommendations
    
    def cluster_recall(self):
        use_gpu = torch.cuda.is_available()
        distances, indices = create_faiss_index(self.user_profiles, use_gpu)
        popularity_recommendations = aggregate_recommendations(self.user_profiles, indices, self.recent_transactions, k=self.top_N)
        
        return popularity_recommendations