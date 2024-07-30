import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from typing import List, Dict
import torch
from collections import Counter

from matplotlib import pyplot as plt
import seaborn as sns

import random
import gensim
assert gensim.models.word2vec.FAST_VERSION > -1

from gensim.models import Word2Vec
import time
import math

from scipy.sparse import coo_matrix
import implicit
import warnings
warnings.filterwarnings("ignore")


def popularity_recall(train: pd.DataFrame, *args, **kwargs):
    """
    Recall the most popular items in the training period
    """
    counts = train['article_id'].value_counts()
    purchase_count = pd.DataFrame(counts).reset_index()

    return purchase_count


def item2vec_recall(train: pd.DataFrame, articles: pd.DataFrame, top_N: int, *args, **kwargs):
    positive_samples = train.groupby('customer_id')['article_id'].agg(list).reset_index()
    all_articles = set(articles['article_id'].astype(str))

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item_list = list(all_articles)
    item_vectors_norm = np.stack([item_vectors[item] for item in item_list])
    norms = np.linalg.norm(item_vectors_norm, axis=1, keepdims=True)
    item_vectors_norm = item_vectors_norm / norms

    user_recommendations = calculate_item_similarities(user_profiles, item_vectors_norm, item_list, top_N, device)

    return user_recommendations

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

def calculate_item_similarities(user_profiles: pd.Series, item_vectors_norm: np.ndarray, item_ids: List[str], top_N: int, device: str) -> Dict[str, List[str]]:
    """
    Calculate item similarities and get top recommendations for each user.

    Parameters:
    user_profiles (pd.Series): Series where index is user_id and value is the average embedding vector.
    item_vectors_norm (np.ndarray): Normalized item vectors.
    item_ids (List[str]): List of item IDs corresponding to the item vectors.
    top_N (int): Number of top recommendations to return for each user.
    device (str): Device to use for computation ('cpu' or 'cuda').

    Returns:
    Dict[str, List[str]]: Dictionary where key is user_id and value is list of recommended article_ids.
    """
    if device == "cpu":
        user_ids = user_profiles.index.tolist()
        user_vectors = np.vstack(user_profiles.values)

        user_norms = np.linalg.norm(user_vectors, axis=1, keepdims=True)
        user_vectors_norm = user_vectors / user_norms

        similarities = np.dot(user_vectors_norm, item_vectors_norm.T)

        top_indices = np.argsort(-similarities, axis=1)[:, :top_N]

        user_recommendations = {
            user_ids[i]: [item_ids[idx] for idx in top_indices[i]]
            for i in range(len(user_ids))
        }
    else:
        item_vectors_norm = torch.tensor(item_vectors_norm, dtype=torch.float, device=device)

        user_ids = user_profiles.index.tolist()
        user_vectors = torch.tensor(list(user_profiles.values), dtype=torch.float, device=device)

        batch_size = 32

        def process_batch(start_idx, end_idx):
            batch_user_vectors = user_vectors[start_idx:end_idx]
            batch_user_vectors_norm = batch_user_vectors / batch_user_vectors.norm(dim=1, keepdim=True)
            
            similarities = torch.mm(batch_user_vectors_norm, item_vectors_norm.t())
            top_indices = torch.topk(similarities, top_N, dim=1).indices
            
            return {user_ids[i]: [item_ids[idx] for idx in top_indices[row_index].cpu().tolist()]
                    for row_index, i in enumerate(range(start_idx, end_idx))}

        user_recommendations = {}
        for start_idx in tqdm(range(0, len(user_vectors), batch_size)):
            end_idx = min(start_idx + batch_size, len(user_vectors))
            user_recommendations.update(process_batch(start_idx, end_idx))

    return user_recommendations


def user_collaborative_recall(train: pd.DataFrame, top_N: int, *args, **kwargs):
    user_item_data = train[['customer_id', 'article_id']]

    # Create a mapping for users and items to integer indices
    user_mapping = {user_id: idx for idx, user_id in enumerate(user_item_data['customer_id'].unique())}
    item_mapping = {item_id: idx for idx, item_id in enumerate(user_item_data['article_id'].unique())}

    # Convert user_ids and item_ids to integer indices
    user_item_data['user_idx'] = user_item_data['customer_id'].map(user_mapping)
    user_item_data['item_idx'] = user_item_data['article_id'].map(item_mapping)

    # Create a sparse matrix in COO format
    sparse_matrix = coo_matrix(([1 for _ in range(len(user_item_data))], 
                                (user_item_data['user_idx'], user_item_data['item_idx'])))

    # Convert the sparse matrix to CSR format
    sparse_matrix_csr = sparse_matrix.tocsr()

    # Initialize the ALS model
    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)

    # Train the model on the sparse matrix
    model.fit(sparse_matrix_csr)

    ids, score = model.recommend([i for i in range(sparse_matrix_csr.shape[0])], sparse_matrix_csr, N=top_N)

    item_inverse_map = {v: k for k, v in item_mapping.items()}
    func = lambda x: item_inverse_map[x]

    # np.moveaxis(np.vectorize(func)(ids), 0, 0)
    recommendations = pd.Series(
        index=list(user_mapping.keys()),
        data=list(np.vectorize(func)(ids))
    )

    print(f"Recommendation shape: {ids.shape, score.shape}")

    return recommendations


def product_code_recall(train: pd.DataFrame, articles: pd.DataFrame, purchase_count: pd.DataFrame, *args, **kwargs):
    prod_code_trn = pd.merge(train, articles[['article_id', 'product_code']])

    prod_code_group = articles[['article_id', 'product_code']].groupby("product_code")

    recent_purchase = []

    for cid, group in tqdm(prod_code_trn.groupby("customer_id")):
        prod_idx = list(group['product_code'].unique())
        recent_purchase.extend([(cid, aid) for aid in prod_idx])

    select_prod = pd.merge(
        pd.DataFrame(recent_purchase, columns=["customer_id", "product_code"]),
        articles[['article_id', 'product_code']],
        on='product_code',
    )

    sorted_prod = pd.merge(purchase_count, select_prod, on=['article_id'])

    sorted_prod_res = {_id: _df for _id, _df in sorted_prod.groupby("customer_id")}

    return sorted_prod_res


def postal_code_recall(train: pd.DataFrame, customers: pd.DataFrame, purchase_count: pd.DataFrame, *args, **kwargs):
    trn_postal = pd.merge(train, customers[['customer_id', 'postal_code']], on=['customer_id'])

    pop_trn_postal = pd.merge(trn_postal, purchase_count, on=['article_id'])
    trn_postal_group = {}

    for postal_code, group in pop_trn_postal.groupby("postal_code"):
        trn_postal_group[postal_code] = group.sort_values("count", ascending=False)

    customers_postal_code_map = pd.Series(data=list(customers['postal_code']), index=customers['customer_id']).to_dict()

    return trn_postal_group, customers_postal_code_map


def image_cluster_recall(train: pd.DataFrame, img_group: pd.DataFrame, purchase_count, *args, **kwargs):
    item_to_cluster = dict(img_group.values)
    cluster_to_item = {}

    for cluster, group in tqdm(img_group.groupby("cluster")):
        cluster_to_item[cluster] = pd.merge(group, purchase_count, how='left').sort_values("count", ascending=False)
    
    img_group_trn = pd.merge(train, img_group, on=['article_id'])

    img_recent_purchase = []

    for cid, group in tqdm(img_group_trn.groupby("customer_id")):
        prod_idx = list(group['cluster'].unique())
        img_recent_purchase.extend([(cid, aid) for aid in prod_idx])

    img_select_prod = pd.merge(
        pd.DataFrame(img_recent_purchase, columns=["customer_id", "cluster"]),
        img_group[['article_id', 'cluster']],
        on='cluster',
    )

    img_sorted_prod = pd.merge(purchase_count, img_select_prod, on=['article_id'])

    img_sorted_prod_res = {_id: _df for _id, _df in tqdm(img_sorted_prod.groupby("customer_id"))}

    return img_sorted_prod_res


class ArticlePairs:
    def __init__(self, df):
        self.article_pairs_matrix = None
        self.article_to_idx = {}
        self.idx_to_article = {}
        self._compute_pairs_matrix(df)
        self.top_n_cache = {}

    def _compute_pairs_matrix(self, df):
        # Create a mapping for articles to integer indices
        unique_articles = df['article_id'].unique()
        self.article_to_idx = {article_id: idx for idx, article_id in enumerate(unique_articles)}
        self.idx_to_article = {idx: article_id for article_id, idx in self.article_to_idx.items()}
        
        num_articles = len(unique_articles)
        
        # Initialize the matrix
        self.article_pairs_matrix = np.zeros((num_articles, num_articles), dtype=int)
        
        # Group by customer_id and iterate through each group
        grouped = df.groupby('customer_id')
        for _, group in grouped:
            articles_bought = group['article_id'].tolist()
            for i in range(len(articles_bought)):
                for j in range(i + 1, len(articles_bought)):
                    idx1 = self.article_to_idx[articles_bought[i]]
                    idx2 = self.article_to_idx[articles_bought[j]]
                    self.article_pairs_matrix[idx1, idx2] += 1
                    self.article_pairs_matrix[idx2, idx1] += 1

    def get_top_n_bought_together(self, article_id, n=50):
        if article_id not in self.article_to_idx:
            raise ValueError(f"Article ID {article_id} not found in the data.")
        
        if article_id not in self.top_n_cache:
            idx = self.article_to_idx[article_id]
            bought_together_counts = self.article_pairs_matrix[idx]
            
            # Use argpartition to get the indices of the top N bought together articles
            top_n_indices = np.argpartition(bought_together_counts, -n)[-n:]
            
            # Get indices of top N bought together articles
            # top_n_indices = np.argsort(bought_together_counts)[::-1][:n]
            top_n_counts = bought_together_counts[top_n_indices]
            
            # Create a dataframe for the result
            result_df = pd.DataFrame({
                'article_id': [self.idx_to_article[i] for i in top_n_indices],
                'num_together': top_n_counts
            })

            self.top_n_cache[article_id] = result_df
        
        return self.top_n_cache[article_id].copy()


def bought_together_recall(train: pd.DataFrame):
    ap = ArticlePairs(train)

    also_bought_res = {}

    for cid, group in tqdm(train.groupby("customer_id")):
        ap_res = []
        for aid in list(group['article_id'].unique()):
            ap_res.append(ap.get_top_n_bought_together(aid, 50))
        also_bought_res[cid] = pd.concat(ap_res)

    return also_bought_res