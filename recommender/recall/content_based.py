import pandas as pd
import numpy as np
import faiss
import os
from tqdm import tqdm


def items_compute_top_n_similarities(df, top_n=500, distance: str = "euclidean"):
    article_ids = df['article_id'].values
    features = np.vstack(df['feature'].values).astype(np.float32)
    
    if distance == "euclidean":
        # Build the L2 index
        index = faiss.IndexFlatL2(features.shape[1])
        index.add(features)
        
        # Perform the search
        distances, indices = index.search(features, top_n + 1)
        
        # Transform distance to similarity (smaller distance = more similar)
        similarity_transform = lambda x: 1 / (1 + x)
    
    elif distance == "cosine":
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(features)
        index = faiss.IndexFlatIP(features.shape[1])
        index.add(features)
        
        # Perform the search
        distances, indices = index.search(features, top_n + 1)
        
        # Cosine similarity directly obtained (larger score = more similar)
        similarity_transform = lambda x: x
    else:
        raise Exception("Unsupported similarity function")
    
    results = {}
    
    for i in range(len(article_ids)):
        target_id = article_ids[i]
        similar_ids = article_ids[indices[i, 1:top_n + 1]].tolist()  # Skip the item itself
        similarity_scores = similarity_transform(distances[i, 1:top_n + 1]).tolist()
        results[target_id] = {
            'similar_items': np.array(similar_ids),
            'similarity_scores': np.array(similarity_scores)
        }
    
    return results


class ContentBased:
    def __init__(
        self,
        image_feature_path: str,
        text_feature_path: str,
        feature_cache_dir: str,
        top_k: int = 300
    ):
        """
        image_feature_path (str): pre-computed Dino-v2 image feature embedding for each item
        text_feature_path (str): pre-computed Glove text feature embedding for each item
        feature_cache_dir (str): directory to store computed similarities
        top_k (int): store top_k closest item for each item
        """
        image_cb_euclidean_path = os.path.join(feature_cache_dir, "image_cb_euclidean.npy")
        text_cb_euclidean_path = os.path.join(feature_cache_dir, "text_cb_euclidean.npy")

        image_cb_cosine_path = os.path.join(feature_cache_dir, "image_cb_cosine.npy")
        text_cb_cosine_path = os.path.join(feature_cache_dir, "text_cb_cosine.npy")

        txt_feat = pd.DataFrame(np.load(text_feature_path, allow_pickle=True).item().items(), columns=['article_id', 'feature'])
        img_feat = pd.DataFrame(np.load(image_feature_path, allow_pickle=True).item().items(), columns=['article_id', 'feature'])

        # Computing top_k similar items for each item with euclidean distance, according to image feature vector
        if not os.path.exists(image_cb_euclidean_path):
            print(f"image content based (euclidean) computing...")
            image_cb_euclidean = items_compute_top_n_similarities(img_feat, top_n=top_k, distance="euclidean")
            np.save(image_cb_euclidean_path, image_cb_euclidean)

        image_cb_euclidean = np.load(image_cb_euclidean_path, allow_pickle=True).item()

        # Computing top_k similar items for each item with cosine similarity, according to image feature vector
        if not os.path.exists(image_cb_cosine_path):
            print(f"image content based (cosine) computing...")
            image_cb_cosine = items_compute_top_n_similarities(img_feat, top_n=top_k, distance="cosine")
            np.save(image_cb_cosine_path, image_cb_cosine)

        image_cb_cosine = np.load(image_cb_cosine_path, allow_pickle=True).item()

        # Computing top_k similar items for each item with euclidean distance, according to text feature vector
        if not os.path.exists(text_cb_euclidean_path):
            print(f"text content based (euclidean) computing...")
            text_cb_euclidean = items_compute_top_n_similarities(txt_feat, top_n=top_k, distance="euclidean")
            np.save(text_cb_euclidean_path, text_cb_euclidean)

        text_cb_euclidean = np.load(text_cb_euclidean_path, allow_pickle=True).item()

        # Computing top_k similar items for each item with cosine similarity, according to text feature vector
        if not os.path.exists(text_cb_cosine_path):
            print(f"text content based (cosine) computing...")
            text_cb_cosine = items_compute_top_n_similarities(txt_feat, top_n=top_k, distance="cosine")
            np.save(text_cb_cosine_path, text_cb_cosine)

        text_cb_cosine = np.load(text_cb_cosine_path, allow_pickle=True).item()

        self.img_feat = img_feat
        self.txt_feat = txt_feat
        self.similarity_dict = {
            ("image", "euclidean"): image_cb_euclidean,
            ("image", "cosine"): image_cb_cosine,
            ("text", "euclidean"): text_cb_euclidean,
            ("text", "cosine"): text_cb_cosine
        }

    def filter_content(self, train: pd.DataFrame, articles: pd.DataFrame):
        """
        Some items in the dataset does not have image. Given a training set, remove all the transcations
        where image is missing
        """
        items_has_content = set(self.img_feat['article_id']).intersection(set(self.txt_feat['article_id']))

        missing_content_articles = set(articles['article_id']) - set(items_has_content)
        train = train[~train['article_id'].isin(missing_content_articles)]

        return train

    def recommend_items(self, train: pd.DataFrame, media: str, dist: str, N):
        """
        Content-Based recommending based on pre-computed item similarity and user's purchase history

        train (pd.DataFrame): training period transcations
        media (str): text-based or image-based
        dist (str): use euclidean or cosine similarity as distance functions
        """
        # select media and distance function
        similarity_dict = self.similarity_dict[(media, dist)]

        # Prepare the recommendation dictionary
        recommendations = {}
        
        users = list(train['customer_id'].unique())
        # Iterate through each unique user
        for user in tqdm(users, desc=f"Content-Based with {media} + {dist}"):
            user_items = train[train['customer_id'] == user]['article_id'].tolist()
            
            candidates = []
            candidates_score = []

            for item in user_items:
                candidates.append(similarity_dict[item]['similar_items'][:N])
                candidates_score.append(similarity_dict[item]['similarity_scores'][:N])
            
            candidates = np.concatenate(candidates)
            candidates_score = np.concatenate(candidates_score)

            # Get the top N items with the highest scores
            top_indices = np.argsort(candidates_score)[::-1]
            
            # Extract corresponding item IDs
            top_items = [candidates[i] for i in top_indices]
            
            top_items = list(dict.fromkeys(top_items))[:N]

            # Store in the recommendation dictionary
            recommendations[user] = top_items
        
        return recommendations