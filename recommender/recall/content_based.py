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

import gensim
assert gensim.models.word2vec.FAST_VERSION > -1

from typing import Dict

from sklearn.preprocessing import normalize
import faiss
from datetime import timedelta

from scipy.sparse import coo_matrix
import implicit
import warnings
warnings.filterwarnings("ignore")


class ContentBased:
    def __init__(
        self,
        train: pd.DataFrame,
        item_cluster: pd.DataFrame,
        purchase_count,
        media: str,
        store_path: str,
        *args,
        **kwargs
    ):
        # item_to_cluster = dict(img_group.values)
        cluster_to_item = {}

        for cluster, group in tqdm(item_cluster.groupby("cluster"), desc=f"{media} cluster construct"):
            cluster_to_item[cluster] = pd.merge(group, purchase_count, how='left').sort_values("count", ascending=False)
        
        item_feature = pd.DataFrame(np.load(store_path, allow_pickle=True).item().items(), columns=['article_id', 'feature'])
        item_cluster = pd.merge(item_cluster, item_feature, on=['article_id'])
        item_cluster_trn = pd.merge(train, item_cluster, on=['article_id'])
        recent_purchased = item_cluster_trn.groupby("customer_id")

        item_in_cluster = []

        for cid, group in tqdm(recent_purchased, desc=f"{media} cluster filter"):
            prod_idx = list(group['cluster'].unique())
            item_in_cluster.extend([(cid, aid) for aid in prod_idx])

        item_select_prod = pd.merge(
            pd.DataFrame(item_in_cluster, columns=["customer_id", "cluster"]),
            item_cluster[['article_id', 'cluster', 'feature']],
            on='cluster',
        )

        # item_feature = {}
        # if media == "image":
        #     for item_id in tqdm(os.listdir(store_path), desc="Loading image feature"):
        #         feature_file = os.path.join(store_path, f"{item_id}")

        #         # 0 for class token, 1 for average of all patch tokens
        #         item_feature[item_id] = np.load(feature_file)[0]
        # elif media == "text":
        #     item_feature = np.load(store_path, allow_pickle=True).item()
        # else:
        #     raise Exception(f"Unexpected media: {media}")
        
        self.item_cluster = item_cluster
        self.item_select_prod = item_select_prod
        self.purchase_count = purchase_count
        self.recent_purchased = recent_purchased
        self.media = media

    def cluster_popularity(self):
        item_sorted_prod = pd.merge(self.purchase_count, self.item_select_prod, on=['article_id'])

        item_sorted_prod_res = {_id: _df for _id, _df in tqdm(item_sorted_prod.groupby("customer_id"), desc=f"{self.media} cluster pop sorting")}

        return item_sorted_prod_res
    
    # def _get_image_vector(self, article_id: int) -> np.ndarray:
    #     # Assumet this is done
    #     pass

    # def _get_text_vector(self, article_id: int) -> np.ndarray:
    #     # Assumet this is done
    #     pass

    # def get_vector(self, article_id: int) -> np.ndarray:
    #     if self.media == "image":
    #         return self._get_image_vector(article_id)
    #     elif self.media == "text":
    #         return self._get_image_vector(article_id)
    #     else:
    #         raise Exception(f"Unexpected media: {self.media}")

    def get_vectors(self, article_id_list: List[int]):
        res = []
        for aid in article_id_list:
            if aid in self.item_feature:
                res.append(self.item_feature[aid])

        return np.array(res)

    def cluster_content_similarity(self):
        len(self.item_select_prod)

        recommendations = {}
        for cid, group in tqdm(self.item_select_prod.groupby("customer_id"), desc=f"{self.media} cluster sim sorting"):
            candidates = group['article_id']
            purchased_vector = np.stack(self.recent_purchased.get_group(cid)['feature'])

            # Step 2: Compute pairwise similarity (dot product) of all candidates and purchased items
            similarity_matrix = np.dot(np.stack(group['feature']), purchased_vector.T)

            # Step 3: Aggregate the similarity score by sum. Each candidate article will have an aggregated score
            aggregated_scores = similarity_matrix.sum(axis=1)

            # Step 4: Sort the candidates according to the aggregated score
            candidate_scores = pd.DataFrame({
                'article_id': candidates,
                'score': aggregated_scores
            }).sort_values(by='score', ascending=False)

            recommendations[cid] = candidate_scores

        return recommendations
    
    # def cluster_content_similarity(self):
    #     recommendations = {}
    #     for cid, group in self.item_select_prod.groupby("customer_id"):
    #         candidates = group['article_id']
    #         purchased = self.recent_purchased[cid]['article_id']

    #         print(candidates)
    #         print(purchased)

    #         # TODO:
    #         # 1. get feature vector for each article in candidates and purchased
    #         # 2. compute pairwise similarity (dot product) of all candidates and purchased items
    #         # 3. aggregate the similarity score by sum. Each candidate article will have a aggregated score 
    #         # 4. sort the candidates according to the aggregated score, store it in recommendations[cid] = pd.DataFrame(..., columns=["article_id", "score"])
    #         exit(0)

    #     return recommendations