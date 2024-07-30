import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from typing import List, Dict
import torch

from matplotlib import pyplot as plt
import seaborn as sns
import itertools

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from recommender.recall import (
    item2vec_recall,
    popularity_recall,
    postal_code_recall,
    product_code_recall,
    image_cluster_recall,
    bought_together_recall,
    user_collaborative_recall
)


def calculate_map_at_n(predictions, ground_truth, top_n=12):
    """
    Calculate Mean Average Precision @ N (MAP@N)
    
    Parameters:
    predictions (list): a list of predicted article_ids.
    ground_truth (list): a list of actual article_ids (ground truth).
    top_n (int): Number of top predictions to consider (default is 12).
    
    Returns:
    float: MAP@N score.
    """
    def precision_at_k(k, predicted, actual):
        if len(predicted) > k:
            predicted = predicted[:k]
        return len(set(predicted) & set(actual)) / k

    def average_precision(predicted, actual):
        if not actual:
            return 0.0
        ap = 0.0
        relevant_items = 0
        for k in range(1, min(len(predicted), top_n) + 1):
            if predicted[k-1] in actual:
                relevant_items += 1
                ap += precision_at_k(k, predicted, actual) * 1
        return ap / min(len(actual), top_n)
    
    return average_precision(predictions, ground_truth)


def rank_calculate_mapk(actual, predicted, k=12):
    mapk = 0
    for user_id, group in predicted.groupby('customer_id'):
        actual_items = set(actual[user_id])
        pred_items = list(group['article_id'])
        score = 0
        hits = 0
        for i, p in enumerate(pred_items[:min(len(pred_items), k)]):
            if p in actual_items:
                hits += 1
                score += hits / (i + 1)
        mapk += score / min(len(actual_items), k)
    mapk /= len(predicted['customer_id'].unique())
    return mapk


def split_last_week_data(df, date_column="t_dat"):
    """
    Splits a DataFrame into two parts: data from the last week and all other data.

    Parameters:
    df (DataFrame): The DataFrame to split.
    date_column (str): The name of the column containing date information.

    Returns:
    tuple: A tuple containing two DataFrames. The first DataFrame contains data from the last week,
           and the second DataFrame contains all other data.
    """
    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Find the maximum date in the DataFrame
    max_date = df[date_column].max()

    # Define the last week range
    last_week_start = max_date - pd.Timedelta(days=6)  # Include the max_date as part of the last week

    # Split the DataFrame into last week and the rest
    last_week_data = df[df[date_column] > last_week_start]
    other_data = df[df[date_column] <= last_week_start]

    return last_week_data, other_data


def filter_data(all_train: pd.DataFrame, all_test: pd.DataFrame, thr: int = 10):
    flit_train = all_train['customer_id'].value_counts()
    train_idx = flit_train[flit_train > thr].index

    flit_test = all_test['customer_id'].value_counts()
    test_idx = flit_test[flit_test > thr].index

    filt_idx = set(train_idx).intersection(set(test_idx))

    trn = all_train[all_train['customer_id'].isin(filt_idx)]
    tst = all_test[all_test['customer_id'].isin(filt_idx)]

    return trn, tst

def get_image_path(item_id):
    item_id_str = str(item_id)
    folder_number = '0' + item_id_str[:2]  # Ensure this logic matches your folder structure
    item_id_str = '0' + item_id_str
    image_url = f'http://localhost:5000/images/{folder_number}/{item_id_str}.jpg'
    # print(image_url)
    return image_url

class RecommenderSystem:
    def __init__(
        self,
        article_path: str,
        customer_path: str,
        train_path: str,
        test_path: str,
        img_cluster_path: str,
        cache_dir: str,
        dev_mode: bool = False,
        recall_top_n: int = 100,
        rank_top_n: int = 12,
        rank_neg_sample: int = 5,
    ):
        self.dev_mode = dev_mode
        self.recall_top_n = recall_top_n
        self.rank_neg_sample = rank_neg_sample
        self.rank_top_n = rank_top_n
        
        os.makedirs(cache_dir, exist_ok=True)
        self.recall_cache_dir = os.path.join(cache_dir, "recall")
        self.ranking_cache_dir = os.path.join(cache_dir, "ranking")

        print(f"Loading data...")
        self.img_cluster = pd.read_csv(img_cluster_path)

        self.articles = pd.read_csv(article_path)
        self.customers = pd.read_csv(customer_path)

        print(f"Initializing data...")
        self._init_data(train_path, test_path)

        print(f"Recalling...")
        recall_res, rank_train, rank_test = self._recall()

        self.recall_res = recall_res
        self.rank_train = rank_train
        self.rank_test = rank_test

        print(f"Ranking...")
        self.recommendations = self._ranking()

    def _init_data(self, train_path, test_path):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        # In in development mode, only select a small portion of users
        if self.dev_mode:
            train, test = filter_data(train, test)

        rank_label, _ = split_last_week_data(train)

        # Use all train period data to recall and train ranking models
        self.train = train

        # Use last week in train period to label purchased for ranking models
        self.rank_label = rank_label

        # Use test period data to test top N recommendation performance
        self.test = test

    def _recall(self):
        metrics_res_path = os.path.join(self.recall_cache_dir, "metric.csv")
        rank_train_path = os.path.join(self.recall_cache_dir, "rank_train.csv")
        rank_test_path = os.path.join(self.recall_cache_dir, "rank_test.csv")

        # If cache exists, use cache
        if os.path.exists(self.recall_cache_dir):
            res = pd.read_csv(metrics_res_path)
            rak_recall_df = pd.read_csv(rank_train_path)
            tst_recall_df = pd.read_csv(rank_test_path)
        else:
            top_n = self.recall_top_n

            purchase_count = popularity_recall(train=self.train)

            item2vec_res = item2vec_recall(
                train=self.train,
                articles=self.articles,
                top_N=top_n
            )

            postal_code_res, customers_postal_code_map = postal_code_recall(
                train=self.train,
                customers=self.customers,
                purchase_count=purchase_count
            )

            product_code_res = product_code_recall(
                train=self.train,
                articles=self.articles,
                purchase_count=purchase_count
            )

            image_cluster_res = image_cluster_recall(
                train=self.train,
                img_group=self.img_cluster,
                purchase_count=purchase_count
            )

            bought_together_res = bought_together_recall(train=self.train)

            user_cf_res = user_collaborative_recall(
                train=self.train,
                top_N=top_n
            )

            metrics = {
                k: {"purchased": [], "hit_num": [], "precision": [], "recall": [], "recall_num": [], "map": []}
                for k in ['postal', 'product', 'pop', 'user_cf', 'img', 'also_buy', 'item2vec', 'together']
            }
            rank_label_recall = []
            test_recall = []

            def register_recall(customer_id, purchased, all_items, recall_pipes: dict, recall_set: list):
                res = {}
                # initialize recall dict for all items
                for item in all_items:
                    res[item] = {}
                    for k in recall_pipes:
                        res[item][k] = 0
                        res[item][f"{k}_score"] = -1
                        res[item]['purchased'] = int(item in purchased)
                        res[item]['customer_id'] = customer_id
                        res[item]['article_id'] = item

                for k in recall_pipes:
                    recalled_num = len(recall_pipes[k])
                    for i, item in enumerate(recall_pipes[k]):
                        res[item][k] = 1
                        res[item][f"{k}_score"] = (recalled_num - i) / recalled_num

                recall_set.extend(res.values())

            def register(k, items, purchased):
                hit = len(set(purchased).intersection(set(items)))
                precision = hit / len(items)
                recall = hit / len(purchased)
                purchased_num = len(purchased)
                
                metrics[k]["purchased"].append(purchased_num)
                metrics[k]["hit_num"].append(hit)
                metrics[k]["precision"].append(precision)
                metrics[k]["recall"].append(recall)
                metrics[k]["recall_num"].append(len(set(items)))
                metrics[k]["map"].append(calculate_map_at_n(list(items)[:top_n], list(purchased)))

            purchase_dict = self.test.groupby('customer_id')['article_id'].agg(list)
            rank_label_purchase_dict = self.rank_label.groupby('customer_id')['article_id'].agg(list)

            test_users = set(self.train['customer_id']).intersection(list(purchase_dict.keys()))

            skip_users = []
            skip_recall = []
            for cid in tqdm(test_users):
                # purchased = set(test[test['customer_id'] == cid]['article_id'])
                purchased = purchase_dict[cid]

                # get user postcode
                postal_code = customers_postal_code_map[cid]

                try:
                    _res = {
                        "also_buy": list(bought_together_res[cid][:top_n]['article_id']),
                        "img": list(image_cluster_res[cid][:top_n]['article_id']),
                        "product": list(product_code_res[cid][:top_n]['article_id']),
                        "pop": list(purchase_count[:top_n]['article_id']),
                        "user_cf": list(user_cf_res[cid][:top_n]),
                        "item2vec": list(item2vec_res[cid][:top_n]),
                        "postal": list(postal_code_res[postal_code][:top_n]['article_id'])
                    }
                except:
                    skip_users.append(cid)
                    continue

                together_recall = set(itertools.chain(*_res.values()))
                
                register_recall(
                    customer_id=cid,
                    purchased=purchased,
                    all_items=together_recall,
                    recall_pipes=_res,
                    recall_set=test_recall
                )
                
                try:
                    rank_label_purchased = rank_label_purchase_dict[cid]
                    register_recall(
                        customer_id=cid,
                        purchased=rank_label_purchased,
                        all_items=together_recall,
                        recall_pipes=_res,
                        recall_set=rank_label_recall
                    )
                except:
                    skip_recall.append(cid)
                
                for k, v in _res.items():
                    register(k, v, purchased)
                
                register("together", together_recall, purchased)
        
            res = pd.DataFrame(metrics).applymap(lambda x: round(np.array(x).mean(), 4)).transpose()
            res['recall_num'] = res['recall_num'].apply(int)
            res['purchased'] = res['purchased'].apply(lambda x: round(x, 2))

            print("Preparing recall for training")
            rak_recall_df = pd.DataFrame(rank_label_recall)

            print("Preparing recall for testing")
            tst_recall_df = pd.DataFrame(test_recall)

            os.makedirs(self.recall_cache_dir, exist_ok=True)
            res.to_csv(metrics_res_path, index=False)
            rak_recall_df.to_csv(rank_train_path, index=False)
            tst_recall_df.to_csv(rank_test_path, index=False)

            print(res)

        return res, rak_recall_df, tst_recall_df
    
    def _ranking(self):
        recommendation_path = os.path.join(self.ranking_cache_dir, "rank.csv")

        if os.path.exists(self.ranking_cache_dir):
            all_pred_prob = pd.read_csv(recommendation_path)
        else:
            # Define the desired ratio
            desired_ratio = 5

            # Function to perform sampling for each group
            def sample_group(group, ratio):
                purchased = group[group['purchased'] == 1]
                not_purchased = group[group['purchased'] == 0]
                
                num_purchased = len(purchased)
                num_not_purchased = min(len(not_purchased), num_purchased * ratio)
                
                sampled_purchased = purchased
                sampled_not_purchased = not_purchased.sample(n=num_not_purchased, random_state=42)
                
                return pd.concat([sampled_purchased, sampled_not_purchased])

            # Apply the sampling function to each customer group
            rank_train = self.rank_train.groupby('customer_id').apply(lambda group: sample_group(group, desired_ratio)).reset_index(drop=True)

            # Split the data into train and test sets
            train_data, test_data = train_test_split(rank_train, test_size=0.2, random_state=42, stratify=rank_train['purchased'])

            # Define features and target
            X = train_data.drop(columns=['customer_id', 'article_id', 'purchased'])
            y = train_data['purchased']

            pred_X = test_data.drop(columns=['customer_id', 'article_id', 'purchased'])
            pred_y = test_data['purchased']

            # Train XGBoost model
            model = XGBClassifier(n_estimators=20, max_depth=10, learning_rate=0.5, objective='binary:logistic', enable_categorical=True)
            model.fit(X, y)

            # Predict and evaluate
            y_pred = model.predict(pred_X)
            y_pred_proba = model.predict_proba(pred_X)[:, 1]

            accuracy = accuracy_score(pred_y, y_pred)
            roc_auc = roc_auc_score(pred_y, y_pred_proba)

            # Print average scores
            print(f'Accuracy: {accuracy}')
            print(f'ROC AUC: {roc_auc}')
            
            self.model = model

            inferece = self.rank_test.copy()
            all_pred_prob = model.predict_proba(self.rank_test.drop(columns=['customer_id', 'article_id', 'purchased']))

            # Record the test set predictions
            all_pred_prob = pd.DataFrame({
                'customer_id': inferece['customer_id'],
                'article_id': inferece['article_id'],
                'predicted_proba': all_pred_prob[:, 1],
            })

            # Reset index of all_test_preds
            all_pred_prob.reset_index(drop=True, inplace=True)

            all_pred_prob.sort_values(by=['customer_id', 'predicted_proba'], ascending=[True, False], inplace=True)

            os.makedirs(self.ranking_cache_dir, exist_ok=True)
            all_pred_prob.to_csv(recommendation_path, index=False)

        recommendations = all_pred_prob.groupby('customer_id').head(self.rank_top_n).reset_index(drop=True)

        purchase_dict = self.test.groupby('customer_id')['article_id'].agg(list)
        map_at_12 = rank_calculate_mapk(purchase_dict, recommendations, self.rank_top_n)
        print(f"MAP@12: {map_at_12}")

        return recommendations
    
    def recommend(self, customer_id: str) -> List[int]:
        if customer_id in self.recommendations:
            return self.recommendations[customer_id]
        else:
            return list(self.articles.sample(self.rank_top_n)['article_id'])
    
    def get_items_by_ids(self, item_ids: List[int]):
        # Filter the DataFrame for the given item IDs
        filtered_items = self.articles[self.articles['article_id'].isin(item_ids)]

        # Convert the updated DataFrame to a list of dictionaries
        item_details = filtered_items.to_dict(orient='records')
        return item_details

    def get_user_purchased(self, customer_id: str) -> list:
        actual_purchases = self.train.groupby('customer_id')['article_id'].agg(list)
        return actual_purchases[customer_id] if customer_id in actual_purchases else []

    def get_items_by_ids(self, item_ids: list) -> List[dict]:
        items = []
        for item_id in item_ids:
            item_details = self.articles[self.articles['article_id'] == item_id].to_dict(orient='records')
            if item_details:
                item = item_details[0]
                item['liked'] = False
                item['image_url'] = get_image_path(item_id)
                items.append(item)
        return items

if __name__ == '__main__':
    system = RecommenderSystem(
        article_path="./dataset/articles.csv",
        customer_path="./dataset/customers.csv",
        train_path="./dataset/split/fold_0/train.csv",
        test_path="./dataset/split/fold_0/test.csv",
        img_cluster_path="./resources/img_cluster_2000.csv",
        dev_mode=True,
        cache_dir="./cache/fold_0"
    )