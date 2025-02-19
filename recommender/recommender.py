import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from typing import List, Dict, Union, Literal
import torch

from matplotlib import pyplot as plt
import seaborn as sns
import itertools

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import argparse

from recommender.utils.general import log, log_configuaration

from recommender.recall import (
    ContentBased,
    Item2VecModel,
    popularity_recall,
    postal_code_recall,
    product_code_recall,
    bought_together_recall,
    user_collaborative_recall,
    age_group_recall
)

from recommender.utils.evaluate import (
    rank_calculate_mapk,
    calculate_map_at_n
)

from recommender.utils.data import (
    filter_data,
    split_last_week_data,
    recall_select,
    cold_start_agg
)

def get_image_path(item_id):
    item_id_str = str(item_id)
    folder_number = '0' + item_id_str[:2]  # Ensure this logic matches your folder structure
    item_id_str = '0' + item_id_str
    image_url = f'http://127.0.0.1:5000/images/{folder_number}/{item_id_str}.jpg'
    # log(image_url)
    return image_url

class RecommenderSystem:
    def __init__(
        self,
        article_path: str,
        customer_path: str,
        train_path: str,
        test_path: str,
        image_cluster_path: str,
        image_feature_path: str,
        text_cluster_path: str,
        text_feature_path: str,
        cache_dir: str,
        production_mode: bool = False,
        recall_top_n: int = 100,
        rank_top_n: int = 12,
        rank_neg_sample: int = 5,
        recall_pipeline: List[
            Literal['postal', 'product', 'pop', 'user_cf', 'img', 'also_buy', 'item2vec_sim', 'item2vec_cls']
        ] = ['postal', 'product', 'pop', 'user_cf', 'img', 'also_buy', 'item2vec_sim', 'item2vec_cls'],
        recall_select: bool = False,
        recall_overwrite_cache: bool = False,
        rank_overwrite_cache: bool = False,
    ):
        log_configuaration(
            article_path=article_path,
            customer_path=customer_path,
            train_path=train_path,
            test_path=test_path,
            image_cluster_path=image_cluster_path,
            image_feature_path=image_feature_path,
            text_cluster_path=text_cluster_path,
            text_feature_path=text_feature_path,
            cache_dir=cache_dir,
            production_mode=production_mode,
            recall_top_n=recall_top_n,
            rank_top_n=rank_top_n,
            rank_neg_sample=rank_neg_sample,
            recall_pipeline=recall_pipeline,
            recall_select=recall_select,
            recall_overwrite_cache=recall_overwrite_cache,
            rank_overwrite_cache=rank_overwrite_cache
        )

        self.production_mode = production_mode
        self.recall_top_n = recall_top_n
        self.rank_neg_sample = rank_neg_sample
        self.rank_top_n = rank_top_n
        self.recall_pipeline = recall_pipeline
        self.recall_select = recall_select
        self.recall_overwrite_cache = recall_overwrite_cache
        self.rank_overwrite_cache = rank_overwrite_cache

        self.image_feature_path = image_feature_path
        self.text_feature_path = text_feature_path

        os.makedirs(cache_dir, exist_ok=True)
        self.recall_cache_dir = os.path.join(cache_dir, "recall")
        self.ranking_cache_dir = os.path.join(cache_dir, "ranking")

        log(f"Loading data...")
        self.img_cluster = pd.read_csv(image_cluster_path)
        self.txt_cluster = pd.read_csv(text_cluster_path)

        self.articles = pd.read_csv(article_path)
        self.customers = pd.read_csv(customer_path)

        log(f"Initializing data...")
        self._init_data(train_path, test_path)

        log(f"Recalling...")
        recall_res, rank_train, rank_test = self._recall()

        self.recall_res = recall_res
        self.rank_train = rank_train
        self.rank_test = rank_test

        log(f"Ranking...")
        self.recommendations = self._ranking().groupby('customer_id')['article_id'].agg(list)

    def _init_data(self, train_path, test_path):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        # If not running for production, only select a small portion of users
        if not self.production_mode:
            train, test = filter_data(train, test)

        rank_label, _ = split_last_week_data(train)

        # Use all train period data to recall and train ranking models
        self.train = train

        # Use last week in train period to label purchased for ranking models
        self.rank_label = rank_label

        # Use test period data to test top N recommendation performance
        self.test = test
        self.actual_purchases = self.train.groupby('customer_id')['article_id'].agg(list)

    def _recall(self):
        metrics_res_path = os.path.join(self.recall_cache_dir, "metric.csv")
        rank_train_path = os.path.join(self.recall_cache_dir, "rank_train.csv")
        rank_test_path = os.path.join(self.recall_cache_dir, "rank_test.csv")
        age_agg_path = os.path.join(self.recall_cache_dir, "age_group.csv")
        postal_agg_path = os.path.join(self.recall_cache_dir, "postal_code.csv")

        # If cache exists, use cache
        if os.path.exists(self.recall_cache_dir) and not self.recall_overwrite_cache:
            res = pd.read_csv(metrics_res_path)
            rak_recall_df = pd.read_csv(rank_train_path)
            tst_recall_df = pd.read_csv(rank_test_path)
            age_group_agg = pd.read_csv(age_agg_path)
            postal_code_agg = pd.read_csv(postal_agg_path)
        else:
            top_n = self.recall_top_n

            purchase_count = popularity_recall(train=self.train)

            image_cb = ContentBased(
                train=self.train,
                item_cluster=self.img_cluster,
                purchase_count=purchase_count,
                store_path=self.image_feature_path,
                media="image"
            )
            image_cluster_pop_res = image_cb.cluster_popularity()
            # image_cluster_sim_res = image_cb.cluster_content_similarity()

            text_cb = ContentBased(
                train=self.train,
                item_cluster=self.txt_cluster,
                purchase_count=purchase_count,
                store_path=self.text_feature_path,
                media="text"
            )
            text_cluster_pop_res = text_cb.cluster_popularity()
            # text_cluster_sim_res = text_cb.cluster_content_similarity()

            item2vec = Item2VecModel(
                train=self.train,
                articles=self.articles,
                top_N=self.recall_top_n
            )

            item2vec_cls_res = item2vec.cluster_recall()
            item2vec_sim_res = item2vec.similarity_recall()
            
            postal_code_res, customers_postal_code_map = postal_code_recall(
                train=self.train,
                customers=self.customers,
                purchase_count=purchase_count
            )

            age_group_res, customers_age_group_map = age_group_recall(
                train=self.train,
                customers=self.customers,
                purchase_count=purchase_count
            )

            product_code_res = product_code_recall(
                train=self.train,
                articles=self.articles,
                purchase_count=purchase_count
            )
            
            bought_together_res = bought_together_recall(train=self.train)

            user_cf_res = user_collaborative_recall(
                train=self.train,
                top_N=top_n
            )

            # Aggregate postal code and age information for user cold start
            postal_code_agg = cold_start_agg(postal_code_res)
            age_group_agg = cold_start_agg(age_group_res)

            metrics = {
                k: {"purchased": [], "hit_num": [], "precision": [], "recall": [], "recall_num": [], "map": []}
                for k in [
                    'postal', 'product', 'pop', 'also_buy', 'age', 'user_cf',
                    'item2vec_sim', 'item2vec_cls',
                    'img_cb_pop', 'txt_cb_pop',
                    'together'
                ]
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
                        res[item][f"{k}_score"] = 0.0
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
            for cid in tqdm(test_users, desc="Aggregrating recall results"):
                purchased = purchase_dict[cid]

                # get user postcode
                postal_code = customers_postal_code_map[cid]
                age_group = customers_age_group_map[cid]

                try:
                    _res = {
                        "also_buy": list(bought_together_res[cid][:top_n]['article_id']),
                        "img_cb_pop": list(image_cluster_pop_res[cid][:top_n]['article_id']),
                        "txt_cb_pop": list(text_cluster_pop_res[cid][:top_n]['article_id']),
                        "product": list(product_code_res[cid][:top_n]['article_id']),
                        "pop": list(purchase_count[:top_n]['article_id']),
                        "user_cf": list(user_cf_res[cid][:top_n]),
                        "item2vec_sim": list(item2vec_sim_res[cid][:top_n]),
                        "item2vec_cls": list(item2vec_cls_res[cid][:top_n]),
                        "postal": list(postal_code_res[postal_code][:top_n]['article_id']),
                        "age": list(age_group_res[age_group][:top_n]['article_id'])
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

            log("Preparing recall for training")
            rak_recall_df = pd.DataFrame(rank_label_recall)

            log("Preparing recall for testing")
            tst_recall_df = pd.DataFrame(test_recall)

            os.makedirs(self.recall_cache_dir, exist_ok=True)
            res.to_csv(metrics_res_path)
            rak_recall_df.to_csv(rank_train_path, index=False)
            tst_recall_df.to_csv(rank_test_path, index=False)
            postal_code_agg.to_csv(postal_agg_path, index=False)
            age_group_agg.to_csv(age_agg_path, index=False)

            log(f"Recall results:")
            print(res)

        if self.recall_select:
            rak_recall_df, tst_recall_df = self._recall_selection(rak_recall_df, tst_recall_df)

        self.age_group_agg = age_group_agg
        self.postal_code_agg = postal_code_agg

        return res, rak_recall_df, tst_recall_df
    
    def _recall_selection(self, rak_recall_df, tst_recall_df):
        rak_recall_records = recall_select(rak_recall_df, self.recall_pipeline)
        tst_recall_records = recall_select(tst_recall_df, self.recall_pipeline)

        purchase_dict = self.test.groupby('customer_id')['article_id'].agg(list)

        top_n = self.rank_top_n
        metrics = {
            "purchased": [], "hit_num": [], "precision": [], "recall": [], "recall_num": [], "map": []
        }
        def register(customer_trans: pd.DataFrame, purchased):
            items = customer_trans['article_id']

            hit = len(set(purchased).intersection(set(items)))
            precision = hit / len(items)
            recall = hit / len(purchased)
            purchased_num = len(purchased)
            
            metrics["purchased"].append(purchased_num)
            metrics["hit_num"].append(hit)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["recall_num"].append(len(set(items)))
            metrics["map"].append(calculate_map_at_n(list(items)[:top_n], list(purchased)))
        
        for cid, customer_df in tst_recall_records.items():
            purchased = purchase_dict[cid]
            register(customer_df, purchased)

        res = pd.DataFrame({f"{'+'.join(self.recall_pipeline)}": metrics}).applymap(lambda x: round(np.array(x).mean(), 4)).transpose()
        res['recall_num'] = res['recall_num'].apply(int)
        res['purchased'] = res['purchased'].apply(lambda x: round(x, 2))

        self.selected_recall_performance = res
        log(f"selected recall performance for {len(tst_recall_records)} users:")
        log(self.selected_recall_performance)

        log(len(pd.concat(tst_recall_records.values())))
        return pd.concat(rak_recall_records.values()), pd.concat(tst_recall_records.values())
    
    def _cold_start_recommend(self, cold_start_user):
        # recommend for customer did not make any purchase
        cold_start_users_df = self.customers[self.customers['customer_id'].isin(cold_start_user)]
        cold_start_users_df['age_group'] = (cold_start_users_df['age'] // 5) * 5
        # Initialize cold start recommendations
        cold_start_recommend = []

        # Process each cold start user
        for idx, user_info in tqdm(cold_start_users_df.iterrows(), total=len(cold_start_users_df), desc="cold start preparing"):
            customer_id = user_info['customer_id']
            user_age_group = user_info['age_group']
            user_postal_code = user_info['postal_code']
            
            # Get top articles from age group
            age_group_recommendations = self.age_group_agg[self.age_group_agg['group'] == user_age_group].nlargest(12, 'count')['article_id'].tolist()
            
            # Get top articles from postal code
            postal_code_recommendations = self.postal_code_agg[self.postal_code_agg['group'] == user_postal_code].nlargest(12, 'count')['article_id'].tolist()
            
            # Combine recommendations and keep top 12 unique items
            combined_recommendations = list(set(age_group_recommendations + postal_code_recommendations))[:12]
            
            # Add to recommendations list
            cold_start_recommend.append([customer_id, combined_recommendations])

        # Create DataFrame for cold start recommendations
        cold_start_recommendations = pd.DataFrame(cold_start_recommend, columns=['customer_id', 'recommendations'])

        return cold_start_recommendations
    
    def _ranking(self):
        recommendation_path = os.path.join(self.ranking_cache_dir, "rank.csv")
        feature_importance_path = os.path.join(self.ranking_cache_dir, "feat_importance.csv")

        if os.path.exists(self.ranking_cache_dir) and not self.rank_overwrite_cache:
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

            # # Split the data into train and test sets
            # train_data, test_data = train_test_split(rank_train, test_size=0.2, random_state=42, stratify=rank_train['purchased'])

            # # Define features and target
            # X = train_data.drop(columns=['customer_id', 'article_id', 'purchased'])
            # y = train_data['purchased']

            # pred_X = test_data.drop(columns=['customer_id', 'article_id', 'purchased'])
            # pred_y = test_data['purchased']

            # # Train XGBoost model
            # model = XGBClassifier(n_estimators=20, max_depth=10, learning_rate=0.5, objective='binary:logistic', enable_categorical=True)
            # model.fit(X, y)

            # # Predict and evaluate
            # y_pred = model.predict(pred_X)
            # y_pred_proba = model.predict_proba(pred_X)[:, 1]

            # accuracy = accuracy_score(pred_y, y_pred)
            # roc_auc = roc_auc_score(pred_y, y_pred_proba)

            X = rank_train.drop(columns=['customer_id', 'article_id', 'purchased'])
            y = rank_train['purchased']

            model = XGBClassifier(n_estimators=20, max_depth=10, learning_rate=0.5, objective='binary:logistic', enable_categorical=True)
            model.fit(X, y)

            # Predict and evaluate
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]

            accuracy = accuracy_score(y, y_pred)
            roc_auc = roc_auc_score(y, y_pred_proba)

            # log average scores
            log(f'Test Accuracy: {accuracy}')
            log(f'Test ROC AUC: {roc_auc}')
            
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

            # Rank according to model prediction score
            all_pred_prob.sort_values(by=['customer_id', 'predicted_proba'], ascending=[True, False], inplace=True)

            # Compute feature importance for interpretations
            feature_importance = model.get_booster().get_score(importance_type='weight')
            importance_df = pd.DataFrame(feature_importance.items(), columns=['Feature', 'Importance'])
            importance_df = importance_df.sort_values(by='Importance', ascending=False)

            # Save model prediction and feature importance
            os.makedirs(self.ranking_cache_dir, exist_ok=True)
            all_pred_prob.to_csv(recommendation_path, index=False)
            importance_df.to_csv(feature_importance_path, index=False)
            
        recommendations = all_pred_prob.groupby('customer_id').head(self.rank_top_n).reset_index(drop=True)

        # cold_start_user = set(self.test['customer_id']) - set(recommendations['customer_id'])
        # cold_recommendations = self._cold_start_recommend(cold_start_user)

        purchase_dict = self.test.groupby('customer_id')['article_id'].agg(list)

        map_at_12 = rank_calculate_mapk(purchase_dict, recommendations, self.rank_top_n)
        # map_at_12_cold = rank_calculate_mapk(purchase_dict, cold_recommendations, self.rank_top_n)

        log(f"MAP@12 for {len(recommendations['customer_id'].unique())} hot users: {map_at_12}")
        # log(f"MAP@12 for {len(cold_recommendations['customer_id'].unique())} cold users: {map_at_12_cold}")

        # return pd.concat([recommendations, cold_recommendations])
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
        return self.actual_purchases[customer_id] if customer_id in self.actual_purchases else []

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Recommender System Configuration')
    
    parser.add_argument('--article_path', type=str, default='./dataset/articles.csv',
                        help='Path to the articles CSV file')
    parser.add_argument('--customer_path', type=str, default='./dataset/customers.csv',
                        help='Path to the customers CSV file')
    parser.add_argument('--train_path', type=str, default='./dataset/split/fold_0/train.csv',
                        help='Path to the train CSV file')
    parser.add_argument('--test_path', type=str, default='./dataset/split/fold_0/test.csv',
                        help='Path to the test CSV file')
    parser.add_argument('--image_cluster_path', type=str, default='./resources/img_cluster_2000.csv',
                        help='Path to the image cluster CSV file')
    parser.add_argument('--image_feature_path', type=str, default='./feature/dino_image_emb.npy',
                        help='Path to the image feature directory')
    parser.add_argument('--text_cluster_path', type=str, default='./resources/text_cluster_2000.csv',
                        help='Path to the text cluster CSV file')
    parser.add_argument('--text_feature_path', type=str, default='./feature/glove_text_emb.npy',
                        help='Path to the text embedding npy')
    parser.add_argument('--cache_dir', type=str, default='./cache/fold_0',
                        help='Directory for cache')
    parser.add_argument('--production_mode', action='store_true',
                        help='Enable production mode')
    parser.add_argument('--rank_overwrite_cache', action='store_true', dest='rank_overwrite_cache',
                        help='Enable overwriting the cache for rank')
    parser.add_argument('--recall_overwrite_cache', action='store_true', dest='recall_overwrite_cache',
                        help='Enable overwriting the cache for recall')
    parser.add_argument('--recall_select', action='store_true',
                        help='Enable selecting certain recall pipeline selection after recall')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    system = RecommenderSystem(
        article_path=args.article_path,
        customer_path=args.customer_path,
        train_path=args.train_path,
        test_path=args.test_path,
        image_cluster_path=args.image_cluster_path,
        image_feature_path=args.image_feature_path,
        text_cluster_path=args.text_cluster_path,
        text_feature_path=args.text_feature_path,
        production_mode=args.production_mode,
        cache_dir=args.cache_dir,
        recall_overwrite_cache=args.recall_overwrite_cache,
        rank_overwrite_cache=args.rank_overwrite_cache,
        recall_select=args.recall_select
    )