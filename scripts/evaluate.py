import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import List, Dict
import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.system import RecommenderSystem
from src.utils.load_cfg import ConfigLoader
from src.const import RecallerKey


def log(msg):
    print(f"[{datetime.now()}]: {msg}")

def load_fold_data(fold_path: str) -> List[dict]:
    fold_data = []

    for p in tqdm(os.listdir(fold_path), desc="Loading fold data"):
        train = pd.read_csv(os.path.join(fold_path, p, "train.csv"))
        train['t_dat'] = pd.to_datetime(train['t_dat'])

        test = pd.read_csv(os.path.join(fold_path, p, "test.csv"))
        test['t_dat'] = pd.to_datetime(train['t_dat'])

        fold_data.append({
            "train": train,
            "test": test
        })
    
    return fold_data

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
    # map_at_n = 0.0
    # for customer_id in ground_truth:
    #     actual = ground_truth[customer_id]
    #     predicted = predictions.get(customer_id, [])
    #     map_at_n += average_precision(predicted, actual)
    
    # return map_at_n / len(ground_truth)

if __name__ == "__main__":
    log(f"Loading dataset...")
    customers = pd.read_csv("./dataset/customers.csv")
    articles = pd.read_csv("./dataset/articles.csv")

    fold_data = load_fold_data("./dataset/split")

    for fold, data in enumerate(fold_data):
        train = data["train"]
        test = data["test"]

        cfg_loader = ConfigLoader("./configs")
        feat_keys = cfg_loader.get_feature("baseline")
        ranker = cfg_loader.get_ranker("xgboost")

        log(f"Initializing recommender system for fold {fold}")
        rs = RecommenderSystem(
            train_data=train,
            user_data=customers,
            item_data=articles,
            image_feat_path="./output",
            cache_dir=f"./cache/fold_{fold}",
            # item_stat_feat_path=f"./cache/fold_{fold}/feature/stat",
            # user_stat_feat_path=f"./cache/fold_{fold}/feature/user_stat",
            # ranking_train_cache=f"./cache/fold_{fold}/train.csv",
            cate_feat_keys=feat_keys["cate_feat_keys"],
            num_feat_keys=feat_keys["num_feat_keys"],
            obj_feat_keys=[],
            recall_cfg=[(RecallerKey.POP, 500)],
            ranker_cfg=[(ranker["model_name"], ranker["params"])]
        )

        log(f"Initializing feature for fold {fold}")
        rs.init_feature()

        log(f"Training ranking models for fold {fold}")
        rs.init_ranking()

        log(f"Evaluating for fold {fold}")
        test_users = set(test['customer_id']).intersection(set(rs.users))

        ranking_input = []
        log(f"Start recalling for fold {fold}")
        for user in tqdm(list(test_users), desc="Recalling"):
            recalled = rs.recall(user)
            pairs = pd.DataFrame({"article_id": list(recalled.keys())})
            pairs["customer_id"] = user

            ranking_input.append(pairs)

        log(f"Start ranking for fold {fold}")
        ranking_input = pd.concat(ranking_input).reset_index(drop=True)
        ranked = rs.batch_ranking(ranking_input)

        log(f"Start evaluating for fold {fold}")
        top_n = 12
        rank_precision = []
        for i, (customer_id, group) in enumerate(ranked.groupby("customer_id")):
            _ranked = group.sort_values("xgboost_y_pred_prob").reset_index()
            purchased = test[test['customer_id'] == user]['article_id']

            # rank_precision.append(
            #     len(set(purchased).intersection(set(_ranked[:top_n]['article_id']))) / top_n
            # )
            rank_precision.append(calculate_map_at_n(list(_ranked[:top_n]['article_id']), list(purchased)))

            if i != 0 and i % 1000 == 0:
                log(f"MAP for {i} users: {np.array(rank_precision).mean():.5f}")

        # log(f"Start evaluating for fold {fold}")
        # top_n = 12
        # rank_precision = []

        # # Group the ranked dataframe by customer_id
        # ranked_grouped = ranked.groupby("customer_id")

        # # Define a function to compute precision for a single customer
        # def compute_precision(customer_id):
        #     group = ranked_grouped.get_group(customer_id)
        #     _ranked = group.sort_values("xgboost_y_pred_prob").reset_index()
        #     purchased = set(test[test['customer_id'] == customer_id]['article_id'])
        #     top_ranked = set(_ranked[:top_n]['article_id'])
        #     precision = len(purchased.intersection(top_ranked)) / top_n
        #     return precision
        
        # # Use ThreadPoolExecutor to parallelize the computation
        # with ThreadPoolExecutor(max_workers=20) as executor:
        #     futures = {executor.submit(compute_precision, customer_id): customer_id for customer_id in ranked_grouped.groups.keys()}
            
        #     for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
        #         rank_precision.append(future.result())

        log(f"Precision: {np.array(rank_precision).mean():.5f}")    