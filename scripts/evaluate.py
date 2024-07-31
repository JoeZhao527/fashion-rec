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

        log(f"Initializing recommender system for fold {fold}")