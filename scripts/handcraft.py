import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from typing import List, Dict
import torch
from datetime import datetime


def get_mode(x):
    mode = x.mode()
    if len(mode) == 0:
        return np.nan
    else:
        return mode[0]

def get_user_basic_features(transactions):
    user_basic_features = transactions.groupby('customer_id').agg(
        num_purchases=('article_id', 'count'),
        avg_price=('price', 'mean'),
        sales_channel_mode=('sales_channel_id', get_mode)
    ).reset_index()
    return user_basic_features

def get_product_basic_features(transactions, customers, articles):
    product_stats = transactions.merge(customers, on='customer_id', how='left').merge(articles, on='article_id', how='left')
    product_basic_features = product_stats.groupby('article_id').agg(
        purchase_count=('t_dat', 'count'),
        avg_price=('price', 'mean'),
        avg_customer_age=('age', 'mean'),
        sales_channel_mode=('sales_channel_id', get_mode),
        active_customer_rate=('Active', 'mean'),
        fn_rate=('FN', 'mean'),
        club_member_status_mode=('club_member_status', get_mode),
        fashion_news_frequency_mode=('fashion_news_frequency', get_mode),
        last_purchase_time=('t_dat', 'max'),
        avg_purchase_interval=('t_dat', lambda x: (x.max() - x.min()).days / len(x) if len(x) > 1 else np.nan)
    ).reset_index()
    return product_basic_features

def get_user_product_combination_features(transactions):
    user_product_stats = transactions.groupby(['customer_id', 'article_id']).agg(
        num_purchases=('t_dat', 'count'),
        last_purchase_time=('t_dat', 'max'),
        avg_purchase_interval=('t_dat', lambda x: (x.max() - x.min()).days / len(x) if len(x) > 1 else np.nan),
        sales_channel_mode=('sales_channel_id', get_mode)
    ).reset_index()
    return user_product_stats

def get_age_product_combination_features(transactions, customers):
    age_product_stats = transactions.merge(customers, on='customer_id', how='left')
    age_product_features = age_product_stats.groupby(['age', 'article_id']).agg(
        purchase_count=('t_dat', 'count')
    ).reset_index()
    return age_product_features

def get_user_repurchase_features(transactions):
    repurchase_stats = transactions.groupby(['customer_id', 'article_id']).agg(
        repurchase=('t_dat', lambda x: 1 if len(x) > 1 else 0)
    ).reset_index()
    return repurchase_stats

def get_product_repurchase_features(transactions):
    product_repurchase_stats = transactions.groupby('article_id').agg(
        repurchase=('t_dat', lambda x: 1 if len(x) > 1 else 0)
    ).reset_index()
    return product_repurchase_stats

# def get_higher_order_combinatorial_features(transactions):
#     next_purchase_prediction = transactions.groupby(['customer_id', 'article_id']).agg(
#         last_purchase_time=('t_dat', 'max')
#     ).reset_index()
#     next_purchase_prediction['next_purchase_time'] = next_purchase_prediction['last_purchase_time'] + pd.DateOffset(days=30)  # Assuming a 30-day purchase cycle for prediction
#     return next_purchase_prediction

def log(msg):
    print(f"[{datetime.now()}]: {msg}")

if __name__ == "__main__":
    out_base = "./hc_feat"
    os.makedirs(out_base)
    
    log("Loading Data...")
    articles = pd.read_csv("./dataset/articles.csv")
    customers = pd.read_csv("./dataset/customers.csv")
    transactions = pd.read_csv("./dataset/transactions_train.csv")

    log("Splitting Training Period...")
    # Convert the 't_dat' column to datetime format
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])

    # Sort transactions by date
    transactions.sort_values('t_dat', inplace=True)

    # Determine the range for the first three months and the next one month
    training_end_date = transactions['t_dat'].min() + pd.DateOffset(months=3)
    validation_end_date = training_end_date + pd.DateOffset(weeks=1)

    # Create training and validation datasets
    train_df = transactions[(transactions['t_dat'] >= transactions['t_dat'].min()) & (transactions['t_dat'] <= training_end_date)]
    validation_df = transactions[(transactions['t_dat'] > training_end_date) & (transactions['t_dat'] <= validation_end_date)]

    log("get_user_basic_features...")
    user_basic_features = get_user_basic_features(train_df)
    log("Saving get_user_basic_features...")
    user_basic_features.to_csv(os.path.join(out_base, "user_basic_features.csv"), index=False)

    print()

    log("get_product_basic_features...")
    product_basic_features = get_product_basic_features(train_df, customers=customers, articles=articles)
    log("Saving get_product_basic_features...")
    product_basic_features.to_csv(os.path.join(out_base, "product_basic_features.csv"), index=False)

    print()

    log("get_user_product_combination_features...")
    user_product_combination_features = get_user_product_combination_features(train_df)
    log("Saving get_user_product_combination_features...")
    user_product_combination_features.to_csv(os.path.join(out_base, "user_product_combination_features.csv"), index=False)

    print()

    log("get_age_product_combination_features...")
    age_product_combination_features = get_age_product_combination_features(train_df, customers=customers)
    log("Saiving get_age_product_combination_features...")
    age_product_combination_features.to_csv(os.path.join(out_base, "age_product_combination_features.csv"), index=False)

    print()

    log("get_user_repurchase_features...")
    user_repurchase_features = get_user_repurchase_features(train_df)
    log("Saving get_user_repurchase_features...")
    user_repurchase_features.to_csv(os.path.join(out_base, "user_repurchase_features.csv"), index=False)

    log("get_product_repurchase_features...")
    product_repurchase_features = get_product_repurchase_features(train_df)
    log("Saving get_product_repurchase_features...")
    product_repurchase_features.to_csv(os.path.join(out_base, "product_repurchase_features.csv"), index=False)