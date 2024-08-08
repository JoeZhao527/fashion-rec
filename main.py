# %%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import faiss
from matplotlib import pyplot as plt
from datetime import datetime

def log(msg):
    print(f"[{datetime.now()}]: {msg}")

log(f"Start importing modules")
from recommender.utils.data import (
    filter_transactions,
    filter_nan_age
)

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
    compute_metrics,
    compare_recommendations,
    visualize_compare_recommendations
)

# %% [markdown]
# # Resource Paths

# %%
# Pre-computed image and text feature vector for each item
# Download by `python scripts/download_emb.py`
image_feature_path = "./feature/dino_image_emb.npy"
text_feature_path = "./feature/glove_text_emb.npy"

# Content-based similarity cache directory
cb_cache_dir = "./feature"

# Items, users, and splited purchase history dataset paths
article_path = "./dataset/articles.csv"
customer_path = "./dataset/customers.csv"
fold_split_data_path = "./dataset/split"

# %% [markdown]
# # [1] Image & Text Content-Based Pre-Computations 
log(f"Content-Based Computation starts")

# %%
# Computing most similar items for each item, with image and text feature
content_based = ContentBased(
    image_feature_path=image_feature_path,
    text_feature_path=text_feature_path,
    feature_cache_dir=cb_cache_dir
)

# %% [markdown]
# # [2] Dataset Loading, Split and Filtering
log(f"Dataset Loading, Split and Filtering")
# %%
# Load item data
articles = pd.read_csv(article_path)

# Load user data
customers = pd.read_csv(customer_path)

# Load transcation data, splitted according to time
fold_data = {}
for _fold in os.listdir(fold_split_data_path):
    if "fold" in _fold:
        fold_data[_fold] = (
            pd.read_csv(os.path.join(fold_split_data_path, _fold, "train.csv")),
            pd.read_csv(os.path.join(fold_split_data_path, _fold, "test.csv"))
        )

# %%
import itertools

class Aggregate:
    """
    Aggregate results from each candidates generation (recall) pipeline
    """
    def __init__(self, recommend):
        self.recommend = recommend
        self.aggregate_recommend = {}

    def register_recall(self, customer_id, purchased, all_items, recall_pipes: dict):
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

        self.aggregate_recommend[customer_id] = res

    def aggregate(self, test_set, train, keys, top_n: int):
        purchase_dict = test_set.groupby('customer_id')['article_id'].agg(list)

        log(len(set(train['customer_id'])))
        test_users = set(train['customer_id']).intersection(list(purchase_dict.keys()))
        log(len(set(test_users)))
        for cid in tqdm(test_users, desc="Aggregrating recall results"):
            purchased = purchase_dict[cid]

            _res = {k: self.recommend[k][cid][:top_n] for k in keys}

            together_recall = set(itertools.chain(*_res.values()))
            
            self.register_recall(
                customer_id=cid,
                purchased=purchased,
                all_items=together_recall,
                recall_pipes=_res
            )

def average_score(agg: Aggregate, key_sets: dict):
    def agg_score_sort(item, keys):
        return sum([item[f"{k}_score"] for k in keys])

    key_sets_recommend = {}
    for k, keys in key_sets.items():
        key_sets_recommend[k] = {}
        for cid, candidates in tqdm(agg.aggregate_recommend.items(), desc=f"aggregating with {keys}"):
            key_sets_recommend[k][cid] = list(dict(sorted(candidates.items(), key=lambda x: agg_score_sort(x[1], keys), reverse=True)).keys())

    return key_sets_recommend

# %%
def prepare_recommendations(train: pd.DataFrame, test_set: pd.DataFrame, new_user_test: pd.DataFrame, recall_top_n=100):
    """
    The main function to make recommendations for each user.

    Recommend to existing users based on their purchase history, and new users based on their age and postal-code (region)
    """
    img_euclidean_recommend = content_based.recommend_items(train, media="image", dist="euclidean", N=recall_top_n)
    # img_cosine_recommend = content_based.recommend_items(train, media="image", dist="cosine", N=recall_top_n)
    txt_euclidean_recommend = content_based.recommend_items(train, media="text", dist="euclidean", N=recall_top_n)
    # txt_cosine_recommend = content_based.recommend_items(train, media="text", dist="cosine", N=recall_top_n)

    purchase_count = popularity_recall(train=train)

    popularity_all = {
        cid: purchase_count['article_id']
        for cid in train['customer_id'].unique()
    }

    product_code_res = product_code_recall(
        train=train,
        articles=articles,
        purchase_count=purchase_count
    )

    product_code_recommend = {
        cid: product_code_res[cid]['article_id']
        for cid in train['customer_id'].unique()
    }
        
    postal_code_res, customers_postal_code_map = postal_code_recall(
        train=train,
        customers=customers,
        purchase_count=purchase_count
    )

    postal_code_recommend = {
        cid: list(postal_code_res[customers_postal_code_map[cid]][:recall_top_n]['article_id'])
        for cid in train['customer_id'].unique()
    }

    age_group_res, customers_age_group_map = age_group_recall(
        train=train,
        customers=customers,
        purchase_count=purchase_count
    )

    age_group_recommend = {
        cid: list(age_group_res[customers_age_group_map[cid]][:recall_top_n]['article_id'])
        for cid in train['customer_id'].unique()
    }

    bought_together_res = bought_together_recall(train=train)
    bought_together_recommend = {
        cid: bought_together_res[cid]['article_id']
        for cid in train['customer_id'].unique()
    }
    
    user_cf_recommend = user_collaborative_recall(
        train=train,
        top_N=recall_top_n,
        model_cfg={
            'factors': 50,
            'alpha': 2.5,
            'iterations': 15,
            'random_state': 42
        }
    )

    item2vec = Item2VecModel(
        train=train,
        articles=articles,
        top_N=recall_top_n,
        model_cfg={
            "window": 9999,
            "seed": 42,
            "vector_size": 256,
            "epochs": 15,
            "negative": 5
        }
    )

    item2vec_cls_recommend = item2vec.cluster_recall()
    item2vec_sim_recommend = item2vec.similarity_recall()

    _recommend = {
        'img_euclidean': img_euclidean_recommend,
        # 'img_cosine': img_cosine_recommend,
        'txt_euclidean': txt_euclidean_recommend,
        # 'txt_cosine': txt_cosine_recommend,
        'popularity_all': popularity_all,
        'product_code': product_code_recommend,
        'postal_code': postal_code_recommend,
        'age_group': age_group_recommend,
        'bought_together': bought_together_recommend,
        'user_cf': user_cf_recommend,
        'item2vec_cls_res': item2vec_cls_recommend,
        'item2vec_sim_res': item2vec_sim_recommend,
    }

    agg = Aggregate(_recommend)
    agg.aggregate(
        test_set=test_set,
        train=train,
        keys=list(_recommend.keys()),
        top_n=recall_top_n
    )

    key_sets = {
        'Aggregation (1)': ['txt_euclidean', 'postal_code', 'item2vec_cls_res'],
        'Aggregation (2)': ['txt_euclidean', 'img_euclidean', 'product_code', 'postal_code', 'item2vec_cls_res', 'item2vec_sim_res'],
        'Aggregation (3)': list(_recommend.keys()),
    }

    key_sets_recommend = average_score(agg, key_sets)

    # Recommendations for new users
    new_user_postal_code_recommend = {}
    new_user_age_group_recommend = {}

    for cid in new_user_test['customer_id'].unique():
        _postal = customers_postal_code_map[cid]
        _age = customers_age_group_map[cid]
        if _postal in postal_code_res and _age in age_group_res:
            new_user_postal_code_recommend[cid] = list(age_group_res[_age][:recall_top_n]['article_id'])
            new_user_age_group_recommend[cid] = list(postal_code_res[_postal][:recall_top_n]['article_id'])

    exist_user_recommend = {
        'Image\n(Euclidean)': img_euclidean_recommend,
        # 'Image\n(Cosine)': img_cosine_recommend,
        'Text\n(Euclidean)': txt_euclidean_recommend,
        # 'Text\n(Cosine)': txt_cosine_recommend,
        'Popularity\n(All)': popularity_all,
        'Popularity\n(Product)': product_code_recommend,
        'Popularity\n(Region)': postal_code_recommend,
        'Popularity\n(Age)': age_group_recommend,
        'Popularity\n(Buy Together)': bought_together_recommend,
        'Collaborative\nFiltering': user_cf_recommend,
        'Item2Vec\n(Cluster Popular)': item2vec_cls_recommend,
        'Item2Vec\n(User-Item Similarity)': item2vec_sim_recommend,
        'Aggregation (1)': key_sets_recommend['Aggregation (1)'],
        'Aggregation (2)': key_sets_recommend['Aggregation (2)'],
        'Aggregation (3)': key_sets_recommend['Aggregation (3)']
    }
    new_user_recommend = {
        'Popularity\n(Region)': new_user_postal_code_recommend,
        'Popularity\n(Age)': new_user_age_group_recommend
    }

    return exist_user_recommend, new_user_recommend

# %% [markdown]
# ### Evaluation Functions

# %%
def collect_metrics(test_set, recommendations, top_n):
    results = {}
    for rec_name, recommendation in recommendations.items():
        results[rec_name] = compute_metrics(recommendation, test_set, top_n=top_n)
    return results

def fold_aggregate_metrics(fold_metrics):
    frames = [
        pd.DataFrame(fold_metrics[i]).transpose()
        for i in range(len(fold_metrics))
    ]  # Add your DataFrames here
    combined_df = pd.concat(frames)

    # Group by index (assuming index are the same and preserve across all dataframes)
    grouped = combined_df.groupby(combined_df.index)

    # Calculate mean and standard deviation
    mean_df = grouped.mean()
    std_df = grouped.std()

    mean_df = pd.DataFrame([mean_df[idx] for idx in combined_df.columns])
    std_df = pd.DataFrame([std_df[idx] for idx in combined_df.columns])

    return mean_df, std_df

# %% [markdown]
# ### Perform Recommendation and Aggregation for each folds
log(f"Perform Recommendation and Aggregation for each folds")
# %%
fold_exist_user_recommend = []
fold_new_user_recommend = []

for fold, _data in fold_data.items():
    log(f"=== Fold {fold} ===")
    # Preparing train and test data for each fold
    train_df, test_df = _data
    train_df = content_based.filter_content(train_df, articles)
    train_df = filter_nan_age(train_df, customers)
    test_df = filter_nan_age(test_df, customers)

    train, (loyal_user_test, regular_user_test, few_purchase_test, new_user_test) = filter_transactions(train_df, test_df, verbose=True)
    test = pd.concat([loyal_user_test, regular_user_test, few_purchase_test])

    # Recommend for existing users and new users
    exist_user_recommend, new_user_recommend = prepare_recommendations(
        train, test_set=test, new_user_test=new_user_test, recall_top_n=100
    )

    # Save recommendation results
    fold_exist_user_recommend.append(exist_user_recommend)
    fold_new_user_recommend.append(new_user_recommend)

log(f"Recommendation done, now saving recommendation results")
# %%
np.save("fold_new_user_recommend.npy", fold_new_user_recommend)
np.save("fold_exist_user_recommend.npy", fold_exist_user_recommend)

log(f"Start evaluation")

# %%
all_exist_user_test_metrics = []
loyal_user_test_metrics = []
regular_user_test_metrics = []
few_purchase_test_metrics = []
new_user_test_metrics = []

for i, (fold, _data) in enumerate(fold_data.items()):
    log(f"Fold {fold}: Computing metrics...")
    train_df, test_df = _data
    train_df = content_based.filter_content(train_df, articles)
    train_df = filter_nan_age(train_df, customers)
    test_df = filter_nan_age(test_df, customers)

    train, (loyal_user_test, regular_user_test, few_purchase_test, new_user_test) = filter_transactions(train_df, test_df, verbose=True)
    test = pd.concat([loyal_user_test, regular_user_test, few_purchase_test])
    
    tst1 = collect_metrics(test_set=test, recommendations=fold_exist_user_recommend[i], top_n=12)
    all_exist_user_test_metrics.append(pd.DataFrame(tst1).transpose())

    tst2 = collect_metrics(test_set=loyal_user_test, recommendations=fold_exist_user_recommend[i], top_n=12)
    loyal_user_test_metrics.append(pd.DataFrame(tst2).transpose())

    tst3 = collect_metrics(test_set=regular_user_test, recommendations=fold_exist_user_recommend[i], top_n=12)
    regular_user_test_metrics.append(pd.DataFrame(tst3).transpose())

    tst4 = collect_metrics(test_set=few_purchase_test, recommendations=fold_exist_user_recommend[i], top_n=12)
    few_purchase_test_metrics.append(pd.DataFrame(tst4).transpose())

    new_user_test = new_user_test[new_user_test['customer_id'].isin(
        list(fold_new_user_recommend[i]['Popularity\n(Region)'].keys())
    )]
    tst5 = collect_metrics(test_set=new_user_test, recommendations=fold_new_user_recommend[i], top_n=12)
    new_user_test_metrics.append(pd.DataFrame(tst5).transpose())

# %% [markdown]
# ### Aggregating metrics of each fold

log(f"Aggregating metrics of each fold")

# %%
all_user_mean, all_user_std = fold_aggregate_metrics(all_exist_user_test_metrics)
loyal_user_mean, loyal_user_std = fold_aggregate_metrics(loyal_user_test_metrics)
regular_user_mean, regular_user_std = fold_aggregate_metrics(regular_user_test_metrics)
few_purch_user_mean, few_purch_user_std = fold_aggregate_metrics(few_purchase_test_metrics)
new_user_mean, new_user_std = fold_aggregate_metrics(new_user_test_metrics)

all_user_mean.to_csv("./all_user_mean.csv")
all_user_std.to_csv("./all_user_std.csv")

loyal_user_mean.to_csv("./loyal_user_mean.csv")
loyal_user_std.to_csv("./loyal_user_std.csv")

regular_user_mean.to_csv("./regular_user_mean.csv")
regular_user_std.to_csv("./regular_user_std.csv")

few_purch_user_mean.to_csv("./few_purch_user_mean.csv")
few_purch_user_std.to_csv("./few_purch_user_std.csv")

new_user_mean.to_csv("./new_user_mean.csv")
new_user_std.to_csv("./new_user_std.csv")