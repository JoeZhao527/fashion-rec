import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def popularity_recall(train: pd.DataFrame, *args, **kwargs):
    """
    Recall the most popular items in the training period
    """
    counts = train['article_id'].value_counts()
    purchase_count = pd.DataFrame(counts).reset_index()

    return purchase_count

def product_code_recall(train: pd.DataFrame, articles: pd.DataFrame, purchase_count: pd.DataFrame, *args, **kwargs):
    prod_code_trn = pd.merge(train, articles[['article_id', 'product_code']])

    recent_purchase = []

    for cid, group in tqdm(prod_code_trn.groupby("customer_id"), desc="product_code recall"):
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
        _group = group.sort_values("count", ascending=False)[['article_id', 'count']].drop_duplicates("article_id")
        trn_postal_group[postal_code] = _group

    customers_postal_code_map = pd.Series(data=list(customers['postal_code']), index=customers['customer_id']).to_dict()

    return trn_postal_group, customers_postal_code_map


def age_group_recall(train: pd.DataFrame, customers: pd.DataFrame, purchase_count: pd.DataFrame, *args, **kwargs):
    # Step 1: Add an age group tag for customers according to customer's `age`, with age interval size 5
    customers['age_group'] = (customers['age'] // 5) * 5
    
    # Step 2: Merge train data with customers' age group
    trn_age_group = pd.merge(train, customers[['customer_id', 'age_group']], on=['customer_id'])

    # Step 3: Merge the result with purchase count
    pop_trn_age_group = pd.merge(trn_age_group, purchase_count, on=['article_id'])

    # Step 4: Group by age group and sort by count
    trn_age_group_group = {}
    for age_group, group in pop_trn_age_group.groupby("age_group"):
        _group = group.sort_values("count", ascending=False)[['article_id', 'count']].drop_duplicates("article_id")
        trn_age_group_group[age_group] = _group

    # Create a map of customers to their age group
    customers_age_group_map = pd.Series(data=list(customers['age_group']), index=customers['customer_id']).to_dict()

    return trn_age_group_group, customers_age_group_map


class ArticlePairs:
    """
    Count how many times that a pair of a items are bought together

    If user bought item `A`, select the most frequently bought items with item `A`, and recommend the most
    popular items within the selected items
    """
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
            top_n_counts = bought_together_counts[top_n_indices]
            
            # Create a dataframe for the result
            result_df = pd.DataFrame({
                'article_id': [self.idx_to_article[i] for i in top_n_indices],
                'num_together': top_n_counts
            })

            self.top_n_cache[article_id] = result_df
        
        return self.top_n_cache[article_id].copy()


def bought_together_recall(train: pd.DataFrame):
    """
    Count how many times that a pair of a items are bought together

    If user bought item `A`, select the most frequently bought items with item `A`, and recommend the most
    popular items within the selected items
    """
    ap = ArticlePairs(train)

    also_bought_res = {}

    for cid, group in tqdm(train.groupby("customer_id"), desc="bought together recall"):
        ap_res = []
        for aid in list(group['article_id'].unique()):
            ap_res.append(ap.get_top_n_bought_together(aid, 50))
        also_bought_res[cid] = pd.concat(ap_res)

    return also_bought_res


