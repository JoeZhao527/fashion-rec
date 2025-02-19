import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from scipy.sparse import coo_matrix
import implicit
import warnings
warnings.filterwarnings("ignore")


def user_collaborative_recall(train: pd.DataFrame, top_N: int, model_cfg: dict, *args, **kwargs):
    print(f"Start matrix factorization:")
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
    model = implicit.als.AlternatingLeastSquares(**model_cfg)

    # Train the model on the sparse matrix
    model.fit(sparse_matrix_csr)

    # Recommend for each user
    ids, score = model.recommend([i for i in range(sparse_matrix_csr.shape[0])], sparse_matrix_csr, N=top_N)

    item_inverse_map = {v: k for k, v in item_mapping.items()}
    func = lambda x: item_inverse_map[x]

    recommendations = pd.Series(
        index=list(user_mapping.keys()),
        data=list(np.vectorize(func)(ids))
    )

    return recommendations