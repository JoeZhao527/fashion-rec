import pandas as pd
from typing import List
from tqdm import tqdm


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


def recall_select(trn, target_cols: List[int]):
    # Create a dictionary to store the new dataframes
    new_dataframes = {cid: [] for cid in trn['customer_id'].unique()}

    for col in tqdm(target_cols, desc="selecting recall pipeline"):
        # Step 2: Create a new dataframe for each selected column where col == 1
        new_df = trn[trn[col] == 1][[col, f"{col}_score", "purchased", "customer_id", "article_id"]]
        
        # Store the new dataframe in the dictionary
        for cid, _df in new_df.groupby("customer_id"):
            new_dataframes[cid].append(_df.reset_index(drop=True))
    
    score_cols = [f"{col}_score" for col in target_cols]
    for cid in tqdm(new_dataframes, desc="Sorting by average recall score"):
        new_df = pd.concat(new_dataframes[cid]).sort_values(f"{col}_score", ascending=False).fillna(0.0)
        new_df['agg_score'] = new_df.apply(lambda x: sum(x[sc] for sc in score_cols), axis=1)
        new_dataframes[cid] = new_df.sort_values("agg_score", ascending=False)

    return new_dataframes

def cold_start_agg(user_groups):
    dataframes = []

    # Iterate through the dictionary
    for group, df in user_groups.items():
        # Add the 'group' column to the DataFrame
        df['group'] = group
        # Drop duplicates based on 'article_id'
        df = df.drop_duplicates(subset=['article_id'])
        # Append the DataFrame to the list
        dataframes.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    return pd.concat(dataframes, ignore_index=True)