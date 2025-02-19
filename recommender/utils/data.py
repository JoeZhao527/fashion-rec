import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
import random


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
    train_idx = set(flit_train[flit_train > thr].index)

    flit_test = all_test['customer_id'].value_counts()
    test_idx = set(flit_test[flit_test > thr].index)

    # test_idx = set(all_test['customer_id'].unique())
    
    filt_idx = train_idx.intersection(test_idx)

    # random.seed(42)
    # cold_start_idx = random.sample(list(train_idx - test_idx), k=len(filt_idx))

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


# def filter_transactions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
#     # Calculate the number of purchases per user in both datasets
#     train_counts = train_df['customer_id'].value_counts()
#     test_counts = test_df['customer_id'].value_counts()

#     # Apply filters based on conditions

#     # Condition 1: Users purchased more than 30 items in training set and more than 10 in testing set
#     condition1_train = train_df[train_df['customer_id'].isin(train_counts[train_counts > 30].index) & train_df['customer_id'].isin(test_counts[test_counts > 10].index)]
#     condition1_test = test_df[test_df['customer_id'].isin(train_counts[train_counts > 30].index) & test_df['customer_id'].isin(test_counts[test_counts > 10].index)]
#     print_condition_summary("Condition 1", condition1_train, condition1_test, "Users purchased more than 30 items in training set and more than 10 in testing set")

#     # Condition 2: Users purchased > 5 items and < 20 items in training set and more than 10 in testing set
#     condition2_train = train_df[train_df['customer_id'].isin(train_counts[(train_counts > 5) & (train_counts < 20)].index) & train_df['customer_id'].isin(test_counts[test_counts > 10].index)]
#     condition2_test = test_df[test_df['customer_id'].isin(train_counts[(train_counts > 5) & (train_counts < 20)].index) & test_df['customer_id'].isin(test_counts[test_counts > 10].index)]
#     print_condition_summary("Condition 2", condition2_train, condition2_test, "Users purchased more than 5 items but less than 20 items in training set and more than 10 in testing set")

#     # Condition 3: Users did not make any purchase in training set, but purchased more than 10 items in testing set
#     condition3_users = test_counts[(test_counts > 10) & ~test_counts.index.isin(train_counts.index)]
#     condition3_test = test_df[test_df['customer_id'].isin(condition3_users.index)]
#     condition3_train = pd.DataFrame(columns=train_df.columns)  # Empty DataFrame as no transactions in training set for these users.
#     print_condition_summary("Condition 3", condition3_train, condition3_test, "Users did not make any purchase in training set, but purchased more than 10 items in testing set")

#     # Concatenate the results for the train DataFrame
#     final_train_df = pd.concat([condition1_train, condition2_train, condition3_train]).drop_duplicates()
#     final_test_df = pd.concat([condition1_test, condition2_test, condition3_test]).drop_duplicates()

#     # Print aggregated information and return
#     print_aggregated_info(final_train_df, final_test_df)
#     return final_train_df, (condition1_test, condition2_test, condition3_test)

def filter_transactions(train_df: pd.DataFrame, test_df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    # Calculate the number of purchases per user in both datasets
    train_counts = train_df['customer_id'].value_counts()
    test_counts = test_df['customer_id'].value_counts()

    # Apply filters based on conditions
    # Condition 1
    condition1_train = train_df[train_df['customer_id'].isin(train_counts[train_counts > 30].index) & train_df['customer_id'].isin(test_counts[test_counts > 10].index)]
    condition1_test = test_df[test_df['customer_id'].isin(train_counts[train_counts > 30].index) & test_df['customer_id'].isin(test_counts[test_counts > 10].index)]

    # Condition 2
    condition2_train = train_df[train_df['customer_id'].isin(train_counts[(train_counts > 5) & (train_counts < 20)].index) & train_df['customer_id'].isin(test_counts[test_counts > 10].index)]
    condition2_test = test_df[test_df['customer_id'].isin(train_counts[(train_counts > 5) & (train_counts < 20)].index) & test_df['customer_id'].isin(test_counts[test_counts > 10].index)]

    # Condition 3
    condition3_train = train_df[train_df['customer_id'].isin(train_counts[train_counts < 5].index) & train_df['customer_id'].isin(test_counts[test_counts > 10].index)]
    condition3_test = test_df[test_df['customer_id'].isin(train_counts[train_counts < 5].index) & test_df['customer_id'].isin(test_counts[test_counts > 10].index)]

    # Condition 4
    condition4_users = test_counts[(test_counts > 10) & ~test_counts.index.isin(train_counts.index)]
    condition4_test = test_df[test_df['customer_id'].isin(condition4_users.index)]
    condition4_train = pd.DataFrame(columns=train_df.columns)  # Empty DataFrame as no transactions in training set for these users.

    # Concatenate results for the final DataFrame
    final_train_df = pd.concat([condition1_train, condition2_train, condition3_train, condition4_train]).drop_duplicates()
    final_test_df = pd.concat([condition1_test, condition2_test, condition3_test, condition4_test]).drop_duplicates()

    if verbose:
        print_condition_summary("Condition 1", condition1_train, condition1_test, "Users purchased more than 30 items in training set and more than 10 in testing set")
        print_condition_summary("Condition 2", condition2_train, condition2_test, "Users purchased more than 5 items but less than 20 items in training set and more than 10 in testing set")
        print_condition_summary("Condition 3", condition3_train, condition3_test, "Users made less than 5 purchases in training set but more than 10 purchases in testing set")
        print_condition_summary("Condition 4", condition4_train, condition4_test, "Users did not make any purchase in training set, but purchased more than 10 items in testing set")

        print_aggregated_info(final_train_df, final_test_df)
    
    return final_train_df, (condition1_test, condition2_test, condition3_test, condition4_test)


def print_condition_summary(condition_name, train_df, test_df, description):
    print(f"{condition_name} - Description: {description}")
    print(f"{condition_name} - Train: {train_df['customer_id'].nunique()} users, {len(train_df)} transactions, {train_df['article_id'].nunique()} articles")
    print(f"{condition_name} - Test: {test_df['customer_id'].nunique()} users, {len(test_df)} transactions, {test_df['article_id'].nunique()} articles")
    print()

def print_aggregated_info(train_df, test_df):
    print("Aggregated Info")
    print(f"Train: {train_df['customer_id'].nunique()} users, {len(train_df)} transactions, {train_df['article_id'].nunique()} articles")
    print(f"Test: {test_df['customer_id'].nunique()} users, {len(test_df)} transactions, {test_df['article_id'].nunique()} articles")


def filter_nan_age(df: pd.DataFrame, customers: pd.DataFrame):
    nan_age_customers = customers[~customers['age'].isna()]['customer_id']

    return df[df['customer_id'].isin(nan_age_customers)]