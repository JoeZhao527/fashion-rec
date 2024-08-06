import pandas as pd
import os
import argparse
from datetime import datetime

def log(msg):
    print(f"[{datetime.now()}]: {msg}")

def split_data(trans_file, output_dir):
    log(f"Loading transaction data...")
    trans = pd.read_csv(trans_file)

    periods = [
        {
            "train": (datetime(2019, 1, 1), datetime(2019, 4, 1)),
            "test": (datetime(2019, 4, 1), datetime(2019, 4, 8))
        },
        {
            "train": (datetime(2019, 4, 1), datetime(2019, 7, 1)),
            "test": (datetime(2019, 7, 1), datetime(2019, 7, 8))
        },
        {
            "train": (datetime(2019, 7, 1), datetime(2019, 10, 1)),
            "test": (datetime(2019, 10, 1), datetime(2019, 10, 8))
        },
        {
            "train": (datetime(2019, 10, 1), datetime(2020, 1, 1)),
            "test": (datetime(2020, 1, 1), datetime(2020, 1, 8))
        }
    ]
    trans['t_dat'] = pd.to_datetime(trans['t_dat'])

    log(f"Splitting according to time...")
    for i, p in enumerate(periods):
        train = trans[(trans['t_dat'] >= p['train'][0]) & (trans['t_dat'] < p['train'][1])]
        test = trans[(trans['t_dat'] >= p['test'][0]) & (trans['t_dat'] < p['test'][1])]
        log(f"Fold {i}: train size: {len(train)}, test size: {len(test)}")

        fold_path = os.path.join(output_dir, f"fold_{i}")
        os.makedirs(fold_path, exist_ok=True)

        train.to_csv(os.path.join(fold_path, "train.csv"), index=False)
        test.to_csv(os.path.join(fold_path, "test.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split transaction data into training and testing sets based on time periods.")
    parser.add_argument('--trans_file', type=str, default="./dataset/transactions_train.csv", help="Path to the transaction data CSV file.")
    parser.add_argument('--output_dir', type=str, default="./dataset/split", help="Directory to save the split data.")
    args = parser.parse_args()

    split_data(args.trans_file, args.output_dir)


# import pandas as pd
# import os
# from datetime import datetime

# def log(msg):
#     print(f"[{datetime.now()}]: {msg}")

# if __name__ == "__main__":
#     log(f"Loading transaction data...")
#     trans = pd.read_csv("./dataset/transactions_train.csv")
#     output_dir = "./dataset/split"

#     periods = [
#         {
#             "train": (datetime(2019, 1, 1), datetime(2019, 4, 1)),
#             "test": (datetime(2019, 4, 1), datetime(2019, 4, 15))
#         },
#         {
#             "train": (datetime(2019, 4, 1), datetime(2019, 7, 1)),
#             "test": (datetime(2019, 7, 1), datetime(2019, 7, 15))
#         },
#         {
#             "train": (datetime(2019, 7, 1), datetime(2019, 10, 1)),
#             "test": (datetime(2019, 10, 1), datetime(2019, 10, 15))
#         },
#         {
#             "train": (datetime(2019, 10, 1), datetime(2020, 1, 1)),
#             "test": (datetime(2020, 1, 1), datetime(2020, 1, 15))
#         }
#     ]
#     trans['t_dat'] = pd.to_datetime(trans['t_dat'])

#     log(f"Splitting according to time...")
#     for i, p in enumerate(periods):
#         train = trans[(trans['t_dat'] >= p['train'][0]) & (trans['t_dat'] < p['train'][1])]
#         test = trans[(trans['t_dat'] >= p['test'][0]) & (trans['t_dat'] < p['test'][1])]
#         print(f"train size: {len(train)}, test size: {len(test)}")

#         fold_path = os.path.join(output_dir, f"fold_{i}")
#         os.makedirs(fold_path, exist_ok=True)

#         train.to_csv(os.path.join(fold_path, "train.csv"), index=False)
#         test.to_csv(os.path.join(fold_path, "test.csv"), index=False)