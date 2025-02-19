from tqdm import tqdm
import os
import numpy as np

if __name__ == '__main__':
    feat_dir = "./feature/dino_img_emb"
    out_path = "./feature/dino_image_emb.npy"
    image_feature = {}

    for item_file in tqdm(os.listdir(feat_dir)):
        feature_file = os.path.join(feat_dir, f"{item_file}")

        item_id = int(item_file.split('.')[0])
        # 0 for class token, 1 for average of all patch tokens
        image_feature[item_id] = np.load(feature_file)[0]

    np.save(out_path, image_feature)