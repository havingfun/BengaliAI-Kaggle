import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import glob


if __name__ == "__main__":
    files = glob.glob("../data/train_*.parquet")
    for f in files:
        print(f)
        image_df = pd.read_parquet(f)
        image_ids = image_df.image_id.values
        image_df = image_df.drop('image_id', axis=1)
        image_array = image_df.values
        for j, image_id in tqdm(enumerate(image_ids), total = len(image_ids)):
            joblib.dump(image_array[j, :], f'../data/images/{image_id}.pkl')

    