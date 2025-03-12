import pickle
import numpy as np
import pandas as pd
from official_data_loader import AudioDataLoader, AudioDataset

if __name__ == "__main__":
    df = pd.read_pickle('/projects/dsci410_510/Kolahi_data_temp/expanded_dataset_v11.pkl')
    data_loader = AudioDataLoader(data_path='/projects/dsci410_510/Kolahi_data_temp/expanded_dataset_v11.pkl')
    train_loader, val_loader, test_loader = data_loader.create_train_val_test_split()
    country_to_idx = {country: idx for idx, country in enumerate(sorted(df['country'].unique()))}

    print(country_to_idx)
    print(train_loader)
