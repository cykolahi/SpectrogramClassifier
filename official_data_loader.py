import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
#from torch.utils import DataLoader
#import h5py

class AudioDataset(Dataset):
    """Custom Dataset class for audio spectrograms"""
    def __init__(self, df):
        self.df = df
        self.country_to_idx = {country: idx for idx, country in enumerate(sorted(df['country'].unique()))}
        
        # Calculate dataset statistics
        all_specs = np.stack(df['spectrogram'].values)
        self.global_mean = np.mean(all_specs)
        self.global_std = np.std(all_specs)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get spectrogram and convert to tensor
        spectrogram = torch.FloatTensor(self.df.iloc[idx]['spectrogram'])
        
        # Use global normalization
        spectrogram = (spectrogram - self.global_mean) / (self.global_std + 1e-6)
        
        # Add channel dimension
        spectrogram = spectrogram.unsqueeze(0)  # Shape becomes (1, 128, 768)
        
        # Get label
        country = self.df.iloc[idx]['country']
        label = torch.tensor(self.country_to_idx[country])
        
        return spectrogram, label

class AudioDataLoader():
    def __init__(self, data_path):
        """
        Initialize the AudioDataLoader.
        
        Args:
            data_path (str): Path to pickle file containing expanded dataset
        """
        self.data_path = data_path
        # Load pickle file instead of CSV
        with open(data_path, 'rb') as f:
            self.expanded_df = pd.read_pickle(f)

        if self.expanded_df is None or len(self.expanded_df) == 0:
            raise ValueError(f"Failed to load data from {data_path} or file is empty")
        
        # Don't create expanded dataset in __init__
        # Let the user call it explicitly when needed
    
    
    def create_train_val_test_split(self, test_size=0.2, random_state=42, batch_size=16):
        """
        Create stratified train/test splits and return DataLoaders.
        
        Args:
            test_size (float): Proportion of dataset for test split
            random_state (int): Random seed
            batch_size (int): Batch size for DataLoader
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        train_df, temp_test_df = train_test_split(
            self.expanded_df,
            test_size=0.3,
            random_state=random_state,
            stratify=self.expanded_df['country']
        )
        val_df, test_df = train_test_split(
            temp_test_df,
            test_size=0.5,
            random_state=random_state,
            stratify=temp_test_df['country']
        )   

        # Create Dataset objects
        train_dataset = AudioDataset(train_df)
        val_dataset = AudioDataset(val_df)
        test_dataset = AudioDataset(test_df)

        # Create DataLoader objects
        train_loader = DataLoader(
            train_dataset, 
            batch_size=16, 
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=1,
            pin_memory=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=1,
            pin_memory=False
        )
        
        print(f"\nDataset split complete:")
        print(f"Training set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        
        return train_loader, val_loader, test_loader

def save_dataset(data, save_path, format='pkl'):
        """
        Save the expanded dataset.
        
        Args:
            save_path (str): Path to save the dataset
            format (str): Format to save in ('pkl' or 'h5')
        """
        if isinstance(data, DataLoader):
            df = data.dataset.df
        else:
            df = data
        if format == 'pkl':
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'audio_paths': df['audio_path'].values,
                    'countries': df['country'].values,
                    'segment_numbers': df['segment_number'].values,
                    'spectrograms': [s for s in df['spectrogram'].values]
                }, f)
        
        elif format == 'h5':
            # Get shapes for spectrograms only
            spec_shapes = [s.shape for s in df['spectrogram'].values]
            max_spec_shape = tuple(max(dim) for dim in zip(*spec_shapes))
            
            # Create padded array for spectrograms
            specs_padded = np.zeros((len(df), *max_spec_shape))
            
            # Fill padded array
            for i, spec in enumerate(df['spectrogram'].values):
                specs_padded[i, :spec.shape[0], :spec.shape[1]] = spec
                
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('audio_paths', data=df['audio_path'].values.astype('S'))
                f.create_dataset('countries', data=df['country'].values.astype('S'))
                f.create_dataset('segment_numbers', data=df['segment_number'].values)
                f.create_dataset('spectrograms', data=specs_padded)


def main():
    data_loader = AudioDataLoader(data_path='/projects/dsci410_510/Kolahi_data_temp/expanded_dataset_v9.pkl')
    train_loader, val_loader, test_loader = data_loader.create_train_val_test_split()
    save_dataset(train_loader, '/projects/dsci410_510/Kolahi_data_temp/train_dataset.pkl')
    save_dataset(val_loader, '/projects/dsci410_510/Kolahi_data_temp/val_dataset.pkl')
    save_dataset(test_loader, '/projects/dsci410_510/Kolahi_data_temp/test_dataset.pkl')

if __name__ == "__main__":
    main()