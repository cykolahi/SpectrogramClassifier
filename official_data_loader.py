import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import pickle
from torch.utils import DataLoader
#import h5py

class AudioDataLoader():
    def __init__(self, data_path):
        """
        Initialize the AudioDataLoader.
        
        Args:
            data_path (str): Path to CSV file containing audio paths and country labels
        """
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)
        self.expanded_df = None  # Initialize as None

        if self.df is None or len(self.df) == 0:
            raise ValueError(f"Failed to load data from {data_path} or file is empty")
        
        # Don't create expanded dataset in __init__
        # Let the user call it explicitly when needed
    
    
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """
        Create stratified train/test splits.
        
        Args:
            test_size (float): Proportion of dataset for test split
            random_state (int): Random seed
            
        Returns:
            tuple: (train_df, test_df)
        """
        train_df, test_df = train_test_split(
            self.expanded_df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.expanded_df['country']
        )
        
        print(f"\nDataset split complete:")
        print(f"Training set: {len(train_df)} samples")
        print(f"Test set: {len(test_df)} samples")

        #print(train_df['country'].value_counts())
        print(f"len(train_df): {len(train_df)}")
        print(f"len(train_df['country'].unique()): {len(train_df['country'].unique())}")
        
        print("\nCountry distribution:")
        print("\nTraining set:")
        print(train_df['country'].value_counts())
        print("\nTest set:")
        print(test_df['country'].value_counts())
        
        return train_df, test_df
    
    def save_dataset(self, save_path, format='pkl'):
        """
        Save the expanded dataset.
        
        Args:
            save_path (str): Path to save the dataset
            format (str): Format to save in ('pkl' or 'h5')
        """
        if format == 'pkl':
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'audio_paths': self.expanded_df['audio_path'].values,
                    'countries': self.expanded_df['country'].values,
                    'segment_numbers': self.expanded_df['segment_number'].values,
                    'spectrograms': [s for s in self.expanded_df['spectrogram'].values]
                }, f)
        
        elif format == 'h5':
            # Get shapes for spectrograms only
            spec_shapes = [s.shape for s in self.expanded_df['spectrogram'].values]
            max_spec_shape = tuple(max(dim) for dim in zip(*spec_shapes))
            
            # Create padded array for spectrograms
            specs_padded = np.zeros((len(self.expanded_df), *max_spec_shape))
            
            # Fill padded array
            for i, spec in enumerate(self.expanded_df['spectrogram'].values):
                specs_padded[i, :spec.shape[0], :spec.shape[1]] = spec
                
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('audio_paths', data=self.expanded_df['audio_path'].values.astype('S'))
                f.create_dataset('countries', data=self.expanded_df['country'].values.astype('S'))
                f.create_dataset('segment_numbers', data=self.expanded_df['segment_number'].values)
                f.create_dataset('spectrograms', data=specs_padded)


def main():
    data_loader = AudioDataLoader(data_path='~/projects/dsci410_510/Kolahi_dataset/Kolahi_data')
    data_loader.create_train_test_split()
    data_loader.save_dataset('~/projects/dsci410_510/Kolahi_dataset/countries_w_spectrograms.pkl')

if __name__ == "__main__":
    main()