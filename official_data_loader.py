import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle
#from torch.utils import DataLoader
#import h5py

class AudioDataset(Dataset):
    """Custom Dataset class for audio spectrograms"""
    def __init__(self, df, transform=False):
        self.df = df
        self.transform = transform
        self.country_to_idx = {country: idx for idx, country in enumerate(sorted(df['country'].unique()))}
        
          # Calculate dataset statistics
        all_specs = np.stack(df['spectrogram'].values)
        self.global_mean = np.mean(all_specs)
        self.global_std = np.std(all_specs)

        # Calculate per-country statistics
        #self.country_stats = {}
        #for country in self.country_to_idx.keys():
        #    country_specs = np.stack(df[df['country'] == country]['spectrogram'].values)
         #   self.country_stats[country] = {
         #       'mean': np.mean(country_specs),
         #       'std': np.std(country_specs) + 1e-6
         #   }

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        spectrogram = torch.FloatTensor(self.df.iloc[idx]['spectrogram'])
        country = self.df.iloc[idx]['country']
        
       
        
        # Normalize using global statistics
        spectrogram = (spectrogram - self.global_mean) / (self.global_std + 1e-6)

        #spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min()) * 2 - 1

        # Data augmentation during training
        if self.transform:
            # Random time masking
            if torch.rand(1) < 0.5:
                time_mask_param = int(spectrogram.shape[1] * 0.1)
                # Librosa doesn't have a direct time masking equivalent, so implement it manually
                mask_start = np.random.randint(0, spectrogram.shape[1] - time_mask_param)
                spectrogram[:, mask_start:mask_start + time_mask_param] = 0
            
            # Random frequency masking
            if torch.rand(1) < 0.5:
                freq_mask_param = int(spectrogram.shape[0] * 0.1)
                # Manually implement frequency masking
                mask_start = np.random.randint(0, spectrogram.shape[0] - freq_mask_param)
                spectrogram[mask_start:mask_start + freq_mask_param, :] = 0
        spectrogram = spectrogram.unsqueeze(0)
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
    
    
    def create_train_val_test_split(self, random_state=42, batch_size=48):
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
        train_dataset = AudioDataset(train_df, transform=True)
        val_dataset = AudioDataset(val_df, transform=True)
        test_dataset = AudioDataset(test_df, transform=True)

        """
        for i in range(len(train_dataset)):
            # Get and process each sample using AudioDataset's __getitem__
            spectrogram, label = train_dataset[i]
            
            # Verify the sample was processed correctly
            if spectrogram is None or label is None:
                print(f"Warning: Sample {i} failed to load")
                continue
                
            # Store processed samples back in dataset
            train_dataset[i] = spectrogram
            train_dataset[i] = label

        for i in range(len(val_dataset)):
            spectrogram, label = val_dataset[i]
            if spectrogram is None or label is None:
                print(f"Warning: Sample {i} failed to load")
                continue
            val_dataset.spectrograms[i] = spectrogram
            val_dataset.labels[i] = label

        for i in range(len(test_dataset)):
            spectrogram, label = test_dataset[i]
            if spectrogram is None or label is None:
                print(f"Warning: Sample {i} failed to load")
                continue
            test_dataset.spectrograms[i] = spectrogram
            test_dataset.labels[i] = label
        """
        # Create DataLoader objects
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=3,
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=3,
            pin_memory=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=3,
            pin_memory=False
        )
        
        print(f"\nDataset split complete:")
        print(f"Training set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        
        return train_loader, val_loader, test_loader

    def create_train_val_split(self, exclude_indices, random_state=42, batch_size=16):
        """
        Create a train/val split from the expanded dataset, excluding specified indices.
    
        Args:
            exclude_indices (list): List of indices (correlate with data from initial test split)to exclude from the split
            random_state (int): Random seed
            batch_size (int): Batch size for DataLoader
        
        Returns:
            tuple: (train_loader, val_loader)
        """
        
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


#def main():
#    data_loader = AudioDataLoader(data_path='/projects/dsci410_510/Kolahi_data_temp/expanded_dataset_v11.pkl')
#    train_loader, val_loader, test_loader = data_loader.create_train_val_test_split()
#    save_dataset(train_loader, '/projects/dsci410_510/Kolahi_data_temp/train_dataset.pkl')
  #  save_dataset(val_loader, '/projects/dsci410_510/Kolahi_data_temp/val_dataset.pkl')
    #save_dataset(test_loader, '/projects/dsci410_510/Kolahi_data_temp/test_dataset.pkl')

#if __name__ == "__main__":
#    main()