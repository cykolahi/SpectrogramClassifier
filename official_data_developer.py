import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import pickle
#import h5py

class AudioDataDeveloper():
    def __init__(self, data_path):
        """
        Initialize the AudioDataDeveloper.
        
        Args:
            data_path (str): Path to CSV file containing audio paths and country labels
        """
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)
        self.expanded_df = None  # Initialize as None
        self.min_samples_per_country = 0

        if self.df is None or len(self.df) == 0:
            raise ValueError(f"Failed to load data from {data_path} or file is empty")
        
        # Don't create expanded dataset in __init__
        # Let the user call it explicitly when needed
        
    def load_data(self):
        """
        Load and preprocess the dataset.
        
        Args:
            min_samples_per_country (int): Minimum number of samples required per country
        """
        # Load the dataset
        self.df = pd.read_csv(self.data_path)
        
        # Remove rows with invalid countries
        self.df = self.df[self.df['country'] != 'Error']
        self.df = self.df[self.df['country'] != 'Unknown']
        
        # Filter countries with minimum samples
        country_counts = self.df['country'].value_counts()
        self.min_samples_per_country = country_counts[2]
        print(f"min_samples_per_country: {self.min_samples_per_country}")
        countries_to_keep = country_counts[country_counts >= self.min_samples_per_country].index
        self.df = self.df[self.df['country'].isin(countries_to_keep)]
        #print(self.df['country'].value_counts())
        print(f"len(self.df): {len(self.df)}")
        print(f"len(self.df['country'].unique()): {len(self.df['country'].unique())}")
        return self.df
    
    def balance_dataset(self, random_state=42):
        """
        Balance the dataset by downsampling majority classes.
        
        Args:
            n_samples_per_country (int): Target number of samples per country
            random_state (int): Random seed for reproducibility
        """
        balanced_df = pd.DataFrame()
        for country in self.df['country'].unique():
            country_data = self.df[self.df['country'] == country]
            if len(country_data) >= self.min_samples_per_country:
                country_data = country_data.sample(n=self.min_samples_per_country, random_state=random_state)
            balanced_df = pd.concat([balanced_df, country_data])
            
        self.df = balanced_df.reset_index(drop=True)

        print(self.df['country'].value_counts())
        print(f"len(self.df): {len(self.df)}")
        print(f"len(self.df['country'].unique()): {len(self.df['country'].unique())}")

        return self.df
    
    def process_audio_file(self, file_path):
        """
        Process audio file into spectrograms.
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
           list of segments
        """
        x, sr = librosa.load(file_path, sr=None, mono=True)
        segment_length = 9 * sr
        segments = []
        
        num_segments = int(len(x) // segment_length)
        
        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = x[start:end]
            stft = np.abs(librosa.stft(segment, n_fft=2048, hop_length=512))
            mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
            log_mel = librosa.power_to_db(mel, ref=np.max)
            # Ensure spectrogram shape is (128, 768) by truncating if needed
            if log_mel.shape[1] > 768:
                log_mel = log_mel[:, :768]
            elif log_mel.shape[1] < 768:
                # Pad with zeros if too short (shouldn't happen with 9 second segments)
                pad_width = ((0, 0), (0, 768 - log_mel.shape[1]))
                log_mel = np.pad(log_mel, pad_width, mode='constant')
            segments.append(log_mel)

        
        return segments
    
    def create_expanded_dataset(self):
        """
        Create expanded dataset with audio segments and spectrograms.
        """
        expanded_data = []
        if self.df is None or len(self.df) == 0:
            raise ValueError(f"Failed to load data from {self.data_path} or file is empty")
        
        for idx, row in self.df.iterrows():
            try:
                segments = self.process_audio_file(row['audio_path'])
                for i, segment in enumerate(segments):
                    expanded_data.append({
                        'audio_path': row['audio_path'],
                        'country': row['country'],
                        'segment_number': i,
                        'spectrogram': segment
                    })
            except Exception as e:
                print(f"Error processing {row['audio_path']}: {str(e)}")
                
        self.expanded_df = pd.DataFrame(expanded_data)
        return self.expanded_df
    
    def save_dataset(self, save_path, format='pkl'):
        self.expanded_df.to_pickle(save_path)

    def examine_dataset(self):
        print(self.expanded_df.head())
        print(self.expanded_df['country'].value_counts())

    def analyze_spectrogram_shapes(self):
        """
        Analyze and print the shapes of spectrograms in the dataset
        """
        if self.expanded_df is None or 'spectrogram' not in self.expanded_df.columns:
            print("No spectrogram data available to analyze")
            return
            
        # Get shapes of all spectrograms
        shapes_and_countries = [(np.shape(spec), country) for spec, country in zip(self.expanded_df['spectrogram'], self.expanded_df['country'])]
        shapes = [shape for shape, _ in shapes_and_countries]

        # Count unique shapes
        shape_counts = {}
        for shape, country in shapes_and_countries:
            if shape == (128, 844):
                print(f"Shape {shape} found for country {country}")
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            
        print("\nSpectrogram shape analysis:")
        print("---------------------------")
        print("Shape counts:")
        for shape, count in shape_counts.items():
            print(f"Shape {shape}: {count} spectrograms")

        print(f"\nTotal spectrograms analyzed: {len(shapes)}")
        

def main():
    data_developer = AudioDataDeveloper(data_path='/projects/dsci410_510/Kolahi_data_temp/temp_tracks_with_countries.csv')
    data_developer.load_data()
    data_developer.balance_dataset()
    data_developer.create_expanded_dataset()
    data_developer.save_dataset('/projects/dsci410_510/Kolahi_data_temp/expanded_dataset_v10.pkl')

    with open('/projects/dsci410_510/Kolahi_data_temp/expanded_dataset_v10.pkl', 'rb') as f:
        expanded_df = pickle.load(f)
    data_developer.examine_dataset()
    data_developer.analyze_spectrogram_shapes()
if __name__ == "__main__":
    main()
    #data_developer.save_dataset('/projects/dsci410_510/Kolahi_dataset/expanded_dataset_v1.pkl')

    
