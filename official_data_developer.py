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
        print("initial country counts:", country_counts)
        self.min_samples_per_country = country_counts.iloc[4]
        # Get countries that have at least min_samples, and take min_samples + 100 if possible
        self.min_samples_each_country = []
        for country in country_counts.index:
            country_count = country_counts[country]
            adjusted_count = country_count - (self.min_samples_per_country + 100)
            if adjusted_count < 0:
                # If negative, use original count
                samples_to_take = country_count
            else:
                # If positive, use min_samples + 100
                samples_to_take = self.min_samples_per_country + 100
            
            if samples_to_take >= self.min_samples_per_country:
                self.min_samples_each_country.append(samples_to_take)
        print(f"min_samples_each_country: {self.min_samples_each_country}") 
        countries_to_keep = country_counts[country_counts >= self.min_samples_per_country].index
        self.df = self.df[self.df['country'].isin(countries_to_keep)]
        #print(self.df['country'].value_counts())
        print(f"len(self.df): {len(self.df)}")
        print(f"len(self.df['country'].unique()): {len(self.df['country'].unique())}")
        return self.df
    
    def balance_dataset(self, random_state=40):
        """
        Balance the dataset by downsampling majority classes.
        
        Args:
            n_samples_per_country (int): Target number of samples per country
            random_state (int): Random seed for reproducibility
        """
        balanced_df = pd.DataFrame()
        for country in range(len(self.min_samples_each_country)):
            country_data = self.df[self.df['country'] == country]
            if len(country_data) > self.min_samples_each_country[country]:
                country_data = country_data.sample(n=self.min_samples_each_country[country], random_state=random_state)
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
           list of audio segments as log-mel spectrograms
        """
        x, sr = librosa.load(file_path, sr=22050, mono=True)  # Fix sample rate
        segment_length = 5 * sr  # Reduce from 9 to 5 seconds
        segments = []
        
        num_segments = int(len(x) // segment_length)
        
        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = x[start:end]
            # Add augmentation
            #if np.random.random() < 0.3:  # 30% chance of pitch shift
                #segment = librosa.effects.pitch_shift(segment, sr=sr, n_steps=np.random.uniform(-2, 2))
            
            stft = np.abs(librosa.stft(segment, n_fft=2048, hop_length=512))
            mel = librosa.feature.melspectrogram(
                sr=sr, 
                S=stft**2,
                n_mels=256,
                fmin=20,
                fmax=8000
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            # Ensure consistent shape
            #if log_mel.shape[1] > 512:  # Adjust target width for 5-second clips
                #log_mel = log_mel[:, :512]
            #elif log_mel.shape[1] < 510:  # Skip segments that are too short
                #continue
            #elif log_mel.shape[1] < 512:  # Pad segments between 510-512
                #pad_width = ((0, 0), (0, 512 - log_mel.shape[1]))
                #log_mel = np.pad(log_mel, pad_width, mode='constant')
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
        #
        # return self.expanded_df
    
    def save_dataset(self, save_path, format='pkl'):
        self.expanded_df.to_pickle(save_path)

    def examine_dataset(self):
        print(self.expanded_df.head())
        print(len(self.expanded_df))
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
            #print(f"Shape {shape} found for country {country}")
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            
        print("\nSpectrogram shape analysis:")
        print("---------------------------")
        print("Shape counts:")
        for shape, count in shape_counts.items():
            print(f"Shape {shape}: {count} spectrograms")

        print(f"\nTotal spectrograms analyzed: {len(shapes)}")
        

def main():
    data_developer = AudioDataDeveloper(data_path='/projects/dsci410_510/Kolahi_data_temp/medium_tracks_with_countries.csv')
    data_developer.load_data()
    data_developer.balance_dataset()
    data_developer.create_expanded_dataset()
    data_developer.save_dataset('/projects/dsci410_510/Kolahi_data_temp/expanded_dataset_v22.pkl')

    #data_developer.examine_dataset()
    data_developer.analyze_spectrogram_shapes()
if __name__ == "__main__":
    main()
    #data_developer.save_dataset('/projects/dsci410_510/Kolahi_dataset/expanded_dataset_v1.pkl')

    
