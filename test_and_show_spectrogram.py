import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import ast
#import torchaudio

import pickle

# Load the pickle file
with open('/projects/dsci410_510/Kolahi_data_temp/expanded_dataset_v20.pkl', 'rb') as f:
    data = pickle.load(f)

def display_random_spectrogram():
    # Get a random index
    idx = np.random.randint(0, len(data['spectrogram']))
    
    # Get the spectrogram and country for that index
    spec_data = data['spectrogram'][idx]
    country = data['country'][idx]
    
    print(spec_data.shape)

    # Calculate correct time axis values
    hop_length = 512  # from your STFT parameters
    sr = 22050       

    
    
    # Display the spectrogram with correct time axis
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(spec_data, fmax=8000)
    plt.colorbar(img, format='%+2.0f dB')
    plt.title(f'Mel-frequency spectrogram - {country}')
    plt.savefig('SpectrogramClassifier/specimgs/random_spectrogram12.png')
    plt.show()

if __name__ == "__main__":
    # Get shapes of all spectrograms
    shapes = [spec.shape for spec in data['spectrogram']]
    
    # Count unique shapes
    shape_counts = {}
    for shape in shapes:
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    
    print("Spectrogram shape counts:")
    for shape, count in shape_counts.items():
        print(f"Shape {shape}: {count} spectrograms")
    print(f"\nTotal spectrograms: {len(shapes)}")
    print()
    display_random_spectrogram()





########################################################################################

#df = pd.read_csv('/Users/cyruskolahi/Documents/DLprojvenv/DL project/data/expanded_dataset_v5.csv')
#df = pd.read_pickle('/Users/cyruskolahi/Documents/DLprojvenv/DL project/data/expanded_dataset_v2.pkl')
#df = pd.read_hdf('/Users/cyruskolahi/Documents/DLprojvenv/DL project/data/expanded_dataset_v2.h5')
#x, sr = librosa.load(file_path, sr=None, mono=True)
#duration = x.shape[-1] / sr
    
    # Split into 9 second segments
#segment_length = 9 * sr
#segments = []
    
    # Calculate number of complete 9-second segments
#num_segments = int(len(x) // segment_length)
    
#for i in range(num_segments):
#    start = i * segment_length
#    end = start + segment_length
#    segment = x[start:end]
        # Create spectrogram
#    stft = np.abs(librosa.stft(segment, n_fft=2048, hop_length=512))
#    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
#    log_mel = librosa.power_to_db(mel)
#    segments.append(log_mel)

def display_random_spectrogram_with_country(csv_path):
    df = pd.read_csv(csv_path)
    random_row = df.sample(n=1).iloc[0]
    
    # Convert string representation back to numpy array
    spec_data = np.array(ast.literal_eval(random_row['spectrogram']))
    
    # Now display the spectrogram
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(spec_data, x_axis='time',
                                 y_axis='mel', sr=22050,
                                 fmax=8000)
    plt.colorbar(img, format='%+2.0f dB')
    plt.title(f'Mel-frequency spectrogram - {random_row["country"]}')
    plt.show()


#display_random_spectrogram_with_country('/Users/cyruskolahi/Documents/DLprojvenv/DL project/data/expanded_dataset_v2.csv')

