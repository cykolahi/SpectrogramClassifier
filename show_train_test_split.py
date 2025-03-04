import pickle 
import pandas as pd
import numpy as np
def main():
    with open('/projects/dsci410_510/Kolahi_data_temp/Kolahi_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    

    print("data.keys(): ", data.keys())
    #print(data['audio_paths'].shape)
    print("countries: ", np.unique(data['countries']))
    #print("data['segment_numbers'][0].shape: ", data['segment_numbers'][0].shape)
    print("spectrogram tensor shape: ", data['spectrograms'][6].shape)
    #print("data['train_indices'].shape: ", data['train_indices'].shape)
    #print("data['test_indices'].shape: ", data['test_indices'].shape)
    #print("number of train samples: ", len(data['train']))
    #print("number of test samples: ", len(data['test']))
    #print("data['train_loader'].shape: ", data['train_loader'].shape)
    #print(data['test_loader'].shape)

if __name__ == "__main__":
    main()