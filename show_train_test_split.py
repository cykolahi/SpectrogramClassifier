import pickle 
import pandas as pd
import numpy as np
def main():
    with open('/projects/dsci410_510/Kolahi_data_temp/train_dataset.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open('/projects/dsci410_510/Kolahi_data_temp/val_dataset.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    with open('/projects/dsci410_510/Kolahi_data_temp/test_dataset.pkl', 'rb') as f:
        test_data = pickle.load(f)
    

    print("train_data.keys(): ", train_data.keys())
    #print(data['audio_paths'].shape)
    print("train_data['countries']: ", np.unique(train_data['countries']))
    #print("data['segment_numbers'][0].shape: ", data['segment_numbers'][0].shape)
    print("spectrogram shape: ", train_data['spectrograms'][6].shape)

    print('train data size: ', len(train_data['countries']))
    print('val data size: ', len(val_data['countries']))
    print('test data size: ', len(test_data['countries']))
    #print("data['train_indices'].shape: ", data['train_indices'].shape)
    #print("data['test_indices'].shape: ", data['test_indices'].shape)
    #print("number of train samples: ", len(data['train']))
    #print("number of test samples: ", len(data['test']))
    #print("data['train_loader'].shape: ", data['train_loader'].shape)
    #print(data['test_loader'].shape)

if __name__ == "__main__":
    main()