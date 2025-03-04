import pickle 
import pandas as pd

def main():
    with open('~/projects/dsci410_510/Kolahi_data_temp/Kolahi_data.pkl', 'rb') as f:
        data = pickle.load(f)

    print(data.keys())
    print(data['audio_paths'].shape)
    print(data['countries'].shape)
    print(data['segment_numbers'].shape)
    print(data['spectrograms'].shape
    print(data['train_indices'].shape)
    print(data['test_indices'].shape)
    print(data['train_dataset'].shape)
    print(data['test_dataset'].shape)
    print(data['train_loader'].shape)
    print(data['test_loader'].shape)

if __name__ == "__main__":
    main()