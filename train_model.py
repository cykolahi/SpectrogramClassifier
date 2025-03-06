import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model.model import SpectrogramCNN
import pickle 
import numpy as np
import pandas as pd
from official_data_loader import AudioDataLoader, AudioDataset

def train_model(train_loader, val_loader, test_loader, num_classes, max_epochs=10):
    # Initialize model
    model = SpectrogramCNN(num_classes=num_classes)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',  # Automatically detect if you have GPU
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
    )
    
    # Train model
    trainer.fit(model, train_loader)

    trainer.validate(model, val_loader)
    # Test model
    trainer.test(model, test_loader)
    
    return model


def main():
    unloaded_data_path = '/projects/dsci410_510/Kolahi_data_temp/expanded_dataset_v9.pkl'
    train_data_path = '/projects/dsci410_510/Kolahi_data_temp/train_dataset.pkl'
    val_data_path = '/projects/dsci410_510/Kolahi_data_temp/val_dataset.pkl'
    test_data_path = '/projects/dsci410_510/Kolahi_data_temp/test_dataset.pkl'

    # Load data
    train_data = pickle.load(open(train_data_path, 'rb'))
    val_data = pickle.load(open(val_data_path, 'rb'))
    test_data = pickle.load(open(test_data_path, 'rb'))

    #print(len(np.unique(train_data['countries'])))
    #print(len(np.unique(val_data['countries'])))
    #print(len(np.unique(test_data['countries'])))
    # or
    data_loader = AudioDataLoader(unloaded_data_path)
    train_loader, val_loader, test_loader = data_loader.create_train_val_test_split()
    
    
    num_classes = len(np.unique(train_data['countries']))
    
    # Train the model
    model = train_model(train_loader, val_loader, test_loader, num_classes) 

if __name__ == "__main__":
    main()

"""
script to run:


"""