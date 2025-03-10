import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model.model import SpectrogramCNN
import torch
import pickle 
import numpy as np
import pandas as pd
from official_data_loader import AudioDataLoader, AudioDataset
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from logger.visualization import TensorboardWriter
import os
import logging


class MetricsCallback(Callback):
    def __init__(self, tensorboard_writer):
        super().__init__()
        self.tensorboard_writer = tensorboard_writer

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        
        # Set step and mode for TensorBoard
        self.tensorboard_writer.set_step(epoch, mode='train')
        
        # Log metrics to console
        print(f"\nEpoch {epoch}")
        print(f"Training Loss: {metrics.get('train_loss', 0):.4f}")
        print(f"Validation Loss: {metrics.get('val_loss', 0):.4f}")
        print(f"Validation Accuracy: {metrics.get('val_acc', 0):.4f}")
        
        # Log training metrics
        self.tensorboard_writer.add_scalar('loss', metrics.get('train_loss', 0))
        
        # Switch to validation mode for logging
        self.tensorboard_writer.set_step(epoch, mode='valid')
        self.tensorboard_writer.add_scalar('loss', metrics.get('val_loss', 0))
        self.tensorboard_writer.add_scalar('accuracy', metrics.get('val_acc', 0))


def train_model(train_loader, val_loader, test_loader, num_classes, max_epochs=100):
    # Initialize model
    if not torch.cuda.is_available():
        print("WARNING: No GPU found. Please check your CUDA installation.")
        #return
        
    #print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    
    model = SpectrogramCNN(num_classes=num_classes)

    # Setup logging
    logger = logging.getLogger('train')
    log_dir = os.path.join(os.getcwd(), 'logs', 'runs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = TensorboardWriter(log_dir, logger, enabled=True)
    
    # Initialize callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='spectrogram-cnn-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    metrics_callback = MetricsCallback(writer)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=[0],
        callbacks=[early_stopping, checkpoint_callback, metrics_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        enable_checkpointing=True,
        logger=True,
        val_check_interval=1.0,
        precision='32-true'
    )
    
    
    trainer.fit(model, train_loader, val_loader)

    trainer.validate(model, val_loader)
 
    trainer.test(model, test_loader)
    
    return model


def main():
    unloaded_data_path = '/projects/dsci410_510/Kolahi_data_temp/expanded_dataset_v10.pkl'
    train_data_path = '/projects/dsci410_510/Kolahi_data_temp/train_dataset.pkl'
    val_data_path = '/projects/dsci410_510/Kolahi_data_temp/val_dataset.pkl'
    test_data_path = '/projects/dsci410_510/Kolahi_data_temp/test_dataset.pkl'

    # Load data
    train_data = pickle.load(open(train_data_path, 'rb'))
    #   val_data = pickle.load(open(val_data_path, 'rb'))
    #   test_data = pickle.load(open(test_data_path, 'rb'))

    #print(len(np.unique(train_data['countries'])))
    #print(len(np.unique(val_data['countries'])))
    #print(len(np.unique(test_data['countries'])))
    # or
    data_loader = AudioDataLoader(unloaded_data_path)
    train_loader, val_loader, test_loader = data_loader.create_train_val_test_split()
    
    
    num_classes = len(np.unique(train_data['countries']))
    
    # Train the model
    model = train_model(train_loader, val_loader, test_loader, num_classes) 

    # Create the directory if it doesn't exist
    save_dir = 'SpectrogramClassifier/models'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model with error handling
    try:
        save_path = os.path.join(save_dir, 'model_v1.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Model successfully saved to: {save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        print(f"Current working directory: {os.getcwd()}")

if __name__ == "__main__":
    main()

"""
script to run:


"""