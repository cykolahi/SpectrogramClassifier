import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model.model import SpectrogramCNN_2d, SpectrogramCNN_1d, SpectrogramCNN_1d_attn, SpectrogramCNN_2d_attn
import torch
import pickle 
import numpy as np
import pandas as pd
from official_data_loader import AudioDataLoader, AudioDataset
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from logger.visualization import TensorboardWriter
import os
import logging

torch.set_float32_matmul_precision('medium')  # or 'medium' for a balance of speed and precision


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
        #print(f"Test Accuracy: {metrics.get('test_acc', 0):.4f}")
        
        # Log training metrics
        self.tensorboard_writer.add_scalar('train_loss', metrics.get('train_loss', 0))

        
        # Switch to validation mode for logging
        self.tensorboard_writer.set_step(epoch, mode='valid')
        self.tensorboard_writer.add_scalar('val_loss', metrics.get('val_loss', 0))
        self.tensorboard_writer.add_scalar('val_acc', metrics.get('val_acc', 0))
        self.tensorboard_writer.add_scalar('test_acc', metrics.get('test_acc', 0))


class DataReloaderCallback(Callback):
    def __init__(self, data_path, test_loader, reload_every_n_epochs=12):
        super().__init__()
        self.data_path = data_path
        self.reload_every_n_epochs = reload_every_n_epochs
        self.current_epoch = 0
        self.test_loader = test_loader
    def find_test_indices(self):
        """
        Find the indices of samples in the test loader dataset.
        
        Returns:
            set: Set of indices from the test dataset
        """
        self.test_indices = set()
        for i in range(len(self.test_loader.dataset)):
            self.test_indices.add(i)
        return self.test_indices
    '''
    def on_epoch_end(self, trainer, pl_module):
        self.current_epoch += 1
        if self.current_epoch % self.reload_every_n_epochs == 0:
            print(f"\nReloading data with new random split at epoch {self.current_epoch}")
            # Create new data loaders, excluding test data
            data_loader = AudioDataLoader(self.data_path)
            self.test_indices = self.find_test_indices()
            new_train_loader, new_val_loader = data_loader.create_train_val_split(
                exclude_indices=self.test_indices,
                random_state=42 + self.current_epoch  # Different seed each time
                batch_size=16)
            
            # Update only train and val dataloaders, keeping test_loader unchanged
            trainer.train_dataloader = lambda: new_train_loader
            trainer.val_dataloaders = lambda: new_val_loader
    '''


def train_model(train_loader, val_loader, test_loader, num_classes, max_epochs, data_path=''):
    # Initialize model
    if not torch.cuda.is_available():
        print("WARNING: No GPU found. Please check your CUDA installation.")
        #return   
    #print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    
    model = SpectrogramCNN_1d_attn(num_classes=3)


    # Setup logging
    logger = logging.getLogger('train')
    log_dir = os.path.join(os.getcwd(), 'logs', 'runs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = TensorboardWriter(log_dir, logger, enabled=True)
    
    # Initialize callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        mode='min',
        min_delta=0.001
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='spectrogram-cnn-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    metrics_callback = MetricsCallback(writer)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        callbacks=[
            checkpoint_callback, metrics_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        enable_checkpointing=True,
        logger=True,
        val_check_interval=1.0,
        accumulate_grad_batches=1,
        #precision='32-true'
        #gradient_clip_val=1.0
        
    )
            
    # Add this debug code before training
    for batch in train_loader:
        x, y = batch
        print("Input shape:", x.shape)
        print("Target shape:", y.shape)
        print("Unique labels:", torch.unique(y))
        break

    trainer.fit(model, train_loader, val_loader)

    trainer.validate(model, val_loader)
 
    trainer.test(model, test_loader)
    
    return model




def main():
    unloaded_data_path = '/projects/dsci410_510/Kolahi_data_temp/expanded_dataset_v21.pkl'
    
    # Load data
    data = pickle.load(open(unloaded_data_path, 'rb'))
    #   val_data = pickle.load(open(val_data_path, 'rb'))
    #   test_data = pickle.load(open(test_data_path, 'rb'))

    #print(len(np.unique(data['countries'])))
    #print(data['countries'])
    #print(f"tensor shape: {data['spectrograms'][0].shape}")
    #print(len(np.unique(val_data['countries'])))
    #print(len(np.unique(test_data['countries'])))
    # or
    data_loader = AudioDataLoader(unloaded_data_path)
    train_loader, val_loader, test_loader = data_loader.create_train_val_test_split(random_state=42)
    
    
    #num_classes = len(np.unique(train_loader['countries']))
    #print(f"Number of classes: {num_classes}")
    #print(f"Number of samples: {len(train_loader)}")
    #print(f"shape of tensor: {train_loader}")
    
    # Train the model
    model = train_model(train_loader, val_loader, test_loader, num_classes=3, max_epochs=250, data_path=unloaded_data_path) 

    # Create the directory if it doesn't exist
    #save_dir = '/projects/dsci410_510/Kolahi_models'

    #os.makedirs(save_dir, exist_ok=True)
    
    # Save the model with error handling
    try:
        save_path = '/projects/dsci410_510/Kolahi_models/model_v24.pth'
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