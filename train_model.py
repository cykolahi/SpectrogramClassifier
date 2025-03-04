import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model.model import SpectrogramCNN

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
    trainer.fit(model, train_loader, val_loader, callbacks=[early_stopping])
    
    # Test model
    trainer.test(model, test_loader)
    
    return model


def main():
    # Example usage
    from official_data_loader import AudioDataLoader
    #data_loader = AudioDataLoader(data_path='/Users/cyruskolahi/Documents/SpectrogramClassifier/data/audio_paths_with_countries.csv')
    data_loader.load_data()  # Load and preprocess the data first
    data_loader.balance_dataset()  # Balance if needed
    data_loader.create_expanded_dataset()  # Now create the expanded dataset
    train_loader, val_loader, test_loader = data_loader.create_train_val_test_split()
    #data_loader.save_dataset('data/expanded_dataset_v6.pkl', format='pkl')
    
    # Get data loaders
    #train_loader, test_loader, val_loader = get_data_loaders()
    
    # Assuming you have 10 classes, modify this based on your dataset
    num_classes = 24
    
    # Train the model
    model = train_model(train_loader, val_loader, test_loader, num_classes) 

if __name__ == "__main__":
    main()

"""
script to run:


"""