import pandas as pd
import matplotlib.pyplot as plt
import os

# Print current directory and available files for debugging
print("Current directory:", os.getcwd())
print("Available files in lightning_logs/version_116/:", os.listdir('lightning_logs/version_31/') if os.path.exists('lightning_logs/version_31/') else "Directory not found")

try:
    # Try to read the metrics file
    df = pd.read_csv('lightning_logs/version_120/metrics.csv')
    print("Data loaded successfully. Shape:", df.shape)
    
    # Group by epoch and calculate mean of losses
    epoch_means = df.groupby('epoch').agg({
        'train_loss': 'mean',
        'val_loss': 'mean', 
        'test_loss': 'mean'
    }).reset_index()

    # Calculate mean accuracies per epoch
    epoch_means['train_acc'] = df.groupby('epoch')['train_acc'].mean()
    epoch_means['val_acc'] = df.groupby('epoch')['val_acc'].mean()
    epoch_means['test_acc'] = df.groupby('epoch')['test_acc'].mean()
    
    # Replace the original dataframe with epoch means
    df = epoch_means
    print("Averaged losses by epoch. Shape:", df.shape)
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], '-', linewidth=2, label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], '-', linewidth=2, label='Validation Loss') 
    plt.plot(df['epoch'], df['test_loss'], '-', linewidth=2, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training, Validation, and Test Loss Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig('SpectrogramClassifier/loss_plots/training_plot120_CNN_attention.png')
    print("Plot saved as training_plot120_CNN_attention.png")
    plt.show()

except FileNotFoundError as e:
    print("Error: Could not find the metrics file.")
    print("Make sure you're running this from the correct directory")
    print("Error details:", e)
except Exception as e:
    print("An error occurred:", e)
