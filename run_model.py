import os
import torch
import glob
import pytorch_lightning as pl
from train_model import AudioDataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

def get_latest_checkpoint():
    """Get the most recent checkpoint file from the checkpoints directory"""
    checkpoint_dir = 'checkpoints'
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in 'checkpoints' directory")
    return max(checkpoint_files, key=os.path.getctime)

def get_model(model_path):
    """ load model from .pth file"""
    from model.model import SpectrogramCNN_1d_attn
    model = SpectrogramCNN_1d_attn(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, test_loader):
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Create confusion matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(all_targets, all_preds)
    
    # Get country names (you'll need to add these)
    country_names = ['Unites States', 'United Kingdom', 'Canada']  # Replace with your actual country names
    
    sns.heatmap(cm,
                annot=True,
                fmt='g',
                xticklabels=country_names,
                yticklabels=country_names)
    
    plt.xlabel('Predicted Country')
    plt.ylabel('True Country')
    plt.title('Country Classification Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    return {
        'test_acc': accuracy,
        'confusion_matrix': cm
    }

def plot_metrics(results):
    """Plot test loss and accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    
    # Plot test loss
    #ax1.bar(['Test Loss'], [results['test_loss']])
    #ax1.set_title('Test Loss')
    #ax1.grid(True, alpha=0.3)
    
    # Plot test accuracy
    ax2.bar(['Test Accuracy'], [results['test_acc']])
    ax2.set_title('Test Accuracy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
  
   
    data_path = '/projects/dsci410_510/Kolahi_data_temp/expanded_dataset_v21.pkl'
    data_loader = AudioDataLoader(data_path)
    model_path = '/projects/dsci410_510/Kolahi_models/model_v23.pth'
    model = get_model(model_path)
    # Load model from checkpoint
    #model = torch.load(checkpoint_path)
    #model.eval()


    #diff_model = get_model('SpectrogramClassifier/models/model_v14.pth')
    #diff_results = evaluate_model(diff_model, test_loader)

    #print("\nTest Results:")
    #print(f"Loss: {diff_results['test_loss']:.4f}")
    #print(f"Accuracy: {diff_results['test_acc']:.4f}")
    # Evaluate model
    accuracies = []
    for random_state in range(42, 50):
        _, _, test_loader = data_loader.create_train_val_test_split(random_state=random_state)
        results = evaluate_model(model, test_loader)
        print(f"Random State: {random_state}")
        print(f"Accuracy: {results['test_acc']:.4f}")
        accuracies.append(results['test_acc'])
    # Calculate and print average accuracy across all random states
    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"\nAverage accuracy across all random states: {avg_accuracy:.4f}")
    # Plot results
    #plot_metrics(results)

if __name__ == "__main__":
    main()
