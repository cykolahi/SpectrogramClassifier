from torch import nn
import torch.nn.functional as F
import torch
import sys
import pytorch_lightning as pl
print("Python path:", sys.executable)
print("Python version:", sys.version)
print("Torch path:", torch.__file__)
print("Torch version:", torch.__version__)
#print('Lightning version:', pl.__version__)


class SpectrogramCNN(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        
        # Increase initial filters and add residual connections
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 4))
        self.dropout1 = nn.Dropout(0.25)
        
        # Deeper network with wider temporal context
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 9), stride=1, padding=(1, 4))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 4))
        self.dropout2 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(3, 15), stride=1, padding=(1, 7))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 4))
        self.dropout3 = nn.Dropout(0.25)
        
        # Improved attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Flatten(2),
            nn.Conv1d(64, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Two-stage pooling
        self.global_freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.global_time_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Wider fully connected layers
        self.fc1 = nn.Linear(64, 1024)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Add label smoothing
        smooth_loss = 0.1 * torch.mean(F.log_softmax(logits, dim=1))
        loss = loss - smooth_loss
        
        # Log metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {'val_loss': loss, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return {'test_loss': loss, 'test_acc': acc}
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # Use a different scheduler that doesn't require steps_per_epoch at init
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Watches validation loss
                "interval": "epoch",    # Updates per epoch instead of per step
                "frequency": 1
            }
        }

    def forward(self, x):
        # Regular convolution path
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Apply temporal attention
        attn = self.attention(x)
        attn = attn.unsqueeze(-1)  # Add dimension back for broadcasting
        x = x * attn
        
        # Two-stage pooling
        x = self.global_freq_pool(x)  # Pool frequency first
        x = self.global_time_pool(x)  # Then pool time
        x = x.view(x.size(0), -1)
        
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x
    
