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

class SpectrogramCNN_1d_attn(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # First attention block after initial feature extraction
        self.attention1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Second attention block before FC layers
        self.attention2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Softmax(dim=1)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(64, 32)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(16, num_classes)

        self.criterion = nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        
        # Add debug prints
        #rint("Val step - y_hat shape:", y_hat.shape)
        #print("Val step - y shape:", y.shape)
        
        loss = self.criterion(y_hat, y)  # CrossEntropyLoss expects [B, C] and [B]
        
        # Calculate accuracy
        pred = torch.argmax(y_hat, dim=1)
        acc = (pred == y).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        # Calculate accuracy
        pred = torch.argmax(y_hat, dim=1)
        acc = (pred == y).float().mean()
        self.log('test_acc', acc)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=0.0001,
                                     weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def forward(self, x):
        x = x.squeeze(1)  # [batch, 128, time]
        
        # First conv block
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        
        # First attention - temporal attention
        attn1 = self.attention1(x)
        x = x * attn1  # Element-wise multiplication
        
        # More conv blocks
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        
        # Global pooling
        x = self.adaptive_pool(x)  # [batch, 64, 1]
        x = x.squeeze(-1)  # [batch, 64]
        
        # Second attention - feature attention
        attn2 = self.attention2(x)
        x = x * attn2
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class SpectrogramCNN_1d(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(64)
        self.pool5 = nn.MaxPool1d(kernel_size=2)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.attention = nn.Sequential(
            #nn.AdaptiveAvgPool1d(512),
            #nn.Flatten(1),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            #nn.Sigmoid()
            nn.Softmax(dim=1)
        )

        self.multihead_attention = nn.MultiheadAttention(embed_dim=64, num_heads=8, dropout=0.1)
        
        # Wider fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(16, 3)

        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        
        # Add debug prints
        #rint("Val step - y_hat shape:", y_hat.shape)
        #print("Val step - y shape:", y.shape)
        
        loss = self.criterion(y_hat, y)  # CrossEntropyLoss expects [B, C] and [B]
        
        # Calculate accuracy
        pred = torch.argmax(y_hat, dim=1)
        acc = (pred == y).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        # Calculate accuracy
        pred = torch.argmax(y_hat, dim=1)
        acc = (pred == y).float().mean()
        self.log('test_acc', acc)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=0.0001,
                                     weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    
    def forward(self, x):
        x = x.squeeze(1)  # Remove channel dim: [batch, 128, time]
        
        # Conv blocks
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)  # [batch, 512, time/4]

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)

        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.pool5(x)
        
        # Global pooling to remove the time dimension
        x = self.adaptive_pool(x)  # [batch, 512, 1]
        # Self-attention
       #   x_att = x.permute(2, 0, 1)  # [time, batch, channels] 
       # x_att, _ = self.multihead_attention(x_att, x_att, x_att)
       # x = x_att.permute(1, 2, 0)  # [batch, channels, time]
        x = x.squeeze(-1)  # [batch, 512]
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)  # This should output [batch, num_classes]
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    
    """   
    def forward(self, x):
        # Input shape: [batch, 1, 128, time]
        x = x.squeeze(1)  # Remove channel dim: [batch, 128, time]
        
        # Convolutional layers with residual connections
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x + F.pad(identity, (0, 0, 128, 0)))  # Pad channels for residual
        x = self.pool1(x)
        
        identity = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x + identity)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Self-attention
        x_att = x.permute(2, 0, 1)  # [time, batch, channels]
        x_att, _ = self.multihead_attention(x_att, x_att, x_att)
        x = x_att.permute(1, 2, 0)  # [batch, channels, time]
        
        # Parallel pooling
        avg_x = self.avg_pool(x).squeeze(-1)
        max_x = self.max_pool(x).squeeze(-1)
        x = torch.cat([avg_x, max_x], dim=1)
        
        # MLP head
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def forward(self, x):
        x = x.squeeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        attn = self.attention(x)
        attn = attn.unsqueeze(-1)  # Add dimension back for broadcasting
        x = x * attn
        
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.dropout4(x)
        x = self.fc2(x)

        return x

    """



class SpectrogramCNN_2d(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        
        # Increase initial filters and add residual connections
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 4))
        self.dropout1 = nn.Dropout(0.25)
        
        # Deeper network with wider temporal context
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 9), stride=1, padding=(1, 4))
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 4))
        self.dropout2 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(3, 15), stride=1, padding=(1, 7))
        self.bn3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 4))
        self.dropout3 = nn.Dropout(0.25)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Wider fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout4 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)
        

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
            patience=10
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x
    

class SpectrogramCNN_2d_attn(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        
        # Increase initial filters and add residual connections
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 4))
        self.dropout1 = nn.Dropout(0.3)

        # First attention block
        self.attention1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Deeper network with wider temporal context
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 9), stride=1, padding=(1, 4))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 4))
        self.dropout2 = nn.Dropout(0.3)

        # Second attention block
        self.attention2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 15), stride=1, padding=(1, 7))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 4))
        self.dropout3 = nn.Dropout(0.3)

        # Third attention block
        self.attention3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Wider fully connected layers
        self.fc1 = nn.Linear(128, 128)
        self.dropout4 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        

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
            lr=0.0001,
            weight_decay=0.01
        )
        
        # Use a different scheduler that doesn't require steps_per_epoch at init
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=8
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
        # First conv block with attention
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Apply first attention
        attn1 = self.attention1(x)
        x = x * attn1
        
        # Second conv block with attention
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Apply second attention
        attn2 = self.attention2(x)
        x = x * attn2
        
        # Third conv block with attention
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Apply third attention
        attn3 = self.attention3(x)
        x = x * attn3
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x
