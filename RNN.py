import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, output_size):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM forward pass (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Get output from last time step
        last_output = lstm_out[:, -1, :]
        
        # Pass through dropout and fully connected layer
        out = self.dropout(last_output)
        out = self.fc(out)
        
        return out.squeeze()

class RNNEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.training_history = []
    
    def train_model(self, train_loader, val_loader, criterion, optimizer, epochs):
        self.model.train()
        training_losses = []
        validation_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            training_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    output = self.model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            validation_losses.append(val_loss)
            
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        self.training_history = {'train_loss': training_losses, 'val_loss': validation_losses}
        return training_losses, validation_losses
    
    def evaluate(self, test_loader):
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                
                outputs = self.model(X_batch)
                
                predictions.append(outputs.cpu().numpy())
                actuals.append(y_batch.numpy())
        
        predictions = np.concatenate(predictions).flatten()
        actuals = np.concatenate(actuals).flatten()
        
        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
        
        print(f"Evaluation Metrics:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        return metrics, predictions, actuals
    
    def visualize_training_history(self, figsize=(10, 6)):
        if not self.training_history:
            raise ValueError("No training history available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        ax.plot(epochs, self.training_history['train_loss'], label='Training Loss')
        ax.plot(epochs, self.training_history['val_loss'], label='Validation Loss')
        
        ax.set_title('Model Loss During Training')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_predictions(self, predictions, actuals, figsize=(12, 6)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot predictions vs actuals
        ax1.scatter(actuals, predictions, alpha=0.5)
        
        # Plot perfect prediction line
        min_val = min(np.min(actuals), np.min(predictions))
        max_val = max(np.max(actuals), np.max(predictions))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax1.set_title('Predictions vs Actual')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.grid(True, alpha=0.3)
        
        # Plot error distribution
        errors = predictions - actuals
        sns.histplot(errors, kde=True, ax=ax2)
        
        ax2.set_title('Error Distribution')
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig