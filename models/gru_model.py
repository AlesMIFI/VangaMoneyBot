"""
GRU модель для прогнозирования временных рядов (PyTorch) - Улучшенная версия
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models.base_model import BaseModel
import logging

logger = logging.getLogger(__name__)


class ImprovedGRUNet(nn.Module):
    """Улучшенная GRU нейронная сеть"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.3):
        super(ImprovedGRUNet, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Более глубокая GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Дополнительные полносвязные слои
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        
        out = self.fc1(last_output)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        
        return out


class GRUModel(BaseModel):
    """GRU модель на PyTorch - Улучшенная версия"""
    
    def __init__(self, n_steps: int = 30):
        super().__init__("GRU")
        self.n_steps = n_steps
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, scaler, **kwargs) -> None:
        """
        Обучение GRU
        
        Args:
            X_train: Признаки [samples, timesteps, features]
            y_train: Целевая переменная [samples]
            scaler: Скейлер для обратного преобразования
        """
        logger.info(f"Training {self.name} on {self.device}")
        
        self.scaler = scaler
        
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        self.model = ImprovedGRUNet(
            input_size=1,
            hidden_size=128,
            num_layers=3,
            dropout=0.3
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=0.0005,
            weight_decay=1e-5
        )
        
        # Scheduler БЕЗ verbose!
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10
            # verbose=True  # ← УДАЛЕНО! Не поддерживается в PyTorch 2.6+
        )
        
        num_epochs = 150
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        best_model_state = None
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            
            # Scheduler step с ручным логированием
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            if old_lr != new_lr:
                logger.info(f"Learning rate changed: {old_lr:.6f} → {new_lr:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, LR: {new_lr:.6f}")
        
        self.is_trained = True
        logger.info(f"{self.name} training completed with best loss: {best_loss:.6f}")
    
    def predict(self, steps: int, last_sequence: np.ndarray, **kwargs) -> np.ndarray:
        """
        Многошаговое прогнозирование
        
        Args:
            steps: Количество шагов прогноза
            last_sequence: Последняя последовательность (масштабированная 0-1)
            
        Returns:
            Массив прогнозов (исходный масштаб в $)
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} is not trained yet")
        
        if steps <= 0:
            raise ValueError(f"Steps must be positive, got {steps}")
        
        if len(last_sequence) != self.n_steps:
            raise ValueError(f"last_sequence must have length {self.n_steps}, got {len(last_sequence)}")
        
        self.model.eval()
        predictions = []
        current_sequence = last_sequence.copy()
        
        with torch.no_grad():
            for _ in range(steps):
                x_input = torch.FloatTensor(current_sequence).reshape(1, self.n_steps, 1).to(self.device)
                
                # Прогноз (в масштабе 0-1)
                pred_scaled = self.model(x_input).cpu().numpy()[0, 0]
                
                # Clip для стабильности
                pred_scaled = np.clip(pred_scaled, 0, 1)
                
                predictions.append(pred_scaled)
                current_sequence = np.append(current_sequence[1:], pred_scaled)
        
        # Обратное масштабирование (0-1 → $)
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        logger.info(f"{self.name} forecast: mean=${predictions.mean():.2f}, "
                   f"min=${predictions.min():.2f}, max=${predictions.max():.2f}")
        
        return predictions
