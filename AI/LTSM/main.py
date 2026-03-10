
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import warnings
import time
import os

# Internal project imports
from dataset_builder import MultivariateStockDataset
from sota_linear import DLinear, NLinear

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
MARKET_DATA_PATH = "../../data/main_data/tech_macro_aligned.csv"
SENTIMENT_DATA_PATH = "../../data/data_scrapping/temp/gdelt_sentiment_bq_aligned.csv"
SEC_DATA_PATH = "../../data/main_data/sec_events.csv"
TARGET_EQUITY = 'AAPL'

# Model Toggle: 'LSTM', 'GRU', 'DLINEAR', or 'NLINEAR'
MODEL_TYPE = 'NLINEAR' 

SEQ_LEN = 256           # ~1 year of trading context
PRED_LEN = 30           # Your 30-day target
INPUT_DIM = 13          # (Including your new Momentum features)
HIDDEN_DIM = 512        
NUM_LAYERS = 1          
BATCH_SIZE = 32         
LEARNING_RATE = 1e-3    # Keep this higher to maintain "aggression"
EPOCHS = 150            
PATIENCE = 30          

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# EARLY STOPPING MECHANISM
# ==========================================
class EarlyStopping:
    """
    Monitors validation loss to terminate training once improvement stagnates.
    Saves optimal parameters to a model-specific persistent file.
    """
    def __init__(self, patience, path='checkpoint.pth'):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# ==========================================
# UNIFIED MODEL WRAPPER
# ==========================================
class ModelFactory(nn.Module):
    """
    Factory class to instantiate the requested architecture while 
    maintaining a unified forward pass interface.
    """
    def __init__(self, model_type, seq_len, pred_len, input_dim, hidden_dim, num_layers):
        super(ModelFactory, self).__init__()
        self.model_type = model_type.upper()
        
        if self.model_type == 'LSTM':
            self.core = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
            self.head = nn.Linear(hidden_dim, pred_len)
        elif self.model_type == 'GRU':
            self.core = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
            self.head = nn.Linear(hidden_dim, pred_len)
        elif self.model_type == 'DLINEAR':
            self.core = DLinear(seq_len, pred_len, input_dim)
        elif self.model_type == 'NLINEAR':
            self.core = NLinear(seq_len, pred_len, input_dim)
        else:
            raise ValueError(f"Architecture {model_type} not recognized.")

    def forward(self, x):
        if self.model_type in ['LSTM', 'GRU']:
            out, _ = self.core(x)
            return self.head(out[:, -1, :]) # Map final hidden state to horizon
        else:
            return self.core(x) # Linear models handle projection internally

# ==========================================
# EXECUTION PIPELINE
# ==========================================
def run_ablation_study():
    print(f"Executing {MODEL_TYPE} training sequence on {DEVICE}...")
    
    # Data Loading
    train_set = MultivariateStockDataset(TARGET_EQUITY, MARKET_DATA_PATH, SENTIMENT_DATA_PATH,SEC_DATA_PATH, split='train')
    test_set = MultivariateStockDataset(TARGET_EQUITY, MARKET_DATA_PATH, SENTIMENT_DATA_PATH,SEC_DATA_PATH, split='test')
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model Initialization
    model = ModelFactory(MODEL_TYPE, SEQ_LEN, PRED_LEN, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=1.5)
    
    # Checkpoint configuration
    weight_path = f"multivariate_{MODEL_TYPE.lower()}_{TARGET_EQUITY}.pth"
    stopper = EarlyStopping(patience=PATIENCE, path=weight_path)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(bx)
                loss = criterion(pred, by)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()
            
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for vx, vy in test_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                v_loss += criterion(model(vx), vy).item()
        
        avg_v = v_loss / len(test_loader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train MSE {t_loss/len(train_loader):.6f} | Val MSE {avg_v:.6f}")
        
        stopper(avg_v, model)
        if stopper.early_stop:
            print(f"Convergence achieved at epoch {epoch+1}. Preserving best weights.")
            break

if __name__ == "__main__":
    run_ablation_study()