
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
# CONFIGURATION & HYPERPARAMETERS (SYNC THESE WITH TUNER RESULTS)
# ==========================================
MARKET_DATA_PATH = "../../data/main_data/tech_macro_aligned.csv"
SENTIMENT_DATA_PATH = "../../data/data_scrapping/temp/gdelt_sentiment_bq_aligned.csv"
SEC_DATA_PATH = "../../data/main_data/sec_events.csv"
TARGET_EQUITY = 'AAPL'

MODEL_TYPE = 'NLINEAR' 
SEQ_LEN = 192           
PRED_LEN = 20           
INPUT_DIM = 13          
HIDDEN_DIM = 512        
NUM_LAYERS = 1          
BATCH_SIZE = 32         
LEARNING_RATE = 1e-3    
EPOCHS = 150            
PATIENCE = 30          

# Optimized weights from tuner (UPDATE THESE AFTER RUNNING TUNER)
# FEATURE_WEIGHTS = { NLINEAR OPTIMZED
#     'ROC_5': 0.65, 
#     'RSI_14': 0.65,
#     '{TICKER}_Sentiment_Tone': 0.65,
#     'SEC_Event': 1.0
# }
FEATURE_WEIGHTS = {
        'ROC_5': 0.65, 
        'RSI_14': 0.65,
        '{TICKER}_Sentiment_Tone': 0.65,
        'SEC_Event': 1.0
    }

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
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

class ModelFactory(nn.Module):
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
            return self.head(out[:, -1, :])
        else:
            return self.core(x)

def train_model(config, verbose=True):
    """
    Function used by the tuner or standard run to train with a specific config.
    """
    m_type = config.get('model_type', MODEL_TYPE)
    s_len = config.get('seq_len', SEQ_LEN)
    p_len = config.get('pred_len', PRED_LEN)
    lr = config.get('lr', LEARNING_RATE)
    bs = config.get('batch_size', BATCH_SIZE)
    epochs = config.get('epochs', EPOCHS)
    
    # Use config weights (from tuner) or global optimized weights (standard run)
    f_weights = config.get('feature_weights', FEATURE_WEIGHTS)
    
    # Data Loading
    train_set = MultivariateStockDataset(TARGET_EQUITY, MARKET_DATA_PATH, SENTIMENT_DATA_PATH, SEC_DATA_PATH, 
                                       seq_len=s_len, pred_len=p_len, split='train', feature_weights=f_weights)
    test_set = MultivariateStockDataset(TARGET_EQUITY, MARKET_DATA_PATH, SENTIMENT_DATA_PATH, SEC_DATA_PATH, 
                                      seq_len=s_len, pred_len=p_len, split='test', feature_weights=f_weights)
    
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)
    
    model = ModelFactory(m_type, s_len, p_len, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Checkpoint configuration
    weight_path = config.get('weight_path', f"multivariate_{m_type.lower()}_{TARGET_EQUITY}.pth")
    stopper = EarlyStopping(patience=PATIENCE, path=weight_path)
    scaler = torch.cuda.amp.GradScaler()

    if verbose:
        print(f"Hyperparameters: SEQ_LEN={s_len}, LR={lr}, WEIGHTS={f_weights}")

    for epoch in range(epochs):
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
        avg_t = t_loss / len(train_loader)
        
        scheduler.step(avg_v)
        
        if verbose and (epoch + 1) % 5 == 0:
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Train Loss {avg_t:.6f} | Val Loss {avg_v:.6f} | LR {curr_lr:.6e}")
        
        stopper(avg_v, model)
        if stopper.early_stop:
            if verbose: print(f"Early stopping at epoch {epoch+1}")
            break
            
    return model, weight_path

if __name__ == "__main__":
    print(f"--- Starting Multivariate {MODEL_TYPE} Training Pipeline ---")
    model, path = train_model({}, verbose=True)
    print(f"--- Training Complete. Best weights saved to: {path} ---")
