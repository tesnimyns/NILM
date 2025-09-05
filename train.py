import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from src.dataset import NILMDataset
from energformer import Energformer
from src.model_nilm import TransformerMultiOutputNILM

# ===========================
# CONFIG
# ===========================
CSV_PATH = "./dataset/channel_5.csv"
WINDOW_SIZE = 256
STRIDE = 1
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4

# ===========================
# LOAD DATA
# ===========================
df = pd.read_csv(CSV_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
appliance_cols = [c for c in df.columns if c not in ["channel_id", "date", "sn", "tot_active_pow"]]

scaler_input = MinMaxScaler()
scaler_target = MinMaxScaler()
df["tot_active_pow"] = scaler_input.fit_transform(df[["tot_active_pow"]])
df[appliance_cols] = scaler_target.fit_transform(df[appliance_cols])

# ===========================
# DATASET & DATALOADER
# ===========================
dataset = NILMDataset(df, input_col="tot_active_pow", target_cols=appliance_cols, window_size=WINDOW_SIZE, stride=STRIDE)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ===========================
# MODEL
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Energformer(input_features=1, d_model=64, n_heads=4, n_layers=4, 
                    num_appliances=len(appliance_cols), dropout=0.1)

#pour l entrainement dU MODELE TransformerMultiOutputNILM
# model = TransformerMultiOutputNILM( d_model=64, n_heads=4, n_layers=3, 
#                     num_appliances=len(appliance_cols), dropout=0.1) 

model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===========================
# TRAIN LOOP
# ===========================
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            val_loss += criterion(preds, y).item()

    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

# ===========================
# Sauvegarde du modèle
# ===========================
torch.save(model.state_dict(), "transformer_energformer_chann5.pt")
print("Modèle sauvegardé sous transformer_energformer.pt")
