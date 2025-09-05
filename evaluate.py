import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from src.dataset import NILMDataset
from src.model_nilm import TransformerMultiOutputNILM
from energformer import Energformer
from sklearn.metrics import mean_absolute_error
import numpy as np



# ===========================
# CONFIG
# ===========================
CSV_PATH = "./dataset/channel_2.csv"
WINDOW_SIZE = 256
STRIDE = 1
BATCH_SIZE = 32
#changer le MODEL_PATH selon le MODELE entrainé et le channel utilisé
MODEL_PATH = "transformer_nilm.pt"

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

dataset = NILMDataset(df, input_col="tot_active_pow", target_cols=appliance_cols, window_size=WINDOW_SIZE, stride=STRIDE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# ===========================
# LOAD MODEL
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#pour l evaluation du MODELE TransformerMultiOutputNILM
# model = TransformerMultiOutputNILM(  d_model=64, 
#     n_heads=4, 
#     n_layers=3,   # <--- essaie 3 au lieu de 4
#     num_appliances=len(appliance_cols), 
#     dropout=0.1)
model = Energformer(input_features=1, d_model=64, n_heads=4, n_layers=4, 
                    num_appliances=len(appliance_cols), dropout=0.1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ===========================
# PREDICTIONS
# ===========================
y_true_all = []
y_pred_all = []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        preds = model(x).cpu()
        y_pred_all.append(preds)
        y_true_all.append(y)

y_pred_all = torch.cat(y_pred_all).numpy()
y_true_all = torch.cat(y_true_all).numpy()

# Dénormalisation
y_pred_all = scaler_target.inverse_transform(y_pred_all)
y_true_all = scaler_target.inverse_transform(y_true_all)

# ===========================
# METRICS
# ===========================
def sae(y_true, y_pred):
    """Signal Aggregate Error = (sum(pred) - sum(true)) / sum(true)"""
    return np.abs(np.sum(y_pred) - np.sum(y_true)) / np.sum(y_true)

def teca(y_true, y_pred):
    """Total Energy Correctly Assigned = sum(min(y_true, y_pred)) / sum(y_true)"""
    return np.sum(np.minimum(y_true, y_pred)) / np.sum(y_true)

def nde(y_true, y_pred):
    """Normalized Disaggregation Error"""
    return np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2)

results = {}

for i, col in enumerate(appliance_cols):
    mae = mean_absolute_error(y_true_all[:, i], y_pred_all[:, i])
    sae_val = sae(y_true_all[:, i], y_pred_all[:, i])
    teca_val = teca(y_true_all[:, i], y_pred_all[:, i])
    nde_val = nde(y_true_all[:, i], y_pred_all[:, i])

    results[col] = {"MAE": mae, "SAE": sae_val, "TECA": teca_val, "NDE": nde_val}

    # Plot préd vs vrai
    plt.figure(figsize=(12, 4))
    plt.plot(y_true_all[:6000, i], label=f"Vrai ({col})")
    plt.plot(y_pred_all[:6000, i], label=f"Prédit ({col})")
    plt.title(f"Prédiction vs Ground Truth - {col}\nMAE={mae:.4f}, SAE={sae_val:.4f}, TECA={teca_val:.4f}, NDE={nde_val:.4f}")
    plt.xlabel("Time Step")
    plt.ylabel("Consommation")
    plt.legend()
    plt.show()

# ===========================
# Résumé des métriques
# ===========================
print("\n==== Résultats des métriques par appareil ====")
for col, vals in results.items():
    print(f"{col}: MAE={vals['MAE']:.4f} | SAE={vals['SAE']:.4f} | TECA={vals['TECA']:.4f} | NDE={vals['NDE']:.4f}")
