"""
Master script: Run weeks 1–6 computations, generate plots & data for mid-semester presentation.
Usage: python run_weeks_1_to_6.py
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from data_loader import load_train, load_test, SENSOR_COLS, SETTING_COLS
from preprocess import (preprocess_pipeline, INFORMATIVE_SENSORS_FD001,
                        apply_savgol_smoothing, fit_scaler, apply_scaler,
                        train_val_split)
from features import extract_features, extract_windowed_features
from train import train_model, compute_metrics
from models.lstm import LSTMModel
from models.cnn_lstm import CNNLSTMModel
from models.cnn_transformer import CNNTransformerModel

os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)
os.makedirs(os.path.join(ROOT, 'checkpoints'), exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

# ═══════════════════════════════════════════════════════════════
# WEEK 1 — EDA
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("WEEK 1 — Exploratory Data Analysis")
print("="*60)

TEMP_W1 = os.path.join(ROOT, 'week01-eda', 'temp')
os.makedirs(TEMP_W1, exist_ok=True)

df_train = load_train(fd_number=1, rul_cap=125)
df_test, rul_true = load_test(fd_number=1)
print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

# --- Engine lifetime distribution ---
cycles_per_engine = df_train.groupby('unit_id')['cycle'].max()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(cycles_per_engine, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(cycles_per_engine.mean(), color='red', linestyle='--', label=f'Mean: {cycles_per_engine.mean():.0f}')
axes[0].set_xlabel('Engine Lifetime (cycles)'); axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Engine Lifetimes (FD001 Train)'); axes[0].legend()
sorted_cycles = cycles_per_engine.sort_values()
axes[1].bar(range(len(sorted_cycles)), sorted_cycles.values, color='steelblue', alpha=0.7)
axes[1].set_xlabel('Engine (sorted)'); axes[1].set_ylabel('Lifetime (cycles)')
axes[1].set_title('Engine Lifetimes — Sorted')
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W1, 'engine_lifetimes.png'), dpi=150, bbox_inches='tight'); plt.close()

# --- RUL distributions ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(df_train['RUL'], bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0].set_xlabel('RUL (capped at 125)'); axes[0].set_ylabel('Count')
axes[0].set_title('RUL Distribution — All Training Rows')
axes[1].hist(rul_true, bins=20, edgecolor='black', alpha=0.7, color='seagreen')
axes[1].set_xlabel('True RUL'); axes[1].set_ylabel('Count')
axes[1].set_title('True Test RUL Distribution')
for uid in [1, 25, 50, 75, 100]:
    unit = df_train[df_train['unit_id'] == uid]
    axes[2].plot(unit['cycle'], unit['RUL'], label=f'Engine {uid}', alpha=0.8)
axes[2].set_xlabel('Cycle'); axes[2].set_ylabel('RUL'); axes[2].set_title('RUL Labeling — Sample Engines'); axes[2].legend(fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W1, 'rul_distributions.png'), dpi=150, bbox_inches='tight'); plt.close()

# --- Sensor variance ---
sensor_variance = df_train[SENSOR_COLS].var().sort_values()
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['red' if v < 1e-4 else 'steelblue' for v in sensor_variance.values]
sensor_variance.plot(kind='bar', ax=ax, color=colors, edgecolor='black', alpha=0.8)
ax.set_ylabel('Variance'); ax.set_title('Sensor Variance (red = near-zero → drop)')
ax.set_yscale('log'); plt.xticks(rotation=45, ha='right')
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W1, 'sensor_variance.png'), dpi=150, bbox_inches='tight'); plt.close()

low_var_sensors = sensor_variance[sensor_variance < 1e-4].index.tolist()
informative_sensors = INFORMATIVE_SENSORS_FD001
print(f"Low-variance sensors to drop: {low_var_sensors}")
print(f"Informative sensors: {informative_sensors}")

# --- Sensor degradation trends ---
sample_engines = [1, 50, 100]
fig, axes = plt.subplots(4, 1, figsize=(16, 14))
show_sensors_groups = [
    ['sensor_2', 'sensor_3', 'sensor_4'],
    ['sensor_7', 'sensor_11', 'sensor_15'],
    ['sensor_12', 'sensor_13', 'sensor_14'],
    ['sensor_17', 'sensor_20', 'sensor_21']
]
for gidx, group in enumerate(show_sensors_groups):
    ax = axes[gidx]
    for sensor in group:
        for eng_id in sample_engines:
            unit = df_train[df_train['unit_id'] == eng_id]
            ax.plot(unit['cycle'], unit[sensor], alpha=0.5, linewidth=0.8)
        # Just add one label per sensor
        ax.plot([], [], label=sensor)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylabel('Value')
    if gidx == 0: ax.set_title('Sensor Degradation Trends (engines 1, 50, 100)')
    if gidx == 3: ax.set_xlabel('Cycle')
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W1, 'sensor_trends.png'), dpi=150, bbox_inches='tight'); plt.close()

# --- Correlation heatmap ---
corr_cols = informative_sensors + ['RUL']
corr_matrix = df_train[corr_cols].corr()
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, square=True)
ax.set_title('Correlation Matrix: Informative Sensors + RUL')
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W1, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight'); plt.close()

rul_corr = corr_matrix['RUL'].drop('RUL').abs().sort_values(ascending=False)

# Save EDA summary
eda_summary = {
    'n_train_engines': int(df_train['unit_id'].nunique()),
    'n_test_engines': int(len(rul_true)),
    'avg_lifetime': float(cycles_per_engine.mean()),
    'min_lifetime': int(cycles_per_engine.min()),
    'max_lifetime': int(cycles_per_engine.max()),
    'informative_sensors': informative_sensors,
    'dropped_sensors': low_var_sensors,
    'rul_cap': 125,
    'top_correlated_sensors': {k: float(v) for k, v in rul_corr.head(6).items()},
    'test_rul_min': int(rul_true.min()),
    'test_rul_max': int(rul_true.max()),
    'test_rul_mean': float(rul_true.mean()),
}
with open(os.path.join(ROOT, 'reports', 'eda_summary.json'), 'w') as f:
    json.dump(eda_summary, f, indent=2)
print("Week 1 EDA complete — plots saved.")

# ═══════════════════════════════════════════════════════════════
# WEEK 2 — Baselines
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("WEEK 2 — Simple ML Baselines")
print("="*60)

TEMP_W2 = os.path.join(ROOT, 'week02-baselines', 'temp')
os.makedirs(TEMP_W2, exist_ok=True)

df_train_raw = load_train(fd_number=1, rul_cap=125)
df_tr, df_vl = train_val_split(df_train_raw, val_fraction=0.2, random_state=42)
sensors = INFORMATIVE_SENSORS_FD001

# Use windowed features (sliding window with stride=10) for train/val
# to get multiple labeled samples per engine (not just last cycle where RUL=0)
feat_train = extract_windowed_features(df_tr, sensors, window=30, stride=10)
feat_val = extract_windowed_features(df_vl, sensors, window=30, stride=10)
feat_test = extract_features(df_test, sensors, window=30)  # test: last window per engine

feature_cols_train = [c for c in feat_train.columns if c not in ['unit_id', 'RUL', 'window_end_cycle']]
feature_cols_test = [c for c in feat_test.columns if c not in ['unit_id', 'RUL', 'total_cycles', 'max_cycle']]
# Align feature columns (windowed doesn't have some cols that single has and vice versa)
common_cols = sorted(set(feature_cols_train) & set(feature_cols_test))

X_train_bl = feat_train[common_cols].values; y_train_bl = feat_train['RUL'].values
X_val_bl = feat_val[common_cols].values; y_val_bl = feat_val['RUL'].values
X_test_bl = feat_test[common_cols].values; y_test_bl = rul_true
feature_cols = common_cols

models_bl = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
}

bl_results = []; bl_preds = {}
for name, model in models_bl.items():
    model.fit(X_train_bl, y_train_bl)
    y_val_p = model.predict(X_val_bl)
    y_test_p = model.predict(X_test_bl)
    vm = compute_metrics(y_val_bl, y_val_p)
    tm = compute_metrics(y_test_bl, y_test_p)
    print(f"  {name}: Val MAE={vm['MAE']:.2f}, Test MAE={tm['MAE']:.2f}, RMSE={tm['RMSE']:.2f}, NASA={tm['NASA_Score']:.0f}")
    bl_results.append({'Model': name, 'Val MAE': vm['MAE'], 'Test MAE': tm['MAE'],
                        'Test RMSE': tm['RMSE'], 'Test NASA Score': tm['NASA_Score']})
    bl_preds[name] = y_test_p

bl_results_df = pd.DataFrame(bl_results)
bl_results_df.to_csv(os.path.join(ROOT, 'reports', 'baseline_results.csv'), index=False)

# --- Baseline predictions scatter ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, (name, y_pred) in enumerate(bl_preds.items()):
    ax = axes[idx // 2, idx % 2]
    ax.scatter(y_test_bl, y_pred, alpha=0.6, s=30, edgecolors='none')
    lims = [0, max(y_test_bl.max(), y_pred.max()) + 10]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel('True RUL'); ax.set_ylabel('Predicted RUL')
    m = compute_metrics(y_test_bl, y_pred)
    ax.set_title(f'{name}\nMAE={m["MAE"]:.1f}, RMSE={m["RMSE"]:.1f}')
plt.suptitle('Baseline Models — Predicted vs True RUL', fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W2, 'baseline_scatter.png'), dpi=150, bbox_inches='tight'); plt.close()

# --- Feature importance ---
rf_model = models_bl['Random Forest']
importance = rf_model.feature_importances_
feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 8))
feat_imp.head(15).plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
ax.set_xlabel('Feature Importance'); ax.set_title('Random Forest — Top 15 Feature Importances')
ax.invert_yaxis()
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W2, 'feature_importance.png'), dpi=150, bbox_inches='tight'); plt.close()
print("Week 2 baselines complete — plots saved.")

# ═══════════════════════════════════════════════════════════════
# WEEK 3 — Health Index
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("WEEK 3 — Health Index Baseline")
print("="*60)

TEMP_W3 = os.path.join(ROOT, 'week03-health-index', 'temp')
os.makedirs(TEMP_W3, exist_ok=True)

df_train_hi = load_train(fd_number=1, rul_cap=None)
scaler_hi = MinMaxScaler()
df_train_hi_s = df_train_hi.copy()
df_train_hi_s[sensors] = scaler_hi.fit_transform(df_train_hi[sensors])

pca = PCA(n_components=1)
df_train_hi_s['HI_pca'] = pca.fit_transform(df_train_hi_s[sensors])

for uid in df_train_hi_s['unit_id'].unique():
    mask = df_train_hi_s['unit_id'] == uid
    hi = df_train_hi_s.loc[mask, 'HI_pca'].values
    if np.corrcoef(np.arange(len(hi)), hi)[0,1] > 0:
        hi = -hi
    hi_min, hi_max = hi.min(), hi.max()
    if hi_max > hi_min:
        df_train_hi_s.loc[mask, 'HI_pca'] = (hi - hi_min) / (hi_max - hi_min)
    else:
        df_train_hi_s.loc[mask, 'HI_pca'] = 0.5

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}")

# Plot health index
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for uid in [1, 25, 50, 75, 100]:
    unit = df_train_hi_s[df_train_hi_s['unit_id'] == uid]
    axes[0].plot(unit['cycle'], unit['HI_pca'], label=f'Engine {uid}', alpha=0.8)
axes[0].set_xlabel('Cycle'); axes[0].set_ylabel('Health Index (PCA)')
axes[0].set_title('PCA Health Index Over Engine Lifetime'); axes[0].legend()
axes[1].scatter(df_train_hi_s['HI_pca'], df_train_hi_s['RUL'], alpha=0.1, s=1)
axes[1].set_xlabel('Health Index'); axes[1].set_ylabel('RUL'); axes[1].set_title('Health Index vs RUL')
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W3, 'health_index.png'), dpi=150, bbox_inches='tight'); plt.close()

# Curve fitting
def exp_model(hi, a, b, c): return a * np.exp(b * hi) + c
sample_hi = df_train_hi_s[['HI_pca', 'RUL']].dropna().sample(n=5000, random_state=42)
try:
    popt, _ = curve_fit(exp_model, sample_hi['HI_pca'], sample_hi['RUL'], p0=[100, 2, 0], maxfev=10000)
    hi_fitted = True
except:
    hi_fitted = False
    popt = None

# Evaluate on test
df_test_hi_s = df_test.copy()
df_test_hi_s[sensors] = scaler_hi.transform(df_test[sensors])
test_last = df_test_hi_s.groupby('unit_id').tail(1).copy()
test_hi_vals = pca.transform(test_last[sensors]).flatten()

if hi_fitted:
    rul_pred_hi = np.clip(exp_model(test_hi_vals, *popt), 0, 200)
else:
    lr_hi = LinearRegression()
    lr_hi.fit(df_train_hi_s['HI_pca'].values.reshape(-1,1), df_train_hi_s['RUL'].values)
    rul_pred_hi = np.clip(lr_hi.predict(test_hi_vals.reshape(-1,1)), 0, 200)

metrics_hi = compute_metrics(rul_true, rul_pred_hi)
print(f"Health Index — MAE: {metrics_hi['MAE']:.2f}, RMSE: {metrics_hi['RMSE']:.2f}, NASA: {metrics_hi['NASA_Score']:.0f}")

hi_results = {'Model': 'Health Index (PCA)', 'Test MAE': metrics_hi['MAE'],
              'Test RMSE': metrics_hi['RMSE'], 'Test NASA Score': metrics_hi['NASA_Score']}
print("Week 3 Health Index complete.")

# ═══════════════════════════════════════════════════════════════
# WEEK 4 — LSTM
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("WEEK 4 — LSTM Model")
print("="*60)

TEMP_W4 = os.path.join(ROOT, 'week04-lstm-and-agent-skeleton', 'temp')
os.makedirs(TEMP_W4, exist_ok=True)

WINDOW = 30; BATCH_SIZE = 256
data = preprocess_pipeline(df_train, df_test, window_size=WINDOW, rul_cap=125)
print(f"X_train: {data['X_train'].shape}, X_test: {data['X_test'].shape}")

train_ds = TensorDataset(torch.tensor(data['X_train']), torch.tensor(data['y_train']))
val_ds = TensorDataset(torch.tensor(data['X_val']), torch.tensor(data['y_val']))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

n_features = data['config']['n_features']
lstm_model = LSTMModel(n_features=n_features, hidden_size=64, n_layers=2, dropout=0.3)
print(f"LSTM parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

lstm_result = train_model(lstm_model, train_loader, val_loader,
                          n_epochs=50, lr=1e-3, patience=10,
                          save_dir=os.path.join(ROOT, 'checkpoints'),
                          model_name='lstm_FD001', device=device)

# Training curves
h = lstm_result['history']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(h['train_loss'], label='Train'); axes[0].plot(h['val_loss'], label='Val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss'); axes[0].set_title('LSTM Training Curves'); axes[0].legend()
axes[1].plot(h['val_mae'], label='MAE'); axes[1].plot(h['val_rmse'], label='RMSE')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Error'); axes[1].set_title('LSTM Validation Metrics'); axes[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W4, 'lstm_training_curves.png'), dpi=150, bbox_inches='tight'); plt.close()

# Test eval
lstm_trained = lstm_result['model']; lstm_trained.eval()
X_test_t = torch.tensor(data['X_test'], dtype=torch.float32).to(lstm_result['device'])
with torch.no_grad():
    lstm_preds = lstm_trained(X_test_t).squeeze(-1).cpu().numpy()
lstm_metrics = compute_metrics(rul_true, lstm_preds)
print(f"LSTM — MAE: {lstm_metrics['MAE']:.2f}, RMSE: {lstm_metrics['RMSE']:.2f}, NASA: {lstm_metrics['NASA_Score']:.0f}")

# Scatter
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(rul_true, lstm_preds, alpha=0.6, s=30)
lims = [0, max(rul_true.max(), lstm_preds.max()) + 10]
ax.plot(lims, lims, 'r--'); ax.set_xlabel('True RUL'); ax.set_ylabel('Predicted RUL')
ax.set_title(f'LSTM — MAE={lstm_metrics["MAE"]:.1f}, RMSE={lstm_metrics["RMSE"]:.1f}')
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W4, 'lstm_scatter.png'), dpi=150, bbox_inches='tight'); plt.close()
print("Week 4 LSTM complete.")

# ═══════════════════════════════════════════════════════════════
# WEEK 5 — Paper A: CNN+LSTM with SG Smoothing
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("WEEK 5 — Paper A: CNN+LSTM with Savitzky-Golay Smoothing")
print("="*60)

TEMP_W5 = os.path.join(ROOT, 'week05-paper-a-cnn-lstm', 'temp')
os.makedirs(TEMP_W5, exist_ok=True)

# SG smoothing visualization
unit1_raw = df_train[df_train['unit_id'] == 1].copy()
unit1_sg = apply_savgol_smoothing(unit1_raw, sensors, window_length=11, polyorder=3)
show_s = ['sensor_2', 'sensor_4', 'sensor_11', 'sensor_13', 'sensor_15', 'sensor_21']
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for idx, sensor in enumerate(show_s):
    ax = axes[idx // 3, idx % 3]
    ax.plot(unit1_raw['cycle'], unit1_raw[sensor], alpha=0.5, label='Raw', linewidth=0.8)
    ax.plot(unit1_sg['cycle'], unit1_sg[sensor], label='SG Smoothed', linewidth=1.5)
    ax.set_title(sensor); ax.legend(fontsize=8)
plt.suptitle('Savitzky-Golay Smoothing — Engine 1', fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W5, 'sg_smoothing_effect.png'), dpi=150, bbox_inches='tight'); plt.close()

# 4 experiments
experiments = [
    {'name': 'LSTM_raw',     'model_class': LSTMModel,    'use_savgol': False},
    {'name': 'LSTM_sg',      'model_class': LSTMModel,    'use_savgol': True},
    {'name': 'CNN_LSTM_raw', 'model_class': CNNLSTMModel, 'use_savgol': False},
    {'name': 'CNN_LSTM_sg',  'model_class': CNNLSTMModel, 'use_savgol': True},
]

papera_results = []
for exp in experiments:
    print(f"\n  Experiment: {exp['name']}")
    data_exp = preprocess_pipeline(df_train, df_test, window_size=WINDOW,
                                   rul_cap=125, use_savgol=exp['use_savgol'])
    tds = TensorDataset(torch.tensor(data_exp['X_train']), torch.tensor(data_exp['y_train']))
    vds = TensorDataset(torch.tensor(data_exp['X_val']), torch.tensor(data_exp['y_val']))
    tl = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(vds, batch_size=BATCH_SIZE, shuffle=False)
    n_f = data_exp['config']['n_features']
    if exp['model_class'] == LSTMModel:
        m = LSTMModel(n_features=n_f, hidden_size=64, n_layers=2, dropout=0.3)
    else:
        m = CNNLSTMModel(n_features=n_f, seq_len=WINDOW, lstm_hidden=64, dropout=0.3)
    res = train_model(m, tl, vl, n_epochs=50, lr=1e-3, patience=10,
                      save_dir=os.path.join(ROOT, 'checkpoints'),
                      model_name=exp['name'], device=device)
    trained = res['model']; trained.eval()
    xt = torch.tensor(data_exp['X_test'], dtype=torch.float32).to(res['device'])
    with torch.no_grad():
        preds = trained(xt).squeeze(-1).cpu().numpy()
    met = compute_metrics(rul_true, preds)
    met['name'] = exp['name']
    papera_results.append(met)
    print(f"    MAE: {met['MAE']:.2f}, RMSE: {met['RMSE']:.2f}, NASA: {met['NASA_Score']:.0f}")

papera_df = pd.DataFrame(papera_results)
papera_df.to_csv(os.path.join(ROOT, 'reports', 'paperA_results.csv'), index=False)

# Bar comparison
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(papera_df))
bars = ax.bar(x_pos, papera_df['MAE'], color=['#4682B4','#5F9EA0','#D2691E','#CD853F'], edgecolor='black')
ax.set_xticks(x_pos); ax.set_xticklabels(papera_df['name'], rotation=20)
ax.set_ylabel('MAE'); ax.set_title('Paper A Experiments — MAE Comparison')
for b, v in zip(bars, papera_df['MAE']): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{v:.1f}', ha='center', fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W5, 'papera_comparison.png'), dpi=150, bbox_inches='tight'); plt.close()
print("Week 5 Paper A complete.")

# ═══════════════════════════════════════════════════════════════
# WEEK 6 — Paper B: CNN-Transformer + MC Dropout
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("WEEK 6 — Paper B: CNN-Transformer + MC Dropout UQ")
print("="*60)

TEMP_W6 = os.path.join(ROOT, 'week06-paper-b-transformer-uq', 'temp')
os.makedirs(TEMP_W6, exist_ok=True)

data_w6 = preprocess_pipeline(df_train, df_test, window_size=WINDOW, rul_cap=125)
ct_model = CNNTransformerModel(n_features=n_features, seq_len=WINDOW,
                                cnn_channels=64, d_model=64, nhead=4,
                                num_encoder_layers=2, dim_feedforward=128,
                                dropout=0.2, mc_dropout=0.1)
print(f"CNN-Transformer params: {sum(p.numel() for p in ct_model.parameters()):,}")

ct_tds = TensorDataset(torch.tensor(data_w6['X_train']), torch.tensor(data_w6['y_train']))
ct_vds = TensorDataset(torch.tensor(data_w6['X_val']), torch.tensor(data_w6['y_val']))
ct_tl = DataLoader(ct_tds, batch_size=BATCH_SIZE, shuffle=True)
ct_vl = DataLoader(ct_vds, batch_size=BATCH_SIZE, shuffle=False)

ct_result = train_model(ct_model, ct_tl, ct_vl, n_epochs=60, lr=1e-3, patience=15,
                        save_dir=os.path.join(ROOT, 'checkpoints'),
                        model_name='cnn_transformer_FD001', device=device)

# Training curves
hc = ct_result['history']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(hc['train_loss'], label='Train'); axes[0].plot(hc['val_loss'], label='Val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('CNN-Transformer Training'); axes[0].legend()
axes[1].plot(hc['val_mae'], label='MAE'); axes[1].plot(hc['val_rmse'], label='RMSE')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Error'); axes[1].set_title('CNN-Transformer Val Metrics'); axes[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W6, 'ct_training_curves.png'), dpi=150, bbox_inches='tight'); plt.close()

# MC Dropout inference
ct_trained = ct_result['model']
X_test_ct = torch.tensor(data_w6['X_test'], dtype=torch.float32).to(ct_result['device'])
MC_SAMPLES = 100
mean_pred, std_pred, all_preds = ct_trained.predict_with_uncertainty(X_test_ct, n_samples=MC_SAMPLES)

ct_metrics = compute_metrics(rul_true, mean_pred)
print(f"CNN-Transformer (MC mean) — MAE: {ct_metrics['MAE']:.2f}, RMSE: {ct_metrics['RMSE']:.2f}, NASA: {ct_metrics['NASA_Score']:.0f}")

# Uncertainty analysis
abs_errors = np.abs(mean_pred - rul_true)
corr = np.corrcoef(std_pred, abs_errors)[0, 1]
print(f"Uncertainty-Error correlation: {corr:.3f}")

# Prediction with uncertainty bands
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
ax.errorbar(rul_true, mean_pred, yerr=1.96*std_pred, fmt='o', alpha=0.5, capsize=2, markersize=4, elinewidth=0.5)
lims = [0, max(rul_true.max(), mean_pred.max()) + 10]
ax.plot(lims, lims, 'r--'); ax.set_xlabel('True RUL'); ax.set_ylabel('Predicted RUL')
ax.set_title('Predictions with 95% CI')
ax = axes[1]
x = np.arange(len(rul_true))
ax.bar(x, rul_true, width=0.4, label='True', alpha=0.7, color='steelblue')
ax.bar(x+0.4, mean_pred, width=0.4, label='Predicted', alpha=0.7, color='coral')
ax.errorbar(x+0.4, mean_pred, yerr=1.96*std_pred, fmt='none', ecolor='red', capsize=1, alpha=0.4)
ax.set_xlabel('Engine'); ax.set_ylabel('RUL'); ax.set_title('Per-Engine Comparison'); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W6, 'ct_predictions_with_ci.png'), dpi=150, bbox_inches='tight'); plt.close()

# Uncertainty sanity checks
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(std_pred, abs_errors, alpha=0.6)
axes[0].set_xlabel('Uncertainty (std)'); axes[0].set_ylabel('Absolute Error')
axes[0].set_title(f'Uncertainty vs Error (corr={corr:.3f})')
axes[1].hist(std_pred, bins=20, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Uncertainty (std)'); axes[1].set_title('Uncertainty Distribution')
axes[2].scatter(rul_true, std_pred, alpha=0.6)
axes[2].set_xlabel('True RUL'); axes[2].set_ylabel('Uncertainty (std)'); axes[2].set_title('True RUL vs Uncertainty')
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W6, 'uncertainty_analysis.png'), dpi=150, bbox_inches='tight'); plt.close()

# MC distribution for sample engines
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
sample_engs = [0, 20, 40, 60, 80, 99]
for idx, eng_idx in enumerate(sample_engs):
    ax = axes[idx // 3, idx % 3]
    ax.hist(all_preds[:, eng_idx], bins=30, edgecolor='black', alpha=0.7, density=True)
    ax.axvline(rul_true[eng_idx], color='red', ls='--', lw=2, label=f'True={rul_true[eng_idx]}')
    ax.axvline(mean_pred[eng_idx], color='blue', ls='-', lw=2, label=f'Mean={mean_pred[eng_idx]:.1f}')
    ax.set_title(f'Engine {eng_idx+1} (σ={std_pred[eng_idx]:.1f})')
    ax.legend(fontsize=8)
plt.suptitle('MC Dropout Prediction Distributions', fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig(os.path.join(TEMP_W6, 'mc_distributions.png'), dpi=150, bbox_inches='tight'); plt.close()

# Save combined summary
w6_summary = {
    'ct_mae': float(ct_metrics['MAE']),
    'ct_rmse': float(ct_metrics['RMSE']),
    'ct_nasa': float(ct_metrics['NASA_Score']),
    'uncertainty_error_corr': float(corr),
    'mean_uncertainty': float(std_pred.mean()),
}
with open(os.path.join(ROOT, 'reports', 'week6_summary.json'), 'w') as f:
    json.dump(w6_summary, f, indent=2)

print("Week 6 CNN-Transformer complete.")

# ═══════════════════════════════════════════════════════════════
# COMBINED RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("COMBINED RESULTS SUMMARY")
print("="*60)

all_model_results = []

# Baselines
for r in bl_results:
    all_model_results.append({'Model': r['Model'], 'MAE': r['Test MAE'],
                               'RMSE': r['Test RMSE'], 'NASA Score': r['Test NASA Score']})

# Health Index
all_model_results.append({'Model': 'Health Index (PCA)', 'MAE': metrics_hi['MAE'],
                            'RMSE': metrics_hi['RMSE'], 'NASA Score': metrics_hi['NASA_Score']})

# LSTM
all_model_results.append({'Model': 'LSTM', 'MAE': lstm_metrics['MAE'],
                            'RMSE': lstm_metrics['RMSE'], 'NASA Score': lstm_metrics['NASA_Score']})

# Paper A results
for r in papera_results:
    all_model_results.append({'Model': r['name'], 'MAE': r['MAE'],
                               'RMSE': r['RMSE'], 'NASA Score': r['NASA_Score']})

# CNN-Transformer
all_model_results.append({'Model': 'CNN-Transformer (MC)', 'MAE': ct_metrics['MAE'],
                            'RMSE': ct_metrics['RMSE'], 'NASA Score': ct_metrics['NASA_Score']})

all_results_df = pd.DataFrame(all_model_results)
all_results_df.to_csv(os.path.join(ROOT, 'reports', 'all_results.csv'), index=False)
print(all_results_df.to_string(index=False))

# Combined bar chart
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
models_list = all_results_df['Model'].values
x_pos = np.arange(len(models_list))

for ax, metric, color in zip(axes, ['MAE', 'RMSE', 'NASA Score'], ['steelblue', 'coral', 'seagreen']):
    vals = all_results_df[metric].values
    bars = ax.bar(x_pos, vals, color=color, edgecolor='black', alpha=0.8)
    ax.set_xticks(x_pos); ax.set_xticklabels(models_list, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(metric); ax.set_title(f'Model Comparison — {metric}')
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.01, f'{v:.1f}', ha='center', fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, 'reports', 'model_comparison.png'), dpi=150, bbox_inches='tight'); plt.close()

print("\n✅ All weeks 1–6 complete! Results saved to reports/")
print("Plots saved to week*/temp/ folders.")

# Save all metrics in JSON for presentation generation
final_data = {
    'eda': eda_summary,
    'baselines': bl_results,
    'health_index': hi_results,
    'lstm': {'MAE': lstm_metrics['MAE'], 'RMSE': lstm_metrics['RMSE'], 'NASA_Score': lstm_metrics['NASA_Score']},
    'paper_a': papera_results,
    'cnn_transformer': w6_summary,
}
with open(os.path.join(ROOT, 'reports', 'all_metrics.json'), 'w') as f:
    json.dump(final_data, f, indent=2, default=float)
