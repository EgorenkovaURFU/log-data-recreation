import lasio
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CausalWellLogDataset
from model import CausalWaveNet
from preprocess import add_synthetic_noise


# Hyperparameters
SEQ_LEN = 256
BATCH_SIZE = 128
EPOCHS = 100
LR = 3e-4


# Loading and preprocessing data
las = lasio.read("../data/159-19BT2_LFP.las")
print("Доступные кривые:", [c.mnemonic for c in las.curves])

df = las.df()
target_curves = ['LFP_GR', 'LFP_NPHI']
df_clean = df[target_curves].copy()
depth = df.index

# Диагностика: смотрим, где пропуски
print("Пропуски в целевых кривых до обработки:")
print(df_clean.isnull().sum())


# Заполняем пропуски линейной интерполяцией
df_clean = df_clean.interpolate(method='linear', limit_direction='both')
# Если остались в начале/конце, заполняем ближайшим значением
df_clean = df_clean.bfill().ffill()

print("Пропуски после обработки:")
print(df_clean.isnull().sum())

# Убедимся, что их больше нет
assert not df_clean.isnull().any().any(), "Остались NaN в целевых кривых!"


scaler = RobustScaler()
scaled_vals = scaler.fit_transform(df_clean)
df_scaled = pd.DataFrame(scaled_vals, columns=target_curves, index=depth)


# Add synthetic noise and create mask for missing values
noisy_arrays = []
mask_arrays = []
for col in target_curves:
    sig = df_scaled[col].values
    noisy, mask = add_synthetic_noise(
        sig,
        noise_std=0.08,
        spike_prob=0.005,
        spike_intensity=4.0,
        missing_interval_prob=0.008,
        missing_interval_length=60
    )
    noisy_arrays.append(noisy)
    mask_arrays.append(mask)

# Create DataFrame for noisy data
df_noisy = pd.DataFrame(
    np.column_stack(noisy_arrays),
    columns=[f'{col}_noisy' for col in target_curves],
    index=depth
)
mask_combined = np.column_stack(mask_arrays)  # (N, 2)


# Dataset and DataLoader
dataset = CausalWellLogDataset(df_scaled, df_noisy, mask_combined, seq_len=SEQ_LEN)
split_idx = int(0.8 * len(dataset))

train_set, val_set = torch.utils.data.random_split(
    dataset,
    [split_idx, len(dataset) - split_idx],
    generator=torch.Generator().manual_seed(42)
)

print(f'Train samples: {len(train_set)}, Validation samples: {len(val_set)}')

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

print("Проверка данных:")
print(f"df_scaled has NaN: {df_scaled.isnull().any().any()}")
print(f"df_noisy has NaN: {df_noisy.isnull().any().any()}")

# Проверка одного батча
for x_batch, y_batch in train_loader:
    print(f"x_batch min/max: {x_batch.min().item():.4f} / {x_batch.max().item():.4f}")
    print(f"x_batch has NaN: {torch.isnan(x_batch).any()}")
    print(f"y_batch has NaN: {torch.isnan(y_batch).any()}")
    break



# Model, optimizer, loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CausalWaveNet(
    in_channels=4,
    hidden_channels=64,
    kernel_size=3,
    out_channels=2
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.L1Loss()


# Train and save the best model based on validation loss
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    # training
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x_batch.size(0)
    train_loss /= len(train_set)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            val_pred = model(x_val)
            v_loss = criterion(val_pred, y_val)
            val_loss += v_loss.item() * x_val.size(0)
    val_loss /= len(val_set)

    print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print('  Model saved as best_model.pth')

print(f' Best Val Loss: {best_val_loss:.4f}')


import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Загружаем лучшие веса
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Готовим "чистые" зашумлённые данные (NaN → 0)
noisy_clean = np.nan_to_num(df_noisy.values, nan=0.0)  # (N, 2)

# Тестовый диапазон (последние 20% скважины)
test_start = int(0.8 * len(depth))
test_indices = range(test_start + SEQ_LEN, len(depth))

depths_test = depth[test_indices]
preds = {col: [] for col in target_curves}
ground_truth = {col: [] for col in target_curves}

with torch.no_grad():
    for idx in test_indices:
        # Формируем окно (SEQ_LEN, 4)
        x_window = np.column_stack([
            noisy_clean[idx - SEQ_LEN:idx, 0],  # GR noisy
            noisy_clean[idx - SEQ_LEN:idx, 1],  # NPHI noisy
            mask_combined[idx - SEQ_LEN:idx, 0],# маска GR
            mask_combined[idx - SEQ_LEN:idx, 1] # маска NPHI
        ]).astype(np.float32)

        x_tensor = torch.from_numpy(x_window).unsqueeze(0).to(device)
        pred = model(x_tensor).cpu().numpy()[0]  # (2,)

        for i, col in enumerate(target_curves):
            preds[col].append(pred[i])
            ground_truth[col].append(df_scaled.iloc[idx][col])

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 10))
for i, col in enumerate(target_curves):
    ax = axes[i]
    d = depths_test
    true = ground_truth[col]
    pred = preds[col]
    noisy_signal = noisy_clean[test_start + SEQ_LEN:, i]
    mask = mask_combined[test_start + SEQ_LEN:, i].astype(bool)

    ax.plot(true, d, 'grey', label='Чистый (таргет)', alpha=0.7)
    ax.plot(noisy_signal, d, 'r-', alpha=0.5, label='Зашумлённый (real-time)')
    ax.plot(pred, d, 'b--', label='Восстановленный моделью')
    # Подсветка пропусков
    ax.fill_betweenx(d, ax.get_xlim()[0], ax.get_xlim()[1],
                     where=mask, color='lightblue', alpha=0.3, label='Пропуски')
    ax.invert_yaxis()
    ax.set_xlabel(col)
    ax.legend(fontsize=8)
    ax.grid(True)

plt.suptitle('Восстановление зашумлённых ГИС (MVP)', fontsize=14)
plt.tight_layout()
plt.show()

# Метрики
for col in target_curves:
    true = ground_truth[col]
    pred = preds[col]
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    print(f"{col} — MAE: {mae:.4f}, RMSE: {rmse:.4f}")

print(type(preds))
print(type(depths_test))
print(type(ground_truth))

import pickle

# Сохранение
def save_data_pickle(data1, data2, data3, filename='data.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((data1, data2, data3), f)


# Сохраняем
save_data_pickle(preds, depths_test, ground_truth, 'mydata.pkl')
