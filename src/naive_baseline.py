import lasio
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocess import add_synthetic_noise
import pickle
from dataset import CausalWellLogDataset


SEQ_LEN = 256


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

dataset = CausalWellLogDataset(df_scaled, df_noisy, mask_combined, seq_len=SEQ_LEN)
test_start = int(0.8 * len(depth))
test_indices = range(test_start + SEQ_LEN, len(depth))
depths_test = depth[test_indices]

def load_data_pickle(filename='data.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
preds, depths_test, ground_truth = load_data_pickle('mydata.pkl')


las = lasio.read("../data/159-19BT2_LFP.las")
print("Доступные кривые:", [c.mnemonic for c in las.curves])

df = las.df()
target_curves = ['LFP_GR', 'LFP_NPHI']
df_clean = df[target_curves].copy()
depth = df.index

scaler = RobustScaler()
scaled_vals = scaler.fit_transform(df_clean)
df_scaled = pd.DataFrame(scaled_vals, columns=target_curves, index=depth)

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

df_noisy = pd.DataFrame(
    np.column_stack(noisy_arrays),
    columns=[f'{col}_noisy' for col in target_curves],
    index=depth
)
mask_combined = np.column_stack(mask_arrays)  # (N, 2)

noisy_clean = np.nan_to_num(df_noisy.values, nan=0.0)  # (N, 2)

# Восстановление baseline'ом
baseline_preds = {col: [] for col in target_curves}

for i, col in enumerate(target_curves):
    sig_noisy = noisy_clean[:, i]
    mask = mask_combined[:, i]
    
    # Медианный фильтр (окно 5)
    filtered = median_filter(sig_noisy, size=5)
    
    # Интерполяция пропусков
    depth_arr = np.arange(len(sig_noisy))
    good_idx = depth_arr[mask == 0]
    if len(good_idx) > 1:
        interp_func = interp1d(good_idx, filtered[good_idx], kind='linear',
                               fill_value='extrapolate', bounds_error=False)
        filled = interp_func(depth_arr)
    else:
        filled = filtered.copy()
    baseline_preds[col] = filled[test_start + SEQ_LEN:]

# --- 1. Восстановление baseline'ом ---
# Будущий массив предсказаний baseline'а (той же длины, что и test_indices)
baseline_preds = {col: [] for col in target_curves}

for i, col in enumerate(target_curves):
    sig_noisy = noisy_clean[:, i]
    mask = mask_combined[:, i]
    
    # Медианный фильтр (окно 5)
    filtered = median_filter(sig_noisy, size=5)
    
    # Интерполяция пропусков на всей длине
    depth_arr = np.arange(len(sig_noisy))  # номера отсчётов
    good_idx = depth_arr[mask == 0]
    if len(good_idx) > 1:
        interp_func = interp1d(good_idx, filtered[good_idx], kind='linear',
                               fill_value='extrapolate', bounds_error=False)
        filled = interp_func(depth_arr)
    else:
        filled = filtered.copy()
    
    # Берём значения ровно для тех же глубин, что и в test_indices
    baseline_preds[col] = filled[test_indices]

# --- 2. Метрики baseline'а ---
print("=== Baseline (Median + Linear Interpolation) ===")
for col in target_curves:
    true = ground_truth[col]
    base = baseline_preds[col]
    mae = mean_absolute_error(true, base)
    rmse = np.sqrt(mean_squared_error(true, base))
    print(f"{col} — MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# --- 3. Сравнительный график (GR) ---
fig, ax = plt.subplots(figsize=(12, 6))
col = 'LFP_GR'
# Нарежем зашумлённый сигнал для test_indices
noisy_test_slice = noisy_clean[test_indices, 0]

ax.plot(ground_truth[col], depths_test, 'grey', label='Чистый', alpha=0.7)
ax.plot(noisy_test_slice, depths_test, 'r-', alpha=0.5, label='Зашумлённый')
ax.plot(preds[col], depths_test, 'b--', label='WaveNet')
ax.plot(baseline_preds[col], depths_test, 'g-.', label='Baseline (Med+Interp)')
ax.invert_yaxis()
ax.set_xlabel(col)
ax.legend()
ax.grid(True)
ax.set_title('Сравнение WaveNet vs Baseline (GR)')
plt.tight_layout()
plt.show()