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