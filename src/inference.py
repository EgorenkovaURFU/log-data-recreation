import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


class RealTimeInpainter:
    def __init__(self, model, scaler, seq_len=256, n_channels=2):
        """
        model: обученная WaveNet (уже на device)
        scaler: обученный RobustScaler для целевых кривых
        seq_len: длина окна
        n_channels: количество кривых (GR, NPHI)
        """
        self.model = model
        self.scaler = scaler
        self.seq_len = seq_len
        self.n_channels = n_channels
        
        # Буферы
        self.noisy_buffer = np.zeros((seq_len, n_channels), dtype=np.float32)
        self.mask_buffer = np.zeros((seq_len, n_channels), dtype=np.float32)
        self.ptr = 0
        self.full = False  # заполнен ли буфер полностью
    
    def update(self, raw_measurements):
        """
        raw_measurements: list/array длиной n_channels (значения или NaN)
        Возвращает исправленное значение (list длиной n_channels) или None, если буфер ещё не полон.
        """
        # Преобразуем в numpy и записываем в кольцевой буфер
        noisy = np.array(raw_measurements, dtype=np.float32)
        mask = np.isnan(noisy).astype(np.float32)
        noisy = np.nan_to_num(noisy, nan=0.0)
        
        self.noisy_buffer[self.ptr] = noisy
        self.mask_buffer[self.ptr] = mask
        self.ptr += 1
        
        if not self.full:
            if self.ptr >= self.seq_len:
                self.full = True
                self.ptr = 0  # переходим в кольцевой режим
            else:
                return None  # ещё нет полного окна
        
        if self.ptr == self.seq_len:
            self.ptr = 0  # закольцовываем
        
        # Подготавливаем входной тензор
        x_in = np.concatenate([self.noisy_buffer, self.mask_buffer], axis=-1)  # (seq_len, 4)
        x_tensor = torch.from_numpy(x_in).unsqueeze(0).to(next(self.model.parameters()).device)
        
        with torch.no_grad():
            pred_normalized = self.model(x_tensor).cpu().numpy()[0]  # (2,)
        
        # Обратное преобразование в исходные единицы
        pred_normalized = pred_normalized.reshape(1, -1)
        pred_original = self.scaler.inverse_transform(pred_normalized)[0]
        
        return pred_original.tolist()  # [GR, NPHI]
    


# Создаём экземпляр обработчика
inpainter = RealTimeInpainter(model, scaler, seq_len=SEQ_LEN, n_channels=2)

# Имитация real-time: идём по тестовому участку по одному замеру
stream_predictions = []
stream_truth = []

for idx in test_indices:
    raw = df_noisy.iloc[idx].values  # [LFP_GR_noisy, LFP_NPHI_noisy] с NaN
    corrected = inpainter.update(raw)
    if corrected is not None:
        stream_predictions.append(corrected)
        stream_truth.append(df_clean.iloc[idx].values)

stream_preds = np.array(stream_predictions)
stream_truth = np.array(stream_truth)

# Визуализация одного канала (GR) в оригинальных единицах
fig, ax = plt.subplots(figsize=(12, 6))
d = depths_test[-len(stream_truth):]  # глубины для тех точек, где уже был вывод
ax.plot(stream_truth[:, 0], d, 'grey', label='Чистый (ориг.)')
ax.plot(stream_preds[:, 0], d, 'b--', label='Потоковый WaveNet')
ax.invert_yaxis()
ax.set_xlabel('GR (API)')
ax.legend()
ax.grid(True)
ax.set_title('Потоковое восстановление в реальном времени (имитация)')
plt.show()

# Метрики в исходных единицах
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae_gr = mean_absolute_error(stream_truth[:, 0], stream_preds[:, 0])
rmse_gr = np.sqrt(mean_squared_error(stream_truth[:, 0], stream_preds[:, 0]))
print(f"Потоковый GR — MAE: {mae_gr:.4f}, RMSE: {rmse_gr:.4f}")