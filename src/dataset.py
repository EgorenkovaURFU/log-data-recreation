import torch
from torch.utils.data import Dataset
import numpy as np


class CausalWellLogDataset(Dataset):

    def __init__(self, clean_df, noisy_df, mask_array, seq_len=256):
        """
        clean_df: DataFrame with clean well log data with columns ['LFP_GR', 'LFP_NPHI']
        noisy_df: DataFrame with noisy well log data with columns ['LFP_GR', 'LFP_NPHI']
        mask_array: NumPy array (N, 2) with True for missing values
        seq_len: Length of the sequence for causal modeling
        """

        self.seq_len = seq_len
        self.clean_df = clean_df.values.astype(np.float32)
        self.noisy_df = noisy_df.values.astype(np.float32)
        self.mask_array = mask_array.astype(np.float32)
        self.length = len(clean_df)

    def __len__(self):
        return self.length - self.seq_len

    def __getitem__(self, idx):
        end = idx + self.seq_len

        # input: concatenate noisy data and mask
        noisy_window = self.noisy_df[idx:end]
        mask_window = self.mask_array[idx:end]

        # repalace NaN with 0 in noisy data
        noisy_window = np.nan_to_num(noisy_window, nan=0.0)
        x = np.concatenate([noisy_window, mask_window], axis=-1) # shape (seq_len, 4)

        # target: clean data at the end of the window (end-1)
        y = self.clean_df[end-1] # shape (2,)
        
        return torch.from_numpy(x), torch.from_numpy(y)
    





        x = self.data.iloc[idx:idx + self.window_size].drop(columns=[self.target_col]).values
        y = self.data.iloc[idx + self.window_size][self.target_col]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)