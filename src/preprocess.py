import numpy as np



def add_synthetic_noise(signal, noise_std=0.1, spike_prob=0.01, spike_intensity=5, 
                        missing_interval_prob=0.008, missing_interval_length=60, seed=42):
    
    rng = np.random.default_rng(seed)
    noisy = signal.copy()
    mask = np.zeros(len(signal), dtype=bool)

    # Gaussian noise
    noisy += rng.normal(0, noise_std, size=signal.shape)

    # Random spikes
    spike_indices = rng.random(len(signal)) < spike_prob
    direction = rng.choice([-1, 1], size=spike_indices.sum())
    noisy[spike_indices] += direction * spike_intensity * (0.5 + rng.random(spike_indices.sum()))

    # Missing intervals
    i = 0
    while i < len(signal):
        if rng.random() < missing_interval_prob:
            gap_length = rng.integers(5, missing_interval_length + 1)
            gap_length = min(gap_length, len(signal) - i)
            mask[i:i+gap_length] = True
            noisy[i:i+gap_length] = np.nan
            i += gap_length
        else:
            i += 1
    return noisy, mask