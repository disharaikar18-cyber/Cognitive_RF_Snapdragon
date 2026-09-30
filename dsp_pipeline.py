import numpy as np
from scipy import signal

def generate_synthetic_iq(signal_type="BPSK", num_samples=1024, snr_db=10):
    t = np.arange(num_samples)
    if signal_type == "BPSK":
        bits = np.random.randint(0, 2, num_samples // 16)
        symbols = 2 * bits - 1
        baseband = np.repeat(symbols, 16)
    elif signal_type == "QPSK":
        bits = np.random.randint(0, 4, num_samples // 16)
        symbols = np.exp(1j * (np.pi / 4 + bits * np.pi / 2))
        baseband = np.repeat(symbols, 16)
    else:  # Noise / CW
        baseband = np.exp(1j * 2 * np.pi * 0.1 * t)
        
    sig_power = np.mean(np.abs(baseband)**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * np.sqrt(noise_power / 2)
    
    return baseband + noise

def iq_to_spectrogram(iq_samples, nperseg=64, target_shape=(64, 64)):
    f, t, Sxx = signal.spectrogram(iq_samples, fs=1.0, nperseg=nperseg, return_onesided=False)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)
    norm_spec = (Sxx_dB - Sxx_dB.min()) / (Sxx_dB.max() - Sxx_dB.min() + 1e-10)
    
    # Ensure exact target shape (64, 64) via padding/truncating
    padded_spec = np.zeros(target_shape, dtype=np.float32)
    h_max = min(norm_spec.shape[0], target_shape[0])
    w_max = min(norm_spec.shape[1], target_shape[1])
    padded_spec[:h_max, :w_max] = norm_spec[:h_max, :w_max]
    
    return padded_spec

if __name__ == "__main__":
    sample_iq = generate_synthetic_iq("BPSK")
    spec = iq_to_spectrogram(sample_iq)
    print(f"Generated Spectrogram Shape: {spec.shape}")