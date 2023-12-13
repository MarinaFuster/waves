import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import stft

def spectral_decomposition(eeg_data, sampling_rate):
    """
    Perform spectral decomposition of EEG data using Short-Time Fast Fourier Transform (STFT).

    Parameters:
    eeg_data (numpy.ndarray): 2D array of EEG data (channels x time).
    sampling_rate (float): Sampling rate of the EEG data in Hz.

    Returns:
    numpy.ndarray: 3D array of spectral power (channels x frequencies x time).
    """

    # Parameters for STFT
    nperseg = 256 * 10  # Length of each segment
    noverlap = nperseg * 0.5  # Number of points to overlap between segments (typically 50%-75% of nperseg)

    # Initialize an array to hold the spectral power
    freqs, times, _ = stft(eeg_data[0], fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
    spectral_power = np.zeros((eeg_data.shape[0], len(freqs), len(times)))

    # Perform STFT for each channel
    for i in range(eeg_data.shape[0]):
        _, _, Zxx = stft(eeg_data[i], fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
        spectral_power[i] = np.abs(Zxx) ** 2

    return spectral_power, freqs, times

# Example usage
eeg_data = np.random.randn(1, 10000)  # Replace with actual EEG data

print(eeg_data.shape)

sampling_rate = 256  # Replace with actual sampling rate
spectral_data, freqs, times = spectral_decomposition(eeg_data, sampling_rate)

print("Spectral data")
print(spectral_data.shape)
print("Freqs")
print(freqs.shape)
print("Times")
print(times.shape)

def plot_spectrogram(spectral_power, freqs, times):
    """
    Plot a spectrogram with logarithmic frequency axis.

    Parameters:
    spectral_power (numpy.ndarray): 2D array of spectral power (frequencies x times).
    freqs (numpy.ndarray): Array of frequencies.
    times (numpy.ndarray): Array of times.
    """
    # Create a figure and a subplot
    _, ax = plt.subplots(figsize=(14, 4))
    
    # Convert power to decibels (dB)
    power_dB = 10 * np.log10(spectral_power)
    
    # Use pcolormesh to plot the spectrogram with a logarithmic scale
    # Note: We add a small value to freqs to avoid taking log of zero
    #pcm = ax.pcolormesh(times, np.log10(freqs + 1e-6), power_dB, shading='gouraud')
    pcm = ax.pcolormesh(times, freqs, power_dB, shading='gouraud')

    
    # Set the y-axis to a logarithmic scale
    #ax.set_yscale('log')
    
    # Set the axis labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    
    # Set the frequency ticks to be formatted as powers of 10
    #ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    #ax.set_yticks([0.1, 1, 10, 100])
    
    # Set the range of the colormap and add a colorbar
    plt.colorbar(pcm, ax=ax, label='Power (dB)')
    
    # Set the title of the plot
    ax.set_title('Frequency Decomposition')
    
    # Show the plot
    plt.show()

# Example usage:
# Assume spectral_power, freqs, and times are defined as per your spectral analysis output.
# Replace these with your actual data before calling the function.
# spectral_power = np.random.rand(100, 1000)  # Example spectral power data
plot_spectrogram(np.squeeze(spectral_data, axis=0), freqs, times)
