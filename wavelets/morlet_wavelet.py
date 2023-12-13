import numpy as np
import matplotlib.pyplot as plt

def morlet_wavelet(frequency, time):
    w0 = 6  # Typical value for the Morlet wavelet
    return np.pi**(-0.25) * np.exp(1j * w0 * time) * np.exp(-time**2 / 2)

# Time vector
t = np.linspace(-2, 2, 1000)

# Frequency
freq = 5  # Hz

# Generate Morlet wavelet
wavelet = morlet_wavelet(freq, t)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t, np.real(wavelet), label='Real Part')
plt.plot(t, np.imag(wavelet), label='Imaginary Part')
plt.title('Morlet Wavelet')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()