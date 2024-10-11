import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.io.wavfile import write

def get_fft_image(image_path):
    # Load the original image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Perform 2D FFT and shift the zero-frequency component to the center
    fft_img = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft_img)
    
    return fft_shifted

def extract_sound_from_fft(fft_2d, sample_rate=44100, video_fps=30):
    """
    Create sound from a 2D frequency domain of an image by taking the mean
    of each column, applying IFFT to create a sound signal, and adjusting the
    time-domain signal by zero-padding or trimming in the frequency domain.

    Args:
        fft_2d (ndarray): 2D FFT of a video frame (image in frequency domain).
        sample_rate (int): The sample rate for the output audio.
        video_fps (int): The frames per second (FPS) of the video.

    Returns:
        audio_signal (ndarray): The time-domain audio signal for one video frame.
    """
    
    # Take the mean of each column to create a 1D frequency signal
    mean_column_frequencies = np.mean(fft_2d, axis=0)
    
    # Get the current length of the frequency data
    current_freq_len = len(mean_column_frequencies)
    
    # Calculate the number of samples needed for one frame duration in the time domain
    num_samples = int(sample_rate / video_fps)
    
    # Adjust the freq domain size to fit the required audio signal samples
    if num_samples > current_freq_len:
        # If more time-domain samples are needed, zero-pad the frequency domain
        pad_length = num_samples - current_freq_len
        # Zero-padding symmetrically on both sides
        padded_frequencies = np.pad(mean_column_frequencies, (pad_length // 2, pad_length - pad_length // 2), 'constant')

    elif num_samples < current_freq_len:
        # If fewer samples are needed, trim the frequency domain
        trim_amount = current_freq_len - num_samples
        # Trim symmetrically from both ends
        trimmed_frequencies = mean_column_frequencies[trim_amount // 2 : current_freq_len - (trim_amount - trim_amount // 2)]
        padded_frequencies = trimmed_frequencies
    else:
        # If the number of samples is exactly the same, no need to pad or trim
        padded_frequencies = mean_column_frequencies
    
    # Apply IFFT to get the time-domain signal
    time_domain_signal = np.fft.ifft(padded_frequencies)
    
    # Take the real part of the IFFT result (since audio signals are real)
    audio_signal = np.real(time_domain_signal)
    
    # Normalize the audio signal to the range [-1, 1]
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    
    # Scale the audio signal to the range of int16 for wav file output
    audio_signal = np.int16(audio_signal * 32767)

    return audio_signal

# Load the FFT image
fft_image = get_fft_image("2Dfft/kaboot.jpg")

# Calculate the mean column frequencies from the FFT image
mean_column_frequencies = (np.mean(fft_image, axis=0))

# Take the magnitude spectrum
magnitude_spectrum = np.abs(mean_column_frequencies)

# Perform the inverse FFT on the mean column frequencies
time_domain_signal = np.fft.ifft(mean_column_frequencies)
real_time_domain_signal = np.real(time_domain_signal)

# Create subplots to display the frequency domain and time domain in a single window
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# Plot the magnitude spectrum (frequency domain)
axs[0].plot(magnitude_spectrum)
axs[0].set_title('Mean Column Frequencies (Magnitude Spectrum)')
axs[0].set_xlabel('Frequency Bin')
axs[0].set_ylabel('Magnitude')
axs[0].grid(True)

# Plot the time-domain signal (inverse 1D FFT)
axs[1].plot(real_time_domain_signal)
axs[1].set_title('Inverse 1D FFT of Mean Column Frequencies (Time Domain)')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Amplitude')
axs[1].grid(True)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

# Generate the audio signal from the FFT
audio_signal = extract_sound_from_fft(fft_2d=fft_image, video_fps=1)
audio_signal = np.array(audio_signal, dtype=np.int16)

# Save the audio as a .wav file
sample_rate = 44100
write('audio1.wav', sample_rate, audio_signal)
