import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.io.wavfile import write




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
    
    # adjust the freq domain size to fit the required audio signal samples
    if num_samples > current_freq_len:
        # If more time-domain samples are needed, zero-pad the frequency domain
        pad_length = num_samples - current_freq_len
        # Zero-padding symetrically on both sides
        padded_frequencies = np.pad(mean_column_frequencies, (pad_length // 2, pad_length - pad_length // 2), 'constant')

    elif num_samples < current_freq_len:
        # If fewer samples are needed, trim the frequency domain
        trim_amount = current_freq_len - num_samples
        # Trim symetrically from both ends
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



def video_to_frames(video_path):
    # Create a VideoCapture object to read the video
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    success, frame = cap.read()
    
    # Loop through the video and extract each frame
    while success:
        frames.append(frame)  # Add the frame to the list
        success, frame = cap.read()  # Read the next frame
    
    cap.release()  # Release the video capture object
    return frames


def fft_compress(img, mask):
    """Returns a tuple of FFT frequency domains of grayscale image"""
    # Check if the image has multiple channels and convert it inot grayscale
    if len(img.shape) != 2:  
        raise Exception("Image not grayscale (has layers)")

    # perform the 2d fft on the image
    # shift zero-frequecites to the center
    img_fft_layer = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(img_fft_layer)

    # apply the mask
    compressed_fft = fft_shifted * mask

    # return frequency domain of the image
    return compressed_fft


def rect_mask(image_shape, compression_factor):
    # get the rows and column pixel counts from a image
    if len(image_shape) == 3:
        rows, cols, _ = image_shape

    else:
        rows, cols = image_shape

    # get image center coords
    center_row, center_col = rows // 2, cols // 2

    # create a mask that retains only part of the image at the center
    mask = np.zeros((rows, cols), dtype=bool)
    mask[int(center_row * (1 - compression_factor)):int(center_row * (1 + compression_factor)),
        int(center_col * (1 - compression_factor)):int(center_col * (1 + compression_factor))] = True
    
    return mask


def gaussian_mask(image_shape, compression_factor, falloff=1.0):
    """
    Creates a circular mask with a falloff in the frequency domain for FFT compression.

    Args:
        image_shape (tuple): Shape of the image (rows, cols) or (rows, cols, channels).
        compression_factor (float): Factor determining the size of the circular mask (0 to 1).
        falloff (float): Controls the smoothness of the falloff (higher means sharper falloff).

    Returns:
        mask (ndarray): A 2D mask with circular falloff.
    """
    
    if len(image_shape) == 3:
        rows, cols, _ = image_shape
    else:
        rows, cols = image_shape

    # Create a grid for distance computation
    y, x = np.ogrid[:rows, :cols]
    
    # Calculate distance from the center of the frequency domain
    center_row, center_col = rows // 2, cols // 2
    distance_from_center = np.sqrt((x - center_col)**2 + (y - center_row)**2)
    
    # Normalize distances so that the maximum distance is 1 (based on the image's size)
    max_distance = np.sqrt(center_row**2 + center_col**2)
    normalized_distance = distance_from_center / max_distance
    
    # Create a circular mask with smooth falloff
    # The falloff is controlled by the compression_factor and falloff parameter
    mask = np.exp(-falloff * (normalized_distance / compression_factor)**2)

    return mask


def fft_decode(compressed_fft):
    # apply the inverse fft to get the compressed image
    compressed_img = np.fft.ifft2(np.fft.ifftshift(compressed_fft))
    compressed_img = np.abs(compressed_img)

    return compressed_img

def add_audio_to_video(video_filename, audio_filename, output_filename):
    """
    Add audio to a video file using moviepy.
    
    Args:
        video_filename (str): The input video file.
        audio_filename (str): The input audio file.
        output_filename (str): The output video file with audio.
    """
    # Load the video and audio files
    video = VideoFileClip(video_filename)
    audio = AudioFileClip(audio_filename)
    
    # Set the audio of the video clip to the audio file
    video_with_audio = video.set_audio(audio)
    
    # Write the final video with audio to a file
    video_with_audio.write_videofile(output_filename, codec='libx264', audio_codec='aac')


def video_compress(video_path, output_filename, fps, sample_rate=44100):
    # split the video into frames and put it into a list
    input_video_frames = video_to_frames(video_path)

    # get info about video
    height, width, layers = input_video_frames[0].shape
    frame_count = len(input_video_frames)

    # setup video encode
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # setup audio buffer
    audio_signal = []

    # compress all video frames
    for i, frame in enumerate(input_video_frames):
        # count progress
        print(i / frame_count * 100, "%")

        # time step generator (0.01 - 0.5)
        dT = 0.1
        t = i * dT
        anim1 = (((math.sin(t) + 1) / 2) * 0.09) ** 3
        anim2 = (((math.cos(t) + 1) / 2) * 0.09) ** 3

        # generate compression masks
        masks = [
            gaussian_mask(frame.shape, 0.1, 0.1),
            rect_mask(frame.shape, anim1),
            rect_mask(frame.shape, anim2),
        ]

        # split the image into its RGB layers
        image_layers = cv2.split(frame)

        # apply fft to RGB layers using different masks
        compressed_frames = [0 for i in range(frame.shape[2])]
        for i, layer in enumerate(image_layers):
            compressed_frames[i] = fft_compress(layer, mask=masks[i])

        # apply inverse fft and merge to single compressed RGB frame
        decompressed_frames = [fft_decode(layer) for layer in compressed_frames]
        merged_frame = cv2.merge(decompressed_frames)

        # normalize the compressed frame
        compressed_frame = cv2.normalize(merged_frame, None, 0, 255, cv2.NORM_MINMAX)
        compressed_frame = compressed_frame.astype(np.uint8)

        # fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # axs[0].imshow(compressed_frame, cmap='gray')
        # plt.show()

        # generate audio signal per frame and extend the complete audio signal
        frame_audio = extract_sound_from_fft(frame, sample_rate, video_fps=fps)
        audio_signal.extend(frame_audio)

        output_video.write(compressed_frame)

    # save the compressed video
    output_video.release()

    # save the audio as 16bit wav
    audio_signal = np.array(audio_signal, dtype=np.int16)
    write('audio.wav', sample_rate, audio_signal)



video_path = "2Dfft/small_kabootar.mp4"
audio_path = "audio.wav"

# video_compress(video_path, output_filename="result.mp4", fps=20)

# add_audio_to_video(video_path, audio_path, output_path="result_a.mp4")


