"""
Detect clap peaks in WAV audio files using amplitude-based peak thresholding.

This script loads one or more WAV audio files, converts stereo audio to mono if needed,
computes absolute amplitude values, and detects strong peaks based on a strict threshold
derived from the highest amplitude in the signal.

Author: Yasaman Piroozfar
License: MIT
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import get_array_type

def detect_strict_extreme_peaks(audio_samples, times, segment_duration, sample_rate, min_factor=5, peak_factor=20):
    highest_peak = np.max(audio_samples)
    mean_amplitude = np.mean(audio_samples)

    if highest_peak < peak_factor * mean_amplitude:
        print(f"No significant peaks detected. Highest peak ({highest_peak:.2f}) is not at least {peak_factor}x the mean ({mean_amplitude:.2f}).")
        return [], None, mean_amplitude  

    strict_threshold = (3/4) * highest_peak
    maxima_list = []
    segment_size = segment_duration * sample_rate

    for i in range(0, len(audio_samples), segment_size):
        segment = audio_samples[i:i + segment_size]
        if len(segment) == 0:
            continue  

        max_index = np.argmax(segment) + i  
        max_value = audio_samples[max_index]

        if max_value > strict_threshold:
            maxima_list.append((times[max_index], max_value))

    return maxima_list, strict_threshold, mean_amplitude

def plot_all_audio_files(audio_files, target_sample_rate=16000, segment_duration=500, 
                         min_factor=5, downsample_factor=100, peak_factor=20):
    num_files = len(audio_files)
    fig, axes = plt.subplots(num_files, 1, figsize=(12, 4 * num_files), sharex=True)

    if num_files == 1:
        axes = [axes]  # Ensure axes is always a list for consistency

    for idx, audio_file in enumerate(audio_files):
        if not os.path.exists(audio_file):
            print(f"File not found: {audio_file}")
            continue

        audio = AudioSegment.from_file(audio_file).set_frame_rate(target_sample_rate)
        sample_rate, sample_width, num_channels = audio.frame_rate, audio.sample_width, audio.channels
        array_type = get_array_type(sample_width * 8)
        audio_samples = np.frombuffer(audio.raw_data, dtype=array_type)

        if num_channels > 1:
            audio_samples = audio_samples.reshape(-1, num_channels).mean(axis=1)

        audio_samples = np.abs(audio_samples)
        total_duration = len(audio_samples) / sample_rate
        times = np.linspace(0, total_duration, num=len(audio_samples))

        maxima_list, strict_threshold, mean_amplitude = detect_strict_extreme_peaks(audio_samples, times, 
                                                                                   segment_duration, sample_rate, 
                                                                                   min_factor=min_factor, 
                                                                                   peak_factor=peak_factor)

        downsampled_times = times[::downsample_factor]
        downsampled_samples = audio_samples[::downsample_factor]

        ax = axes[idx]
        ax.plot(downsampled_times, downsampled_samples, color='blue', alpha=0.7, label="Waveform")

        if maxima_list:
            maxima_times, maxima_values = zip(*maxima_list)
            ax.scatter(maxima_times, maxima_values, color='red', label=f"Strict Peaks (Above {strict_threshold:.2f})")

            top_two_maxima = maxima_list[:2]
            for i, (time, value) in enumerate(top_two_maxima):
                ax.scatter(time, value, color='green', s=100, label=f"Top Maxima {i+1}")
                ax.text(time, value, f"  {time:.2f}s", fontsize=9, color='green')

        ax.set_title(f"Audio Waveform: {audio_file}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid()

    plt.tight_layout(rect=[0,0,1,1.95])
    plt.show()

if __name__ == "__main__":
    # Example usage:
    # Replace these paths with your own WAV files.

    audio_files = [
        "example_audio_with_clap.wav",
        "example_audio_without_clap.wav"
    ]

    plot_all_audio_files(
        audio_files,
        target_sample_rate=16000,
        segment_duration=500,
        min_factor=5,
        downsample_factor=100,
        peak_factor=25
    )
