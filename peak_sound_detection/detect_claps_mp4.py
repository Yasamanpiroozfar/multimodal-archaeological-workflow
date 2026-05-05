import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from pydub import AudioSegment
from pydub.utils import get_array_type

def extract_audio(video_file, output_audio):
    """
    Extracts audio from a video file using ffmpeg.
    """
    output_dir = os.path.dirname(output_audio)
    
    if output_dir:  # Only create directory if it's not empty
        os.makedirs(output_dir, exist_ok=True)

    cmd = f"ffmpeg -i \"{video_file}\" -q:a 0 -map a \"{output_audio}\" -y"
    subprocess.run(cmd, shell=True, check=True)
    print(f"Audio extracted to {output_audio}")

def detect_strict_extreme_peaks(audio_samples, times, segment_duration, sample_rate, min_factor=5, peak_factor=20):
    """
    Detects strict extreme peaks in the audio signal.
    """
    highest_peak = np.max(audio_samples)
    mean_amplitude = np.mean(audio_samples)

    if highest_peak < peak_factor * mean_amplitude:
        print(f"\nNo significant peaks detected in the audio.")
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

def plot_all_video_audio(video_files, video_folder, target_sample_rate=16000, segment_duration=500, 
                         min_factor=5, downsample_factor=100, peak_factor=20):
    """
    Extracts audio from multiple video files and plots their waveforms with detected peaks.
    """
    num_files = len(video_files)
    fig, axes = plt.subplots(num_files, 1, figsize=(12, 4 * num_files), sharex=True)

    if num_files == 1:
        axes = [axes]  # Ensure axes is always a list for consistency

    for idx, video_file in enumerate(video_files):
        video_full_path = os.path.join(video_folder, video_file)
        audio_output_path = os.path.splitext(video_full_path)[0] + ".wav"

        if not os.path.exists(video_full_path):
            print(f"File not found: {video_full_path}")
            continue

        print(f"Extracting audio from {video_file}...")
        extract_audio(video_full_path, audio_output_path)

        # Load extracted audio
        audio = AudioSegment.from_file(audio_output_path).set_frame_rate(target_sample_rate)
        sample_rate, sample_width, num_channels = audio.frame_rate, audio.sample_width, audio.channels
        array_type = get_array_type(sample_width * 8)
        audio_samples = np.frombuffer(audio.raw_data, dtype=array_type)

        if num_channels > 1:
            audio_samples = audio_samples.reshape(-1, num_channels).mean(axis=1)

        audio_samples = np.abs(audio_samples)
        total_duration = len(audio_samples) / sample_rate
        times = np.linspace(0, total_duration, num=len(audio_samples))

        # Detect peaks
        maxima_list, strict_threshold, mean_amplitude = detect_strict_extreme_peaks(
            audio_samples, times, segment_duration, sample_rate, min_factor=min_factor, peak_factor=peak_factor
        )

        # Downsample for visualization
        downsampled_times = times[::downsample_factor]
        downsampled_samples = audio_samples[::downsample_factor]

        # Plot
        ax = axes[idx]
        ax.plot(downsampled_times, downsampled_samples, color='blue', alpha=0.7, label="Waveform")

        if strict_threshold is None:
            ax.text(0.5, 0.5, "No significant peaks detected", fontsize=14, color='red', ha='center', va='center', 
                    transform=ax.transAxes)
        else:
            maxima_times, maxima_values = zip(*maxima_list)
            ax.scatter(maxima_times, maxima_values, color='red', label=f"Strict Peaks (Above {strict_threshold:.2f})")

            top_two_maxima = maxima_list[:2]
            for i, (time, value) in enumerate(top_two_maxima):
                ax.scatter(time, value, color='green', s=100, label=f"Top Maxima {i+1}")
                ax.text(time, value, f"  {time:.2f}s", fontsize=9, color='green')

        ax.set_title(f"Audio Waveform: {video_file}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Please replace paths with your own video files.")

    # List of video files
    video_files = [
        "video_1.MP4", "video_2.mp4", "video_3.mp4", "video_4.mp4", "video_5.mp4"
    ]

    # Path to videos
    video_folder_path = "path/to/your/video/folder"

    plot_all_video_audio(video_files, video_folder_path, target_sample_rate=16000, segment_duration=500, 
                        min_factor=5, downsample_factor=100, peak_factor=70)

