# multimodal-archaeological-workflow

A multimodal workflow for processing archaeological video and audio data.  
This repository includes tools for:

- detecting clap peaks in audio and video  
- synchronizing multimedia data  
- generating subtitles  
- interactive color-based tracking and enhancement in videos  

## Author
Yasaman Piroozfar

## License
MIT

---

## Requirements

Install the following Python libraries:

```bash
pip install opencv-python numpy matplotlib pydub tqdm
```

Additional requirement:

- **FFmpeg** must be installed and available in your system PATH (used for audio extraction from video)

---

## Repository Structure

### `color_tracking.py`

Interactive color-based tracking and enhancement pipeline.

**What it does:**
- Step 0: select a time range (in seconds)  
- Step 1: choose a frame from that range  
- Step 2: select one or more color points (HSV is extracted automatically)  
- Step 3: define ROIs (regions of interest)  
- Tracks and highlights the selected colors across the video  

**How it works:**
- Converts frames to HSV color space  
- Builds dynamic color masks around selected HSV values  
- Finds contours and highlights strongest matching regions  

**Run:**
```bash
python color_detection.py video.mp4 output.MP4 --slow_factor 1 --sensitivity 8
```

---

### `peak_sound_detection/detect_claps_wav.py`

Detects clap peaks directly from `.wav` audio files.

**What it does:**
- Loads audio  
- Converts to mono if needed  
- Computes amplitude signal  
- Detects strong peaks using a strict threshold  

**How it works:**
- Uses peak vs mean amplitude comparison  
- Segments signal and finds local maxima  

---

### `peak_sound_detection/detect_claps_mp4.py`

Detects clap peaks from video files by extracting audio first.

**What it does:**
- Extracts audio using FFmpeg  
- Runs peak detection on extracted audio  
- Supports **single or multiple videos**  

**How it works:**
- Uses subprocess + FFmpeg for extraction  
- Applies same amplitude-based peak detection  
- Visualizes waveform and detected peaks  

---

### `subtitle_sync/`

Subtitle synchronization and generation tools.

**What it does:**
- Reads annotation/transcription files  
- Applies time shifts  
- Generates subtitle files  

**Outputs:**
- `language1.srt`  
- `language2.srt`  
- combined subtitles if needed  

---

## Notes

- Clap detection scripts support both **single and multiple inputs**  
- Color tracking is fully **interactive**  
- Sensitivity parameter controls tolerance around selected HSV values  
- FFmpeg is required for video-based clap detection  
