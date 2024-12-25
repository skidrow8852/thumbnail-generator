# Video Thumbnail Generator
An automated video thumbnail generation tool powered by AI and computer vision. It analyzes video frames, detects key objects, and generates visually appealing thumbnails enriched with metadata and AI-generated prompts. It supports both GPU-accelerated and CPU-based processing, ensuring compatibility across diverse systems.


![Designer](https://github.com/user-attachments/assets/267c19b7-2272-4481-bfc2-ef6c051c58ca)


## Key Features:

- **AI-Powered Object Detection**: Identifies key elements in video frames to generate contextual thumbnails.
- **Customizable Metadata Integration**: Incorporates metadata for personalized and relevant outputs.
- **GPU and CPU Support**: Automatically adapts to your system's capabilities for seamless performance.
- **Easy-to-Use**: Intuitive functionality for both beginners and advanced users.
- **Scalable Design**: Handles videos of varying lengths and resolutions efficiently.

## Installation

1. To set up the project, install the required libraries by running:

```bash
pip install -r requirements.txt
```

2. Install FFmpeg, This package requires **FFmpeg** to be installed on your system. Follow the instructions below based on your operating system:

##### For Windows:
1. Download FFmpeg from the official website: [FFmpeg Download](https://ffmpeg.org/download.html)
2. Choose the Windows version and download the zip file.
3. Extract the zip file to this directory ```(C:\ffmpeg)```.

##### For Linux:
On Ubuntu/Debian-based systems, install FFmpeg via APT:

```bash
sudo apt update
sudo apt install ffmpeg
```

# Usage
1. Run the main script:

```py
python main.py
```
2. When prompted, provide the path to the video file you want to process.

3. The tool will analyze the video, generate thumbnails, and save them to the ```Output``` folder.
