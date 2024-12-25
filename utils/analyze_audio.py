import librosa
import numpy as np
from moviepy import VideoFileClip
import whisper
import os

def analyze_audio(video_file):
    try:
        print("*********** Analyzing audio **************")
        
        # Verify that the video file exists
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")

        # Extract audio from the video using moviepy
        video = VideoFileClip(video_file)
        audio_file = os.path.join(os.getcwd(), "temp_audio.wav")
        
        video.audio.write_audiofile(audio_file, fps=44100, codec='pcm_s16le', logger=None)

        # Verify that the audio file was extracted
        if not os.path.exists(audio_file):
            raise Exception("Failed to extract audio.")
        
        # Load the extracted audio using librosa
        waveform, sr_rate = librosa.load(audio_file, sr=None)

        # Analyze duration and loudness
        duration = librosa.get_duration(y=waveform, sr=sr_rate)
        loudness = np.mean(librosa.feature.rms(y=waveform))

        # Enhance loudness description
        if loudness < 0.01:
            loudness_desc = "The audio is very quiet, suggesting silence or minimal sound."
        elif loudness < 0.1:
            loudness_desc = "The audio is soft, possibly indicating a calm or low-energy environment."
        elif loudness < 0.3:
            loudness_desc = "The audio is moderate, suggesting natural background noise or casual conversation."
        else:
            loudness_desc = "The audio is loud, potentially indicating intense or chaotic sounds."

        # Analyze pitch
        pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr_rate)
        dominant_pitch = np.mean([np.max(p) for p in pitches if np.max(p) > 0])

        # Enhance pitch description
        if dominant_pitch > 250:
            pitch_desc = "The audio contains high-pitched sounds, possibly birds, alarms, or music."
        elif dominant_pitch > 100:
            pitch_desc = "The audio contains mid-range pitched sounds, likely speech or background chatter."
        else:
            pitch_desc = "The audio contains low-pitched sounds, which could be deep voices, machinery, or ambient noise."

        # Generate audio description based on analysis
        audio_desc = f"{loudness_desc} {pitch_desc}"

        # Use Whisper to transcribe speech if present
        model = whisper.load_model("base") 

        # Transcribe the audio with Whisper
        result = model.transcribe(audio_file)

        # Add detected speech to the description
        speech_text = result['text']
        if speech_text:
            audio_desc += f" Detected speech: '{speech_text}'."
        else:
            audio_desc += " No clear speech detected."

        # Cleanup temporary audio file
        os.remove(audio_file)

        # Return the final description
        return audio_desc

    except Exception as e:
        return f"Audio analysis failed: {e}"
