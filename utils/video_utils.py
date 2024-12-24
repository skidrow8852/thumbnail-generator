import cv2
from moviepy import VideoFileClip

def extract_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // num_frames * i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def get_video_metadata(video_path):
    clip = VideoFileClip(video_path)
    duration = int(clip.duration)
    clip.close()
    return {"duration": duration}
