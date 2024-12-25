import os
from utils.video_utils import extract_frames, get_video_metadata
from utils.object_detection import load_model, analyze_frame
from utils.face_detection import detect_faces
from utils.thumbnail_generation import choose_best_frame, generate_thumbnail,generate_thumbnail_with_model
import time


def main():
    ts = time.time()
    video_path = input("Enter the path to the video file: ").strip()
    output_path = "outputs/thumbnail.jpg"
    output_path_generated = f"outputs/thumbnail_generated_{ts}.jpg"

    if not os.path.exists(video_path):
        print("Video file does not exist.")
        return

    # Create the outputs folder if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Load object detection model
    print("************* Loading object detection model *************")
    model = load_model()

    # Extract frames from the video
    print("************* Extracting frames *************")
    frames = extract_frames(video_path, num_frames=5)

    # Analyze frames for object detection
    print("************* Analyzing frames *************")
    predictions = [analyze_frame(frame, model) for frame in frames]

    # Choose the best frame (prioritize faces, fallback to objects)
    print("************* Choosing the best frame *************")
    best_frame, best_index = choose_best_frame(frames, predictions)

    # Extract video metadata
    metadata = get_video_metadata(video_path)

    # Generate and save the thumbnail
    print("************* Generating thumbnail *************")
    generate_thumbnail(best_frame, predictions[best_index], output_path, metadata=metadata)

    print(f"Thumbnail generated and saved at: {output_path}")

    # Load Stable Diffusion model
    print("************* Generating thumbnail with Stable Diffusion *************")
    generate_thumbnail_with_model(best_frame, predictions, output_path_generated, metadata=metadata, video_file=video_path)
    print(f"Thumbnail generated and saved at: {output_path_generated}")

if __name__ == "__main__":
    main()
