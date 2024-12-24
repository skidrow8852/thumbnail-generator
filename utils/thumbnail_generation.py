import cv2
from PIL import Image, ImageDraw, ImageFont
from utils.face_detection import detect_faces
from diffusers import StableDiffusionPipeline,StableDiffusionXLImg2ImgPipeline
import numpy as np
import torch

def load_base_model():
    base_model = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)
    return base_model

def load_refiner_model():
    refiner_model = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    refiner_model.to(device)
    return refiner_model

# generate an image from a text prompt using Stable Diffusion
def generate_image_from_prompt(prompt):
    # Load the base pipeline
    base_model = load_base_model()
    refiner_model = load_refiner_model()

# Generate an image with the base model
    image = base_model(prompt).images[0]
    refined_image = refiner_model(prompt=prompt, image=image).images[0]
    return refined_image

def choose_best_frame(frames, predictions, object_threshold=0.5):
    best_frame = None
    best_index = -1
    max_faces = 0

    # First, try to choose the frame with the most faces detected
    for idx, frame in enumerate(frames):
        faces = detect_faces(frame)
        if len(faces) > max_faces:
            max_faces = len(faces)
            best_frame = frame
            best_index = idx

    # If no faces were detected, fall back to object detection
    if best_frame is None:
        print("No faces detected, falling back to object detection...")
        max_objects = 0
        for idx, (frame, prediction) in enumerate(zip(frames, predictions)):
            object_count = sum(score > object_threshold for score in prediction['scores'])
            if object_count > max_objects:
                max_objects = object_count
                best_frame = frame
                best_index = idx

    return best_frame, best_index


def generate_thumbnail_with_model(frame, predictions, output_path, object_threshold=0.5, metadata=None):
    # Create a textual prompt based on extracted features for Stable Diffusion
    prompt = create_video_description(frame, predictions, object_threshold, metadata)
    
    # Call Stable Diffusion to generate a thumbnail image based on the prompt
    generated_image = generate_image_from_prompt(prompt)
    
    # Save the generated image as the thumbnail
    generated_image.save(output_path)
    print(f"Thumbnail saved at {output_path}")

def create_video_description(frame, predictions, object_threshold, metadata):
    # Generate a prompt description for image generation based on frames and predictions
    description = "A scene from a video with the following features: "
    
    # Analyze frames and predictions to create a more detailed description.
    faces = detect_faces(frame)
    
    # Explicitly check if any faces were detected
    if len(faces) > 0:  # If faces is a list
        description += "There are faces in the scene. "
    else:
        description += "No faces detected in the scene. "

    # Check if predictions is a list of dictionaries
    if isinstance(predictions, list):
        for prediction in predictions:
            if "boxes" in prediction and "scores" in prediction:
                for box, score in zip(prediction['boxes'], prediction['scores']):
                    if score > object_threshold:
                        description += "There are objects detected. "
    else:
        print(f"Unexpected structure for predictions: {predictions}")

    # Add additional metadata
    if metadata:
        description += f"Duration: {metadata['duration']} seconds. "
    
    return description


def generate_thumbnail(frame, predictions, output_path, object_threshold=0.5, metadata=None):
    # Ensure the frame is in RGB format for Pillow
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)

    # Highlight detected objects
    #for box, score in zip(predictions['boxes'], predictions['scores']):
    #    if score > object_threshold:
    #       x1, y1, x2, y2 = map(int, box)
    #       draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)

    # Highlight detected faces
    #faces = detect_faces(frame_rgb)  # Pass the RGB frame for face detection
    #for (x, y, w, h) in faces:
    #    draw.rectangle(((x, y), (x + w, y + h)), outline="blue", width=3)

    # Add metadata text
    try:
        # Try to use a TrueType font, fallback if not available
        font = ImageFont.truetype("arial.ttf", size=30)
    except IOError:
        print("Warning: 'arial.ttf' not found. Using default font.")
        font = ImageFont.load_default()

    if metadata:
        duration_text = f"Duration: {metadata['duration']} sec"
        #draw.text((10, 10), duration_text, fill="blue", font=font)

    # Save the thumbnail image
    image.save(output_path)
