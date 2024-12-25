import cv2
from PIL import Image, ImageDraw, ImageFont
from utils.face_detection import detect_faces
from diffusers import StableDiffusionPipeline,StableDiffusionXLImg2ImgPipeline
import numpy as np
import torch
from utils.analyze_audio import analyze_audio
import threading

def load_base_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32  # Use float32 for CPU
    )
    base_model.to(device)
    return base_model

def load_refiner_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    refiner_model = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32  # Use float32 for CPU
    )
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



def generate_thumbnail_with_model(frame, predictions, output_path, object_threshold=0.5, metadata=None, video_file=None):
    # Create a textual prompt based on extracted features for Stable Diffusion
    prompt = create_video_description(frame, predictions, object_threshold, metadata, video_file)
    
    # Function to generate image and save the prompt text to file
    def generate_and_save():
        # Call Stable Diffusion to generate a thumbnail image based on the prompt
        generated_image = generate_image_from_prompt(prompt)
        
        # Save the generated image as the thumbnail
        generated_image.save(output_path)
        print(f"Thumbnail saved at {output_path}")
    
    # Function to save the prompt to a file
    def save_prompt():
        prompt_filename = output_path.replace('.jpg', '_prompt.txt')  
        with open(prompt_filename, 'w') as file:
            file.write(prompt)
        print(f"Prompt saved at {prompt_filename}")

    # Create threads for both tasks
    generate_image_thread = threading.Thread(target=generate_and_save)
    #save_prompt_thread = threading.Thread(target=save_prompt)

    # Start both threads
    generate_image_thread.start()
    #save_prompt_thread.start()

    # Wait for both threads to finish
    generate_image_thread.join()
    #save_prompt_thread.join()

    print("Both tasks completed.")

def create_video_description(frame, predictions, object_threshold, metadata, video_file=None):

    prompt = "A visually and audibly engaging video scene that includes: "

    # Analyze the visual content for faces
    faces = detect_faces(frame)
    if len(faces) > 0:
        prompt += f"{len(faces)} person(s) are visible in the scene. "
    else:
        prompt += "No people are visible in the scene. "


    # Check predictions 
    objects = 0
    if isinstance(predictions, list):
        for prediction in predictions:
            if "boxes" in prediction and "scores" in prediction:
                for box, score in zip(prediction['boxes'], prediction['scores']):
                    if score > object_threshold:
                        objects += 1
                        
    prompt += f"There are {objects} objects detected. "
    # Add contextual details from metadata
    if metadata:
        if "location" in metadata:
            prompt += f"The scene appears to be set in {metadata['location']}. "
        if "weather" in metadata:
            prompt += f"The weather is {metadata['weather']}. "
        if "time_of_day" in metadata:
            prompt += f"The time of day is {metadata['time_of_day']}. "
        if "duration" in metadata:
            prompt += f"The video lasts approximately {metadata['duration']} seconds. "

    # Include audio context if the video file is provided
    if video_file:
        try:
            audio_analysis = analyze_audio(video_file)
            if audio_analysis:
                prompt += f"The audio indicates: {audio_analysis} "
            else:
                print("The audio does not contain notable elements to describe. ")
        except Exception as e:
            print(f"Audio analysis could not be performed due to an error: {e}. ")

    return prompt



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
