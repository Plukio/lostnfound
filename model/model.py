from inference.models.yolo_world.yolo_world import YOLOWorld
import supervision as sv
import torch
from transformers import CLIPModel, CLIPProcessor
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import os
import spacy

en_nlp = spacy.load('en_core_web_sm')



# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# Load YOLOWorld model
yolo_model = YOLOWorld(model_id="yolo_world/l")

def extract_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frames = []
    while success:
        frames.append(image)
        success, image = vidcap.read()
    return frames

def find_object_in_frame(clip_model, clip_processor, yolo_model, frame, text_input, device, confidence_threshold=0.01):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    doc = en_nlp(text_input)
    sentence = next(doc.sents)
    for word in sentence:
        if word.dep_ == 'ROOT':
            root = str(word)
            break

    # Set classes for YOLO model
    classes = [text_input, root]
    yolo_model.set_classes(classes)

    # YOLO object detection
    results = yolo_model.infer(frame_pil, confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results).with_nms(threshold=0.01)

    text_tokens = clip_processor(text=text_input, return_tensors="pt", padding=True).input_ids.to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(text_tokens).cpu().numpy()

    max_similarity = 0
    best_bbox = None

    for bbox in detections.xyxy:
        x1, y1, x2, y2 = bbox
        cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
        cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        image_inputs = clip_processor(images=cropped_image_pil, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs).cpu().numpy()

        similarity = np.dot(text_features, image_features.T).item()

        if similarity > max_similarity:
            max_similarity = similarity
            best_bbox = (int(x1), int(y1), int(x2), int(y2))

    return best_bbox, max_similarity

def process_video(video_path, text_input, output_path, confidence_threshold=0.01):
    frames = extract_frames(video_path)
    detected_frames = []

    cnt = 0
    for frame in tqdm(frames, desc="Processing frames"):
        if cnt % 10:
            bbox, similarity = find_object_in_frame(clip_model, clip_processor, yolo_model, frame, text_input, device, confidence_threshold)
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                detected_frames.append((frame, similarity))
        cnt += 1

    detected_frames = sorted(detected_frames, key=lambda x: x[1], reverse=True)[:3]
    top_frames = [frame for frame, _ in detected_frames]

    save_detected_frames(top_frames, output_path)

def save_detected_frames(frames, output_path):
    os.makedirs(output_path, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(os.path.join(output_path, f"frame_{idx}.jpg"), frame)