import streamlit as st
import os
from PIL import Image
from model import process_video, save_detected_frames

st.title('🎥 🕵️‍♂️ Video Object Finder with CLIPxYOLOworld Duo!')

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
text_input = st.text_input("Enter text to detect objects", "cat")

if uploaded_file is not None and st.button("Process Video"):
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("Video uploaded successfully.")

    output_path = "output_frames"
    process_video("uploaded_video.mp4", text_input, output_path)

    st.write(f"Frames processed and saved to {output_path}.")
    
    for frame_file in sorted(os.listdir(output_path)):
        frame_path = os.path.join(output_path, frame_file)
        st.image(Image.open(frame_path), caption=frame_file)
