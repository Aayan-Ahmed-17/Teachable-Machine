import streamlit as st
import requests
import pandas as pd
import io
import os

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Teachable Machine Clone", layout="wide")

st.title("Teachable Machine Clone")
st.write("Train a custom image classification model directly in your browser.")

# Initialize session state
if "classes" not in st.session_state:
    st.session_state.classes = ["Class 1", "Class 2"]
if "is_trained" not in st.session_state:
    st.session_state.is_trained = False

def fetch_classes_info():
    try:
        resp = requests.get(f"{BACKEND_URL}/classes/info")
        if resp.status_code == 200:
            return resp.json().get("classes", {})
    except:
        pass
    return {}

classes_info = fetch_classes_info()

# Sync session_state with backend
backend_classes = list(classes_info.keys())
for c in backend_classes:
    if c not in st.session_state.classes:
        st.session_state.classes.append(c)

# Layout Columns
col_left, col_mid, col_right = st.columns([2.5, 1.2, 1.3])

# Column 1: Classes
with col_left:
    st.header("Classes")
    
    # Render each class card
    for idx, cls in enumerate(list(st.session_state.classes)):
        num_samples = classes_info.get(cls, 0)
        with st.container(border=True):
            # Class header: Name and Delete button
            c_header, c_del = st.columns([5, 1])
            new_name = c_header.text_input(
                "Class Name", 
                value=cls, 
                key=f"name_{cls}_{idx}",
                label_visibility="collapsed"
            )
            
            # Handle renaming
            if new_name != cls and new_name.strip() != "":
                st.session_state.classes[idx] = new_name
                st.session_state.is_trained = False
                st.rerun()
                
            if c_del.button("🗑️", key=f"del_{cls}_{idx}"):
                try:
                    requests.delete(f"{BACKEND_URL}/class/{cls}")
                except Exception:
                    pass
                if cls in st.session_state.classes:
                    st.session_state.classes.remove(cls)
                st.session_state.is_trained = False
                st.rerun()

            st.write(f"**Samples:** {num_samples}")

            # Input methods: Webcam vs File Upload
            tab_cam, tab_file = st.tabs(["📷 Webcam", "📁 Upload"])
            
            with tab_cam:
                camera_img = st.camera_input(f"Capture for {cls}", key=f"cam_{cls}_{idx}", label_visibility="collapsed")
                if camera_img:
                    files = {"image_data": camera_img.getvalue()}
                    data = {"class_name": cls}
                    resp = requests.post(f"{BACKEND_URL}/upload-sample", files=files, data=data)
                    if resp.status_code == 200:
                        st.success("Uploaded!")
                        st.session_state.is_trained = False
                        st.rerun()
                
            with tab_file:
                uploaded_files = st.file_uploader(
                    "Choose images", 
                    accept_multiple_files=True, 
                    type=['jpg', 'jpeg', 'png'], 
                    key=f"file_{cls}_{idx}",
                    label_visibility="collapsed"
                )
                if uploaded_files:
                    if st.button("Upload Files", key=f"btn_{cls}_{idx}"):
                        with st.spinner("Uploading..."):
                            success_count = 0
                            for f in uploaded_files:
                                files = {"image_data": f.getvalue()}
                                data = {"class_name": cls}
                                resp = requests.post(f"{BACKEND_URL}/upload-sample", files=files, data=data)
                                if resp.status_code == 200:
                                    success_count += 1
                            st.success(f"Uploaded {success_count} files!")
                            st.session_state.is_trained = False
                            st.rerun()

            # Show existing samples via the API
            try:
                samples_resp = requests.get(f"{BACKEND_URL}/samples/{cls}")
                if samples_resp.status_code == 200:
                    samples = samples_resp.json().get("samples", [])
                    if samples:
                        st.write("---")
                        # Display a scrollable-like grid
                        cols = st.columns(4)
                        for s_idx, s_url in enumerate(samples[:8]):  # Show up to 8 previews
                            cols[s_idx % 4].image(s_url, use_column_width=True)
            except Exception:
                pass

    # Add Class Button
    if st.button("➕ Add a class"):
        new_cls = f"Class {len(st.session_state.classes) + 1}"
        st.session_state.classes.append(new_cls)
        st.session_state.is_trained = False
        st.rerun()

# Column 2: Training
with col_mid:
    st.header("Training")
    with st.container(border=True):
        st.subheader("Model Training")
        st.write("Train your model on the collected classes.")

# Column 3: Preview
with col_right:
    st.header("Preview")
    with st.container(border=True):
        st.subheader("Inference")
        st.write("Test your trained model here.")

