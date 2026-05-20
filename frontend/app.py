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
        with st.container(border=True):
            # Class header: Name and Delete button
            c_header, c_del = st.columns([4, 1])
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

            # Input methods: Webcam vs File Upload
            tab_cam, tab_file = st.tabs(["📷 Webcam", "📁 Upload"])
            
            with tab_cam:
                st.write("Record webcam images...")
                
            with tab_file:
                st.write("Upload file samples...")

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

