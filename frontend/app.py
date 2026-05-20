import streamlit as st
import requests
import pandas as pd
import io

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Teachable Machine Clone", layout="wide")

st.title("Teachable Machine Clone")
st.write("Train a custom image classification model directly in your browser.")

# Initialize session state
if "classes" not in st.session_state:
    st.session_state.classes = []
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

# Sidebar for managing classes
st.sidebar.header("Manage Classes")
new_class_name = st.sidebar.text_input("New Class Name")
if st.sidebar.button("Add Class"):
    if new_class_name and new_class_name not in st.session_state.classes:
        st.session_state.classes.append(new_class_name)
        st.session_state.is_trained = False
        st.sidebar.success(f"Class '{new_class_name}' added!")
        st.rerun()

st.sidebar.write("Current Classes:")
for cls in list(st.session_state.classes):
    col1, col2 = st.sidebar.columns([3, 1])
    num_samples = classes_info.get(cls, 0)
    col1.write(f"- **{cls}** ({num_samples})")
    if col2.button("Del", key=f"del_{cls}"):
        try:
            requests.delete(f"{BACKEND_URL}/class/{cls}")
        except:
            pass
        if cls in st.session_state.classes:
            st.session_state.classes.remove(cls)
        st.session_state.is_trained = False
        st.rerun()

# Main area: Data Collection
st.header("1. Data Collection")
if not st.session_state.classes:
    st.info("Add at least one class from the sidebar to start collecting data.")
else:
    selected_class = st.selectbox("Select Class to add samples to:", st.session_state.classes)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Webcam Capture")
        camera_img = st.camera_input(f"Capture for {selected_class}", key=f"cam_{selected_class}")
        if camera_img:
            # Upload to backend
            files = {"image_data": camera_img.getvalue()}
            data = {"class_name": selected_class}
            resp = requests.post(f"{BACKEND_URL}/upload-sample", files=files, data=data)
            if resp.status_code == 200:
                st.success(f"Image added to {selected_class}!")
                st.session_state.is_trained = False
            else:
                st.error("Failed to upload image.")
                
    with col2:
        st.subheader("File Upload")
        uploaded_files = st.file_uploader(f"Upload files for {selected_class}", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'], key=f"file_{selected_class}")
        if uploaded_files:
            if st.button("Upload Files", key=f"btn_{selected_class}"):
                with st.spinner("Uploading..."):
                    success_count = 0
                    for f in uploaded_files:
                        files = {"image_data": f.getvalue()}
                        data = {"class_name": selected_class}
                        resp = requests.post(f"{BACKEND_URL}/upload-sample", files=files, data=data)
                        if resp.status_code == 200:
                            success_count += 1
                    st.success(f"Successfully uploaded {success_count} files to {selected_class}!")
                    st.session_state.is_trained = False

st.divider()

# Main area: Training
st.header("2. Training")
if len(st.session_state.classes) < 2:
    st.warning("You need at least 2 classes to train a model.")
else:
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model using Transfer Learning... This might take a moment."):
            try:
                resp = requests.post(f"{BACKEND_URL}/train", timeout=120)
                if resp.status_code == 200:
                    st.success("Model trained successfully!")
                    st.session_state.is_trained = True
                else:
                    st.error(f"Training failed: {resp.json().get('message')}")
            except Exception as e:
                st.error(f"Error during training: {e}")

st.divider()

# Main area: Prediction
st.header("3. Preview & Predict")
if not st.session_state.is_trained:
    st.info("Train the model first to enable predictions.")
else:
    st.write("Test your model with new images.")
    col_test1, col_test2 = st.columns([1, 1])
    
    test_image = None
    
    with col_test1:
        tab1, tab2 = st.tabs(["Webcam", "File Upload"])
        with tab1:
            test_camera_img = st.camera_input("Test with Webcam", key="test_cam")
            if test_camera_img:
                test_image = test_camera_img
        with tab2:
            test_file_img = st.file_uploader("Test with File", type=['jpg', 'jpeg', 'png'], key="test_file")
            if test_file_img:
                test_image = test_file_img
                # Use st.image to fix the error of reading bytes
                st.image(test_image.getvalue(), caption="Uploaded Image", use_column_width=True)

    with col_test2:
        if test_image:
            with st.spinner("Predicting..."):
                files = {"image_data": test_image.getvalue()}
                try:
                    resp = requests.post(f"{BACKEND_URL}/predict", files=files)
                    if resp.status_code == 200:
                        result = resp.json()
                        pred = result["prediction"]
                        probs = result["probabilities"]
                        
                        st.subheader(f"Prediction: **{pred}**")
                        
                        # Plot probabilities
                        df = pd.DataFrame(list(probs.items()), columns=["Class", "Probability"])
                        df.set_index("Class", inplace=True)
                        st.bar_chart(df)
                    else:
                        st.error(f"Prediction failed: {resp.json().get('message')}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
