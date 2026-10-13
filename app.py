import streamlit as st
import requests
import io

# Streamlit page config
st.set_page_config(
    page_title="AgroDoc - Plant Disease Detection", 
    page_icon="🌿", 
    layout="centered"
)

st.title("🌿 AgroDoc: Plant Disease Detector")
st.write("Upload a plant leaf image to detect diseases using PyTorch & ResNet18 model.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image preview dikhana
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)
    
    if st.button("Predict Disease"):
        with st.spinner("Analyzing leaf health... Please wait..."):
            try:
                # FastAPI endpoint par image bytes bhejna
                bytes_data = uploaded_file.getvalue()
                files = {"file": (uploaded_file.name, bytes_data, uploaded_file.type)}
                
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    disease = result["predicted_disease"]
                    confidence = result["confidence_score"]
                    
                    st.success("Analysis Complete!")
                    st.markdown(f"### 🦠 Predicted Condition: **{disease}**")
                    st.markdown(f"### 📊 Confidence Score: **{confidence}%**")
                else:
                    st.error(f"Error from server: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI server. Make sure `uvicorn main:app --reload` is running!")