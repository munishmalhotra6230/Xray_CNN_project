import streamlit as st
from PIL import Image

from model import load_model,pipeline
@ st.cache_resource
def get_model():
    return load_model()
st.set_page_config(page_title="Chest X-Ray Classifier", page_icon="🫁")
st.title("🫁 Chest X-Ray Pneumonia Detector")
st.markdown("Upload a chest X-ray image and the model will predict whether it shows **Pneumonia** or is **Normal**.")
file=st.file_uploader("Upload the file",type=["jpg", "jpeg", "png"])
if file is not None:
    image=Image.open(file)
    st.image(image,"The uploaded Image",use_column_width=True)
    with st.spinner("Analyzing......"):
        model=get_model()
        prediction,confidence=pipeline(image,model)
    if prediction==1:
        st.error("The Image shows pneumonia symptoms")
        st.header("Results")
        st.subheader(f"confidence score is {confidence}")
    else:
        st.success("No disease found") 
        st.header("Results")
        st.subheader(f"confidence score is {confidence}")


        


