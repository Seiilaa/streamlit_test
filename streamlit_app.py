import streamlit as st
import torch
from PIL import Image
from realesrgan import RealESRGAN
import numpy as np
from io import BytesIO


def run_model(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model = RealESRGAN(device, scale=2)
    model.load_weights('weights/RealESRGAN_x2.pth')
    # enhance the image
    image_array = Image.open(image).convert('RGB')
    sr_image = model.predict(np.array(image_array))
    return sr_image


st.set_page_config(layout='centered', page_icon='🔎', page_title='Image upscaler')
st.title('🔎 Image upscaler 2x')
st.write('Get higher resolution using this app.')
st.write('It uses RealESRGAN to upscale the image resolution 2x. Give it a try!')

uploaded_file = st.file_uploader("Upload your image")

if uploaded_file:
    st.write('Upscaling the image 🏃‍♀')
    st.write('This might take up to 15 minutes depending on the image size.')

    left, right = st.columns(2)
    left.write("Input image")
    left.image(uploaded_file, width=300)
    right.write("Upscaled image")

    result_file = run_model(uploaded_file)
    right.image(result_file, width=300)
    buf = BytesIO()
    result_file.save(buf, format="png")
    byte_im = buf.getvalue()
    st.download_button(
        label="Download Image",
        data=byte_im,
        file_name="upscale_image.png",
        mime="image/png",
    )