import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from torchvision.transforms import ToTensor

from models.cell import model_cell

device = 'cpu'


st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)



st.title('Cell detector')

from PIL import  Image

image = st.file_uploader("Choose a file")
if image is not None:
    # To read file as bytes:

    st.image(image)
    model_cell.to(device)
    model_cell.eval()

    img = ToTensor(Image.open(image))

    # st.write(img.shape)







st.page_link("main.py", label="Home", icon="🏠")
# st.page_link("pages/page_2.py", label="Page 2", icon="2️⃣", disabled=True)