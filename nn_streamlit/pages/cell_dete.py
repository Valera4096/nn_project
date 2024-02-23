import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from torchvision.transforms import ToTensor
from torchvision import transforms as T
import torch.nn.functional as F
import torch

import requests
import shutil

import datetime

from models.cell import model_cell
model = model_cell
device = 'cpu'
model.to(device)

st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)





from PIL import  Image

resize = T.Resize((224, 224))

to_tensor = T.ToTensor()

idx2class = {0: 'EOSINOPHIL', 1: 'LYMPHOCYTE', 2: 'MONOCYTE', 3: 'NEUTROPHIL'}
def get_prediction(path: str):
    img = resize(path)
    img = to_tensor(img).unsqueeze(0)
    with torch.no_grad():
        pred_classes = F.softmax(model(img).to(device), dim=1)

    lst = pred_classes.tolist()[0]

    def f(lst):
        a = []
        for i,e in enumerate(lst):
            a.append((i,e))
        a = sorted(a, key = lambda x: x[1], reverse=True)
        b = []
        for i in a:
            b.append((idx2class[i[0]], round(i[1],4) ))


        return b

    return f(lst)


st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

st.title('Cell detector')

# uploaded_file = None

# on = st.toggle('Activate feature')

# if on:
#     st.title('Вставьте ссылку')
#     link = st.text_input('Movie title', 'Life of Brian')
#     img_url = 'https://file-examples.com/storage/fefbfe84f862d721699d168/2017/10/file_example_PNG_500kB.png'
#     path = 'download.png'
#     r = requests.get(link, stream=True)
#     if r.status_code == 200:
#         with open(path, 'wb') as f:
#             r.raw.decode_content = True
#             shutil.copyfileobj(r.raw, f)
#             st.write('Image Downloaded Successfully')
            
#             uploaded_file = Image.open(f)


# else:
uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

#title = st.text_input('Movie title', 'Life of Brian')

if uploaded_file is not None:

    now = datetime.datetime.now()
    
    # Отображение загруженного изображения
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    # Предсказание по загруженному изображению
    prediction = get_prediction(image)
    st.success(f"Предсказание: %")

    X = []
    Y = []
    for i in prediction:
        X.append(i[0])
        Y.append(i[-1])
    fig, ax = plt.subplots()
    ax = plt.bar(X, Y)
    finish = datetime.datetime.now()
    st.write(fig)
    elapsed_time = finish - now

    st.write(f' Затраченное время на обработку:  Ч : М : С : МС ----> {elapsed_time}.')


# path = !!!
# !wget -O image.jpg https://petsi.net/images/dogbreed/big/10.jpg
# img = resize(io.read_image('image2.jpg')/255)



st.page_link("main.py", label="Home", icon="🏠")
# st.page_link("pages/page_2.py", label="Page 2", icon="2️⃣", disabled=True)