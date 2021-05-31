import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
from tensorflow.keras.preprocessing import image 
from PIL import Image,ImageOps
import numpy as np


model = load_model('model.h5')

def main():

    #st.set_page_config(layout="wide")

    html_template = """
        <div style ="background-color:#1B9CFC;padding:5px">
        <h2 style ="color:white;text-align:center;">DIAGNOSTICO DE COVID-19 POR RAIO-X</h2>
        <h4 style ="color:white;text-align:center;">Christiano Piccinin, Lucenildo Cerqueira e Hugo Brandão</h3>
        </div>
    """

    # Função do stramlit que faz o display da webpage
    st.markdown(html_template, unsafe_allow_html = True)

    st.write('')
    st.write('1º Clique sobre o botão Browse files e submeta a imagem para ser analisada')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=False, key=None, help="Envie uma imagem de raio-x do pulmão para nosso sistema de IA. Só é permitido uma imagem por vez!")
    
    if uploaded_file is not None:
        Img = Image.open(uploaded_file)
        st.image(Img, caption='Imagem Carregada', width=600)
        st.write(uploaded_file)
        st.write(uploaded_file.name)
        
        #x = f'C:/Users/devch/OneDrive/Área de Trabalho/Hackthon_COVID/src/COVID/{uploaded_file.name}'
        img = image.load_img(uploaded_file.name, target_size=(299, 299))
        img = image.img_to_array(img)
        #img = img/255
        img = img.reshape(1,299,299,3)

        prediction = model.predict(img)

        if(np.argmax(prediction) == 3.0):
            st.error('A probabilidade de ser COVID-19 é  {} %'.format(prediction))
        else:
            st.success(prediction)
            st.success(prediction[0])
            st.success('Não existe a probabilidade de ser COVID-19')
        

if __name__=='__main__':
    main()
