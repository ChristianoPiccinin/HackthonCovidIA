import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
from typing import Tuple

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=Tuple)

def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

with st.spinner('Carregando modelo..'):
    model = load_model()


def import_and_predict(image_data, model):
    size = (299,299)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction
    

def main():
    
    class_names = ['covid','normal','lung_opacity','pneumonia']
       
    html_template = """
            <div style ="background-color:#1B9CFC;padding:5px">
            <h2 style ="color:white;text-align:center;">DIAGNOSTICO DE COVID-19 POR RAIO-X</h2>
            <h4 style ="color:white;text-align:center;">Christiano Piccinin, Lucenildo Cerqueira e Hugo Brandão</h3>
            </div>
        """

    # Função do stramlit que faz o display da webpage
    st.markdown(html_template, unsafe_allow_html = True)

    st.write('')
    

    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"],
                                 accept_multiple_files=False,
                                 key=None)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, model)
        score = tf.nn.softmax(predictions[0])
        #st.write(score)
        st.success('A probabilidade de ser {} é {:.2f} %'.format(class_names[np.argmax(score)], 100 * np.max(score)))

if __name__=='__main__':
    main()
