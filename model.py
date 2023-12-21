import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
import pickle
from keras import layers
from keras import layers
from tensorflow.keras.models import Sequential, model_from_json
from PIL import Image, ImageOps
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


file = open('modeljson.json', 'r')
model = file.read()
file.close()
lm = model_from_json(model)
lm.load_weights('modelweights.h5')
lm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



st.header('Dogs and Cats recognition')
st.subheader('Now we can recognize for you wether its cat a dog!')

im = st.file_uploader('pick a file')



def import_and_predict(image_data, model):
    size = (130,130)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ..., cv2.IMREAD_GRAYSCALE]
    prediction = model.predict(img_reshape)
    return prediction

if st.button('submit image'):
    image = Image.open(im)
    st.image(image)
    prediction= import_and_predict(image,lm)
    st.subheader('Prediction:')

    if prediction[0]:

        st.write('its a cat')
    else:

        st.write('its a dog')

st.write('---')

diab = pd.read_csv('diabetes.csv')
X= diab.drop('Outcome', axis=1).values
y = diab['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=2)
scaling = StandardScaler()
X_train_scaled = scaling.fit_transform(X_train)
X_test_scaled = scaling.fit_transform(X_test)
model3 = LogisticRegression()
model3.fit(X_train, y_train)

st.header('this is diebetes prediction area!')
st.subheader('lets see you are save or not..')
st.write('enter the following details: ')

preg = st.number_input('number of pregnencies',1,5)
gluc = st.number_input('write your glucose',0,31)
bp = st.number_input('write your blood pressure',40,150)
sb = st.number_input('what is your skin thickness',0,500)
insu = st.number_input('insulin level',0,100)
bmi = st.number_input('enter your bmi level')
de = st.number_input('write probabilty of diebetes patients in your family')
age = st.number_input('write your age',0,100)
if st.button('submit'):
    prediction = model3.predict([[int(preg), int(gluc), int(bp), int(sb), int(insu), int(bmi), int(de), int(age)]])

    yn = prediction[0]
    if yn == 0:
        st.success('you are safe according to me')
    elif yn == 1:
        st.warning('see a doctor as soon as possible')
st.write('---')

st.write("[Contact here](https://www.instagram.com/fly_fazaia/)")
