import streamlit as st
import pandas as pd
import cv2
import numpy as np
data = pd.read_csv('Reviews.csv')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Score'], test_size=0.20)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train,y_train)
st.title('Customer Satisfaction Prediction')


review = st.text_area('Enter your Food review',key="2")
num=clf.predict([review])

    

option = st.selectbox('Please give rating to the product',(1,2,3,4,5),key="3")

if option >3:
    num-=1

while st.button('Submit',key="1"):
    st.header('Prediction:')
    if num>3:
        filename = ("D:\IT vedant\Machine Learning\Screenshot 2021-12-17 183226 (2).jpg")
        img = cv2.imread(filename, 1)
        image = np.array([img])
        original_title = '<p style="font-family:Courier; color:White; font-size: 20px;">satisfied</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        st.image(image, channels="BGR")
    elif num==3:
        filename = ("D:\IT vedant\Machine Learning\Screenshot 2021-12-17 175702.jpg")
        img = cv2.imread(filename, 1)
        image = np.array([img])
        original_title = '<p style="font-family:Courier; color:White; font-size: 20px;">Neutral</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        st.image(image, channels="BGR")
    else:
        filename = ("D:\IT vedant\Machine Learning\Screenshot 2021-12-17 183119 (2).jpg")
        img = cv2.imread(filename, 1)
        image = np.array([img])
        original_title = '<p style="font-family:Courier; color:White; font-size: 20px;">Unsatisfied</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        st.image(image, channels="BGR")

