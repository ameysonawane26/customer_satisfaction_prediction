import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

data = pd.read_excel('Reviews_New.xlsx')
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


review = st.text_area('Enter your Food review',key=2)
num=clf.predict([review])

    

option = st.selectbox('Please give rating to the product',(1,2,3,4,5),key=3)

if option >3:
    num-=1

while st.button('Submit',key=1):
    st.header('Prediction:')
    if num>3:
        img = Image.open("Screenshot 2021-12-17 183226 (2).jpg")
        st.image(img)
    elif 2<num<4:
        img1 = Image.open("Screenshot 2021-12-17 175702.jpg")
        st.image(img1)
    else:
        img2 = Image.open("Screenshot 2021-12-17 183119 (2).jpg")
        st.image(img2)

