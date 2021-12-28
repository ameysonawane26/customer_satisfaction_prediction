import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


data = pd.read_excel('Data/Reviews_all_new.xlsx')

X1 = data['Summary'].values.astype('U')
X2= data['Text']
y = data['Score']

@st.cache(suppress_st_warning=True)
def fit_model():
    clf1 = Pipeline([('vectorizer', CountVectorizer()),('lr', LogisticRegression(solver='lbfgs', max_iter=400))])
    clf2 = Pipeline([('vectorizer', CountVectorizer()),('lr', LogisticRegression(solver='lbfgs', max_iter=400))])
    clf1.fit(X1,y)
    clf2.fit(X2,y)

fit_model()


st.title('Customer Satisfaction Prediction')

summary = st.text_input('Enter Title of Your review',key='3')

review = st.text_area('Enter your Food review',key='2')

option = st.selectbox('Please give rating to the product',(1,2,3,4,5),key=4)

num1=clf1.predict([summary])
num2=clf2.predict([review])

num3 = (num1+option)/2

if (num3/num2)*100 >=70:
    number = (num1+num2+option)/3
else:
    number = (num1+option)/2

press = st.button('Submit',key=1)

while press:
    st.header('Prediction:')
    if number>3:
        img = Image.open("Data/Screenshot 2021-12-17 183226 (2).jpg")
        st.image(img)
        break
    elif 2<number<4:
        img1 = Image.open("Data/Screenshot 2021-12-17 175702.jpg")
        st.image(img1)
        break
    else:
        img2 = Image.open("Data/Screenshot 2021-12-17 183119 (2).jpg")
        st.image(img2)
        break
        

  
