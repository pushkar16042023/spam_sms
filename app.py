import streamlit as st
import sklearn
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import PorterStemmer

port_stemmer = PorterStemmer()

feature_extraction = pickle.load(open('vectorizer.pkl', 'rb'))
vc = pickle.load(open('model.pkl', 'rb'))

# Create a function to generate cleaned data from raw text
def transform_message(message):
  message = message.lower()
  message = nltk.word_tokenize(message)

  ps = PorterStemmer()

  processed_words=[]

  for i in message:
    if i.isalnum():
      processed_words.append(i)

  for i in message:
    if i not in stopwords.words('english') and i not in string.punctuation:
      processed_words.append(i);
  message = processed_words[:]
  processed_words.clear()
  for i in message:
    processed_words.append(ps.stem(i))
  return " ".join(processed_words)




st.title('SMS Spam Classifier')

input_sms = st.text_input("Enter the Message")

if st.button('Predict'):

    if input_sms == "":
        st.header('Please Enter Your Message !!!')

    else:

        # 1. Preprocess
        transformed_sms = transform_message(input_sms)

# 2. Vectorize
        vect_input = feature_extraction.transform([transformed_sms])

# 3. Predict
        result = vc.predict(vect_input)[0]

        # 4. Display

        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")