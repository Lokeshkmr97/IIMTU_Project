import streamlit as st
import pickle
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# ---------------- Load Model & Vectorizer ----------------
with open('rf_model.pkl', 'rb') as f:
    rf = pickle.load(f)

with open('cv.pkl', 'rb') as f:
    cv = pickle.load(f)

# ---------------- Preprocessing ----------------
def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent').replace('$', ' dollar ').replace('₹', ' rupee ').replace('€', ' euro ').replace('@', ' at ')
    q = q.replace('[math]', '')
    q = re.sub(r'\W', ' ', BeautifulSoup(q, 'html.parser').get_text())
    return q

# ---------------- Feature Functions ----------------
def query_point_creator(q1,q2):
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    
    features = []
    # basic features
    features.append(len(q1))
    features.append(len(q2))
    features.append(len(q1.split()))
    features.append(len(q2.split()))
    
    w1 = set(q1.split())
    w2 = set(q2.split())
    features.append(len(w1 & w2))
    features.append(len(w1) + len(w2))
    features.append(round(len(w1 & w2)/(len(w1)+len(w2)+1e-5),2))
    
    # bag-of-words
    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()
    
    return np.hstack((np.array(features).reshape(1,7), q1_bow, q2_bow))

# ---------------- Streamlit UI ----------------
st.title("Quora Question Pair Duplicate Checker")
st.write("Enter two questions to check if they are duplicates:")

q1_input = st.text_area("Enter Question 1")
q2_input = st.text_area("Enter Question 2")

if st.button("Check Duplicate"):
    if q1_input.strip() == "" or q2_input.strip() == "":
        st.warning("Please enter both questions.")
    else:
        features = query_point_creator(q1_input, q2_input)
        prediction = rf.predict(features)[0]
        if prediction == 1:
            st.success("These questions are duplicates")
        else:
            st.error("These questions are NOT duplicates")
