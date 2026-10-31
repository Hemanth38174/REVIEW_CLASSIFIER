import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the saved model and vectorizer
@st.cache_resource
def load_assets():
    with open('bnb_model.pkl', 'rb') as m_file:
        model = pickle.load(m_file)
    with open('tfidf_vectorizer.pkl', 'rb') as v_file:
        vectorizer = pickle.load(v_file)
    return model, vectorizer

model, vectorizer = load_assets()

# Text Cleaning Preprocessing Logic
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# Streamlit App UI
st.set_page_config(page_title="Restaurant Review Sentiment Analyzer", page_icon="🍔", layout="centered")

st.title("🍔 Restaurant Review Sentiment Analyzer")
st.write("Enter a restaurant review below, and our Naïve Bayes model will predict whether the feedback is Positive or Negative!")

# Input area
user_input = st.text_area("Enter Review:", placeholder="The food was delicious and the staff was friendly...")

if st.button("Predict Sentiment", use_container_width=True):
    if user_input.strip() != "":
        # 1. Clean
        cleaned_input = clean_text(user_input)
        
        # 2. Vectorize
        vectorized_input = vectorizer.transform([cleaned_input]).toarray()
        
        # 3. Predict
        prediction = model.predict(vectorized_input)[0]
        
        # Display Results
        st.subheader("Analysis Results:")
        st.markdown(f"**Cleaned words analyzed:** `'{cleaned_input}'`")
        
        if prediction == 1:
            st.success("**Positive Sentiment detected!** 👍")
        else:
            st.error("**Negative Sentiment detected!** 👎")
    else:
        st.warning("Please write a review before predicting!")
