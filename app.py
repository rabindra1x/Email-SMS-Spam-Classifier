import streamlit as st
import pickle
import re
import nltk
# Download necessary NLTK resources
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer

# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to clean and transform text
def transform_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)            # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)          # Remove punctuation
    text = re.sub(r'\d+', '', text)              # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()     # Remove extra whitespace

    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# Load models
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("🚫 Spam")
    else:
        st.header("✅ Not Spam")