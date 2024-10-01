import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Integrate the Model into the Streamlit App
# 1. Load the Pre-trained Model and TF-IDF Vectorizer.
# 2. Create a User Interface in Streamlit.
# 3. Preprocess the Input Text.
# 4. Use the Vectorizer to Transform the Input.
# 5. Predict the Sentiment Using the Trained Model.
# 6. Display the Result.


# Load the pre-trained model and vectorizer
with open('../models/logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('../vectorizers/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)


# Load spaCy for text processing
nlp = spacy.load('en_core_web_sm')



# Function to preprocess input text (tokenization, lemmatization, etc.)
def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])


# Title for the app
st.title("Sentiment Analysis App")

# Text input from the user
user_input = st.text_area("Enter a sentence or paragraph for sentiment analysis:")


# Button to make a prediction
if st.button("Analyze"):
    if user_input:
        # Preprocess the input text
        cleaned_input = preprocess_text(user_input)
        
        # Transform the input text using the trained TF-IDF vectorizer
        input_vector = tfidf.transform([cleaned_input])
        
        # Predict the sentiment using the trained model
        prediction = model.predict(input_vector)[0]
        
        # Convert numeric prediction to sentiment label
        sentiment_label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        predicted_sentiment = sentiment_label[prediction]
        
        # Display the prediction
        st.write(f"**Predicted Sentiment:** {predicted_sentiment}")
    else:
        st.write("Please enter some text for analysis.")
