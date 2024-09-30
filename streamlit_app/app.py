import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import spacy

# Load your trained model (replace with your actual model loading code)
model = LogisticRegression()  # Or load a pre-trained model if saved

# Initialize the TF-IDF vectorizer (replace with the actual vectorizer you've trained)
tfidf = TfidfVectorizer(max_features=5000)

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
