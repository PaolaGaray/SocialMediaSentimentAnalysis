# Integrate the Model into the Streamlit App
# 1. Load the Pre-trained Model and TF-IDF Vectorizer.
# 2. Create a User Interface in Streamlit.
# 3. Preprocess the Input Text.
# 4. Use the Vectorizer to Transform the Input.
# 5. Predict the Sentiment Using the Trained Model.
# 6. Display the Result.

import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import spacy

# Load spaCy's small English model for tokenization and lemmatization
nlp = spacy.load('en_core_web_sm')

# Function to preprocess input text (tokenization, lemmatization, etc.)
def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Load your trained model and vectorizer
with open('../models/logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('../vectorizers/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Load y_test and y_pred
with open('../data/clean/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

with open('../data/clean/y_pred.pkl', 'rb') as f:
    y_pred = pickle.load(f)

# Sidebar for navigation
option = st.sidebar.selectbox(
    "Choose what you want to do:",
    ("Sentiment Analysis", "Model Performance")
)

if option == "Sentiment Analysis":
    st.title("Sentiment Analysis App")
    user_input = st.text_area("Enter a sentence or paragraph for sentiment analysis:")
    
    if st.button("Analyze"):
        if user_input:
            # Preprocess the user input text
            cleaned_input = preprocess_text(user_input)
            input_vector = tfidf.transform([cleaned_input])
            prediction = model.predict(input_vector)[0]
            
            # Map prediction to sentiment label
            sentiment_label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            predicted_sentiment = sentiment_label[prediction]
            
            st.write(f"**Predicted Sentiment:** {predicted_sentiment}")
        else:
            st.write("Please enter some text for analysis.")

elif option == "Model Performance":
    st.title("Model Performance Metrics")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)
    st.write("**Accuracy:**", report['accuracy'])
    st.write("**Precision:**", report['macro avg']['precision'])
    st.write("**Recall:**", report['macro avg']['recall'])
    st.write("**F1-Score:**", report['macro avg']['f1-score'])

    # Additional Metric (Optional): Displaying Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy:** {accuracy:.2f}")

