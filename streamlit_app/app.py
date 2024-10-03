# Integrate the Model into the Streamlit App
# 1. Load the Pre-trained Model and TF-IDF Vectorizer.
# 2. Create a User Interface in Streamlit.
# 3. Preprocess the Input Text.
# 4. Use the Vectorizer to Transform the Input.
# 5. Predict the Sentiment Using the Trained Model.
# 6. Display the Result.

import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg  # <-- Importing the image module
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import spacy
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Load the spaCy model for text processing
nlp = spacy.load('en_core_web_sm')

# Preprocess function to clean text
def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Load the trained model and vectorizer
with open('../models/logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('../vectorizers/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Load y_test and y_pred for performance analysis
with open('../data/clean/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

with open('../data/clean/y_pred.pkl', 'rb') as f:
    y_pred = pickle.load(f)

# Sidebar for navigation

option = st.sidebar.radio(
    "Choose an action:",
    ("Sentiment Analysis", "Mass Sentiment Analysis", "Sentiment Timeline", "Model Performance")
)


# Sentiment Analysis Section
if option == "Sentiment Analysis":
    st.title("Sentiment Analysis App")
    user_input = st.text_area("Enter a sentence or paragraph for sentiment analysis:")
    
    if st.button("Analyze"):
        if user_input:
            cleaned_input = preprocess_text(user_input)
            input_vector = tfidf.transform([cleaned_input])
            prediction = model.predict(input_vector)[0]
            
            sentiment_label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            predicted_sentiment = sentiment_label[prediction]
            
            st.write(f"**Predicted Sentiment:** {predicted_sentiment}")
            
            # Load the corresponding icons based on the predicted sentiment
            if predicted_sentiment == 'Positive':
                icon_img = './assets/promoter.png'
            elif predicted_sentiment == 'Neutral':
                icon_img = './assets/passive.png'
            else:
                icon_img = './assets/detractor.png'
            
            # Display the corresponding sentiment icon
            st.image(icon_img, width=50)  # Adjust the width of the icon as necessary
        else:
            st.write("Please enter some text for analysis.")




# Upload CSV for Sentiment Analysis Section
elif option == "Mass Sentiment Analysis":
    st.title("Mass Sentiment Analysis")

    # File uploader for CSV files
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the uploaded CSV file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Ensure the file has a 'text' column
        if 'text' not in df.columns:
            st.error("The CSV file must contain a column named 'text'.")
        else:
            # Preprocess the text column
            df['cleaned_text'] = df['text'].apply(preprocess_text)

            # Transform the cleaned text into TF-IDF vectors
            tfidf_vectors = tfidf.transform(df['cleaned_text'])

            # Predict the sentiment of each text using the model
            df['predicted_sentiment'] = model.predict(tfidf_vectors)

            # Map numeric predictions to sentiment labels
            sentiment_label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            df['sentiment_label'] = df['predicted_sentiment'].map(sentiment_label)

            # Count the number of each sentiment for visualization
            sentiment_counts = df['sentiment_label'].value_counts()

            # ---- Overall Sentiment Overview (Primer) ----
            # Mapping sentiment labels to Promoters, Passives, and Detractors
            sentiment_label_map = {
                'Positive': 'Promoters',
                'Neutral': 'Passives',
                'Negative': 'Detractors'
            }

            # Replace sentiment labels with Promoters, Passives, and Detractors
            df['group_label'] = df['sentiment_label'].map(sentiment_label_map)

            # Count the number of each group
            group_counts = df['group_label'].value_counts()

            # Calculate percentages for each group
            group_percentages = [
                (group_counts.get('Promoters', 0) / group_counts.sum()) * 100,
                (group_counts.get('Passives', 0) / group_counts.sum()) * 100,
                (group_counts.get('Detractors', 0) / group_counts.sum()) * 100
            ]

            # Create the horizontal bar chart for the Promoters/Passives/Detractors
            categories = ['Promoters', 'Passives', 'Detractors']
            colors = ['green', 'orange', 'red']
            fig, ax = plt.subplots(figsize=(8, 5))

            # Plot the horizontal bar chart
            bars = ax.barh(categories, group_percentages, color=colors)

            # Add percentage labels on the bars
            for i, (bar, percentage) in enumerate(zip(bars, group_percentages)):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{percentage:.2f}%', va='center', fontweight='bold')

            # Add the total responses label at the top
            total_responses = group_counts.sum()
            ax.text(0.5, 1.05, f'{total_responses} responses', ha='center', va='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='orange', edgecolor='none', boxstyle='round,pad=0.5'))

            # Customize labels and title
            st.subheader("Overall Sentiment Overview")
            ax.set_xlabel('Percentage (%)')
            #ax.set_title('Overall Sentiment Overview')

            # Show the horizontal bar plot
            st.pyplot(fig)

            # ---- Predicted Sentiments for Uploaded Data (Segundo) ----
            st.subheader("Predicted Sentiments for Uploaded Data")
            st.write(df[['text', 'sentiment_label']])

            # ---- Sentiment Distribution (Último con gráfico más pequeño) ----
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(5, 3))  # Reduced figure size

            # Define custom colors for each sentiment
            colors = ['red' if label == 'Negative' else 'orange' if label == 'Neutral' else 'green' for label in sentiment_counts.index]

            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors, ax=ax)
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            st.pyplot(fig)




# Upload CSV for Sentiment Timeline Section
elif option == "Sentiment Timeline":
    st.title("Upload CSV for Sentiment Timeline Analysis")

    # File uploader for CSV files
    uploaded_file_timeline = st.file_uploader("Choose a CSV file with text and date (year, month, day)", type="csv")

    if uploaded_file_timeline is not None:
        # Load the uploaded CSV file into a pandas DataFrame
        df_timeline = pd.read_csv(uploaded_file_timeline)

        # Ensure the file has 'text', 'year', 'month', and 'day' columns
        required_columns = ['text', 'year', 'month', 'day']
        if not all(col in df_timeline.columns for col in required_columns):
            st.error(f"The CSV file must contain the following columns: {', '.join(required_columns)}.")
        else:
            # Preprocess the text column
            df_timeline['cleaned_text'] = df_timeline['text'].apply(preprocess_text)

            # Transform the cleaned text into TF-IDF vectors
            tfidf_vectors_timeline = tfidf.transform(df_timeline['cleaned_text'])

            # Predict the sentiment of each text using the model
            df_timeline['predicted_sentiment'] = model.predict(tfidf_vectors_timeline)

            # Map numeric predictions to sentiment labels
            sentiment_label_timeline = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            df_timeline['sentiment_label'] = df_timeline['predicted_sentiment'].map(sentiment_label_timeline)

            # Create a 'date' column using only year and month for easier plotting
            df_timeline['date'] = pd.to_datetime(df_timeline[['year', 'month']].assign(day=1))

            # Group the data by year-month and count sentiment labels
            sentiment_counts_timeline = df_timeline.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0)

            # Calculate average sentiment score for each month
            sentiment_score_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
            df_timeline['sentiment_score'] = df_timeline['sentiment_label'].map(sentiment_score_mapping)
            avg_sentiment_score = df_timeline.groupby('date')['sentiment_score'].mean()

            # Plot the sentiment timeline
            st.subheader("Sentiment Timeline")

            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Stacked bar plot for sentiment counts per month
            sentiment_counts_timeline.plot(kind='bar', stacked=True, ax=ax1, color=['red', 'orange', 'green'])
            ax1.set_ylabel("Number of Comments")
            ax1.set_xlabel("Date (Year-Month)")
            ax1.set_title("Sentiment Timeline (Monthly)")
            ax1.legend(title="Sentiment", loc='upper left')

            # Line plot for average sentiment score per month
            ax2 = ax1.twinx()
            ax2.plot(avg_sentiment_score.index, avg_sentiment_score.values, color='blue', marker='o', linestyle='-', label="Average Sentiment Score")
            ax2.set_ylabel("Average Sentiment Score")
            ax2.set_ylim(0, 2)
            ax2.legend(loc='upper right')

            # Change the X-axis to show both year and month
            ax1.set_xticklabels([date.strftime('%Y-%m') for date in avg_sentiment_score.index], rotation=45)

            st.pyplot(fig)

            # Show the DataFrame with predictions (optional)
            st.subheader("Predicted Sentiments for Uploaded Data")
            st.write(df_timeline[['text', 'year', 'month', 'day', 'sentiment_label']])




# Model Performance Section
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
