# Sentiment Analysis App

## Project Overview

The **Sentiment Analysis App** is a machine learning-based tool designed to classify text data into three categories: **positive**, **neutral**, and **negative** sentiment. This project combines Natural Language Processing (NLP) techniques with machine learning algorithms to analyze and understand the sentiments conveyed in text data. It is built using Python, spaCy, scikit-learn, and Streamlit for a highly interactive and visually appealing web application.

The primary objective of this project is to offer a scalable solution for businesses that need to analyze customer feedback, social media data, reviews, and survey responses. By understanding customer sentiments, companies can better tailor their strategies, make data-driven decisions, and improve overall customer experience.

---

## Business Applications & Use Cases

Sentiment analysis is a powerful tool for businesses across multiple industries. This application can be particularly useful for the following purposes:

### 1. **Benchmarking & Competitive Analysis**
   - Companies can use sentiment analysis to compare their brand's public perception with that of their competitors.
   - Analyzing reviews and social media mentions of competing products can provide insights into areas of strength and weaknesses, helping companies differentiate their offerings.

### 2. **Campaign Effectiveness Evaluation**
   - Marketing teams can assess the success of their campaigns by analyzing customer feedback and social media posts.
   - This app allows companies to measure sentiment trends before and after a campaign launch to understand customer reactions in real-time.

### 3. **Product Development & Improvement**
   - By analyzing reviews and customer feedback, product teams can identify areas for improvement.
   - Positive sentiment can highlight features that customers love, while negative sentiment can pinpoint pain points or issues that need attention.

### 4. **Customer Support & Satisfaction Analysis**
   - Sentiment analysis can be applied to customer service interactions, emails, or survey responses to gauge customer satisfaction.
   - By identifying and addressing negative sentiments early, companies can improve customer retention and brand loyalty.

### 5. **Social Media Monitoring**
   - Brands can monitor social media platforms in real-time to track public sentiment regarding their products, services, or company as a whole.
   - This helps in quickly responding to customer concerns or capitalizing on positive trends.

---

## Technical Details

### Project Structure

- **Preprocessing**: Text data is preprocessed using **spaCy**, where it undergoes cleaning (removing stopwords, punctuation, etc.), tokenization, and lemmatization to ensure that the text is in its simplest and most informative form.
  
- **Feature Extraction**: The **TF-IDF vectorizer** is employed to convert the cleaned and tokenized text into numerical values (features) that can be fed into a machine learning model. This approach emphasizes words that are more unique and relevant within the dataset.

- **Model**: The **Logistic Regression** model is used for classifying the sentiment of the text. The model is trained using a dataset of customer feedback labeled with positive, neutral, and negative sentiments.

- **Evaluation Metrics**: 
  - **Confusion Matrix**: Used to visualize the true vs. predicted values, helping in understanding the performance of the model.
  - **Classification Report**: Includes precision, recall, F1-score, and accuracy metrics for a more detailed evaluation.
  
- **Interactive Dashboard**: The web app is built using **Streamlit**, allowing users to input text and receive instant sentiment predictions. It also provides real-time visualization of model performance through confusion matrices and key evaluation metrics.

### Technology Stack

- **Programming Language**: Python 3.8+
- **Libraries**:
  - **spaCy**: For text preprocessing (tokenization, lemmatization, and stopword removal)
  - **scikit-learn**: For machine learning (TF-IDF vectorization, Logistic Regression model)
  - **matplotlib & seaborn**: For visualizing the confusion matrix
  - **Streamlit**: For building the interactive web application
  - **pickle**: For saving/loading models and vectorizers
- **Data**: The dataset contains 41,642 customer reviews across three sentiment categories: positive, neutral, and negative.

---

## Dataset

The dataset used for this project includes customer feedback that has been labeled with one of three possible sentiment labels:

1. **Positive** (Label: `2`)
2. **Neutral** (Label: `1`)
3. **Negative** (Label: `0`)

Each data entry has the following columns:
- `id`: A unique identifier for each review.
- `text`: The actual text of the customer feedback.
- `target`: The numerical label for sentiment (0 for negative, 1 for neutral, 2 for positive).
- `sentiment`: The sentiment as a string (positive, neutral, or negative).

**Sample Data**:

| id   | text                                                   | target | sentiment |
|------|--------------------------------------------------------|--------|-----------|
| 9536 | Cooking microwave pizzas, yummy                        | 2      | positive  |
| 6135 | Any plans of allowing sub tasks to show up in...        | 1      | neutral   |
| 17697| I love the humor, I just reworded it. Like saying...    | 2      | positive  |
| 14182| Naw idk what ur talkin about                           | 1      | neutral   |
| 17840| That sucks to hear. I hate days like that               | 0      | negative  |

The full dataset contains more than 40,000 rows, offering a substantial amount of data for training and testing the sentiment analysis model.

---

## Usage

### Running the App Locally

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourUsername/Sentiment-Analysis-App.git
   cd Sentiment-Analysis-App
