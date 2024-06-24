# Cyberbullying and Hate Detection Project

## Overview

This project aims to detect cyberbullying and hate speech in tweets using Natural Language Processing (NLP) and machine learning techniques. The dataset used consists of tweets labeled as either hate speech or non-hate speech. The project involves data preprocessing, visualization, and building a classification model using Logistic Regression.

## Requirements

- Python 3.x
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Seaborn
- Matplotlib
- WordCloud

## Setup

1. Install the required libraries using pip:
    ```bash
    pip install pandas numpy nltk scikit-learn seaborn matplotlib wordcloud
    ```

2. Download the dataset `hateDetection_train.csv` and place it in the project directory.

## Steps

### 1. Data Loading and Exploration
- Load the dataset using Pandas.
- Display the first few rows of the dataset and its information.
- Print random tweets to get an initial understanding of the data.

### 2. Data Preprocessing
- Convert tweets to lowercase.
- Remove URLs, mentions, hashtags, and special characters.
- Tokenize the tweets and remove stop words.
- Lemmatize the words in the tweets.
- Drop duplicate tweets.

### 3. Data Visualization
- Plot the distribution of labels (hate vs. non-hate) using a count plot and pie chart.
- Generate word clouds for the most frequent words in non-hate and hate tweets.

### 4. Feature Extraction
- Use `TfidfVectorizer` to convert tweets into numerical features.
- Extract n-grams (unigrams, bigrams, and trigrams).

### 5. Model Building
- Split the data into training and testing sets.
- Train a Logistic Regression model.
- Evaluate the model using accuracy, confusion matrix, and classification report.
- Perform hyperparameter tuning using GridSearchCV.

### 6. Model Evaluation
- Display the confusion matrix.
- Print the classification report.
- Display the best hyperparameters found by GridSearchCV.
