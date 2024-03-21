Here's a README file for your code:

---

# YouTube Video Comments Sentiment Analysis

This project aims to perform sentiment analysis on comments from a YouTube video using Natural Language Processing (NLP) techniques. The sentiment analysis helps in understanding the overall sentiment expressed in the comments, whether positive, negative, or neutral.

## Requirements

- Python 3.x
- Libraries:
  - googleapiclient
  - numpy
  - pandas
  - matplotlib
  - nltk
  - scikit-learn
  - python-dotenv

## Usage

The script retrieves comments from a specified YouTube video and performs sentiment analysis on those comments. The sentiment analysis classifies each comment as positive, negative, or neutral based on its sentiment score.

## Implementation Details

- The script uses the Google API client library to interact with the YouTube Data API for retrieving comments from a video.
- Sentiment analysis is performed using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool from the NLTK library.
- The data processing involves text preprocessing techniques such as removing stopwords, punctuation, and lemmatization.
- The processed data is then encoded using label encoding and upsampled to handle class imbalance.
- Gaussian Naive Bayes classifier is trained on the processed data for sentiment classification.
