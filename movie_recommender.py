"""
movie_recommender.py
Main script for building a movie recommender system using topic modeling techniques.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

# Use a built-in English stopword list (no NLTK required)
ENGLISH_STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

def load_data(path=None):
    """Load movie data from a CSV file. If no path is given, load the sample movies.csv."""
    if path is None:
        path = 'movies.csv'
    return pd.read_csv(path)

def preprocess(texts):
    """Preprocess movie descriptions: lowercase, remove stopwords, keep simple tokenization."""
    stop_words = ENGLISH_STOPWORDS
    return [
        ' '.join([word for word in str(text).lower().split() if word.isalpha() and word not in stop_words])
        for text in texts
    ]

def build_topic_model(descriptions, n_topics=6):
    """
    Build topic model using TF-IDF and NMF.
    n_topics: number of topics (adjust for dataset size/diversity; default=6 for small/medium datasets)
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(descriptions)
    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_
    return vectorizer, nmf, W, H

def recommend_movies(movie_idx, W, top_n=5):
    """Recommend movies based on topic similarity (cosine similarity in topic space)."""
    similarities = cosine_similarity([W[movie_idx]], W)[0]
    similar_indices = similarities.argsort()[::-1][1:top_n+1]
    return similar_indices

# Note: For best results, tune n_topics in build_topic_model for your dataset size and diversity.
# For small datasets, 4-8 topics is typical; for larger, try 10-20.

if __name__ == "__main__":
    # Example usage
    # df = load_data('movies.csv')
    # descriptions = preprocess(df['description'])
    # vectorizer, nmf, W, H = build_topic_model(descriptions)
    # recommendations = recommend_movies(0, W)
    pass
