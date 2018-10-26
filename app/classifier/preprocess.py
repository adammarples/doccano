"""
Preprocessor.
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize(text):
    """Tokenize text.

    Args:
        text: string.

    Returns:
        A list of words.
    """
    return text.strip().split(' ')


def build_vectorizer():
    return TfidfVectorizer(tokenizer=tokenize)
