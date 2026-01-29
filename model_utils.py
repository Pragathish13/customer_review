import re
from collections import Counter

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

def extract_complaints(negative_reviews, top_n=5):
    words = []
    for review in negative_reviews:
        words.extend(review.split())

    common = Counter(words).most_common(top_n)
    return common
