from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rank_results(query, documents):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query] + documents)
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    ranked_indices = cosine_similarities.argsort()[::-1]
    ranked_results = [documents[i] for i in ranked_indices]
    return ranked_results
