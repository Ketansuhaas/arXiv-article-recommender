import joblib
from sklearn.cluster import KMeans
import re
from nltk.stem import WordNetLemmatizer
ps = WordNetLemmatizer()

def train_model(num_clusters, corpus_embeddings):
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    joblib.dump(clustering_model, "knn.joblib")
    return cluster_assignment

def clean_abstract(df):
    corpus = []
    for line in df:
        review = re.sub('[^a-zA-Z]', ' ', line)
        review = review.lower()
        review = review.split()
        review = [ps.lemmatize(word) for word in review]
        review = ' '.join(review)
        corpus.append(review)    
    return corpus