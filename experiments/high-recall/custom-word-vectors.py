import time

import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from utils import load_dataset, train_save_word2vec, vectorize, check_performance

WORD2VEC_TRAINED = True
N_NEIGHBOURS = 100  # number of similar questions
METRIC = "cosine"  # metric to be used in KNN, e.g. "euclidean", "cityblock", "cosine"


def custom_word_vectors() -> None:
    """
    Main logic of the module, training, saving and performance check-up of word vectors model

    :return:
        None
    """
    start_time = time.time()
    df = load_dataset("../../data/insurance_qna_dataset.csv")
    questions = np.unique(df.iloc[:, 0].to_numpy())
    questions = [list(tokenize(question.lower())) for question in questions]

    if WORD2VEC_TRAINED:
        _, wv = Word2Vec.load("word2vec.model"), KeyedVectors.load(
            "word2vec.wordvectors", mmap="r"
        )
        print("Word2Vec model loaded")
    else:
        _, wv = train_save_word2vec(questions)
        print("Word2Vec model trained and saved")

    tfidf_vectorizer = TfidfVectorizer(token_pattern="(?u)\\b\\w+\\b")
    tfidf_vectorizer.fit_transform([" ".join(question) for question in questions])

    vectorized_questions = np.asarray(
        [vectorize(wv, tfidf_vectorizer, question) for question in questions]
    )
    print("Questions vectorized")

    knn = NearestNeighbors(n_neighbors=N_NEIGHBOURS, metric=METRIC).fit(
        vectorized_questions
    )
    print("KNN fitted")

    score = check_performance(wv, knn, tfidf_vectorizer, questions)
    print(f"Score: {score:.2f} | ETA: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    custom_word_vectors()
