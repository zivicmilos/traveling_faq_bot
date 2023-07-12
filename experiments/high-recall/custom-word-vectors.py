import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from utils import load_dataset, train_save_word2vec, vectorize, check_performance

WORD2VEC_TRAINED = True
N_NEIGHBOURS = 100  # number of similar questions
METRIC = "cosine"  # metric to be used in KNN, e.g. "euclidean", "cityblock", "cosine"
SENTENCE_VECTOR_STRATEGY = "sum"  # strategy of transforming word vectors to sentence vector, e.g. "sum", "average"


def custom_word_vectors() -> None:
    """
    Main logic of the module, training, saving and performance check-up of word vectors model

    :return:
        None
    """
    df = load_dataset("../../data/insurance_qna_dataset.csv")
    questions = np.unique(df.iloc[:, 0].to_numpy())
    questions = [list(tokenize(question.lower())) for question in questions]

    tfidf_vectorizer = TfidfVectorizer(token_pattern="(?u)\\b\\w+\\b")
    tfidf_questions = [" ".join(question) for question in questions]
    tfidf_vectorizer.fit_transform(tfidf_questions)

    if WORD2VEC_TRAINED:
        model, wv = Word2Vec.load("word2vec.model"), KeyedVectors.load(
            "word2vec.wordvectors", mmap="r"
        )
        print("Word2Vec model loaded")
    else:
        model, wv = train_save_word2vec(questions)
        print("Word2Vec model trained and saved")

    vectorized_questions = np.asarray(
        [
            vectorize(wv, question, SENTENCE_VECTOR_STRATEGY, tfidf_vectorizer)
            for question in questions
        ]
    )

    knn = NearestNeighbors(n_neighbors=N_NEIGHBOURS, metric=METRIC).fit(
        vectorized_questions
    )
    print("KNN fitted")

    score = check_performance(
        wv, knn, questions, SENTENCE_VECTOR_STRATEGY, tfidf_vectorizer
    )
    print(f"Score: {score:.2f}")


if __name__ == "__main__":
    custom_word_vectors()
