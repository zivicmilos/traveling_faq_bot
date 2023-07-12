import json

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads dataset

    :param path:
        dataset path
    :return:
        dataframe
    """
    df = pd.read_csv(path, sep="\t")
    df.drop(columns=df.columns[0], axis=1, inplace=True)

    return df


def train_save_word2vec(corpus: list[list[str]]) -> tuple[Word2Vec, KeyedVectors]:
    """
    Trains and saves Word2Vec model

    :param corpus: list[list[str]]
        input corpus of documents
    :return:
        tuple of Word2Vec model and its word vectors
    """
    model = Word2Vec(
        sentences=corpus, vector_size=100, window=5, min_count=1, workers=4, epochs=50
    )
    model.save("word2vec.model")

    word_vectors = model.wv
    word_vectors.save("word2vec.wordvectors")

    return model, model.wv


def vectorize(
    wv: KeyedVectors,
    document: list[str],
    strategy: str,
    tfidf_vectorizer: TfidfVectorizer,
) -> np.ndarray:
    """
    Transforms documents to vectors

    :param wv: KeyedVectors
        vectors of all words from vocabulary
    :param document:
        input document from corpus
    :param strategy: str
        strategy of transforming word vectors
    :param tfidf_vectorizer: TfidfVectorizer
        TF-IDF vectorizer
    :return:
        vector representation of question
    """
    token_positions = [tfidf_vectorizer.vocabulary_[token] for token in document]
    idf = np.asarray(
        [tfidf_vectorizer.idf_[token_position] for token_position in token_positions]
    )
    document = np.asarray([wv[token] for token in document])
    document = idf[:, np.newaxis] * document

    if strategy == "sum":
        document = np.sum(document, axis=0)
    elif strategy == "average":
        document = np.mean(document, axis=0)
    else:
        raise ValueError(
            f"Strategy {strategy} is not supported. Try 'sum' or 'average'"
        )

    return document


def check_performance(
    wv: KeyedVectors,
    knn: NearestNeighbors,
    questions: list,
    strategy: str,
    tfidf_vectorizer: TfidfVectorizer,
) -> float:
    """
    Calculate performance of finding similar questions

    :param wv: KeyedVectors
        vectors of all words from vocabulary
    :param knn: NearestNeighbors
        K-nearest neighbors
    :param questions: list
        input questions
    :param strategy: str
        strategy of transforming word vectors
    :param tfidf_vectorizer: TfidfVectorizer
        TF-IDF vectorizer
    :return:
        score (lesser is better)
    """
    print("Performance check started")
    with open("../../data/test_questions_json.json") as json_file:
        json_data = json.load(json_file)

    test_questions = json_data["question"]
    original = json_data["original"]

    test_questions = [list(tokenize(tq.lower())) for tq in test_questions]
    for i, tq in enumerate(test_questions):
        test_questions[i] = list(filter(lambda x: x in wv.index_to_key, tq))
    test_questions = np.asarray(
        [vectorize(wv, tq, strategy, tfidf_vectorizer) for tq in test_questions]
    )
    _, indices = knn.kneighbors(test_questions)

    original = [list(tokenize(o.lower())) for o in original]

    indices_original = np.asarray([questions.index(o) for o in original])

    rank = np.where(indices == indices_original[:, None])[1]
    penalization = (indices_original.shape[0] - rank.shape[0]) * 2 * knn.n_neighbors
    score = (rank.sum() + penalization) / indices_original.shape[0]

    return score
