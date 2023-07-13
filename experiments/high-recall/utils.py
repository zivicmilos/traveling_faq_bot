import json
import re

import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from spacy.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r"\S+").match)

SENTENCE_VECTOR_STRATEGY = "sum"  # strategy of transforming word vectors to sentence vector, e.g. "sum", "average"
SENTENCE_VECTOR_WEIGHT = "none"  # weight type applied to word vector, e.g. "idf", "pos", "ner", "pos+ner"
POS = {"NOUN": 5.0, "PROPN": 6.0, "VERB": 2.0, "ADJ": 4.0}
NER = {"MONEY": 6.0, "CARDINAL": 5.0, "DATE ": 4.0, "FAC ": 4.0}


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
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4, epochs=50)
    model.save("word2vec.model")

    word_vectors = model.wv
    word_vectors.save("word2vec.wordvectors")

    return model, model.wv


def vectorize(
    wv: KeyedVectors, document: list[str], tfidf_vectorizer: TfidfVectorizer = None
) -> np.ndarray:
    """
    Transforms documents to vectors

    :param wv: KeyedVectors
        vectors of all words from vocabulary
    :param document:
        input document from corpus
    :param tfidf_vectorizer: TfidfVectorizer
        TF-IDF vectorizer
    :return:
        vector representation of question
    """
    if SENTENCE_VECTOR_WEIGHT == "idf":
        idf = np.asarray(
            [
                tfidf_vectorizer.idf_[tfidf_vectorizer.vocabulary_[token]]
                for token in document
            ]
        )
        document = np.asarray([wv[token] for token in document])
        document = idf[:, np.newaxis] * document
    elif SENTENCE_VECTOR_WEIGHT == "pos":
        doc = nlp(" ".join(document))
        pos = np.asarray([POS.get(token.pos_, 1.0) for token in doc])

        document = np.asarray([wv[token] for token in document])
        document = pos[:, np.newaxis] * document
    elif SENTENCE_VECTOR_WEIGHT == "ner":
        doc = nlp(" ".join(document))
        ner = np.asarray([NER.get(token.ent_type_, 1.0) for token in doc])

        document = np.asarray([wv[token] for token in document])
        document = ner[:, np.newaxis] * document
    elif SENTENCE_VECTOR_WEIGHT == "pos+ner":
        doc = nlp(" ".join(document))
        pos = np.asarray([POS.get(token.pos_, 1.0) for token in doc])
        ner = np.asarray([NER.get(token.ent_type_, 1.0) for token in doc])
        pos_ner = pos + ner

        document = np.asarray([wv[token] for token in document])
        document = pos_ner[:, np.newaxis] * document
    else:
        document = np.asarray([wv[token] for token in document])

    if SENTENCE_VECTOR_STRATEGY == "sum":
        document = np.sum(document, axis=0)
    elif SENTENCE_VECTOR_STRATEGY == "average":
        document = np.mean(document, axis=0)
    else:
        raise ValueError(
            f"Strategy {SENTENCE_VECTOR_STRATEGY} is not supported. Try 'sum' or 'average'"
        )

    return document


def check_performance(
    wv: KeyedVectors,
    knn: NearestNeighbors,
    corpus: list[list[str]],
    tfidf_vectorizer: TfidfVectorizer = None
) -> float:
    """
    Calculate performance of finding similar questions

    :param wv: KeyedVectors
        vectors of all words from vocabulary
    :param knn: NearestNeighbors
        K-nearest neighbors
    :param corpus: list
        input corpus of documents
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
        [vectorize(wv, tq, tfidf_vectorizer) for tq in test_questions]
    )
    _, indices = knn.kneighbors(test_questions)

    original = [list(tokenize(o.lower())) for o in original]
    indices_original = np.asarray([corpus.index(o) for o in original])

    rank = np.where(indices == indices_original[:, None])[1]
    penalization = (indices_original.shape[0] - rank.shape[0]) * 2 * knn.n_neighbors
    score = (rank.sum() + penalization) / indices_original.shape[0]

    return score
