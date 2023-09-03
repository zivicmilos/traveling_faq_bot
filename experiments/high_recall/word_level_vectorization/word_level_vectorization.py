import json
import os
import re
import time

import numpy as np
import spacy
from gensim import downloader
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from spacy.tokenizer import Tokenizer

from experiments.high_recall.HighRecall import HighRecall
from experiments.high_recall.word_level_vectorization.utils import load_dataset, train_save_word2vec


POS = {"NOUN": 5.0, "PROPN": 6.0, "VERB": 2.0, "ADJ": 4.0}
NER = {"MONEY": 6.0, "CARDINAL": 5.0, "DATE ": 4.0, "FAC ": 4.0}
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "data"))


class WordLevelVectorization(HighRecall):
    """
    Represents WordLevelVectorization model

    :attr: train: bool
        train Word2Vec model
    :attr: n_neighbours: int
        number of similar documents
    :attr: metric: str
        metric to be used in KNN, e.g. "euclidean", "cityblock", "cosine"
    :attr: logging: bool
        log during execution
    :attr: word_vectors: str
        word vectors strategy, e.g. "custom" or "pretrained"
    :attr: strategy: str
        strategy of transforming word vectors to sentence vector, e.g. "sum", "average"
    :attr: weight: str
        weight type applied to word vector, e.g. "idf", "pos", "ner", "pos+ner"
    :attr: questions: np.ndarray
        input questions
    :attr: documents: list[list[str]]
        tokenized input questions
    :attr: wv: KeyedVectors
        word vectors
    :attr: nlp: spacy.lang.en.English
        spacy language
    """

    def __init__(
        self,
        train: bool = False,
        n_neighbours: int = 100,
        metric: str = "cosine",
        logging: bool = True,
        word_vectors: str = "custom",
        strategy: str = "sum",
        weight: str = None,
    ):
        """
        Initialize CustomWord2Vec class

        :param train: bool
            train Word2Vec model
        :param n_neighbours: int
            number of similar documents
        :param metric: str
            metric to be used in KNN, e.g. "euclidean", "cityblock", "cosine"
        :param logging: bool
            log during execution
        :param word_vectors: str
            word vectors strategy, e.g. "custom" or "pretrained"
        :param strategy: str
            strategy of transforming word vectors to sentence vector, e.g. "sum", "average"
        :param weight: str
            weight type applied to word vector, e.g. "idf", "pos", "ner", "pos+ner"
        """
        self.train = train
        self.n_neighbours = n_neighbours
        self.metric = metric
        self.logging = logging
        self.word_vectors = word_vectors
        self.strategy = strategy
        self.weight = weight

        df = load_dataset(os.path.join(DATA_DIR, "traveling_qna_dataset.csv"))
        self.questions = np.unique(df.iloc[:, 0].to_numpy())
        self.documents = [
            list(tokenize(question.lower())) for question in self.questions
        ]

        if self.train:
            _, self.wv = train_save_word2vec(self.documents)
            if self.logging:
                print("Word2Vec model trained and saved")
        else:
            if self.word_vectors == "custom":
                _, self.wv = Word2Vec.load(os.path.join(CURRENT_DIR, "word2vec.model")), KeyedVectors.load(
                    os.path.join(CURRENT_DIR, "word2vec.wordvectors"), mmap="r"
                )
                if self.logging:
                    print("Custom Word2Vec model loaded")
            elif self.word_vectors == "pretrained":
                self.wv = downloader.load("glove-wiki-gigaword-50")
                for i, document in enumerate(self.documents):
                    self.documents[i] = list(
                        filter(lambda x: x in self.wv.index_to_key, document)
                    )
                if self.logging:
                    print("Pretrained Word2Vec model loaded")
            else:
                raise ValueError(
                    f"Word vectors {self.word_vectors} are not supported. Try 'custom' or 'pretrained'"
                )

        if weight == "idf":
            self.tfidf_vectorizer = TfidfVectorizer(token_pattern="(?u)\\b\\w+\\b")
            self.tfidf_vectorizer.fit_transform(
                [" ".join(question) for question in self.documents]
            )
        elif weight in ("pos", "ner", "pos+ner"):
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp.tokenizer = Tokenizer(
                self.nlp.vocab, token_match=re.compile(r"\S+").match
            )

    def transform(self, document: list[str]) -> np.ndarray:
        """
        Transforms documents to vectors

        :param document:
            input document from corpus
        :return:
            vector representation of question
        """
        if self.weight == "idf":
            idf = np.asarray(
                [
                    self.tfidf_vectorizer.idf_[self.tfidf_vectorizer.vocabulary_[token]]
                    for token in document
                ]
            )
            document = np.asarray([self.wv[token] for token in document])
            document = idf[:, np.newaxis] * document
        elif self.weight == "pos":
            doc = self.nlp(" ".join(document))
            pos = np.asarray([POS.get(token.pos_, 1.0) for token in doc])

            document = np.asarray([self.wv[token] for token in document])
            document = pos[:, np.newaxis] * document
        elif self.weight == "ner":
            doc = self.nlp(" ".join(document))
            ner = np.asarray([NER.get(token.ent_type_, 1.0) for token in doc])

            document = np.asarray([self.wv[token] for token in document])
            document = ner[:, np.newaxis] * document
        elif self.weight == "pos+ner":
            doc = self.nlp(" ".join(document))
            pos = np.asarray([POS.get(token.pos_, 1.0) for token in doc])
            ner = np.asarray([NER.get(token.ent_type_, 1.0) for token in doc])
            pos_ner = pos + ner

            document = np.asarray([self.wv[token] for token in document])
            document = pos_ner[:, np.newaxis] * document
        else:
            document = np.asarray([self.wv[token] for token in document])

        if self.strategy == "sum":
            document = np.sum(document, axis=0)
        elif self.strategy == "average":
            document = np.mean(document, axis=0)
        else:
            raise ValueError(
                f"Strategy {self.strategy} is not supported. Try 'sum' or 'average'"
            )

        return document

    def check_performance(self, knn: NearestNeighbors) -> float:
        """
        Calculate performance of finding similar questions

        :param knn: NearestNeighbors
            K-nearest neighbors
        :return:
            score (lesser is better)
        """
        print("Performance check started")
        with open((os.path.join(DATA_DIR, "test_questions_json.json"))) as json_file:
            json_data = json.load(json_file)

        test_questions = json_data["question"]
        original = json_data["original"]

        test_questions = [list(tokenize(tq.lower())) for tq in test_questions]
        for i, tq in enumerate(test_questions):
            test_questions[i] = list(filter(lambda x: x in self.wv.index_to_key, tq))
        test_questions = np.asarray([self.transform(tq) for tq in test_questions])
        _, indices = knn.kneighbors(test_questions)

        original = [list(tokenize(o.lower())) for o in original]
        indices_original = np.asarray([self.documents.index(o) for o in original])

        rank = np.where(indices == indices_original[:, None])[1]
        penalization = (indices_original.shape[0] - rank.shape[0]) * 2 * knn.n_neighbors
        score = (rank.sum() + penalization) / indices_original.shape[0]

        return score

    def get_n_similar_documents(self, document: str, n_neighbours: int = None) -> list[str]:
        """
        Gives N documents similar to input document

        :param document: str
            input document
        :param n_neighbours: int
            number of similar documents
        :return:
            list of N documents similar to input document
        """
        start_time = time.time()

        vectorized_questions = np.asarray(
            [self.transform(question) for question in self.documents]
        )
        if self.logging:
            print("Questions vectorized")

        if n_neighbours is None:
            n_neighbours = self.n_neighbours
        knn = NearestNeighbors(n_neighbors=n_neighbours, metric=self.metric).fit(
            vectorized_questions
        )
        if self.logging:
            print("KNN fitted")

        if self.logging:
            score = self.check_performance(knn)
            print(f"Score: {score:.2f} | ETA: {time.time() - start_time:.2f}s")

        document = tokenize(document.lower())
        document = list(filter(lambda x: x in self.wv.index_to_key, document))
        document = np.asarray(self.transform(document)).reshape(1, -1)

        _, indices = knn.kneighbors(document)

        similar_documents = [self.questions[i] for i in indices]
        similar_documents = similar_documents[0].tolist()

        return similar_documents


if __name__ == "__main__":
    word_level_vectorization = WordLevelVectorization(
        train=False,
        n_neighbours=100,
        metric="cosine",
        logging=True,
        word_vectors="custom",
        strategy="sum",
        weight=None,
    )
    similar_documents = word_level_vectorization.get_n_similar_documents(
        "Can you fly around the globe with just one bag?"
    )
    print(similar_documents)
