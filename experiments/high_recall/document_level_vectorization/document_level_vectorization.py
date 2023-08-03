import json
import os
import time
from typing import Iterable, Tuple

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from experiments.high_recall.document_level_vectorization.preprocessing import stem_document, lemmatize_document


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data"))


class DocumentLevelVectorization:
    """
    Represents DocumentLevelVectorization model

    :attr: n_neighbours: int
        number of similar documents
    :attr: metric: str
        metric to be used in KNN, e.g. "euclidean", "cityblock", "cosine"
    :attr: logging: bool
        log during execution
    :attr: vectorizer_type: str
        vectorizer type, e.g. "tf", "tf-idf"
    :attr: preprocessing: str
        preprocessing technique, e.g. "stemming", "lemmatization"
    :attr: stemmer: str
        stemmer type, e.g. "porter", "snowball", "lancaster"
    :attr: stop_words: str
        stop words, e.g. "english" or None
    :attr: ngram_range: Tuple[int, int]
        N-gram range, e.g. (1, 2) for unigrams and bigrams
    :attr: vectorizer: sklearn.feature_extraction.text
        vectorizer model
    :attr: questions: Iterable
        input questions
    :attr: vectorized_questions: Iterable
        vectorized representation of input questions
    """

    def __init__(
        self,
        n_neighbours: int = 100,
        metric: str = "cosine",
        logging: bool = True,
        vectorizer_type: str = "tf",
        preprocessing: str = "stemming",
        stemmer: str = "snowball",
        stop_words: str = None,
        ngram_range: Tuple[int, int] = (1, 1),
    ):
        """
        Initialize DocumentLevelVectorization class

        :param n_neighbours: int
            number of similar documents
        :param metric: str
            metric to be used in KNN, e.g. "euclidean", "cityblock", "cosine"
        :param logging: bool
            log during execution
        :param vectorizer_type: str
            vectorizer type, e.g. "tf", "tf-idf"
        :param preprocessing: str
            preprocessing technique, e.g. "stemming", "lemmatization"
        :param stemmer: str
            stemmer type, e.g. "porter", "snowball", "lancaster"
        :param stop_words: str
            stop words, e.g. "english" or None
        :param ngram_range: Tuple[int, int]
            N-gram range, e.g. (1, 2) for unigrams and bigrams
        """
        self.n_neighbours = n_neighbours
        self.metric = metric
        self.logging = logging
        self.vectorizer_type = vectorizer_type
        self.preprocessing = preprocessing
        self.stemmer = stemmer
        self.stop_words = stop_words
        self.ngram_range = ngram_range

        nltk.download("punkt")  # used for tokenization
        nltk.download("wordnet")  # used for lemmatization

        df = pd.read_csv(os.path.join(DATA_DIR, "insurance_qna_dataset.csv"), sep="\t")
        df.drop(columns=df.columns[0], axis=1, inplace=True)

        if self.vectorizer_type == "tf":
            self.vectorizer = CountVectorizer(
                lowercase=True, ngram_range=ngram_range, stop_words=stop_words
            )
        elif self.vectorizer_type == "tf-idf":
            self.vectorizer = TfidfVectorizer(
                lowercase=True, ngram_range=ngram_range, stop_words=stop_words
            )
        else:
            raise ValueError(
                f"Vectorizer type '{self.vectorizer_type}' is not supported. Try with 'tf' or 'tf-idf'."
            )

        self.questions = df.iloc[:, 0].to_numpy()
        self.vectorized_questions = self.preprocess_documents(self.questions)
        self.vectorized_questions = self.vectorizer.fit_transform(self.vectorized_questions)
        self.vectorized_questions = np.unique(
            self.vectorized_questions.toarray(), axis=0
        )
        if self.logging:
            print("TF applied")

    def preprocess_documents(self, documents: Iterable[str]) -> Iterable[str]:
        """
        Applies preprocessing to iterable of documents

        :param documents: Iterable[str]
            iterable of documents
        :return:
            processed iterable of documents
        """
        if self.preprocessing == "stemming":
            documents = np.asarray(
                [stem_document(document, self.stemmer) for document in documents]
            )
        elif self.preprocessing == "lemmatization":
            documents = np.asarray(
                [lemmatize_document(document) for document in documents]
            )

        return documents

    def check_performance(self, knn: NearestNeighbors) -> float:
        """
        Calculate performance of finding similar questions

        :param knn: NearestNeighbors
            K-nearest neighbors
        :return:
            score (lesser is better)
        """
        print("Performance check started")
        with open(os.path.join(DATA_DIR, "test_questions_json.json")) as json_file:
            json_data = json.load(json_file)

        test_questions = json_data["question"]
        original = json_data["original"]

        test_questions = self.preprocess_documents(test_questions)
        test_questions = self.vectorizer.transform(test_questions)
        _, indices = knn.kneighbors(test_questions.toarray())

        original = self.preprocess_documents(original)
        original = self.vectorizer.transform(original)

        original = list(map(set, self.vectorizer.inverse_transform(original)))
        vectorized_questions = list(
            map(set, self.vectorizer.inverse_transform(self.vectorized_questions))
        )

        indices_original = np.asarray([vectorized_questions.index(o) for o in original])

        rank = np.where(indices == indices_original[:, None])[1]
        penalization = (indices_original.shape[0] - rank.shape[0]) * 2 * knn.n_neighbors
        score = (rank.sum() + penalization) / indices_original.shape[0]

        return score

    def get_n_similar_documents(self, document: str) -> list[str]:
        """
        Gives N documents similar to input document

        :param document: str
            input document
        :return:
            list of N documents similar to input document
        """
        start_time = time.time()

        knn = NearestNeighbors(n_neighbors=self.n_neighbours, metric=self.metric).fit(
            self.vectorized_questions
        )
        if self.logging:
            print("KNN fitted")

        if self.logging:
            score = self.check_performance(knn)
            print(f"Score: {score:.2f} | ETA: {time.time() - start_time:.2f}s")

        document = self.preprocess_documents(document)
        if type(document) == str:
            document = [document]
        document = self.vectorizer.transform(document)

        _, indices = knn.kneighbors(document)

        similar_documents = [self.questions[i] for i in indices]
        similar_documents = similar_documents[0].tolist()

        return similar_documents


if __name__ == "__main__":
    document_level_vectorization = DocumentLevelVectorization(
        n_neighbours=100,
        metric="cosine",
        logging=True,
        vectorizer_type="tf",
        preprocessing="stemming",
        stemmer="snowball",
        stop_words="english",
        ngram_range=(1, 1),
    )
    similar_documents = document_level_vectorization.get_n_similar_documents(
        "Why Do They Take Bloods And Urine For Lifes Insurance?"
    )
    print(similar_documents)
