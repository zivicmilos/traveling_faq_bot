import json
import os
from typing import Collection

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import BertTokenizer, BertModel

from experiments.high_recall.HighRecall import HighRecall

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "data"))


def batch(iterable: Collection, n: int = 1) -> Collection:
    """
    Yields n batches from input iterable

    :param iterable: Collection
        input iterable
    :param n: int
        number of batches
    :return:
        batch iterable
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


class TourBERTEmbeddings(HighRecall):
    """
    Represents TourBERT model

    :attr: train: bool
        train Word2Vec model
    :attr: n_neighbours: int
        number of similar documents
    :attr: metric: str
        metric to be used in KNN, e.g. "euclidean", "cityblock", "cosine"
    :attr: logging: bool
        log during execution
    :attr: questions: list[str]
        input questions
    :attr: embeddings: np.ndarray
        embeddings of input questions
    :attr: tokenizer: BertTokenizer
        TourBERT tokenizer
    :attr: model: TFBertModel
        TourBERT model
    """

    def __init__(
        self,
        train: bool = False,
        n_neighbours: int = 100,
        metric: str = "cosine",
        logging: bool = True,
    ):
        """
        Initialize TourBERTEmbeddings class

        :param train: bool
            train Word2Vec model
        :param n_neighbours: int
            number of similar documents
        :param metric: str
            metric to be used in KNN, e.g. "euclidean", "cityblock", "cosine"
        :param logging: bool
            log during execution
        """
        self.train = train
        self.n_neighbours = n_neighbours
        self.metric = metric
        self.logging = logging
        self.embeddings = []

        df = pd.read_csv(os.path.join(DATA_DIR, "traveling_qna_dataset.csv"), sep="\t")
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        self.questions = np.unique(df.iloc[:, 0].to_numpy())
        self.questions = self.questions.tolist()

        self.tokenizer = BertTokenizer.from_pretrained("veroman/TourBERT")
        self.model = BertModel.from_pretrained("veroman/TourBERT")

        self.transform()

    def transform(self) -> None:
        """
        Transforms input documents into embeddings

        :return:
            None
        """
        if self.train:
            # Transform and save input questions
            for i, q in enumerate(batch(self.questions, 32)):
                encoded_text = self.tokenizer(q, return_tensors="pt", padding=True)
                with torch.no_grad():
                    output = self.model(**encoded_text)[1].numpy().reshape(-1)
                output = np.array_split(output, len(q))
                self.embeddings += output
                print((i + 1) * 32)

            self.embeddings = np.asarray(self.embeddings)
            np.save(os.path.join(CURRENT_DIR, "tourbert_emmbedings.npy"), self.embeddings)
        else:
            self.embeddings = np.load(os.path.join(CURRENT_DIR, "tourbert_emmbedings.npy"))

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

        if self.train:
            # Transform and save test questions
            encoded_text = self.tokenizer(
                test_questions, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                output = self.model(**encoded_text)[1].numpy().reshape(-1)
            output = np.array_split(output, 60)
            tq = np.asarray(output)
            np.save(os.path.join(CURRENT_DIR, "tourbert_emmbedings_test.npy"), tq)
        else:
            tq = np.load(os.path.join(CURRENT_DIR, "tourbert_emmbedings_test.npy"))

        _, indices = knn.kneighbors(tq)

        indices_original = np.asarray([self.questions.index(o) for o in original])

        rank = np.where(indices == indices_original[:, None])[1]
        penalization = (indices_original.shape[0] - rank.shape[0]) * 2 * knn.n_neighbors
        score = (rank.sum() + penalization) / indices_original.shape[0]

        print(f"Score: {score}")

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
        if n_neighbours is None:
            n_neighbours = self.n_neighbours
        knn = NearestNeighbors(n_neighbors=n_neighbours, metric=self.metric).fit(
            self.embeddings
        )

        if self.logging:
            self.check_performance(knn)

        encoded_text = self.tokenizer(document, return_tensors="pt", padding=True)
        with torch.no_grad():
            output = self.model(**encoded_text)[1].numpy().reshape(-1)
        output = np.asarray(output)

        _, indices = knn.kneighbors(output.reshape(1, -1))

        similar_documents = [self.questions[i] for i in indices[0]]

        return similar_documents


if __name__ == "__main__":
    tourbert_embeddings = TourBERTEmbeddings(
        train=False,
        n_neighbours=100,
        metric="cosine",
        logging=True,
    )
    similar_documents = tourbert_embeddings.get_n_similar_documents(
        "Can you fly around the globe with just one bag?"
    )
    print(similar_documents)
