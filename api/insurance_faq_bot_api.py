import itertools
import pickle

import nltk
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from keras.layers import Input, Embedding, LSTM, Lambda
from keras.models import Model
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from pydantic import BaseModel

from experiments.high_precision.high_precision_modeling import (
    exponent_neg_manhattan_distance,
    text_to_word_list,
)
from experiments.high_recall.document_level_vectorization.document_level_vectorization import DocumentLevelVectorization
from experiments.high_recall.finbert_embeddings.finbert_embeddings import FinBERTEmbeddings
from experiments.high_recall.word_level_vectorization.word_level_vectorization import (
    WordLevelVectorization,
)


def get_model(model: str):
    if model == "custom":
        embedding_dim = 100
        embeddings = np.load(
            "../experiments/high_precision/embeddings/custom_wv_embeddings.npy"
        )
    else:
        embedding_dim = 300
        embeddings = np.load(
            "../experiments/high_precision/embeddings/pretrained_wv_embeddings.npy"
        )
    max_seq_length = 212
    n_hidden = 20

    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype="int32")
    right_input = Input(shape=(max_seq_length,), dtype="int32")

    embedding_layer = Embedding(
        len(embeddings),
        embedding_dim,
        weights=[embeddings],
        input_length=max_seq_length,
        trainable=False,
    )

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    shared_lstm = LSTM(n_hidden)

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    # Calculates the distance as defined by the MaLSTM model
    malstm_distance = Lambda(
        function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
        output_shape=lambda x: (x[0][0], 1),
    )([left_output, right_output])

    # Pack it all up into a model
    return Model([left_input, right_input], [malstm_distance])


def find_answer(question: str) -> str:
    df = pd.read_csv("../data/insurance_qna_dataset.csv", sep="\t")
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    answer = df.loc[df["Question"] == question]["Answer"].tolist()

    return " ".join(answer)


app = FastAPI()

allowed_origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


class Question(BaseModel):
    question: str
    model: str
    preprocessing: str
    weight: str


@app.post("/faq/questions")
def read_root(question: Question):
    if question.model in ["custom", "pretrained"]:
        high_recall_model = WordLevelVectorization(
            train=False,
            n_neighbours=100,
            metric="cosine",
            logging=False,
            word_vectors=question.model,
            strategy="sum",
            weight=question.weight,
        )
    elif question.model in ["tf", "tf-idf"]:
        high_recall_model = DocumentLevelVectorization(
            n_neighbours=100,
            metric="cosine",
            logging=False,
            vectorizer_type=question.model,
            preprocessing=question.preprocessing,
            stemmer="snowball",
            stop_words="english",
            ngram_range=(1, 1),
        )
    else:
        high_recall_model = FinBERTEmbeddings(
            train=False,
            n_neighbours=100,
            metric="cosine",
            logging=False,
        )

    candidates = high_recall_model.get_n_similar_documents(question.question)
    candidates_ = candidates.copy()
    questions = [question.question for _ in range(len(candidates))]

    if question.model == "custom":
        with open(
            "../experiments/high_precision/vocabulary/vocabulary_custom_wv.pkl", "rb"
        ) as f:
            vocabulary = pickle.load(f)
    else:
        with open(
            "../experiments/high_precision/vocabulary/vocabulary_pretrained_wv.pkl",
            "rb",
        ) as f:
            vocabulary = pickle.load(f)

    for i, c in enumerate(candidates):
        candidates[i] = [
            vocabulary.get(word, 0)
            for word in text_to_word_list(c)
            if word not in stops
        ]
    for i, q in enumerate(questions):
        questions[i] = [
            vocabulary.get(word, 0)
            for word in text_to_word_list(q)
            if word not in stops
        ]
    candidates = pad_sequences(candidates, maxlen=212)
    questions = pad_sequences(questions, maxlen=212)

    high_precision_model = get_model(question.model)
    if question.model == "custom":
        high_precision_model.load_weights(
            "../experiments/high_precision/weights/malstm_weights_custom_wv.h5"
        )
    else:
        high_precision_model.load_weights(
            "../experiments/high_precision/weights/malstm_weights_pretrained_wv.h5"
        )
    output = high_precision_model.predict([questions, candidates])
    output = list(itertools.chain.from_iterable(output))

    index_max = np.argmax(output)

    return find_answer(candidates_[index_max])


if __name__ == "__main__":
    nltk.download("stopwords")
    stops = set(stopwords.words("english"))

    uvicorn.run(app, host="127.0.0.1", port=8000)
