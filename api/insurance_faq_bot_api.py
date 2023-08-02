import itertools
import pickle

import nltk
import numpy as np
import uvicorn
from fastapi import FastAPI
from keras.layers import Input, Embedding, LSTM, Lambda
from keras.models import Model
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from pydantic import BaseModel

from experiments.high_precision.high_precision_modeling import exponent_neg_manhattan_distance, text_to_word_list
from experiments.high_recall.word_level_vectorization.word_level_vectorization import WordLevelVectorization


WORD_VECTORS = "pretrained"


def get_model():
    if WORD_VECTORS == "custom":
        embedding_dim = 100
        embeddings = np.load("../experiments/high_precision/embeddings/custom_wv_embeddings.npy")
    elif WORD_VECTORS == "pretrained":
        embedding_dim = 300
        embeddings = np.load("../experiments/high_precision/embeddings/pretrained_wv_embeddings.npy")
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


app = FastAPI()


class Question(BaseModel):
    question: str


@app.post("/faq/questions")
def read_root(question: Question):
    word_level_vectorization = WordLevelVectorization(
        train=False,
        n_neighbours=100,
        metric="cosine",
        logging=False,
        word_vectors="custom",
        strategy="sum",
        weight=None,
    )
    candidates = word_level_vectorization.get_n_similar_documents(question.question)
    candidates_ = candidates.copy()
    questions = [question.question for _ in range(len(candidates))]

    for i, c in enumerate(candidates):
        candidates[i] = [vocabulary.get(word, 0) for word in text_to_word_list(c) if word not in stops]
    for i, q in enumerate(questions):
        questions[i] = [vocabulary.get(word, 0) for word in text_to_word_list(q) if word not in stops]
    candidates = pad_sequences(candidates, maxlen=212)
    questions = pad_sequences(questions, maxlen=212)

    model = get_model()
    if WORD_VECTORS == "custom":
        model.load_weights("../experiments/high_precision/weights/malstm_weights_custom_wv.h5")
    elif WORD_VECTORS == "pretrained":
        model.load_weights("../experiments/high_precision/weights/malstm_weights_pretrained_wv.h5")
    output = model.predict([questions, candidates])
    output = list(itertools.chain.from_iterable(output))

    index_max = np.argmax(output)

    return f"Most similar question to '{question.question}' is: '{candidates_[index_max]}'"


if __name__ == "__main__":
    nltk.download("stopwords")
    stops = set(stopwords.words("english"))

    if WORD_VECTORS == "custom":
        with open('../experiments/high_precision/vocabulary/vocabulary_custom_wv.pkl', 'rb') as f:
            vocabulary = pickle.load(f)
    elif WORD_VECTORS == "pretrained":
        with open('../experiments/high_precision/vocabulary/vocabulary_pretrained_wv.pkl', 'rb') as f:
            vocabulary = pickle.load(f)

    uvicorn.run(app, host="127.0.0.1", port=8000)

