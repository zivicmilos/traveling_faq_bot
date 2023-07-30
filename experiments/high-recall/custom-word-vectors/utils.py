import pandas as pd
from gensim.models import Word2Vec, KeyedVectors


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
