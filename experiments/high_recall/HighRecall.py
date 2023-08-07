import abc


class HighRecall(abc.ABC):
    @abc.abstractmethod
    def transform(self, *args):
        pass

    @abc.abstractmethod
    def check_performance(self, knn):
        pass

    @abc.abstractmethod
    def get_n_similar_documents(self, document):
        pass
