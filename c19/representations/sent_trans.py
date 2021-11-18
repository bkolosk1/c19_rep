""" sentence BERT transformation """
from sentence_transformers import SentenceTransformer


class BERTTransformer():
    """[summary]
    """
    def __init__(self, model='distilbert-base-nli-mean-tokens'):
        """Inits the model.

        Args:
            model (str, optional):  Sentence-Transfomer model to be used from here
            https://www.sbert.net/docs/pretrained_models.html
            Defaults to 'distilbert-base-nli-mean-tokens'.
        """
        self._model_name = model
        self._model = None

    def fit(self, texts):
        """Fits the Sentence Transformers representation for a given model.

        Args:
            texts ([str]): Textual data to be transformed to numerical representation.
        """
        del texts
        self._model = SentenceTransformer(self._model_name)
        return self


    def transform(self, texts):
        """[summary]

        Args:
            texts ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self._model.encode(texts)

    def fit_transform(self,                    texts):
        """[summary]

        Args:
            texts ([type]): [description]

        Returns:
            [type]: [description]
        """
        self.fit(texts)
        return self.transform(texts)
