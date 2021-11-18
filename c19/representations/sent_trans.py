from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
from nltk import word_tokenize
from string import punctuation
from sentence_transformers import SentenceTransformer


class BERTTransformer():
    def __init__(self):
        self._model = None

    def fit(self, texts, model_name='distilbert-base-nli-mean-tokens'):
        """Fits the Sentence Transformers representation for a given model.

        Args:
            texts ([str]): Textual data to be transformed to numerical representation.
            model_name (str, optional): Sentence-Transfomer model to be used from here https://www.sbert.net/docs/pretrained_models.html . Defaults to 'distilbert-base-nli-mean-tokens'.
        """
        self._model = SentenceTransformer(model_name)

    def transform(self, texts):
        X = self._model.encode(texts)
        return X

    def fit_transform(self,
                      texts,
                      model_name='distilbert-base-nli-mean-tokens'):
        self.fit(texts, model_name=model_name)
        return self.transform(texts)
