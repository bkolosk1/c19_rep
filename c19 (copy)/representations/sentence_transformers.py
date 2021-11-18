
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
        self.X = None
        
    def fit(self, texts , model_name = 'distilbert-base-nli-mean-tokens'):
        self._model = SentenceTransformer(self.model_name)
        self.X = self._model.encode(texts)  
        return self.X 

    def to_matrix(self):
        return np.array(self.X)
    
