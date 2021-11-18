import numpy as np
from .feature_construction import get_features, build_dataframe
from sklearn.decomposition import TruncatedSVD
import pickle

class SVD():
    def __init__(self):
        self.tokenizer = None 
        self.reducer = None
        
    def fit(self, texts, nfeats=10000,dims=512):
        dataframe = build_dataframe(texts)
        tokenizer, feature_names, _ = get_features(dataframe, max_num_feat = nfeats)
        reducer = TruncatedSVD(n_components = min(dims, nfeats * len(feature_names)-1))
        self.tokenizer = tokenizer
        data_matrix = self.tokenizer.transform(dataframe)
        self.reducer =  reducer.fit(data_matrix) 

    def transform(self, texts):
        dataframe = build_dataframe(texts)
        data_matrix = self.tokenizer.transform(dataframe)
        reduced_matrix = self.reducer.transform(data_matrix)
        return reduced_matrix

    def fit_transform(self, texts, nfeats=10000,dims=512):
        dataframe = build_dataframe(texts)
        self.fit(texts,nfeats=nfeats,dims=dims)
        return self.transform(texts)

class SVD():
    def __init__(self):
        self.tokenizer = None 
        self.reducer = None
        
    def fit(self, texts, nfeats=10000,dims=512):
        dataframe = build_dataframe(texts)
        tokenizer, feature_names, _ = get_features(dataframe, max_num_feat = nfeats)
        reducer = TruncatedSVD(n_components = min(dims, nfeats * len(feature_names)-1))
        self.tokenizer = tokenizer
        data_matrix = self.tokenizer.transform(dataframe)
        self.reducer =  reducer.fit(data_matrix) 

    def transform(self, texts):
        dataframe = build_dataframe(texts)
        data_matrix = self.tokenizer.transform(dataframe)
        reduced_matrix = self.reducer.transform(data_matrix)
        return reduced_matrix

    def fit_transform(self, texts, nfeats=10000,dims=512):
        dataframe = build_dataframe(texts)
        self.fit(texts,nfeats=nfeats,dims=dims)
        return self.transform(texts)