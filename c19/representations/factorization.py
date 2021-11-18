import numpy as np
from .feature_construction import get_features, build_dataframe
from sklearn.decomposition import TruncatedSVD


class SVD():
    def __init__(self, nfeats, dims):
        """Initializes the representation
        Args:
            nfeats (int, optional): Number of n-gram features both character and word. Defaults to 10000.
            dims (int, optional): Dimension of final factorized space.
            """
        self.tokenizer = None
        self.reducer = None
        self.nfeats = nfeats
        self.dims = dims

    def fit(self, texts):
        """Fits the SVD representation from nfeats to dims.

        Args:
            texts ([str]): Textual data to be transformed to numerical representation.
         """
        dataframe = build_dataframe(texts)
        tokenizer, feature_names, _ = get_features(dataframe,
                                                   max_num_feat=self.nfeats)
        reducer = TruncatedSVD(
            n_components=min(self.dims, 
            self.nfeats * len(feature_names) - 1))
        self.tokenizer = tokenizer
        data_matrix = self.tokenizer.transform(dataframe)
        self.reducer = reducer.fit(data_matrix)

    def transform(self, texts):
        dataframe = build_dataframe(texts)
        data_matrix = self.tokenizer.transform(dataframe)
        reduced_matrix = self.reducer.transform(data_matrix)
        return reduced_matrix

    def fit_transform(self, texts):
        dataframe = build_dataframe(texts)
        self.fit(texts)
        return self.transform(texts)

