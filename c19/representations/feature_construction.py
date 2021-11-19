"""
Preprocessing and feature construction for the SVD.
"""
import string
import re
import logging
import multiprocessing as mp
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.sem.logic import TypeException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

np.random.seed(42)


def remove_punctuation(text):
    """
    This method removes punctuation
    """

    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    return text


def remove_stopwords(text):
    """
    This method removes stopwords
    """

    stops = set(stopwords.words("english"))
    text = text.split()
    text = [x.lower() for x in text if x.lower() not in stops]
    return " ".join(text)


def remove_mentions(text, replace_token):
    """
    This method removes mentions (relevant for tweets)
    """

    return re.sub(r'(?:@[\w_]+)', replace_token, text)


def remove_hashtags(text, replace_token = "HASH"):
    """
    This method removes hashtags
    """

    return re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", replace_token, text)


def remove_url(text, replace_token = "URL"):
    """
    Removal of URLs
    """

    regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex, replace_token, text)


def get_affix(text):
    """
    This method gets the affix information
    """

    return " ".join(
        [word[-4:] if len(word) >= 4 else word for word in text.split()])


def get_pos_tags(text):
    """
    This method yields pos tags
    """
    tokens = word_tokenize(text)
    tgx = " ".join([x[1] for x in pos_tag(tokens)])
    return tgx


def ttr(text):
    """Trim text

    Args:
        text (str): Text to be trimmed.

    Returns:
        [type]: trim size
    """
    if len(text.split(" ")) > 1 and len(text.split()) > 0:
        return len(set(text.split())) / len(text.split())
    return 0


class TextCol(BaseEstimator, TransformerMixin):
    """
    A helper processor class
    """

    def __init__(self, key):
        """[summary]

        Args:
            key ([type]): [description]
        """
        self.key = key

    def fit(self, _, __=None):
        """[summary]

        Args:
            _ ([type]): [description]
            __ ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        return self

    def transform(self, data_dict):
        """[summary]

        Args:
            data_dict ([type]): [description]

        Returns:
            [type]: [description]
        """
        return data_dict[self.key]


def parallelize(data, method):
    """
    Helper method for parallelization
    """

    cores = mp.cpu_count()
    data_split = np.array_split(data, cores)
    with mp.Pool(cores) as pool:
        data = pd.concat(pool.map(method, data_split))
    return data


def build_dataframe(data_docs):
    """
    One of the core methods responsible for construction of a dataframe object.
    """

    df_data = pd.DataFrame({'text': data_docs})
    df_data['no_punctuation'] = df_data['text'].apply(remove_punctuation)
    df_data['no_url'] = df_data['no_punctuation'].apply(remove_url)
    df_data['no_hash'] = df_data['no_url'].apply(remove_hashtags)
    df_data['no_stopwords'] = df_data['no_hash'].apply(remove_stopwords)
    df_data['text_clean'] = df_data['text']
    df_data['pos_tag_seq'] = df_data['text_clean'].apply(get_pos_tags)
    return df_data



def get_features(df_data, max_num_feat=1000):
    """
    Method that computes various TF-IDF-alike features.
    """

    tfidf_word_unigram = TfidfVectorizer(ngram_range=(1, 2),
                                         max_features=max_num_feat)

    tfidf_char_unigram = TfidfVectorizer(analyzer='char',
                                         ngram_range=(2, 3),
                                         max_features=max_num_feat)

    features = [
        ('word',
         Pipeline([('s1', TextCol(key='no_stopwords')),
                   ('word_tfidf', tfidf_word_unigram)])),
        ('char',
         Pipeline([('s2', TextCol(key='no_stopwords')),
                   ('char_tfidf', tfidf_char_unigram)]))
    ]

    feature_names = [x[0] for x in features]
    matrix = Pipeline([('union',
                        FeatureUnion(transformer_list=features,
                                     n_jobs=8)),
                       ('normalize', Normalizer())])

    try:
        data_matrix = matrix.fit_transform(df_data)
        tokenizer = matrix

    except TypeException as e_s:
        print(e_s, "Feature construction error.")
        tokenizer = None

    return tokenizer, feature_names, data_matrix
