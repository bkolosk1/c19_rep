"""
Evolution of AutoBOT. Skrlj 2019
"""
import string
import re
import logging
import multiprocessing as mp
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, Normalizer
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


def remove_URL(text):
    """
    This method removes mentions (relevant for tweets)
    """

    return re.sub(r'#URL#', '', text)


def remove_HASH(text):
    """
    This method removes mentions (relevant for tweets)
    """

    return re.sub(r'#HASHTAG#', '', text)


def remove_hashtags(text, replace_token):
    """
    This method removes hashtags
    """

    return re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", replace_token, text)


def remove_url(text, replace_token):
    """
    Removal of URLs
    """

    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
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
    if len(text.split(" ")) > 1 and len(text.split()) > 0:
        return len(set(text.split())) / len(text.split())
    else:
        return 0


class text_col(BaseEstimator, TransformerMixin):
    """
    A helper processor class
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


# fit and transform numeric features, used in scikit Feature union
class digit_col(BaseEstimator, TransformerMixin):
    """
    Dealing with numeric features
    """

    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops = [
            'text', 'no_punctuation', 'no_stopwords', 'text_clean', 'affixes',
            'pos_tag_seq'
        ]
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        scaler = MinMaxScaler().fit(hd_searches)
        return scaler.transform(hd_searches)


def parallelize(data, method):
    """
    Helper method for parallelization
    """

    cores = mp.cpu_count()
    data_split = np.array_split(data, cores)
    pool = mp.Pool(cores)
    data = pd.concat(pool.map(method, data_split))
    pool.close()
    pool.join()
    return data


def build_dataframe(data_docs):
    """
    One of the core methods responsible for construction of a dataframe object.
    """

    df_data = pd.DataFrame({'text': data_docs})
    df_data['no_punctuation'] = df_data['text'].map(
        lambda x: remove_punctuation(x))
    df_data['no_url'] = df_data['no_punctuation'].map(lambda x: remove_URL(x))
    df_data['no_hash'] = df_data['no_url'].map(lambda x: remove_HASH(x))
    df_data['no_stopwords'] = df_data['no_hash'].map(
        lambda x: remove_stopwords(x))
    df_data['text_clean'] = df_data['text']
    df_data['pos_tag_seq'] = df_data['text_clean'].map(
        lambda x: get_pos_tags(x))
    return df_data


class FeaturePrunner:
    """
    Core class describing sentence embedding methodology employed here.
    """

    def __init__(self, max_num_feat=2048):

        self.max_num_feat = max_num_feat

    def fit(self, input_data, y=None):

        return self

    def transform(self, input_data):
        return input_data

    def get_feature_names(self):

        pass


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
         Pipeline([('s1', text_col(key='no_stopwords')),
                   ('word_tfidf', tfidf_word_unigram)])),
        ('char',
         Pipeline([('s2', text_col(key='no_stopwords')),
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

    except Exception as e_s:
        print(e_s, "Feature construction error.")
        tokenizer = None

    return tokenizer, feature_names, data_matrix
