import numpy as np
import pandas as pd
from sklearn import preprocessing
from nltk import word_tokenize
from string import punctuation


class Stat():
    def __init__(self):
        pass

    def fit(self, texts, normalize=True):
        df_data = {}
        df_data['w_based'] = np.array(list(map(self.count_word_based, texts)))
        df_data['c_based'] = np.array(list(map(self.count_char_based, texts)))
        feature_mat = np.concatenate((df_data['w_based'], df_data['c_based']),
                                     axis=1)
        X = preprocessing.scale(feature_mat) if normalize else feature_mat
        return X

    def fit_transform(self, texts, normalize=True):
        return self.fit(texts, normalize=True)

    def transform(self, texts, normalize=True):
        return self.fit(texts, normalize=True)

    def get_features():
        f = [
            'min_len', 'max_len', 'upper', 'lower', 'mean', 'digits',
            'letters', 'spaces', 'punct'
        ]
        return f

    def count_vowels(self, text):
        vowels = "aeiou"
        v_dict = {}
        for v in vowels:
            v_dict[v] = text.count(v)
        return v_dict

    def count_word_based(self, text):
        word_stats = {}
        stat = []
        upper = 0
        lower = 0
        parsed = word_tokenize(text)
        stat = [len(word) for word in parsed]
        min_len = min(stat)
        max_len = max(stat)
        upper = sum([1 if word[0].isupper() else 0 for word in parsed])
        lower = len(stat) - upper
        stat = np.array(stat)
        word_stats['min_len'] = min_len
        word_stats['max_len'] = max_len
        word_stats['upper'] = upper
        word_stats['lower'] = lower
        word_stats['mean'] = np.mean(stat)
        word_stats['std'] = np.std(stat)
        return np.array(list(word_stats.values()))

    def count_char_based(self, text):
        char_stats = {'digits': 0, 'letters': 0, 'spaces': 0, 'punct': 0}
        for c in text:
            if c.isdigit():
                char_stats['digits'] = char_stats['digits'] + 1
            elif c.isalpha():
                char_stats['letters'] = char_stats['letters'] + 1
            elif c.isspace():
                char_stats['spaces'] = char_stats['spaces'] + 1
            elif c in punctuation:
                char_stats['punct'] = char_stats['punct'] + 1
        char_stats.update(self.count_vowels(text))
        return np.array(list(char_stats.values()))
