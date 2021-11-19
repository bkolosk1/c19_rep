"""
    Document representation based on statistical distributions.
"""
from string import punctuation
import numpy as np
from sklearn.preprocessing import scale
from nltk import word_tokenize

def get_features():
    """Returns the feature names

    Returns:
        list(str): list of strings containing names
    """
    return ['min_len', 'max_len', 'upper', 'lower', 'mean', 'digits',
        'letters', 'spaces', 'punct']


def count_vowels(text):
    """Counts vowels

    Args:
        text ([type]): [description]

    Returns:
        [type]: [description]
    """
    vowels = "aeiou"
    v_dict = {}
    for vow in vowels:
        v_dict[vow] = text.count(vow)
    return v_dict
def count_word_based(text):
    """Counts word based statistics.

    Args:
        text ([type]): [description]

    Returns:
        [type]: [description]
    """
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

def count_char_based(text):
    """Counts char based statistics.

    Args:
        text ([type]): [description]

    Returns:
        [type]: [description]
    """
    char_stats = {'digits': 0, 'letters': 0, 'spaces': 0, 'punct': 0}
    for char in text:
        if char.isdigit():
            char_stats['digits'] = char_stats['digits'] + 1
        elif char.isalpha():
            char_stats['letters'] = char_stats['letters'] + 1
        elif char.isspace():
            char_stats['spaces'] = char_stats['spaces'] + 1
        elif char in punctuation:
            char_stats['punct'] = char_stats['punct'] + 1
    char_stats.update(count_vowels(text))
    return np.array(list(char_stats.values()))


class Stat():
    """
        Statistical representation.
    """
    def __init__(self,normalize=True):
        """Inits the model.

        Args:
            normalize (bool, optional): Normalizaiton of model. Defaults to True.
        """
        self.normalize = normalize

    def fit(self, texts):
        """[summary]

        Args:
            texts ([type]): [description]
            normalize (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        df_data = {}
        df_data['w_based'] = np.array(list(map(count_word_based, texts)))
        df_data['c_based'] = np.array(list(map(count_char_based, texts)))
        feature_mat = np.concatenate((df_data['w_based'], df_data['c_based']),
                                     axis=1)
        final_rep = scale(feature_mat) if self.normalize else feature_mat
        return final_rep

    def fit_transform(self, texts):
        """[summary]

        Args:
            texts ([type]): [description]
            normalize (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        return self.fit(texts)

    def transform(self, texts):
        """[summary]

        Args:
            texts ([type]): [description]
            normalize (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        return self.fit(texts)
