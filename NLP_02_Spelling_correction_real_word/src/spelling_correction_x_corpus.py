import string
import re

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import twitter_samples


class Tweets:
    def __init__(self):
        self.data_str = None
        self.data = self.load_data()
        self.words = self.preprocess_data()

    def load_data(self):
        """
        Load tweet data from nltk
        """

        # select the set of positive and negative tweets
        all_positive_tweets = twitter_samples.strings('positive_tweets.json')
        all_negative_tweets = twitter_samples.strings('negative_tweets.json')

        # concatenate the lists, 1st part is the positive tweets followed by the negative
        return all_positive_tweets + all_negative_tweets

    def preprocess_data(self):
        """
        Preprocess
        """
        # Low case and remove punctuation
        self.data = [t.lower().translate(str.maketrans('', '', string.punctuation)) for t in self.data]

        # Remove non-ASCII characters
        self.data = [re.sub("[^A-Z0-9 ]", "", t, 0, re.IGNORECASE) for t in self.data]

        # Remove multiple whitespaces
        self.data = [re.sub("\s\s+", " ", t) for t in self.data]

        # ------------------
        # Create vocabulary
        self.data_str = ''.join(self.data)
        self.data_str = re.sub("\s\s+", " ", self.data_str)

        # Remove extra whitespaces
        return self.data_str.split(" ")

    def transform_data_for_lm(self):
        """
        Transform raw data to be used by nltk language model.
        Expected format : [['a', 'b', 'c'], ['a', 'c', 'd', 'c', 'e', 'f']]
        where sentences are split to words [[word, word], [word, word]]
        """

        tokenized_text = [list(map(str.lower, word_tokenize(sent)))
                          for sent in sent_tokenize(self.data_str)]

        return tokenized_text
