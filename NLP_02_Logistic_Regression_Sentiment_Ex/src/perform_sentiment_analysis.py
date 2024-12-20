
import pandas as pd
import re
import math
import string
import numpy as np

FILENAME_REVIEW = './data/example01.txt'
FILENAME_POSITIVE_LEXICON = './data/positive_lexicon.txt'
FILENAME_NEGATIVE_LEXICON = './data/negative_lexicon.txt'

class SentimentAnalysis():
    def __init__(self):
        self.review, self.positive_lexicon, self.negative_lexicon = self.load_data()
        self.create_support_df()

    def load_data(self):
        """
        1 - Load data from txt file
        2 - Load positive lexicon
        3 - Load negative lexicon
        """

        # Load as one line
        with open(FILENAME_REVIEW, 'r') as file:
            review = file.read().rstrip()

        # As multiple lines
        with open(FILENAME_POSITIVE_LEXICON, "r", encoding="utf-8") as file:
            positive_lexicon = file.readlines()

        with open(FILENAME_NEGATIVE_LEXICON, 'r') as file:
            negative_lexicon = file.readlines()

        # Remove punctuation
        review = review.translate(str.maketrans('', '', string.punctuation))

        # Low case
        review = review.lower()

        # Remove douple spaces
        review = re.sub(' {2,}', ' ', review)

        # Remove new line
        positive_lexicon = [s.rstrip() for s in positive_lexicon]
        negative_lexicon = [s.rstrip() for s in negative_lexicon]

        return review, positive_lexicon, negative_lexicon

    def create_support_df(self):
        """
        Create support dataframe for the analysis
        :return: df
        """
        # Initialize data of lists
        columns_ = ['definition', 'value', 'weight']
        index_ = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

        df = pd.DataFrame(columns=columns_, index=index_)

        # Count of positive lexicon in document
        x1 = len([w for w in self.positive_lexicon if self.review.count(w)>0])

        # Count of negative lexicon in document
        x2 = len([w for w in self.negative_lexicon if self.review.count(w) > 0])

        # If "no" is present in the document
        x3 = int(' no' in self.review)

        # Count first and second pronouns
        pronouns_lst = ['i', 'me', 'you']
        x4 = len([w for w in pronouns_lst if self.review.count(w) > 0])

        # Count ! exclamation marks
        x5 = int('!' in self.review)

        # Log (word count of documents)
        # Split by space, also could be split by ' (i'll - i will)
        words = self.review.split(' ')
        word_count = len(words)

        x6 = round(math.log(word_count),2)

        # Fill dataframe values
        df['value'] = [x1, x2, x3, x4, x5, x6]

        # Definitions
        definitions_ = ['count (positive lexicon) in doc',
                      'count (negative lexicon) in doc',
                      '1 if "no" in doc, 0 overwise',
                      'count (1st and 2nd pronouns in doc)',
                      '1 if "!" in doc, 0 otherwise',
                      'log (word count of doc)']
        df['definition'] = definitions_

        # Weights
        # How important features are for positive, negative decision
        df['weight'] = [2.5, -5, -1.2, 0.5, 2.0, 0.7]

        # Bias
        bias_ = 0.1
        result = round(df['value'].dot(df['weight']),2) + bias_

        # Support sigmoid functon
        def sig(x):
            return 1 / (1 + np.exp(-x))

        probability_ = sig(result)

        print(self.review)
        print(df)
        print(f'Probability of review being positive : {round(probability_,2)}')
        print(f'Probability of review being negative : {round(1-probability_,2)}')

        # Cross-entropy loss, for y = 1 : - log(y_hat)
        cross_entropy_loss_1 = - math.log(probability_)
        cross_entropy_loss_0 = - math.log(1-probability_)
        print(f'Cross entropy loss for y = 0 : {round(cross_entropy_loss_0, 2)}')
        print(f'Cross entropy loss for y = 1 : {round(cross_entropy_loss_1,2)}')
