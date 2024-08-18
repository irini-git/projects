from string import punctuation

import pandas as pd
import numpy as np
from functools import reduce
import re

from numpy.ma.core import negative

# FILENAME = "simple_data.csv"
FILENAME = "another_simple_data.csv"
FILEDIRECTORY = "./data/"

class SentimentData():
    def __init__(self):
        self.test_words = None
        self.vocabulary = None
        self.training_subset = None
        self.test_subset = None
        self.UNK = None

        self.df = self.load_data()
        self.binary_naives_bayes = self.get_user_input_if_binary()
        self.prior = self.compute_prior_probability()
        self.drop_unknown_words()
        self.likelihoods = self.compute_likelihoods()
        self.score_test_set()

    def load_data(self):
        df = pd.read_csv(FILEDIRECTORY + FILENAME,
                         sep=";")

        # Dealing with negation ------------
        # Add NOT_ to every word between negation and following punctuation.
        # Simplified to word "didn't"
        # check if any document contain "didn't"

        # low case to all documents
        df['documents'] = df['documents'].str.lower()

        # Locate text till next punctuation
        negation = "didn't"
        punctuation_pattern = ","

        # for d in df['documents']:
        def change_if_negation(d):
            """
            Change the entry if it contains a negation.
            We use simplified version for negation equal to 'didn't' and punctuation is comma ','
            Case 1 : no punctuation, the phrase starts with negation
            Case 2 : no punctuation, negation in the middle
            Case 3 : there is punctuation, the phrase starts with negation
            Case 4 : there is punctuation, negation in the middle
            """
            if negation in d:
                # There is negation

                res = re.findall(f'{negation}([\S\s]*){punctuation_pattern}', d)
                if not res:
                    # Case 1
                    # - no punctuation after negation
                    # - negation has leading position

                    if d.startswith(negation):
                        # negation has leading position
                        res = re.findall(f'{negation}([\S\s]*)', d)[0]

                        # remove leading space
                        res = res.lstrip()

                        # add NOT_ to each word after negation
                        neg_d = ["NOT_"+w for w in res.split(" ")]

                        # recreate the phrase
                        d_new = negation + " " + " ".join(neg_d)

                        # print(f'No punctuation, leading negation : {d} -> {d_new}')
                        return d

                    elif not d.endswith(negation):
                        # Case 2
                        # - no punctuation after negation
                        # - negation not in the end of the phrase
                        # if negation in the end do nothing, it is only applicable to words between negation and next punctuation

                        res = re.findall(f'([\S\s]*){negation}([\S\s]*)', d)

                        # remove leading space for part after negation
                        after = res[0][1].lstrip()

                        # add NOT_ to each word after negation
                        neg_d = ["NOT_" + w for w in after.split(" ")]

                        # recreate the phrase
                        d_new = res[0][0] + negation + " " + " ".join(neg_d)
                        # print(f'No punctuation, negation in the middle : {d} -> {d_new}')
                        return

                else :
                    # Case 3
                    # - there is punctuation after negation
                    # - negation has leading position

                    if d.startswith(negation):
                        # negation has leading position
                        res = re.findall(f'{negation}([\S\s]*){punctuation_pattern}([\S\s]*)', d)[0]

                        # remove leading space for part after negation
                        start = res[0].lstrip()

                        # add NOT_ to each word after negation
                        neg_d = ["NOT_"+w for w in start.split(" ")]

                        # recreate the phrase
                        # not recreating punctuation
                        d_new = negation + " " + " ".join(neg_d) + res[1]

                        # print(f'Punctuation, leading negation : {d} -> {d_new}')
                        return d

                    else:
                        # Case 4
                        # - there is punctuation after negation
                        # - negation has not leading position

                        res = re.findall(f'([\S\s]*){negation}([\S\s]*){punctuation_pattern}([\S\s]*)', d)[0]

                        # remove left and right spaces from matches
                        res = [r.lstrip().rstrip() for r in res]

                        # add NOT_ to each word after negation and before punctuation
                        neg_d = ["NOT_"+w for w in res[1].split(" ")]

                        # recreate the phrase
                        # not recreating punctuation
                        d_new = res[0] + " " + negation + " " + " ".join(neg_d) + " " + res[2]
                        return d
            else:
                # no negation, return the initial entry
                return d

        df['doc_'] = df['documents'].apply(lambda row : change_if_negation(row))

        return df

    def get_user_input_if_binary(self):
        """
        Get user input whether to use Binary multinomial Naive Bayes.
        """
        question = "Do you want to use Binary multinomial Naive Bayes (y) or Naive Bayes (n): "
        text_if_error = "Expecting 'y' or 'n'. "

        x = input(f"{question}").lower()
        if x not in ['n', 'y']:
            x = input(f"{text_if_error}{question}").lower()

        if x == 'y':
            print('Binary multinomial Naive Bayes is applied.')
            return True

        else:
            print('Naive Bayes is applied.')
            return False

    def compute_prior_probability(self):
        """ Compute prior probability of the class
        P(cj) = Ncj / Ntotal
        number of documents in that class divided by all documents in all classes
        :return: prior probability of all classes
        """

        # based on training dataset only as {category : prior}
        # find distinct categories

        # Create training and test subset
        self.training_subset = self.df.query("dataset == 'training'")
        self.test_subset = self.df.query("dataset == 'test'")

        # Get counts per category
        data = (self.training_subset
                .groupby(["category"])
                .agg({'documents': ['count']})
                .reset_index()
                )
        data.columns = ['category', 'count']

        # Add column for count per class divided by total number of docs
        Ndoc = len(self.training_subset)
        data['prior'] = data['count'] / Ndoc

        # counts = counts.to_dict(orient='records')
        return data

    def drop_unknown_words(self):
        """
        Drop all words that exist in test set and not in training set.
        :return: cleaned test set
        """

        def get_unque_words(df):
            # Concatenate all documents in one string
            data = ' '.join(df["documents"])

            # Create list of unique words
            data = list(set(data.split(" ")))

            return data

        self.vocabulary = get_unque_words(self.training_subset)

        test_words = get_unque_words(self.test_subset)

        # UNK words : words in test and NOT in training
        self.UNK = np.setdiff1d(test_words, self.vocabulary)
        print(f"UNK words : {self.UNK}")

        if self.binary_naives_bayes:
            # Drop UNK words from test set
            self.test_words = list(set(test_words).intersection(self.vocabulary))
        else:
            data = ' '.join(self.test_subset["documents"])
            data = list(data.split(" "))
            self.test_words = [x for x in data if x not in self.UNK]

        print(f'Test set without UNK words : {self.test_words}')

    def compute_likelihoods(self):
        """Compute likelihoods from training set
        P(wi|c) = count(wi,c) + 1 / (SUM count(w,c) + |Vocabulary|)
        count(wi,c) count for any word in the class, how often the word occurs
        SUM count(w,c) sum for all words in the vocabulary their counts in this class

        For the given word,  we need to know
            - its count in the current class
            - the SUM of all words in that class
            - vocabulary size
        """

        likelihoods = pd.DataFrame(self.vocabulary, columns=['words'])

        def words_in_class(category, df=self.training_subset):
            df_class = df.query('category == @category')
            words = ' '.join(df_class['documents'])

            if self.binary_naives_bayes:
                # Create list of unique words
                data = list(set(words.split(" ")))
            else:
                # Create list of words
                data = list(words.split(" "))

            # How many words in the classes
            count = len(data)

            return data, count

        words_plus, words_plus_count = words_in_class("+")
        words_minus, words_minus_count = words_in_class("-")

        def word_count(x, category):
            if category == "+":
                numerator = words_plus.count(x) + 1
                denominator = words_plus_count + len(self.vocabulary)

            elif category == "-":
                numerator = words_minus.count(x) + 1
                denominator = words_minus_count + len(self.vocabulary)

            else:
                pass

            return numerator / denominator

        likelihoods['+'] = likelihoods.apply(lambda x: word_count(x['words'], "+"), axis=1)
        likelihoods['-'] = likelihoods.apply(lambda x: word_count(x['words'], "-"), axis=1)

        print("\nProbabilities based on train set")
        print(likelihoods)

        return likelihoods

    def score_test_set(self):
        """
        Compute the probability of negative and positive class
        and take whichever is higher
        probability = prior * likelihood
        P(-)P(S|-) and P(+)P(S|+)
        :return: Probability of test data in class
        """

        def calculate_probability(category):

            likelihood = [self.likelihoods.query("words == @w")[category].values[0] for w in self.test_words]
            likelihood = reduce(lambda x, y: x * y, likelihood)

            prior = self.prior.query('category=="+"')['prior'].values[0]

            probability = prior * likelihood

            return probability

        print(f'\nFor {self.test_words} the category is (+/-) ... ')

        if calculate_probability("+") > calculate_probability("-"):
            print("category +")
        elif calculate_probability("+") < calculate_probability("-"):
            print("category -")
        elif calculate_probability("+") == calculate_probability("-"):
            print("Same probability for both categories")
        else:
            pass
