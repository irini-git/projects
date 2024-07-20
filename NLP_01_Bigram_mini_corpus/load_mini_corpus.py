import pandas as pd

sentence_01 = "<s> frère Jacques </s>"
sentence_02 = "<s> dormez vous </s>"
sentence_03 = "<s> sonnez les matines </s>"
sentence_04 = "<s> ding dang dong </s>"

CORPUS = [sentence_01, sentence_01, sentence_02, sentence_02, sentence_03, sentence_03, sentence_04, sentence_04]


class Mini_corpus:
    def __init__(self):
        self.mini_corpus = CORPUS
        self.words = None
        self.all_bigrams = list()

        self.calculate_probabilities()

    def calculate_probabilities(self):
        """
            the function calculate the conditional probability of the next word using bigram model

            :param self.words: unique words in the corpus
            :param df: dataframe with probabilities with index is history (previous word) and column is next word
            :param self.all_bigrams: all bigrams in the corpus incl. special symbols for the beginning and the end
            :return: print the dataframe with conditional probabilities
            """

        # Return unique words in the corpus
        words = set()
        for c in self.mini_corpus:
            words = words.union(set(c.split()))

        self.words = sorted(list(words))

        # Create dataframe for probabilities
        df = pd.DataFrame(columns=self.words,
                          index=self.words)

        # Find all bigrams
        for c in self.mini_corpus:
            c = c.split()
            self.all_bigrams += [[c[i], c[i + 1]] for i in range(len(c) - 1)]

        # iterate through index - column
        # Find bigrams with specific history - first element
        # and assign to a specific element in dataframe
        # Example : df.loc["<s>", "I"] = 0.66

        for index_ in self.words:
            for column_ in self.words:
                bigram = index_ + ' ' + column_

                # Count of the bigram C(wn−1wn)
                count_of_bigram = sum([bigram in c for c in self.mini_corpus])

                # Count of bigrams that start with the same first word
                count_of_bigram_start_with = sum([b[0] == index_ for b in self.all_bigrams])

                # Calculate probability
                # Count of the bigram C(wn−1wn) normalized by the sum of all the bigrams
                # that share the same first word wn−1:
                # Rounding is for visibility
                try :
                    bigram_probability = round(count_of_bigram/count_of_bigram_start_with,2)

                except ZeroDivisionError:
                    # Catch the error when dividing by zero, i.e. nothing stats with </s>
                    bigram_probability = 0

                # Assign value
                df.loc[index_, column_] = bigram_probability

        # Print full output
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(df)
