import pandas as pd
import regex as re
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated, Laplace

INPUTFILECONFUSIONMATRIX = "./data/count_1edit.txt"


class EditConfusionMatrix:
    def __init__(self):
        self.df = self.load_data()
        self.create_substitution_confusion_matrix()
        self.df_subset = None

    def load_data(self):
        """
        Load counts for all single-edit spelling correction edits,
        from the file spell-errors.txt by Peter Norvig
        :return: data of single-edit
        """

        df = pd.read_csv(INPUTFILECONFUSIONMATRIX,
                         encoding="ISO-8859-1",
                         sep='\t',
                         header=None,
                         names=['letters', 'count'])

        # Data cleaning
        # One row index 415 has '|'
        # We will assume it is means to change space to empty string.
        # Operations with spaces do not bring value for the current exercise,
        # and this entry will be removed
        df.drop(index=[415], inplace=True)

        # Remove space in column values using replace() function
        df['letters'] = df['letters'].apply(lambda x: x.replace(' ', ''))

        # Create explicit columns for a correct and an incorrect letter
        df['correct'] = [re.search(r'(.*)\|(.*)', l).group(1) for l in df['letters']]
        df['incorrect'] = [re.search(r'(.*)\|(.*)', l).group(2) for l in df['letters']]

        # Drop the original column with correct pipe incorrect letters
        df = df.drop('letters', axis=1)

        # Rearrange columns order
        df = df[['correct', 'incorrect', 'count']]

        return df

    def create_substitution_confusion_matrix(self):
        """
        Create substitution confusion matrix from the list of one edit.
        Memo : substitution [x,y] : count (x typed as y)
        :return: substitution confusion matrix
        """

        # One letter changed to one letter
        # alternative: self.df['correct_len'] = self.df[['correct']].apply(lambda x: x.str.len(), axis=1)
        self.df['correct_len'] = [len(x) for x in self.df.correct]
        self.df['incorrect_len'] = [len(x) for x in self.df.incorrect]

        # Define operation
        def define_operation(a, b):
            # 1. correct is -1 to incorrect
            if len(b) - len(a) == 1:
                return 'ins'
            # 2. correct and incorrect have length of one: e -> i
            elif len(a) == len(b) and len(a) == 1:
                return 'sub'
            # 3. correct and incorrect have length of two: bh -> hb
            elif len(a) == len(b) and len(a) == 2:
                return 'trans'
            # 4. correct is +1 to incorrect
            elif len(a) - len(b) == 1:
                return 'del'
            # check for unknown operations
            else:
                return 'unk'

        self.df['operator'] = self.df.apply(lambda x: define_operation(x['correct'], x['incorrect']), axis=1)

        # with pd.option_context('display.max_rows', None, ):
        # print(self.df.head(10))
        # print(self.df.query('correct_len == 2 & incorrect_len==2'))

        # Optional : check we assigned operations to all entries
        # operator_unk = 'unk'
        # print(self.df.query('operator == @operator_unk'))

        # TO DO: explore and save to log file
        # Exploration
        # print(self.df.operator.value_counts())

    def calculate_channel_model(self, corpora):
        """Generate probabilities from the confusion matrix
            P(x|w) =
            del   [wi-1, wi] / count(wi-1, w) if deletion
            ins   [wi-1, xi] / count(wi-1)    if insertion
            sub   [xi, wi]   / count(wi)      if substitution
            trans [wi, wi+1] / count(wi,wi+1) if transposition
        """

        # probability to be calculated based on the corpora

        def get_counts(correct):
            """
            Get counts for different operations based on the corpora
            :param operator: one of four transformations
            :param correct: correct letter
            :param incorrect: incorrect letter - typo
            :return: counts to calculate probability
            """
            return sum([w.count(correct) for w in corpora])

        # Denominator for probability
        self.df['total'] = self.df.apply(lambda x: get_counts(x['correct']), axis=1)

        # Calculate a channel model, count / total
        # Potentially to use log here
        self.df['channel_model'] = self.df['count'] / self.df['total']

    def get_probability(self):
        """
        Calculate probability of word using in-built nltk unigram language model.
        """
        # rows in scope
        # -------------------------------
        # candidate | correct | error |
        # actress   | t       | -     |
        # cress     | -       | a     |
        # caress    | ca      | ac    |
        # access    | c       | r     |
        # across    | o       | e     |
        # acres     | -       | s     |
        # -------------------------------

        # print(self.df.query('correct == "" & incorrect == "a"'))
        value01 = self.df.index[(self.df['correct'] == "") & (self.df['incorrect'] == "a")]

        # print(self.df.query('correct == "ca" & incorrect == "ac"'))
        value02 = self.df.index[(self.df['correct'] == "ca") & (self.df['incorrect'] == "ac")]

        # print(self.df.query('correct == "c" & incorrect == "r"'))
        value03 = self.df.index[(self.df['correct'] == "c") & (self.df['incorrect'] == "r")]

        # print(self.df.query('correct == "o" & incorrect == "e"'))
        value04 = self.df.index[(self.df['correct'] == "o") & (self.df['incorrect'] == "e")]

        # Compile results
        rows_ = [value01[0], value02[0], value03[0], value04[0]]
        columns = ['correct', 'incorrect', 'operator', 'channel_model']
        self.df_subset = self.df[columns].loc[rows_].reset_index(drop=True)

        # words
        # candidate | correct | error |
        # actress   | t       | -     |
        # cress     | -       | a     |
        # caress    | ca      | ac    |
        # access    | c       | r     |
        # across    | o       | e     |
        # acres     | -       | s     |
        candidate = ['cress', 'caress', 'access', 'across']
        self.df_subset.insert(0, 'candidate', candidate)

    def calculate_word_model(self, text, column):
        """Calculate word model
        Using built-in nltk language models.
        We have chosen Lapcace model to cater to unknown words.
        At least one of the possible corrections (actress) does not apper in STM corpus """

        # Define the text for vocabulary and ngram counts.
        train, vocab = padded_everygram_pipeline(2, text)

        # Train model
        lm = Laplace(2)

        # Fit model
        lm.fit(train, vocab)

        # Fit model
        lm.fit(train, vocab)

        # Possible corrections
        # cress     | -       | a     |
        # caress    | ca      | ac    |
        # access    | c       | r     |
        # across    | o       | e     |
        # acres     | -       | s     |
        possible_corrections = ['cress', 'caress', 'access', 'across']

        # Get log probability

        for w in possible_corrections:
            # The chance that "b" is preceded by "a" >>> lm.score("b", ["a"])
            # Initial phrase : versatile acress whose
            # 1. alternative is to use log not score
            overall_score = lm.score(w, ["versatile"]) * lm.score("whose", [w])
            self.df_subset.loc[self.df_subset['candidate'] == w, [column + " word model"]] = overall_score

        # Multiply channel model and word model
        self.df_subset[column + " probability"] = self.df_subset['channel_model'] * self.df_subset[column + " word model"]

    def get_correct_spelling(self):
        """
        Print output in the form :
        For the misspelled word "acress", based on STM corpus the most probable word is "".
        """

        for corpus in ['STM', 'Abdication']:

            index_max_value = self.df_subset[corpus + ' probability'].idxmax()
            word = self.df_subset['candidate'].loc[index_max_value]

            print(f'For the misspelled word "acress", based on {corpus} corpus the most probable word is "{word}".')

