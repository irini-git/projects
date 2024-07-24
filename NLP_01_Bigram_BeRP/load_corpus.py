import pandas as pd
import regex as re
from collections import Counter
from wordcloud import WordCloud
import altair as alt
from functools import reduce

FILENAMELOADTXT = "NLP_01_Bigram_BeRP/data/transcript.txt"
# raw data is available via https://github.com/wooters/berp-trans
# raw data is not uploaded to this project, and is a part of gitingore
# for the filename, the alternative is to save it as environmental variable

FILETOSAVE = r"NLP_01_Bigram_BeRP/data/test.txt"
# Save to file - cleaned data in txt format

FILEFIGHISTLENMESSAGERAW = "NLP_01_Bigram_BeRP/fig/hist_message_len_raw.png"
# Filename to save chart for distribution of message length

FILEFIGHISTLENMESSAGECLEANED= "NLP_01_Bigram_BeRP/fig/hist_message_len_cleaned.png"
# Filename to save chart for distribution of message length

FILESAVEWORDCLOUD = r"NLP_01_Bigram_BeRP/fig/wordcloud_messages.png"
# Filename to save word cloud for raw messages

class BeRP_corpus:
    def __init__(self):
        self.corpus = None
        self.words = None
        self.all_bigrams = list()
        self.corpus_str = None
        self.corpus_str_len_raw = None
        self.corpus_str_len_cleaned = None
        self.df = None

        self.load_data()
        self.explore_data()
        self.preprocess_data()
        #self.visualize_data()
        self.calculate_probabilities()
        self.compute_probabilities()

    def load_data(self):
        """
        the function loads data from text file as list and
        :return: save data to 'self.corpus'
        """
        # Load data as dataframe
        self.corpus = pd.read_csv(FILENAMELOADTXT,
                                  header=None,
                                  names=['txt'])

        # Create columns for id and message
        self.corpus['id'] = self.corpus['txt'].str[:9]
        self.corpus['message'] = self.corpus['txt'].str[10:]

        # Remove raw message
        self.corpus.drop(columns=['txt'], inplace=True)

        # Set id as index
        self.corpus.set_index('id', inplace=True)

    def explore_data(self):
        """
        the function explores dataset
        :return: list insights
        """

        # Describe
        # Totally, 8566 messages and 6580 unique
        print('Describe dataframe:')
        print(self.corpus.describe())

        # What are repeated messages?
        # Most common messages are
        # - to start over
        # - to specify time of the meal (lunch, dinner)
        # - to say 'do not mind'
        c = Counter(list(zip(self.corpus.message)))
        print(f'\nMost common messages:\n{c.most_common(10)}')

        # What are the most common words?
        # By the way, the raw dataset has a histogram for words
        # Most common words are
        # - i
        # - to
        # - like
        # - food
        # - about
        # - the
        # - . period

        self.corpus_str = [m for m in self.corpus['message']]
        self.corpus_str = ' '.join(self.corpus_str)
        self.words = self.corpus_str.split()

        resulting_count = Counter(self.words)
        print(f'\nMost common words:\n{resulting_count.most_common(20)}')

        # What is the length of the phrase / message
        # The message is not cleaned, do the length includes all add. words like 'um', etc.
        self.corpus_str_len_raw = [len(m) for m in self.corpus['message']]
        print(f'\nCommon number of words in message:\n{Counter(self.corpus_str_len_raw).most_common(10)}')

        # TO DO : What is the duration of conversation
        # how many messages with same leading id

        # Potential improvement is to remove stop words as many are commonly used
        # It is not done due to the specifics of the current project

    def plot_wordcloud(self, text):
        """
        plot word cloud for all messages
        :return: save a chart of word cloud
        """

        # Create and generate a word cloud image
        # lower max_font_size, change the maximum number of word and lighten the background:
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)

        # Save the image in the img folder
        wordcloud.to_file(FILESAVEWORDCLOUD)

    def plot_hist_message_len(self, x, filename):
        """
        the function plot chart for length of the messages
        as a histogram with a global mean overlay
        :return: save chart to file
        """

        df_chart = pd.DataFrame(x, columns=['words'])

        x_label = "Number of words in message"
        chart_title = "Length of messages"
        chart_subtitle = ["How many words in a message"]

        base = alt.Chart(df_chart)

        bar = alt.Chart(
            df_chart,
            title=alt.Title(
                chart_title,
                subtitle=chart_subtitle
            )
        ).mark_bar().encode(
            alt.X("words:Q", bin=alt.Bin(extent=[0, 200], step=10), title=x_label),
            y='count()',
            color=alt.value("lightgray")
        )

        rule = base.mark_rule(color='tomato').encode(
            x='mean(words):Q',
            size=alt.value(2)
        )

        bar + rule

        chart = bar + rule

        chart.save(filename)

    def preprocess_data(self):
        """
        the function preprocesses data
        - Remove leading id and space.
        :return: cleaned data is saved to 'self.corpus'
        """

        # Clean text

        # 1. Remove asterisks, keep the word inside.Example: i * understand * -> i understand
        self.corpus['message'] = [re.sub(r'\*', r'', c) for c in self.corpus['message']]

        # Optional : test if the transformation was successful
        # idx = [716]
        # print(self.corpus.iloc[idx])

        # 2. Remove angle brackets <> and what is inside
        pattern_verbal_repairs = r'\<.+\>'
        self.corpus['message'] = [re.sub(pattern_verbal_repairs, r'', c)
                                  for c in self.corpus['message']]

        # Optional : test if the transformation was successful
        # idx = [9]
        # print(self.corpus.iloc[idx])

        # 3. Remove parenthesis and a hyphen before or after for word-fragments
        pattern_word_fragments = r'[\)\-*]|[\(]'
        self.corpus['message'] = [re.sub(pattern_word_fragments, r'', c)
                                 for c in self.corpus['message']]

        # Optional : test if the transformation was successful
        # idx = [777]
        # print(self.corpus.iloc[idx])

        # 4. Remove filled pauses and other noises
        pattern_filled_pauses = r'\[.*\]'
        self.corpus['message'] = [re.sub(pattern_filled_pauses, r'', c)
                                 for c in self.corpus['message']]

        # Optional : test if the transformation was successful
        # idx = [1]
        # print(self.corpus.iloc[idx])

        # 5. Remove leading space using .lstrip()
        self.corpus['message'] = [c.lstrip() for c in self.corpus['message']]

        # Optional : test if the transformation was successful
        # idx = [1]
        # print(self.corpus.iloc[idx])

        # 6. Add special symbols for beginning and end of message
        # Some messages consist of more than one sentence
        # In this exercise we keep the period as is in the message
        # The alternative is to split the message into sub-messages
        special_symbol_start = '<s> '
        special_symbol_end = ' </s>'
        self.corpus['message'] = [special_symbol_start + c + special_symbol_end for c in self.corpus['message']]

        # Save to file
        self.corpus.to_csv(FILETOSAVE, sep=' ', mode='w+')

        # Remove stop words ---------------
        # self.word_token = [w for w in words if w not in stopwords]  # tokens, all words in document

        # Length of cleaned messages - how many words are there
        self.corpus_str_len_cleaned = [len(m) for m in self.corpus['message']]

    def visualize_data(self):
        """
        Plot and save visualisations
        - word cloud
        - length of words
        :return: charts as png files
        """
        # Plot word cloud
        self.plot_wordcloud(self.corpus_str)

        # Histogram for length of message - raw
        self.plot_hist_message_len(self.corpus_str_len_raw, FILEFIGHISTLENMESSAGERAW)

        # Histogram for length of message - cleaned
        self.plot_hist_message_len(self.corpus_str_len_cleaned, FILEFIGHISTLENMESSAGECLEANED)

        # Potential improvement : plot histograms side by side

    def calculate_probabilities(self):
        """
            the function calculate the conditional probability of the next word using bigram model

            :param self.words: unique words in the corpus
            :param df: dataframe with probabilities with index is history (previous word) and column is next word
            :param self.all_bigrams: all bigrams in the corpus incl. special symbols for the beginning and the end
            :return: print the dataframe with conditional probabilities
            """

        # Return unique words in the corpus
        # words = set()
        # for c in self.corpus['message']:
        #     words = words.union(set(c.split()))
        # self.words = list(words)
        # Subset of dataset following the example from the book
        self.words = ['<s>', 'i', 'want', 'to', 'eat', 'chinese', 'english', 'italian', 'food', 'lunch', 'spend', '</s>']

        # Create dataframe for probabilities
        self.df = pd.DataFrame(columns=self.words,
                          index=self.words)

        # Find all bigrams
        for c in self.corpus['message']:
            c = c.split()
            self.all_bigrams += [[c[i], c[i + 1]] for i in range(len(c) - 1)]

        # iterate through index - column
        # Find bigrams with specific history - first element
        # and assign to a specific element in dataframe
        # Example : df.loc["<s>", "I"] = 0.66

        V = len(self.words) # total word types in the vocabulary V

        for index_ in self.words:
            for column_ in self.words:
                bigram = index_ + ' ' + column_

                # Count of the bigram C(wn−1wn)
                count_of_bigram = sum([bigram in c for c in self.corpus['message']])

                # Count of bigrams that start with the same first word
                count_of_bigram_start_with = sum([b[0] == index_ for b in self.all_bigrams])

                # Calculate probability ----------------------------------------
                # Vanilla : count of the bigram C(wn−1wn) normalized by the sum of all the bigrams
                # that share the same first word wn−1: C(wn−1wn) / C(wn−1)
                #
                # Laplace Smoothing :
                # For add-one smoothed bigram counts, we augment the unigram count by
                # the number of total word types in the vocabulary V:
                # C(wn−1wn) +1 / C(wn−1) +V
                # Comment : The effect of Laplace Smoothing is not very visible due to the small subset
                #
                try :
                    bigram_probability = (count_of_bigram + 1) / (count_of_bigram_start_with + V)

                except ZeroDivisionError:
                    # Catch the error when dividing by zero, e.g. nothing stats with </s>
                    bigram_probability = 0

                # Assign value
                self.df.loc[index_, column_] = bigram_probability

        # Print output
        # The summary table is somewhat different from the one provided in the book
        # The misalignment might be caused by different preprocessing
        print(f'\n{self.df}')

    def compute_probabilities(self):
        """
        Compute probability of the sentence
        by multiplying the appropriate bigram probabilities together:

        Example : <s> I want English food </s>
        P(<s> i want english food </s>) =
            P(i|<s>)P(want|i)P(english|want)P(food|english)P(</s>|food)

        :return: probability of the sentence
        """

        # Mind to add the words to the subset of df when it is created
        # Potential improvement : how to deal with unknown words UNK
        # - find a known word whose vector in this space is closest to the unknown word using some distance metrics.
        text = "I want chinese"

        # Lower case
        sentence = text.lower()

        special_symbol_start = '<s> '
        special_symbol_end = ' </s>'
        sentence = special_symbol_start + sentence + special_symbol_end

        c = sentence.split()
        bigrams = {c[i] : c[i + 1] for i in range(len(c) - 1)}

        test_probabilities = [self.df.loc[key, value] for key, value in bigrams.items()]

        result = reduce(lambda x,y:x*y,test_probabilities)
        print(f"\nThe probability of '{text}' is {round(result, 4)}.")

