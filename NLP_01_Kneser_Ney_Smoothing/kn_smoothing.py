import re
import nltk
from nltk.tokenize import sent_tokenize
from wordcloud import WordCloud
from pathlib import Path
import pickle
from collections import Counter
from itertools import tee
import string

stopwords = nltk.corpus.stopwords.words('french')

FILEINPUTDATA_18jun1940 = "./data/L'appel du 18 juin du général de Gaulle.txt"
FILEINPUTDATA_22jun1940 = "./data/L'appel du 22 juin 1940.txt"
FILEINPUTDATA_03aug1940 = "./data/A tous les Francais 02 aout.txt"

FILENAMEINPUTDATAPICKLE = "./data/preprocessed_input_data.pkl"

FILESAVEWORDCLOUD = "./fig/de Gaulle 1940 - wordcloud.png"

DISCOUNT = 0.75


class Kneser_Ney_Smoothing:
    def __init__(self):

        self.text = self.load_data()

        self.all_bigrams = list()
        self.unigram_dict = {}
        self.lambda_ = {}
        self.p_continuation = {}
        self.c = None

        self.calculate_kn_smoothing_parameters()


    def load_data_txt(self):
        """
        Load raw data from local txt files
        :return: raw data
        """
        # Reads the file as an individual string
        with open(FILEINPUTDATA_18jun1940, encoding="utf-8") as f:
            text_18jun1940 = f.read()

        with open(FILEINPUTDATA_22jun1940, encoding="utf-8") as f:
            text_22jun1940 = f.read()

        with open(FILEINPUTDATA_03aug1940, encoding="utf-8") as f:
            text_03aug1940 = f.read()

        text = text_18jun1940 + '\n' + text_22jun1940 + '\n' + text_03aug1940

        return text

    def load_data(self):
        """
        If a pickle file exists : load data from it,
        otherwise load data from local txt files and preprocess
        :return: save preprocessed data to self-variable
        """
        my_file = Path(FILENAMEINPUTDATAPICKLE)

        if my_file.is_file():
            # file exists:
            # Load from pickle file if exists
            with open(FILENAMEINPUTDATAPICKLE, 'rb') as f:
                text = pickle.load(f)


        else:
            # Load from local files
            text = self.load_data_txt()

            # Preprocess data from local txt files
            text = self.preprocess_data(text)

            # Save preprocessed data to file
            with open(FILENAMEINPUTDATAPICKLE, 'wb') as f:
                pickle.dump(text, f, protocol=pickle.HIGHEST_PROTOCOL)

        return text

    def preprocess_data(self, text):
        """
        Preprocess data :
        remove punctuation,
        low case,
        add special symbols in the beginning and at the end of each sentence
        :return: cleaned text
        """
        # Remove commas and change to a low case
        text = re.sub(r",", "", text.lower())

        # Remove multiple spaces
        text = re.sub("\\s+", " ", text)

        # Add special symbols in the beginning and at the end of the sentence
        special_symbol_beginning = '<s> '
        special_symbol_end = ' </s>'
        text = sent_tokenize(text.lower())

        # Train corpus has punctuation in the end
        text = [special_symbol_beginning + t[:-1] + special_symbol_end for t in text]

        return  text

    def pairwise(self,iterable):
            """Support function to create pairs. Used by two other self.functions"""
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

    def calculate_kn_smoothing_parameters(self):
        """
        Calculate probability of next word given previous word, use self.parameters
        :return: probability of next word given previous word
        """

        # P(wi | wi - 1) = max(c(wi - 1, wi) - d, 0) / c(wi - 1) +
        # lambda (wi-1) P_continuation(wi)

        # 1. Extract unique words (or type of words) from the corpus
        text_str = ' '.join(self.text)
        # Split to words
        text_words = text_str.split(" ")
        # Unique words
        words = list(set(text_words))

        # 2. Find all bigrams
        # All bigrams in the corpus
        self.all_bigrams = list(map(lambda x: (x[0], x[1]), self.pairwise(text_words)))

        # Bigram types, or unique bigrams
        bigram_types = list(set(self.all_bigrams))

        # 3. Unigrams c(wi-1)
        # Count of bigrams (not types, all bigrams) that starts with the same first word wi-1
        # Ex. for bigram "Simon says", how many bigrams start with "Simon"
        for w in words:
            self.unigram_dict[w] = sum([b[0] == w for b in self.all_bigrams])

        # 4. Bigrams c(wi - 1, wi)
        # Count of occurrence of bigrams with words wi-1 and wi
        # Ex. "Vae victis" > word "vae" is word wi-1, word "victis" is wi
        self.c = dict(Counter(self.all_bigrams))

        # 5. Bigrams correction / discount
        # max(c(wi - 1, wi) - d, 0)
        self.c = {key: max(value - DISCOUNT, 0) for key, value in self.c.items()}

        # 6. Novel continuation
        # a number of bigram types the word completes
        # P_continuation (wi)
        # Potential improvement: use nested dictionary to combine continuation and unigram count
        # not self.all_bigrams but unique
        word_types_wi_minus_1 = {}

        for w in words:
            self.p_continuation[w] = sum([b[1] == w for b in bigram_types])
            word_types_wi_minus_1[w] = sum([b[0] == w for b in bigram_types])

        # Normalizes p_continuation
        # Divide by is a total number of word bigram types | {wj-1,wj : c(wj-1,wj)>0} |
        self.p_continuation = {key: value / len(bigram_types) for key, value in self.p_continuation.items()}

        # 7. Lambda
        # All probability mass from normalized discount
        # from higher order probabilities
        # and use it to assign to unigram probabilities
        #
        # lambda(wi-1) = d / c(wi-1)
        #     |{w : c(wi-1,w)>0}|
        # where d / c(wi-1) is normalized discount
        # |{w : c(wi-1,w)>0}| - total number of word types that can follow the context of wi-1
        #
        # how many word types we discounted
        # how many types we have applied the normalized discount
        # when multiplied together, we know how much probability mass total to continuation of the word
        # c(wi-1) has been calculated as unigram_dict[w] in 3.Unigrams c(wi-1)
        # There is an empty string, and simply d / value return float division by zero error, using if-else
        # [f(x) if condition else g(x) for x in sequence]
        normalized_discount = {key: 0 if value == 0 else DISCOUNT / value for key, value in self.unigram_dict.items()}

        # Lambda
        # sudo : self.lambda_ = [normalized_discount[w] / word_types_wi_minus_1[w]
        # Python {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}

        self.lambda_ = {k: normalized_discount.get(k, 0) + word_types_wi_minus_1.get(k, 0) for k in set(normalized_discount) | set(word_types_wi_minus_1)}


    def calculate_probability(self):
        """Calculate probability of the text
        """

        # 1. Prompt for user input
        user_text = input('Enter the text:\n').lower()
        # print(user_text)
        # user_text = 'Pourtant, je suis convaincu, et ce n’est pas un faux optimisme, que dans ce contexte de crises peut naître le meilleur.'
        # user_text = 'La France'

        # Add period in the end if no punctuation
        # To be able to re-use the same pre-process function
        if user_text[-1] not in string.punctuation:
            user_text = user_text + '.'

        # 2. Preprocess text
        user_text = self.preprocess_data(user_text)

        # After preprocessing the text is saved as a list of one element
        # transform list to string
        user_text = user_text[0]

        # Split to words
        user_text = user_text.split(" ")

        # 3. Find all bigrams in corpus
        all_bigrams = list(map(lambda x: (x[0], x[1]), self.pairwise(user_text)))

        # 4. Calculate probability for every pair and multiply

        # Probability is calculated iterating through bigrams
        # the first element is set to one to unsure multiplication is correct
        probability = 1

        # Iterate through pairs and return joint probability
        for b in all_bigrams:
            word_w_minus_1 = b[0]
            word_w = b[1]
            probability *= self.c[b] / self.unigram_dict[word_w_minus_1] + self.lambda_[word_w_minus_1]*self.p_continuation[word_w]

        # TO DO : handle unknown words UNK
        return probability


    def plot_wordcloud(self):
        """
        plot word cloud
        :return: save a chart of word cloud
        """

        # Load a raw text
        text = self.load_data_txt()

        # Word cloud for a raw text has "de", "la", "le" as most popular words
        # For better insights let's remove stopwords

        # Remove punctuation and change to a low case
        text = re.sub(r"[,?!.]*", "", text.lower())

        # Tokenize text
        words = text.split()

        # Remove stop words
        words = [w for w in words if w not in stopwords]

        # Concatenate words in one string - as required for the chart
        text = ' '.join(words)

        # Create and generate a word cloud image
        # We use load_data as the input since it requires text as a string, not lost
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)

        # Save the image in the img folder
        wordcloud.to_file(FILESAVEWORDCLOUD)
