import re
import glob
import os
import nltk
from nltk import word_tokenize, sent_tokenize

TXTFILEFOLDER = "./data/"


class SpellingCorrectionCorpora():
    def __init__(self, filenametxt):
        self.filenametxt = filenametxt
        self.text = self.load_txt_files()
        self.words = self.preprocess_data()

    def load_txt_files(self):
        """
        Load all available articles in txt format from the specified folder,
        and concatenate to one string.
        :return: self.text will raw data
        """
        txt_files = glob.glob(os.path.join(TXTFILEFOLDER, f'{self.filenametxt}.txt'))

        text = ''

        for file in txt_files:
            with open(file, 'r', encoding='utf-8') as f:
                text += f.read()

        return text

    def transform_data_for_lm(self):
        """
        Transform raw data to be used by nltk language model.
        Expected format : [['a', 'b', 'c'], ['a', 'c', 'd', 'c', 'e', 'f']]
        where sentences are split to words [[word, word], [word, word]]
        """

        text = self.load_txt_files()
        tokenized_text = [list(map(str.lower, word_tokenize(sent)))
                          for sent in sent_tokenize(text)]

        return tokenized_text

    def preprocess_data(self):
        """
        Preprocess data :
        remove punctuation, low case, remove special characters, remove multiple whitespaces
        :return: cleaned text, not split to words
        """

        # Remove group notation, e.g (Gn,∘)
        self.text = re.sub("\\s\\(\\w+,∘\\)", "", self.text)

        # Remove some mathematical notation : ⩾ ∈
        self.text = re.sub("\\S+⩾\\S+", "", self.text)

        # Change to a low case
        self.text = self.text.lower()  #.translate(str.maketrans('', '', string.punctuation))

        # Replace new line with whitespace
        self.text = re.sub("\\n+", " ", self.text)

        # Remove special characters
        self.text = re.sub("[►]", "", self.text)

        # Remove expressions like n∈n to denote relation "is an element of"
        # sff′∈hf′
        # ff′∈⋂cφ¯scφ
        self.text = re.sub("\\w(′)?∈(⋂)?\\w(′)?", "", self.text)

        # Remove degrees, °n or °
        self.text = re.sub("°(n)?", "", self.text)

        # Remove numbers
        self.text = re.sub("\\d+", "", self.text)

        # Remove multiple whitespaces
        # Might be a result of "of Fig. 1 with" -> "of Fig  with"
        self.text = re.sub("\\s+", " ", self.text)

        # split
        return self.text.split(" ")
