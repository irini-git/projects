import re

import pandas as pd
import altair as alt

import unidecode

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint

# pip install tensorflow-text

from sklearn.model_selection import train_test_split

FILENAME_SENTENCES_ALLAGREE = './data/Sentences_AllAgree.txt'

# import pandas as pd

class FSAData():
    def __init__(self):
        self.load_phrasebank()


    def load_phrasebank(self):

        # Load
        sentences_allagree = pd.read_csv(FILENAME_SENTENCES_ALLAGREE,
                                         engine='python',
                                         sep='.@', encoding='latin-1',
                                         header=None, names=['sentence', 'sentiment'])

        # self.explore_data(sentences_allagree)
        self.clean_data(sentences_allagree)

    def clean_data(self, df):



        def preprocess_text(text):
            # Tokenize the text

            # Manually remove special characters
            text = re.sub(r"([`!,'%-.$()+é:0-9=;®/£¦¼ó?])"," ", text)
            #text = re.sub(u"[àáâñ]", " ", text)
            #text = re.sub("ál", "al", text)
            # replace accented characters?
            text = unidecode.unidecode(text)
            tokens = word_tokenize(text.lower())

            # raise SystemExit(0)

            # Remove stop words
            filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

            # Lemmatize the tokens
            # lemmatizer = WordNetLemmatizer()
            # lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

            # Stemmer
            ps = PorterStemmer()
            stemmed_tokens = [ps.stem(token) for token in filtered_tokens]

            # raise SystemExit(0)

            # Join the tokens back into a string
            processed_text = ' '.join(stemmed_tokens)

            return processed_text

        df['sentence_text'] = df['sentence'].apply(preprocess_text)

        # Choose custom max words and max length
        str_ = ' '.join(df['sentence_text'])
        max_words = len(list(set(str_.split())))

        max_len = max([len(s.split()) for s in df['sentence_text']])
        max_len = int(max_len/2)

        # Word embedding
        data = df['sentence_text'].tolist()
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(data)
        sequences = tokenizer.texts_to_sequences(data)
        phrases = pad_sequences(sequences, maxlen=max_len)

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            # print(df['sentence_text'].values)
            # print(phrases)
            # print('-'*30)

        # Create embedding layer
        embedding_layer = Embedding(1000, 64)


    def text_classification(self, df):
        X_train, X_test, y_train, y_test = train_test_split(df['sentence_text'],
                                                            df['sentiment'],
                                                            test_size=0.1,
                                                            random_state=42)

    def explore_data(self, df):
        # Metadata
        print(f'Info : {df.info()}')
        print(f'Shape : {df.shape}')
        print(f"{df['sentiment'].value_counts()}")

        # Visualize
        source = df['sentiment'].value_counts(sort=True).to_frame('counts').reset_index()

        chart = alt.Chart(source).mark_bar().encode(
            y=alt.Y('sentiment:O', title='',sort='descending'),
            x=alt.X('counts:Q', title=''),
        ).properties(
            title='When all agree, what is sentiment distribution?'
        )

        chart.save('./fig/FSA_phrasebank_allagree.png')


