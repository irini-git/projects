import pandas as pd
import re
import numpy as np

from pathlib import Path

# from stanfordcorenlp import StanfordCoreNLP
import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import shakespeare, brown
from functools import reduce
import math
from gensim.models import Word2Vec
import multiprocessing

from numpy.ma.core import negative
from regex import split

# nltk.download('shakespeare')
# nltk.download('brown')

# Uncomment if run for the first time
# nltk.download('words')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('averaged_perceptron_tagger_eng')

# Constants
TERMS = ['battle', 'good', 'fool', 'wit']
WORD_WINDOW = 2


class WordVector:
    def __init__(self):
        self.nltk_skipgram()

    def skipgram_model(self):
        """
        Train a classifier that is given a candidate (word, context) pair
            (apricot, jam) -> P(+|apricot, jam) high
            (apricot, aardvark) -> P(-|apricot, aardvark) high
        And assigns each pair a probability
        c is a real context word for +
            P(+|w,c)
            P(-|w,c) = 1 - P(+|w,c)
        """
        target_word = 'apricot'
        train_sentence = 'lemon, a tablespoon of apricot jam, a pinch'
        print('-'*20)
        print(f'Target word : {target_word}')
        print(f'Train sentence : {train_sentence}')
        print(f'Word window : +/- {WORD_WINDOW} context words')

        # target word
        train_sentence_lst = train_sentence.split()
        print(train_sentence_lst)
        target_ind = train_sentence_lst.index('apricot')
        context_words = train_sentence_lst[target_ind-WORD_WINDOW:target_ind] + train_sentence_lst[target_ind+1:target_ind+WORD_WINDOW+1]
        print(f'Positive examples : {context_words}')


    def nltk_skipgram(self):
        sentences = brown.sents()
        EMB_DIM = 300

        # train model
        # sentences=None, size=100, alpha=0.025, window=5,
        # min_count=5, max_vocab_size=None, sample=0.001,
        # seed=1, workers=3, min_alpha=0.0001, sg=0,
        # hs=0, negative=5, cbow_mean=1,
        # hashfxn=<built-in function hash>, iter=5, null_word=0,
        # trim_rule=None, sorted_vocab=1, batch_words=10000)

        w2v = Word2Vec(sentences, vector_size=EMB_DIM, window=5, min_count=5,
                       negative=15, workers=multiprocessing.cpu_count())

        word_vectors = w2v.wv
        result = word_vectors.similar_by_word('Saturday')
        print("Most similar to 'Saturday':\n", result[:3])

        result = word_vectors.similar_by_word('money')
        print("Most similar to 'money':\n", result[:3])

        result = word_vectors.similar_by_word('child')
        print("Most similar to 'child':\n", result[:3])

        result = word_vectors.most_similar(positive=['child'], negative=['person'])
        print("Most similar to 'child' but dissimilar to 'person':\n", result[:3])

        # not present in vocabulary
        # result = word_vectors.most_similar(positive=['king, woman'], negative=['man'])
        # print("Most similar to 'king' and 'woman' but dissimilar to 'man':\n", result[:3])





    def cosine_example(self):
        """
        COmpute the similarity
        btw cherry and information
        """

        columns = ['pie', 'data', 'computer']
        data = [(442, 8, 2), (5, 1683, 1670), (5, 3982, 3325)]
        index_ = ['cherry', 'digital', 'information']
        df = pd.DataFrame(data, columns=columns, index=index_)

        def calculate_cosine(a,b):
            up = np.dot(df.loc[a].values, df.loc[b].values)
            down = np.linalg.norm(df.loc[a].values) * np.linalg.norm(df.loc[b].values)

            print(f'Cosine similariry between "{a}" and "{b}" is {round(up/down,4)}.')
            return

        calculate_cosine(a='cherry', b='information')
        calculate_cosine(a='digital', b='information')


    def load_local_data(self):

        print(f'Which plays of Shakespeare contain the words?')
        df = pd.DataFrame(index=TERMS)

        # Load four plays
        as_you_like_it_txt = Path('./data/as_you_like_it.txt').read_text()
        twelve_night_txt = Path('./data/twelve_night.txt').read_text()
        julius_caesar_txt = Path('./data/julius_caesar.txt').read_text()
        henri_v_txt = Path('./data/henri_v.txt').read_text()


        for txt, title_ in zip([as_you_like_it_txt,twelve_night_txt,julius_caesar_txt,henri_v_txt],
                               ['as_you_like_it_txt','twelve_night_txt','julius_caesar_txt','henri_v_txt']):
            column_if_present = [1 if t.lower() in txt else 0 for t in TERMS]
            column_tf = [len(re.findall(f'\\b{t.lower()}\\b', txt.lower())) if t.lower() in txt else 0 for t in TERMS]
            column_log_tf = [math.log10(t+1) if t > 0 else 0 for t in column_tf] # log10(count (t,d) + 1)
            df[f'{title_}_boolean'] = column_if_present
            df[f'{title_}_TF'] = column_tf
            df[f'{title_}_LOGTF'] = column_log_tf

        N = 4
        boolean_columns = [c for c in df.columns if 'boolean' in c]
        df['df'] = df[boolean_columns].sum(axis=1, numeric_only=True)

        df['idf'] = np.log10(N / df['df'])
        TF_columns = [c for c in df.columns if '_LOGTF' in c]

        for c in TF_columns:
            title = re.search(r'^(.*?)\_LOGTF', c).group(1)
            df[f'{title}_TFIDF'] = df[c] * df['idf']

        with pd.option_context('display.max_rows', None, 'display.max_columns',None):
            print(round(df,2))

    def load_data(self):
        # return term document matrix
        plays = shakespeare.fileids()
        print(f'Which plays of Shakespeare contain the words?')
        df = pd.DataFrame(index=TERMS)

        for p in plays:

            def list_to_string(list_):
                # Support function to preprocess list of strings
                temp = ' '.join(list_)
                temp = re.sub(r'\n?', '', temp)
                temp = temp.lower()
                return temp

            # Load a play
            play = shakespeare.xml(f'{p}')

            # Title is the only element, use [0][0] to extract it
            title_ = [list(p.itertext()) for p in play if p.tag == 'TITLE'][0][0]
            full_text = [list(p.itertext()) for p in play if p.tag == 'ACT']
            # Flatten the list
            text_ = reduce(lambda x, y: x + y, full_text)
            # Apply custom function to clean
            txt = list_to_string(text_)

            # Create a column for if a term is present in play
            column_if_present = [1 if t.lower() in txt else 0 for t in TERMS]

            column_tf = [len(re.findall(f'\\b{t.lower()}\\b', txt.lower())) if t.lower() in txt else 0 for t in TERMS]
            column_log_tf = [1 + math.log10(t) if t > 0 else 0 for t in column_tf]

            df[f'{title_}_boolean'] = column_if_present
            df[f'{title_}_TF_'] = column_tf
            df[f'{title_}_TF'] = column_log_tf

        # Calculate Inversed Document frequency / one value per collection
        # We have a different collection compared to example in lecture, so final numbers differ
        N = len(plays)
        boolean_columns = [c for c in df.columns if 'boolean' in c]
        df['df'] = df[boolean_columns].sum(axis=1, numeric_only=True)

        df['idf'] = np.log10(N / df['df'])  # should have checked for zero

        TF_columns = [c for c in df.columns if '_TF' in c]

        for c in TF_columns:
            title = re.search(r'^(.*?)\_TF', c).group(1)
            df[f'{title}_TFIDF'] = df[c] * df['idf']

        print('Document matrix ---------- ')
        columns_name_tf_idf = [c for c in df.columns if '_TFIDF' in c]
        columns_name_tf = [c for c in df.columns if '_TF_' in c]

        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(df[columns_name_tf])

        return



class EntityData:
    def __init__(self):
        self.extract_entity()

    def extract_entity(self):

        sample_text = """Jane Villanueva of United, a unit of United Airlines Holding, said the fare applies to the Chicago route."""

        # Tokenization: Split the sample_text into a list of words or tokens
        tokens = nltk.word_tokenize(sample_text)

        # Tagging
        tagged_tokens = nltk.pos_tag(tokens)

        # Extract entities
        entities = nltk.ne_chunk(tagged_tokens)

        print('-'*30)
        print('BIO tagging')
        print(f'Sample text : {sample_text}')

        words = [e.leaves()[0][0] if type(e) is nltk.tree.tree.Tree else e[0] for e in entities]
        labels = [e.label() if type(e) is nltk.tree.tree.Tree else e[1] for e in entities]

        df = pd.DataFrame({'words': words, 'labels': labels})
        print(df)

    def do_part_of_speech_tagging(self):
        txt = """There were 70 children there. Preliminary findings were reported in today's New England Journal of Medicine."""

        tokenized = sent_tokenize(txt)

        for i in tokenized:
            wordsList = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(wordsList)
            print(tagged)

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                           timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
                    'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
                    'pipelineLanguage': 'en',
                    'outputFormat': 'json'
                }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens


class NRC_VAD:
    def __init__(self):
        self.arousal = pd.read_csv('./data/arousal-NRC-VAD-Lexicon.txt',
                                   sep='\t', header=None, names=['word', 'score'])
        self.dominance = pd.read_csv('./data/dominance-NRC-VAD-Lexicon.txt',
                                     sep='\t', header=None, names=['word', 'score'])
        self.valence = pd.read_csv('./data/valence-NRC-VAD-Lexicon.txt',
                                   sep='\t', header=None, names=['word', 'score'])

    def search(self, word_):
        print(f'Scores for "{word_}"')
        print(f"  - arousal {self.arousal.query('word==@word_')['score'].values[0]}")
        print(f"  - dominance {self.dominance.query('word==@word_')['score'].values[0]}")
        print(f"  - valence {self.valence.query('word==@word_')['score'].values[0]}")








# ----------------