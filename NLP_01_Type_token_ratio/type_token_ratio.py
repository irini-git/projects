import re
from collections import Counter
import nltk
from os.path import exists
import pandas as pd
import plotly.express as px

FILEPATH = './data/'
stopwords = nltk.corpus.stopwords.words('english')
FILETOSAVE = './output/type_token_ratio.csv'
PLOT_TEMPLATE = 'plotly_white'

# Constants
AUTHOR = "Agatha Christie"


class Documents():
    """This class is responsible for loading data from local file,
    for preprocessing data and for calculating
    type-token ration (TTR)."""

    def __init__(self, filename):
        self.txt = None
        self.word_token = None
        self.filename = filename

        self.load_data()
        self.preprocess_data()
        self.calculate_rrt()

    def load_data(self):
        with open(self.filename, errors="ignore") as f:
            # Read the contents of the file into a variable
            self.txt = f.read()

    def remove_book_details(self):
        """Extract the pure text of the book omitting details added by publisher"""
        # Initializing substrings

        # Check if Gutenberg or English e-reader
        if "*** END" in self.txt:
            # Gutenberg text is surrounded by *** disclamaers https://www.gutenberg.org/ebooks/
            sub1 = " ***"
            sub2 = "*** END"

            # Getting index of substrings
            idx1 = self.txt.index(sub1)
            idx2 = self.txt.index(sub2)

            # Length of substring 1 is added to
            # get string from next character
            self.txt = self.txt[idx1 + len(sub1) + 1: idx2]

        else:
            # English e-reader text https://english-e-reader.net/
            sub2 = '- THE END -'

            self.txt = self.txt.split(sub2)[0]

    def preprocess_data(self):
        """Preprocess data"""

        # Clean text
        self.txt = re.sub(r'[^\x00-\x7f]', r' ', self.txt)  # remove non-ascii, like ï»
        self.txt = re.sub(r"\d+", "", self.txt)  # remove digits and currencies
        self.txt = re.sub(r"\_", "", self.txt)  # remove underscore that are used for French

        self.remove_book_details()  # Remove information added by publisher

        # Remove Latin Numerals
        # https://stackoverflow.com/questions/68048675/remove-roman-numbers
        pattern = r"\b(?=[MDCLXVIΙ])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})([IΙ]X|[IΙ]V|V?[IΙ]{0,3})\b\.?"
        self.txt = re.sub(pattern, '&', self.txt)

        # Normalize text ---------------
        self.txt = re.sub(r"[^a-zA-Z0-9]", " ", self.txt.lower())

        # Tokenize text ---------------
        words = self.txt.split()

        # Remove stop words ---------------
        self.word_token = [w for w in words if w not in stopwords]  # tokens, all words in document

    def calculate_rrt(self):
        """Calculate type / token ratio"""
        word_type = Counter(self.word_token).keys()  # types, unique words in document
        ttr = round(len(word_type) / len(self.word_token), 2)  # type / token ratio

        title = self.filename.split('-')[1]
        year = self.filename.split('-')[2][:-4]

        text_to_write = f'{AUTHOR};{title};{year};{len(word_type)};{len(self.word_token)};{ttr}'

        file_exists = exists(FILETOSAVE)

        if not file_exists:
            # File does not exist
            with open(FILETOSAVE, 'a') as f:
                print('author;title;year;type;token;ttr', file=f)  # Print header
                print(text_to_write, file=f)  # Print data

        else:
            # Print data to file
            with open(FILETOSAVE, 'a') as f:
                print(text_to_write, file=f)


class Graphs:
    """This class is responsible for plotting type-token ration (TTR)."""

    def __init__(self):
        self.filename = FILETOSAVE
        self.df = None
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.filename, sep=';')

    def ttr_scatter_plot(self):
        fig = (px.scatter(self.df, x="year", y="ttr"
                          , color="token", color_continuous_scale='brbg'
                          , template=PLOT_TEMPLATE
                          , labels={
                                    "year": "",
                                    "ttr": "Type Token Ratio",
                                    "token" : "N of tokens"
                                    }
                          ,title="Vocabulary of detective stories by A. Christie"
                          )
               .update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')
               )

        fig.update_traces(marker=dict(size=14
                                      ,line=dict(width=1,color='WhiteSmoke')
                                      )
                          )
        fig.write_image(file='./fig/ttr_scatter_plot.png', format='png')
