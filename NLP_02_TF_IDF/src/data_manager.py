from nltk.corpus import shakespeare
import re
import pandas as pd
from functools import reduce
from collections import Counter
import numpy as np
import math
import altair as alt

# import nltk
# nltk.download('shakespeare')

# Constants
TERMS = ['Antony', 'Brutus', 'Caesar', 'Calpurnia', 'Cleopatra', 'mercy', 'worser']
text1 = 'Brutus'.lower()
text2 = 'Caesar'.lower()
text3 = 'Calpurnia'.lower()
COLLECTION_SIZE = 0 # collection size placeholder

# Ex. from lecture, from Hamlet
DOC_ONE = "I did enact Julius Caesar I was killed i' the Capitol; Brutus killed me."

# Ex. from lecture, from Julius Caesar
DOC_TWO = "So let it be with Caesar. The noble Brutus hath told you Caesar was ambitious"



class DataPlays:
    def __init__(self):
        self.find_brutus()

    def find_brutus(self):
        plays = shakespeare.fileids()
        print(f'Which plays of Shakespeare contain the words "Brutus" and "Caesar" but not "Calpunia"?')

        # Create a placeholder for Dataframe
        # Index : terms
        # Columns : plays
        # cells 0 or 1 for if a term is present in the play
        df = pd.DataFrame(index=TERMS)

        for p in plays:

            def list_to_string(list_):
                # Support function to preprocess list of strings
                # Combine into one string
                temp = ' '.join(list_)
                # Remove new line
                temp = re.sub(r'\n?', '', temp)
                # Lower
                temp = temp.lower()
                return temp

            # Load a play
            play = shakespeare.xml(f'{p}')

            # Title is the only element, use [0][0] to extract it
            title_ = [list(p.itertext()) for p in play if p.tag == 'TITLE'][0][0]
            personae_ = [list(p.itertext()) for p in play if p.tag == 'PERSONAE'][0]

            full_text = [list(p.itertext()) for p in play if p.tag == 'ACT']
            # Using reduce to flatten the list
            text_ = reduce(lambda x, y: x + y, full_text)
            # Apply custom function to clean
            txt = list_to_string(text_)

            # Using re
            # m = re.search(f'.*({text1})+.*({text2})+|.*({text2})+.*({text1})+.*', txt)

            # Create a column for if a term is present in play
            column_if_present = [1 if t.lower() in txt else 0 for t in TERMS]

            column_tf = [len(re.findall(t.lower(), txt)) if t.lower() in txt else 0 for t in TERMS]
            column_log_tf = [1 + math.log10(t) if t > 0 else 0 for t in column_tf]

            df[f'{title_}_boolean'] = column_if_present
            # df[f'{title_}_TF_'] = column_tf
            df[f'{title_}_TF'] = column_log_tf
            # df[f'{title_}_TF_IDF'] = column_tf_idf

            # Support code to write to filt for manual check
            # if title_ == 'The Tragedy of Antony and Cleopatra':
            #     with open("./data/a_c_text01.txt", "w") as text_file:
            #         print(f"{txt}", file=text_file)

            if (text1 in txt)*(text2 in txt)*(text3 not in txt):
                print(f'    Yes : {title_}.')
            # else:
                # print(f'    No : {title_}.')

            # Output
            # dataframe words as index and plays as columns

        # Calculate Inversed Document frequency / one value per collection
        # We have a different collection compared to example in lecture, so final numbers differ
        N = len(plays)
        boolean_columns = [c for c in df.columns if 'boolean' in c]
        df['df'] = df[boolean_columns].sum(axis=1, numeric_only=True)

        df['idf'] = np.log10(N/df['df']) # should have checked for zero

        TF_columns = [c for c in df.columns if '_TF' in c]

        for c in TF_columns:
            title = re.search(r'^(.*?)\_TF',c).group(1)
            df[f'{title}_TFIDF'] = df[c] * df['idf']

        print('Document matrix ---------- ')
        columns_name_tf_idf = [c for c in df.columns if '_TFIDF' in c]

        with pd.option_context('display.max_rows', None, 'display.max_columns',
                                None):  # more options can be specified also
             print(df[columns_name_tf_idf])



        # Incidence vectors, use booleans
        #  "Brutus" and "Caesar" but not "Calpurnia"?')
        print('-'*30)

        condition1 = (df.loc[f'{text1.title()}'] == 1).values
        condition2 = (df.loc[f'{text2.title()}'] == 1).values
        condition3 = (df.loc[f'{text3.title()}'] == 0).values
        conditions = condition1 * condition2 * condition3
        res = conditions * df.columns.values
        print(res)


    def construct_inverted_index(self):
        # preprocess - each of documents is a sequence of word token
        # - tokenization
        # - linguistic modules
        # - fed tokens into indexer

        # 1. Attribute words to doc id
        print('Construct inverted index', '-'*30)

        # Create a placeholder for a dataframe with word - doc id
        df_terms = pd.DataFrame(columns=['Term', 'docID'])

        def simple_preprocess(document, docID):
            # text preprocess
            # - remove punctuation (comma, period and semicolon)
            # - transform string to a list of strings
            document = document.lower()
            term_column = re.sub('[.,;]', '', document)
            term_column = term_column.split(' ')

            df_temp = pd.DataFrame(term_column, columns=['term'])
            df_temp['docID'] = docID
            return df_temp

        # Concat dfs
        df1 = simple_preprocess(DOC_ONE, 1)
        df2 = simple_preprocess(DOC_TWO, 2)

        df_terms = pd.concat([df1, df2])

        # Sort
        # - alphabetical order for terms
        # - secondary: numerical order for doc id

        df_terms.sort_values(by=['term', 'docID'],
                             ascending=[True, True],
                             inplace=True)
        # print(df_terms)

        # Create dictionaty
        # term : document frequency
        # print(df_terms['term'].value_counts())

        # Posting list
        # List of documents where it occurs

        posting_lists = {t:df_terms.query('term==@t')['docID'].unique() for t in df_terms['term']}
        print(posting_lists)

        def locate_brutus_and_caesar(brutus='brutus', caesar='caesar'):
            # Locate documents where str1 and str2 are present

            # Start from startng posting list.

            a = posting_lists[brutus]
            b = posting_lists[caesar]

            temp = list(set(a) & set(b))

            return [str(x) for x in temp]

        print(f'Locate documents where both "Brutus" and "Caesar" occur, docID: {locate_brutus_and_caesar()}.')


    def score_with_jaccard_coef(self):
        """
        Ex for scoring with the Jaccard Coefficient
        """
        # Quote of the days
        quote_of_the_day_1 = "Happiness is not a matter of intensity but of balance, order, rhythm and harmony."
        quote_of_the_day_2 = "The best preparation for tomorrow is doing your best today."
        quote_of_the_day_3 = "God gave you a gift of 86,400 seconds today. Have you used one to say 'thank you?'"

        query_ = "ides of march"
        document_1 = "caesar died in march"
        document_2 = "the long ides march"

        def jaccard_coef(str1,str2):

            a = str1.split()
            b = str2.split()

            intersection = len(list(set(a) & set(b)))
            union = len(list(set(a).union(b)))

            print('-' * 20)
            print(f"Set A: {str1}")
            print(f"Set B: {str2}")
            print(f'Jaccard coefficient: {round(intersection / union,2)}.')

            return intersection / union

        def score_example(query, doc1, doc2):
            jaccard_coef_1 = jaccard_coef(query, doc1)
            jaccard_coef_2 = jaccard_coef(query, doc2)

            print('-'*10)
            print(f'Query : {query}')
            print(f'Doc1 : {doc1}')
            print(f'Doc2 : {doc2}')

            if jaccard_coef_1 > jaccard_coef_2:
                best_match = doc1
            else:
                best_match = doc2

            print(f'Best match "{best_match}"')


        _ = jaccard_coef(quote_of_the_day_1, quote_of_the_day_1)
        _ = jaccard_coef(quote_of_the_day_1, quote_of_the_day_2)

        score_example(query_, document_1, document_2)

    def explore_bag_of_words(self):

        print('-'*30)
        print('Explore Bag of words')

        doc1 = 'John is quicker than Mary'
        doc2 = 'Mary is quicker than John'

        # Placeholder for dataframe
        df = pd.DataFrame()

        # Create a bag of words
        def create_df(doc_):
            word_counts = Counter(doc_.split())

            # Create df from counter, sort by index for comparison
            df_temp = pd.DataFrame.from_dict(word_counts,
                                                 orient='index').sort_index()

            return df_temp

        def compare_bag_of_words(doc1=doc1, doc2=doc2):
            df1 = create_df(doc1)
            df2 = create_df(doc2)

            print(f'Document 1 : {doc1}')
            print(f'Document 2 : {doc2}')
            print(f'Have same vectors : {df1.equals(df2)}')

        compare_bag_of_words()

        def plot_log_frequency_weighting():

            term_frequency = np.linspace(0., 1000., 1000).tolist()
            weighted_frequency = [1 + math.log10(tf) if tf > 0 else 0 for tf in term_frequency]
            df = pd.DataFrame(list(zip(term_frequency, weighted_frequency)),
                              columns =['Term frequency', 'Log-frequency weighting'])

            print('-'*30)
            print('Term frequency')
            print(df.head(3))

            chart = alt.Chart(df).mark_line().encode(
                x='Term frequency',
                y='Log-frequency weighting'
            )

            chart.save('./fig/weighted_frequency.png')

        plot_log_frequency_weighting()

        def inverted_document_frequency():
            # initialize data of lists.
            data = {'term': ['calpurnia', 'animal', 'sunday', 'fly', 'under', 'the'],
                    'df_t': [1, 100, 1000, 10000, 100000, 1000000]}

            # Create DataFrame
            df = pd.DataFrame(data)

            # collection
            N = 1000000

            # Calculate inverse document frequency
            df['idf_t'] = np.log10(N/df['df_t'])

            print('-' * 30)
            print('Inverse document frequency')
            print(df)

        inverted_document_frequency()

    def vector_space_model(self):
        """
        Formalize vector space proximity
        How similar are the novels
        - Sense and Sensibility, SaS
        - Pride and Prejudice, PaP
        - Wuthering Heights, WH
        """
        # File names
        DOC_SAS = './data/SaS.utf-8'
        DOC_PAP = './data/PaP.utf-8'
        DOC_WH = './data/WH.utf-8'

        # Load data from file
        import codecs
        def load_from_utf(document=DOC_SAS):
            f =  codecs.open(document, 'r', 'UTF-8')
            f_str =  [''.join(line) for line in f]
            return ''.join(f_str)

        SAS = load_from_utf(document=DOC_SAS)
        PAP = load_from_utf(document=DOC_PAP)
        WH = load_from_utf(document=DOC_WH)

        # terms to consider
        TERMS_VECTOR = ['affection', 'jealous', 'gossip', 'wuthering']

        # No text preprocessing

        # Term frequency
        def calculate_term_frequency(df, txt, title, terms = TERMS_VECTOR):
            column_tf = [len(re.findall(t.lower(), txt.lower())) if t.lower() in txt.lower() else 0 for t in terms]
            column_log_tf = [1 + math.log10(t) if t > 0 else 0 for t in column_tf]
            column_norm_log_tf = column_log_tf / (np.linalg.norm(column_log_tf) + 1e-16)

            df[f'{title}_TF'] = column_tf
            df[f'{title}_LOG_TF'] = column_log_tf
            df[f'{title}_NORM_LOG_TF'] = column_norm_log_tf

        # Placeholder for df
        df = pd.DataFrame(index=TERMS_VECTOR)

        calculate_term_frequency(df, SAS, 'SAS')
        calculate_term_frequency(df, PAP, 'PAP')
        calculate_term_frequency(df, WH, 'WH')

        columns_log_tf = [c for c in df.columns if '_LOG_TF' in c]

        print('-'*30)
        print('Vector Space Model')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df[columns_log_tf])
            print('')


        # What is similarity
        def find_similarity(doc1, doc2):

            def decrypt_title(t):
                if t=='PAP':
                    return 'Pride and Prejudice'
                elif t == 'SAS':
                    return 'Sense and Sensibility'
                elif t == 'WH':
                    return 'Wuthering Heights'

            a = df[f'{doc1}_NORM_LOG_TF']
            b = df[f'{doc2}_NORM_LOG_TF']
            res = np.dot(a, b)
            print(f'Similarity between "{decrypt_title(doc1)}" and "{decrypt_title(doc2)}" is {round(res,2)}.')

        # Find similarity between the documents
        find_similarity('SAS', 'PAP')
        find_similarity('SAS', 'WH')
        find_similarity('PAP', 'WH')


    def smart_notation(self):
        """
        TF IDF example
        """

        document1 = "car insurance auto insurance"
        query1 = "best car insurance"

        TERMS_EX = ['auto', 'best', 'car', 'insurance']

        # df placeholder
        df = pd.DataFrame(index=TERMS_EX)

        # lnc.ltc

        # Query : ltc ------------------
        # term frequency raw
        df['query_tf_raw'] = [1 if t.lower() in query1.lower() else 0 for t in TERMS_EX]
        # term frequency log weighted
        df['query_tf_wt'] = [1 + math.log10(t) if t > 0 else 0 for t in df['query_tf_raw']]
        # document frequency, manually to be aligned with the ex
        df['query_df'] = [5000, 50000, 10000, 1000]
        # inversed document frequency, idf, manually as we do not know N
        df['query_idf'] = [2.3, 1.3, 2.0, 3.0]

        # Calculate N and check
        N = 1000000
        df['query_idf_test'] = np.log10(N/df['query_df'])

        # Calculate N
        # log(N/df) = log(N) - log(df)
        # log(N) = log(N/df) + log(df)
        # N = 10 ^ (3 + log(1000))
        # N = 10 ^ (3 + 3)
        # N = 1000000

        # wt
        df['query_wt'] = df['query_tf_wt'] * df['query_idf']
        # unit vector using cosine normalization
        df['n_lize'] = df['query_wt'] / (np.linalg.norm(df['query_wt']) + 1e-16)

        # Document
        df['doc_tf_raw'] = [len(re.findall(t.lower(), document1.lower())) if t.lower() in document1.lower() else 0 for t in TERMS_EX]
        df['doc_tf_wt'] = [1 + math.log10(t) if t > 0 else 0 for t in df['doc_tf_raw']]

        # no idf component in the document, same weitghs
        df['doc_wt'] = df['doc_tf_wt']
        df['doc_n_lize'] = df['doc_wt'] / (np.linalg.norm(df['doc_wt']) + 1e-16)

        # product
        df['prod'] = np.multiply(df['n_lize'], df['doc_n_lize'])

        # Result on the screen
        print('-'*30)
        print('TF IDF example')

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(np.round(df, decimals=2))
            print(f'Document : {document1}')
            print(f'Query : {query1}')
            print(f'Similarity : {round(df["prod"].sum(),1)}')

    def evaluate_search_engine(self):
        """Ex to evaluate search engine"""

        # There is a collection of 10 relevant documents
        doc_rel = ['R', 'N', 'N', 'R', 'R', 'N', 'R', 'N', 'N', 'N']
        df = pd.DataFrame(doc_rel, columns=['Relevance'])

        # recall : tp / (tp + fn)
        # precision : fp / fp + tn

        recall = [len(df.loc[:i].query('Relevance=="R"')['Relevance'])/10 for i in list(range(0, 10))]
        precision = [len(df.loc[:i].query('Relevance=="R"')['Relevance'])/(i+1) for i in list(range(0, 10))]
        df['recall'] = recall
        df['precision'] = precision

        df = df.reset_index()

        #chart = alt.Chart(df.reset_index()).mark_line().encode(
        #     x='index',
        #     y='recall'
        # )

        COLORHEX_GREY = '#767676'
        COLORHEX_ASCENT = '#ff4d00'

        base = alt.Chart().mark_line().encode(x=alt.X("index").title(''))
        chart = alt.layer(*[base.encode(y=alt.Y(col).title(''), color=alt.value(color_))
                            for col, color_ in zip(['recall', 'precision'],
                                                   [COLORHEX_GREY, COLORHEX_ASCENT])],
                          data=df).properties(
                            width=800,
                            title = {
                                "text": ["Cumulative recall (grey) and precision (orange)"],
                            }
                        )

        chart.save('./fig/evaluate_search_engine.png')

        print('-' * 30)
        print('Evaluate Search Engine')

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df.index)
            print(round(df,1))

# ----------------