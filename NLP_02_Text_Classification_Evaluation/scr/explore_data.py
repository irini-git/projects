import os

from bs4 import BeautifulSoup
import re
from datetime import datetime
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from collections import Counter
import altair as alt

FILEPATH = "./data/reuters21578/"
FILEPATHPICKLE = "./data/"
DATE_FORMAT = '%d-%b-%Y %H:%M:%S.%f'
FILENAMEPICKLE_RAW = "data_raw.pkl"
FILENAMEPICKLE_PREPROCESSED = "data_preprocessed.pkl"
FILESAVEBARCHART_TOPIC = "./fig/topics - barchart.png"
FILESAVEBARCHART_TOPIC_SPLIT = "./fig/topics - barchart - split.png"


class ClassificationDataset:
    """Reuters Data for classification"""

    def __init__(self):
        # self.get_current_directory()
        self.topics = ""
        self.topics_per_article = None
        self.df = self.load_data()
        # self.get_stats()
        # self.explore_data()

    def explore_data(self):
        """Explore data
        """
        # 1. Information about the dataset
        print(f'Columns : {self.df.columns}')
        print(f'\nShape : {self.df.shape[0]}')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(f'\nSample :\n{self.df.head(2)}')

        # 2. Information about topics
        #
        # 2a. Filter dataframe so it only contains non-empty topics
        df = self.df.query('topic.str.len() != 0')
        print(f'\nN of articles with non-empty topics: {df.shape[0]}.')

        # 2b. Visualize most popular topics (bar chart)
        # Flatter lists
        topics_list = df['topic'].tolist()

        def flatten_comprehension(matrix):
            return [item for row in matrix for item in row]

        topics = flatten_comprehension(topics_list)

        # Use Counter to find the most popular topics
        counts = Counter(topics)
        df_counts = pd.DataFrame.from_dict(counts, orient='index')
        df_counts = df_counts.rename(columns={'index': 'topic', 0: 'count'})

        # Sort dataframe
        df_counts = df_counts.sort_values(by=['count'], ascending=False).reset_index()

        # Plot bar chart for most popular topics
        self.plot_bar_chart_topics(df_counts, top_number=5)

        # Word count for the most popular topic
        columns = ['topic', 'text']
        # self.plot_wordcloud_per_topic(df[columns], df_counts, n_topics = 3)

        # 3. Topics with the smallest number of texts
        #
        df_counts_asc = df_counts.sort_values(by=['count'], ascending=True).reset_index()
        print(f'Example of less popular topics : {df_counts_asc.head(6)}')

    def plot_wordcloud_per_topic(self, df, df_counts, n_topics=3):
        """
        For a specific topic, concatenate all articles, and plot word count.
        :return: save a chart of word cloud
        """
        # For the selected number of top topics,
        # extract topic names.
        topics_in_scope = df_counts['index'].head(n_topics).to_list()

        print(f'Perform analysis for {n_topics} topics : {topics_in_scope}.')

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(f'\nSample :\n{df.head(1)}')

        for topic in topics_in_scope:
            # Create a subset of articles of that topic
            mask = df.apply(lambda row: topic in row['topic'], axis=1)
            df_articles_per_topic = df[mask]

            # Flatter lists
            articles_text = df['text'].tolist()

            def flatten_comprehension(matrix):
                return [item for row in matrix for item in row]

            text = flatten_comprehension(articles_text)
            text = " ".join(text)

            # Create and generate a word cloud image
            # We use load_data as the input since it requires text as a string, not lost
            wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)

            # Save the image in the img folder
            filename_ = f"./fig/topics - wordcloud - {topic}.png"
            wordcloud.to_file(filename_)

    def plot_bar_chart_topics(self, df, top_number=10):
        """
        Plot bar chart for frequency of topics of the articles
        df : database with topics and counts
        top_number : number of leading topics to visualize, default 10
        :return:
        """

        # 0. Prepare data : filter top N topics
        sorted_df = df.head(top_number)

        # 1. Plot most popular topics
        base = alt.Chart(sorted_df).mark_bar(color='firebrick').encode(
            x=alt.X('count:Q').title('number of articles'),
            y=alt.Y('index:N').title('').sort('-x'),
            text='count'  # add label
        )

        chart = base.mark_bar() + base.mark_text(align='right', dx=2)

        # Save the image in the img folder
        chart.save(FILESAVEBARCHART_TOPIC)

        # 2. So we have the view on most popular topics in total
        # Now what is the view on most popular topics per split (train, test)

        # List of topics in scope
        col_one_list = sorted_df['index'].tolist()

        # Extract a subset dataframe for topics mentioned
        columns = ['id', 'split', 'topic']
        df_topic_per_split = self.df[columns].copy()

        # Clean data
        # Column "split" has values "not-used", "test" and "train": remove "not-used"
        df_topic_per_split.query('split != "not-used"', inplace=True)

        # For all topics in scope, check if it is associated with the article,
        # Create a column for every topic with True when a topic mentioned, and False otherwise
        # Return a number not a boolean: 1*(boolean) maps a boolean to 1 (True) or 0 (False)
        for t in col_one_list:
            df_topic_per_split[t] = df_topic_per_split.apply(lambda row: 1 * (t in row.topic), axis=1)

        # Count the number of articles per topic for train / test split.
        df_grouped = df_topic_per_split.groupby(['split'], as_index=True)[col_one_list].sum()

        # Transpose dataframe and reset index
        df_grouped = df_grouped.T
        df_grouped.reset_index(inplace=True)

        # Rename column
        df_grouped.rename(columns={'index': 'topic'},
                          inplace=True)

        # Now we need to ensure the data has the correct format to plot a chart
        # For this, we first prepare test and train data separately and then union it.
        df_test = df_grouped[['topic', 'test']]
        df_test['split'] = 'test'
        df_test.rename(columns={'test': 'count'}, inplace=True)

        df_train = df_grouped[['topic', 'train']]
        df_train['split'] = 'train'
        df_train.rename(columns={'train': 'count'}, inplace=True)

        df_grouped = pd.concat([df_train, df_test], ignore_index=True)

        # Plot a stacked bar chart
        bars = alt.Chart(df_grouped).mark_bar().encode(
            x=alt.X('count:Q').title('number of articles'),
            y=alt.Y('topic:N').title('').sort('-x'),
            color='split'
        ).properties(
            width=800,
            height=300
        )

        text = alt.Chart(df_grouped).mark_text(dx=-10, dy=0, color='black').encode(
            x=alt.X('count:Q').title('number of articles'),
            y=alt.Y('topic:N').title('').sort('-x'),
            detail='split:N',
            text=alt.Text('count:Q', format=',.0f')
        )

        chart = bars + text

        # Save the image in the img folder
        chart.save(FILESAVEBARCHART_TOPIC_SPLIT)

    def load_data(self):
        """
        Load data from original sgm files, clean and save to pickle files (if the first run),
        or load data from pickle files (if other than first run).
        :return: dataframe with a subset of input data, applicable for classification
        """

        filename_pkl = FILEPATHPICKLE + FILENAMEPICKLE_RAW
        filename_pkl_preprocessed = FILEPATHPICKLE + FILENAMEPICKLE_PREPROCESSED

        # Check if a preprocessed file exists with cleaned data ready for classification
        if os.path.isfile(filename_pkl_preprocessed):
            return pd.read_pickle(filename_pkl_preprocessed)
        # Check if a pickle file exists with raw data
        elif os.path.isfile(filename_pkl):
            # Load from pickle
            self.df = pd.read_pickle(filename_pkl)
            # Preprocess text for classification
            self.preprocess_data()
            return self.df
        else:
            # load from original input files
            self.df = self.load_data_from_sgm()
            self.load_data()

    def preprocess_data(self):
        """
        Preprocess input data:
        - sort by id
        - clean text
        :return: Preprocessed data for classification
        """

        # Sort by id and reindex
        self.df.sort_values(by="id", ascending=True, inplace=True)
        self.df = self.df.reset_index(drop=True)

        # Extract a subset of data with topics
        # The classification task aims to assign a topic to the article
        df = self.df[self.df.topic.apply(len) > 0].copy()

        def clean_text(s):
            """
            Cleans text, applied to all articles in the dataframe
            :param s: article text
            :return: cleaned article text
            """
            # 1.
            # replace multiple new lines or multiple spaces with one space
            s = re.sub(r'\n+', ' ', s)

            # lower case
            s = s.lower()

            # remove punctuation
            s = re.sub("[^0-9A-Za-z]", " ", s)

            # remove multiple spaces
            s = re.sub(r'\s+', ' ', s)

            # 2. Remove stop words
            s = s.split()
            stopwords = nltk.corpus.stopwords.words('english')
            s = [word for word in s if not word in stopwords]

            # 3. Stemming
            stem = PorterStemmer()
            s = [stem.stem(token) for token in s]

            # 4. List to sting
            # As a result of previous data manipulations, article text is in the format of lists
            # Perform below transformation
            # as is: [word1, word2, word3]
            # to be: [word1 word2 word3]
            s = ' '.join(s)

            return s

        # clean text
        self.df['text'] = self.df['article'].apply(clean_text)

        # Save preprocessed data to a pickle file
        self.df.to_pickle(FILEPATHPICKLE + FILENAMEPICKLE_PREPROCESSED)

    def load_data_from_sgm(self):
        # iterate through all files in the directory

        df_final = pd.DataFrame(columns=['id', 'file', 'split', 'topic', 'article'])

        for filename in os.listdir(FILEPATH):

            # Load data from file
            if filename.endswith(".sgm"):

                with open(FILEPATH + filename, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f, 'html.parser')

                def string2date(date_str):
                    """
                    Convert date as string to datetime.
                    Current format: '5-MAR-1987 09:21:58.67' (string)
                    Converted format: standard datetime element
                    :return: date in datetime format
                    """

                    # Extract text from tag
                    date_str = date_str.getText()

                    # Remove leading space for dates with single number in date (ex. 1 Mar)
                    date_str = date_str.lstrip()

                    # Remove multiple spaces
                    # Some dates have extra space between year and hour
                    date_str = re.sub(r'\s\s+', ' ', date_str)

                    # Manual correction for typo in id 17192
                    if date_str == "31-MAR-1987 605:12:19.12":
                        date_str = "31-MAR-1987 05:12:19.12"

                    # Remove &#5;&#5;&#5;RM or some other characters in the end
                    # date_pattern = '27-MAR-1987 00:03:35.68'
                    # There might be one or more degits for seconds
                    data_pattern = "^[0-9]{1,2}-[A-Z]{3}-[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{1,2}"
                    date_str = re.findall(data_pattern, date_str)[0]

                    # Convert to datetime object
                    date_obj = datetime.strptime(date_str, DATE_FORMAT)

                    return date_obj

                def taglist2list(t):
                    # Transform list of tags to list of string
                    topics_with_tags = t.find_all('d')
                    topics_list = [tt.getText() for tt in topics_with_tags]
                    return topics_list

                def clean_title(t):
                    # Clean title: use low case, remove new line, remove characters in <>, replace multiple spaces

                    # lower text
                    t = t.getText().lower()

                    # remove new line
                    t = t.replace("\n", "")

                    # remove space in the beginning and in the end
                    t = t.strip('')

                    # remove < any character >
                    t = re.sub(r'<[\s\S]*>', '', t)

                    # remove multiple spaces
                    t = re.sub(r'\s\s+', ' ', t)

                    return t

                # Extract data as tags
                soup.find_all('reuters')
                topics = soup.find_all('topics')  # <d>My home address</d>
                dates = soup.find_all('date')
                places = soup.find_all('places')
                people = soup.find_all('people')
                orgs = soup.find_all('orgs')
                exchanges = soup.find_all('exchanges')
                companies = soup.find_all('companies')
                titles = soup.find_all('title')
                articles = soup.find_all('text')

                # Some information from original read.me
                # Training Set (13,625 docs): LEWISSPLIT="TRAIN";  TOPICS="YES" or "NO"
                # Test Set (6,188 docs):  LEWISSPLIT="TEST"; TOPICS="YES" or "NO"
                # Unused (1,765): LEWISSPLIT="NOT-USED" or TOPICS="BYPASS"
                a_tags_with_lewissplit = soup.find_all('reuters', attrs={'lewissplit': True})
                lewissplit = [tag['lewissplit'].lower() for tag in a_tags_with_lewissplit]

                # new id as int
                a_tags_with_newid = soup.find_all('reuters', attrs={'newid': True})
                newid = [int(tag['newid']) for tag in a_tags_with_newid]

                # Clean data
                dates = [string2date(d) for d in dates]
                places = [taglist2list(p) for p in places]
                topics = [taglist2list(t) for t in topics]
                people = [taglist2list(p) for p in people]
                orgs = [taglist2list(o) for o in orgs]
                exchanges = [taglist2list(e) for e in exchanges]
                companies = [taglist2list(c) for c in companies]
                titles = [clean_title(t) for t in titles]
                articles = [a.getText() for a in articles]

                # Optional: explore data:
                # Not all tags are mandatory, for example, title might be empty.
                # For this exercise, we only use mandatory tags that are present for all entries.
                # for l in [topics, articles, lewissplit]:
                #     print(f'Shape {len(l)} : {l[:2]}')

                # Create a dataframe using the columns that are mandatory
                df = pd.DataFrame(list(zip(newid, lewissplit, topics, articles)),
                                  columns=['id', 'split', 'topic', 'article'])

                # add column to id load file
                df.insert(1, 'file', filename[:-4])

                # Append to self dataframe and reset index
                df_final = pd.concat([df, df_final], ignore_index=True)

        # Save data to a pickle file
        df_final.to_pickle(FILEPATHPICKLE + FILENAMEPICKLE_RAW)

        return df_final

    def get_stats(self):
        """ Optional function to return stats about the data.
        """
        n_articles = 0  # initiate number of articles
        n_articles_with_topics = 0  # initiate number of articles with topics

        for filename in os.listdir(FILEPATH):

            def deduplicate_topics(topics_str):
                # remove duplicates
                result = list(set(topics_str.split(',')))

                # sort the list alphabetically
                result.sort()

                # transform a list to str
                result = ','.join(result)
                return result

            if filename.endswith(".sgm"):
                #Load data from sgm (Standard Generalized Markup) files using beautiful soup

                def clean_topics(topic_string):
                    """
                    Transform bs tag element to string and clean
                        replace middle separator </d><d> by comma
                        remove leader </d></topics>
                        remove closing <topics><d>
                    :param topic_string: tag with topics as Bs tag element
                    :return: cleaned topics as string
                    """
                    result = re.sub('</d><d>', ',', str(topic_string))
                    result = re.sub('<topics><d>', '', result)
                    result = re.sub('</d></topics>', '', result)

                    return result

                with open(FILEPATH + filename, 'r', encoding='utf-8', errors='ignore') as f:
                    # print(filename)
                    soup = BeautifulSoup(f, 'html.parser')

                    articles = soup.find_all('reuters')
                    topics = soup.find_all('topics')  # <td>My home address</td>

                    # [ variable for variable in sequence if(condition) ]
                    articles_with_topics = [t for t in topics if len(t.contents) != 0]

                    # apply function to all elements in the list
                    all_topics = list(map(clean_topics, articles_with_topics))

                    # Check if it is possible to have multiple topics per article
                    # [f(x) for x in sequence if condition]
                    if self.topics_per_article is None:
                        # Set variable to True if there is any occurence of multiple topics in article
                        self.topics_per_article = any([True for t in all_topics if "," in t])
                    topics_str = (','.join(all_topics))
                    self.topics += deduplicate_topics(topics_str)

                    n_articles += len(articles)
                    n_articles_with_topics += len(articles_with_topics)

        self.topics = deduplicate_topics(self.topics)

        # extract some elements to use as the example, result is a list of stringg
        topics_example = re.findall(",([a-z]*),", self.topics)[:3]
        # transform the result into string
        topics_example = ", ".join(topics_example)

        n_categories = len(self.topics.split(','))
        print(f'Total N of articles : {n_articles}')
        print(f'Total N of articles with topics : {n_articles_with_topics}')
        print(f'Totally, there are {n_categories} topics: ex. {topics_example}.')

        if self.topics_per_article:
            print("It is possible to have more than one topic per article.")
        else:
            print("It is NOT possible to have more than one topic per article.")

    @staticmethod
    def get_current_directory():
        cwd = os.getcwd()
        print(cwd)
