import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import logging

logging.basicConfig(level=logging.INFO, filename="log.log", filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
FILENAME = './data/SMSSpamCollection'


class SMS_Dataset:
    """This class is responsible for loading data from local file,
    for preprocessing data and for modeling.
    This class also contains a small example for Bag of Words."""

    def __init__(self):
        self.df = None

    def load_txt(self, ):
        # Load data from txt file
        self.df = pd.read_csv(FILENAME,
                              sep="	",
                              header=None,
                              names=['label', 'sms_message'])

        # Convert the values in the 'label' column to numerical values
        label_mapping = {'ham': 0, 'spam': 1}
        self.df['label'] = self.df['label'].map(label_mapping, na_action=None)
        # Convert the labels to numerical values

        logging.info(f"Data size : {self.df.shape}\n")
        logging.info(f'Value counts : \n{self.df.label.value_counts()}\n')
        logging.info(f'% of spam : {round(100 * self.df.label.sum() / self.df.shape[0])}\n')

    def example_bag_of_words(self):
        """Preprocess text using bag of words"""
        documents = ['Hello, how are you!',
                     'Win money, win from home.',
                     'Call me now.',
                     'Hello, Call hello you tomorrow?']

        vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w\\w+\\b',
                                     lowercase=True,
                                     stop_words='english')

        # Create a matrix with each row representing one of the 4 documents,
        # and each column representing a word (feature name).
        # Each value in the matrix will represent the frequency of the word in that column
        # occurring in the particular document in that row.
        doc_array = vectorizer.fit_transform(documents).toarray()

        # Convert the 'doc_array' we created into a dataframe,
        # with the column names as the words (feature names).Call the dataframe 'frequency_matrix'.
        frequency_matrix = pd.DataFrame(data=doc_array, columns=vectorizer.get_feature_names_out())

        #
        logging.info(f'Small example ')
        logging.info(f'documents : {documents}\n')
        logging.info(f'vectorizer : {vectorizer}\n')
        logging.info(f'features : {vectorizer.get_feature_names_out()}\n')
        logging.info(f'doc_array : {doc_array}\n')
        logging.info(f'frequency_matrix : {frequency_matrix}\n')
        logging.info(f'-'*30)

    def preprocess_model(self):
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(self.df['sms_message'],
                                                            self.df['label'],
                                                            test_size=0.3,
                                                            random_state=0)

        logging.info(f'Number of rows in the total set: {self.df.shape[0]}')
        logging.info(f'Number of rows in the training set: {X_train.shape[0]}')
        logging.info(f'Number of rows in the test set: {X_test.shape[0]}')

        # Vectorizer
        vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w\\w+\\b',
                                     lowercase=True,
                                     stop_words='english')

        # Vectorize dataset
        vectorized_data = vectorizer.fit_transform(X_train)

        # Choose model
        classification_model = input("Choose the model. Enter 'm' for MultinomialNB or c for ComplementNB: ").lower()
        if classification_model not in ['m','c']:
            classification_model = input("Only two options are available (m/c). "
                                         "Print 'm' for MultinomialNB or 'c' for ComplementNB: ").lower()
        # Define model
        if classification_model == 'm':
            # Naive Bayes classifier for multinomial models
            logging.info ('Naive Bayes classifier for multinomial models (MultinomialNB)')
            model = MultinomialNB(alpha=0.1)
        else:
            # Complement Naive Bayes classifier described in Rennie et al.(2003)
            logging.info ('Complement Naive Bayes classifier (ComplementNB)')
            model = ComplementNB(alpha=0.1)

        model.fit(vectorized_data, Y_train)

        # Evaluate model
        predictions = model.predict(vectorizer.transform(X_test))

        accuracy = accuracy_score(Y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(Y_test, predictions)
        precision = precision_score(Y_test, predictions)

        # When data sets are unbalanced, like this sample,
        # the accuracy score could be a misleading metric for evaluation.
        # Precision, on the other hand,
        # will help us minimize the number of false positives
        # (that is, the number of non-spam texts that end up in spam).
        logging.info(f'Accuracy: {round(100 * accuracy, 2)} %')
        logging.info(f'Balanced accuracy: {round(100*balanced_accuracy,2)} %')
        logging.info(f'Precision: {round(100*precision,2)} %')

        cm = confusion_matrix(Y_test, predictions, labels=model.classes_)
        logging.info(f'Confusion Matrix: {cm}')


