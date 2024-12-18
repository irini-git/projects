from pydoc_data.topics import topics

import numpy as np
import pickle

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

FILENAMEPICKLE_PREPROCESSED = "./data/data_preprocessed.pkl"
FEATURES_TFIDF = 2000 # Increase to get higher accuracy

FILEPATHPICKLE = "./data/"
FILENAMEPICKLE_Y_PRED_TEST = f"{FEATURES_TFIDF}_y_pred_test.pkl"
FILENAMEPICKLE_Y_TEST = "y_test.pkl"
FILENAMEPICKLE_TOPICS_MLB = "mlb_topics.pkl"

# Set a random seed for reproducibility
RANDOM_SEED = 6

class ClassificationModel:
    """Reuters Data for classification"""
    def __init__(self, df):
        self.df = df
        self.main_pipe = self.create_model()

    def prepare_data(self):
        """
        1. Transform topics for classifier
        2. Extract features from the training data using TfidfVectorizer.
        :return: X, y
        """

        # 0. Drop articles without topic
        # To classify, need at least 2 classes.

        # Support functon to replace empty entries with nulls
        def replace_empty_entry(t):
            if not t:
                return np.nan
            else:
                return t

        self.df['topic_'] = self.df.apply(lambda row: replace_empty_entry(row['topic']), axis=1)
        # Drop entries with null vales
        df_cleaned = self.df.dropna(subset=['topic_'])

        # 1. Transform topics for the classifier, simplified example of the change
        # As is: [earn, acq]
        # To be: [0 0 0 1 0 1]
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(df_cleaned['topic_'])

        # import numpy as np
        # np.set_printoptions(threshold=np.inf)
        # print(y[:2])

        # Retrieve labels
        # topics = mlb.inverse_transform(y)
        # print(topics)

        # 2. Extract features from the training data using TfidfVectorizer
        docs = list(df_cleaned['text'])
        tfidf_vectorizer = TfidfVectorizer(
                                use_idf=True,
                                max_features = FEATURES_TFIDF,
                                max_df=0.95)
        tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(docs)
        X = tfidf_vectorizer_vectors.toarray()

        # View X
        # print(X[0])

        return X, y, mlb

    def create_model(self):
        """
        Create the preprocessing pipelines for numeric and categorical data
        :return: classifier
        """

        # Split data into train and test sets
        X, y, mlb = self.prepare_data()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                            shuffle=True,
                                            random_state=RANDOM_SEED
                                        )

        # Classifier
        estimators = MultiOutputClassifier(
            estimator=XGBClassifier(n_estimators=175, max_depth=20)
        )

        # np.set_printoptions(threshold=np.inf)
        # print(y_train[:3])

        # Parameters search ---------------
        # params = {
        #     'estimator__n_estimators': [i for i in range(50, 225, 25)],
        #     'estimator__max_depth': [10, 20, 30, 40, 50]
        # }
        # {'estimator__n_estimators': 175, 'estimator__max_depth': 20}
        # search = RandomizedSearchCV(estimators, params, cv=5, return_train_score=False)
        # Create a gridsearch, fit the best model
        # best_model = search.fit(X_train, y_train)
        # Print the best set of hyperparameters and the corresponding score
        # print(best_model.best_params_)
        # Create a gridsearch of the pipeline, the fit the best model
        # best_model = estimators.fit(X_train, y_train)
        # ---------------

        best_model = estimators.fit(X_train, y_train)

        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        print("\nTraining Accuracy score:", accuracy_score(y_train, y_pred_train))
        print("Testing Accuracy score:", accuracy_score(y_test, y_pred_test))

        # Save y data to a pickle file
        with open(FILEPATHPICKLE + FILENAMEPICKLE_Y_PRED_TEST, "wb") as f:
            pickle.dump(y_pred_test, f, protocol=pickle.HIGHEST_PROTOCOL)

        # We use random seed, only save y test and mlb once
        # with open(FILEPATHPICKLE + FILENAMEPICKLE_Y_TEST, "wb") as f:
        #     pickle.dump(y_test, f, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # with open(FILEPATHPICKLE + FILENAMEPICKLE_TOPICS_MLB, "wb") as f:
        #    pickle.dump(mlb, f, protocol=pickle.HIGHEST_PROTOCOL)

        return

