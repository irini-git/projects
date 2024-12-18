import pickle
from sklearn.metrics import accuracy_score
import itertools
from collections import Counter
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

FILENAMEPICKLE_Y_PRED_TEST = "./data/2000_y_pred_test.pkl"
FILENAMEPICKLE_Y_TEST = "./data/y_test.pkl"
FILENAMEPICKLE_TOPICS = "./data/mlb_topics.pkl"
TOP_TOPICS = 6 # Number of true topics in confusion table

class ClassificationResults:
    def  __init__(self):
        self.y_assigned, self.y_true = self.load_data()
        self.confusion_matrix = self.custom_confusion_matrix()
        self.calculate_metrics()

    def load_data(self):
        with open(FILENAMEPICKLE_Y_PRED_TEST, 'rb') as f:
            y_pred_test = pickle.load(f)

        with open(FILENAMEPICKLE_Y_TEST, 'rb') as f:
            y_test = pickle.load(f)

        with open(FILENAMEPICKLE_TOPICS, 'rb') as f:
             mlb = pickle.load(f)

        # Retrieve labels
        y_assigned = mlb.inverse_transform(y_pred_test)
        y_true = mlb.inverse_transform(y_test)

        # Print Accuracy scores
        print("Accuracy score:",round(accuracy_score(y_test, y_pred_test),3))

        return y_assigned, y_true

    def custom_confusion_matrix(self):

        # For the confusion matrix
        # - rows are true topics

        # Retrieve unique true topics
        unique_topics_true = list(set(itertools.chain(*self.y_true)))
        unique_topics_assigned = list(set(itertools.chain(*self.y_assigned)))

        # Topics never assigned, or topics never true
        print(f'N of true topics : {len(unique_topics_true)}.')
        print(f'N of assigned topics : {len(unique_topics_assigned)}.')
        print(f'True but never assigned : {list(set(unique_topics_true).difference(unique_topics_assigned))}')
        print(f'Assigned but not true : {list(set(unique_topics_assigned).difference(unique_topics_true))}')

        # Top popular true topics
        c_topics_true = Counter(self.y_true).most_common(TOP_TOPICS)
        topics_top = [elt[0][0] for elt_id, elt in enumerate(c_topics_true)]

        # Placeholder for dataframe
        df_confusion_matrix = pd.DataFrame(None, index=topics_top, columns=topics_top)

        # Support function to calculate entries
        def fill_matrix(topic_='earn'):
            """
            Custom function to fill confusion matrix for specific elements
            For the illustration purposes, earn is used as a part of variable names.
            :return: updates df
            """

             # Earn example
            idx_earn = [c_ind for c_ind, c in enumerate(self.y_true) if c==(f'{topic_}',)]
            assigned_earn = [c for c_ind, c in enumerate(self.y_assigned) if all([c_ind in idx_earn, len(c) == 1])]

            c_earn_assigned = Counter(assigned_earn)

            for k,v in c_earn_assigned.items():
                df_confusion_matrix.at[topic_, k[0]] = v

        # Apply custom function for top 6 topics
        for t in topics_top:
            fill_matrix(t)

        # Replace NaN with 0 and convert to int
        # Disabled Future warning
        df_confusion_matrix.fillna(0, inplace=True)
        df_confusion_matrix = df_confusion_matrix.astype('int')

        # Optional: It is possible to have multiple topics
        # Only keep the columns same as top list
        # df_confusion_matrix = df_confusion_matrix.drop(columns=[col for col in df_confusion_matrix if col not in topics_top])

        # Print output
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(df_confusion_matrix)

        return df_confusion_matrix

    def calculate_metrics(self):
        """
        Custom function to calculate recall, precision and accuracy.
        :return:
        """

        # Recall : fraction of docs in class i classified correctly
        # [row-wise or class-oriented] Ex. out of all documents classified as 'earn'
        # How many are actually about 'earn'.
        print(f'\nExamples for metrics {"-"*10}')
        print('\n1. Recall')
        total = self.confusion_matrix.loc['earn'].sum()
        n_assigned = self.confusion_matrix['earn']['earn']

        print(f'Total documents of class earn : {total}' )
        print(f'Assigned documents of class earn : {n_assigned}')
        print(f'Recall of class earn is {n_assigned}/{total} = {round(n_assigned/total,2)}')

        # Precision: fraction of focs assigned class i that are actually about class i
        # Column - wise
        # For the documents we said it was about 'earn',
        # How many times it was indeed about 'earn'
        print('\n2. Precision')
        # n_assigned = self.confusion_matrix['earn']['earn']
        total = self.confusion_matrix['earn'].sum()
        n_true = self.confusion_matrix['earn']['earn']
        print(f'Total documents of assigned to class earn : {total}' )
        print(f'N of documents assigned indeed about it {n_true}')
        print(f'Precision of class earn is {n_true}/{total} = {round(n_true/total,2)}')

        # Accuracy: fraction on documents classified correctly
        # Sum of diagonal entries divided by the sum of entries by the confusion matrix
        print(self.confusion_matrix)
        total = self.confusion_matrix.values.sum()
        sum_diag = np.diag(self.confusion_matrix.to_numpy()).sum()

        print(f'Sum of diagonal entries : {sum_diag}')
        print(f'Sum of entries by the confusion matrix : {total}')
        print(f'Accuracy : {sum_diag}/{total} = {round(sum_diag/total,2)}')

    def perform_micro_macro_averaging(self):
        """
        Support function for micro vs. macro averaging.
        :return:
        """

        # Initialize data to Dicts of series.
        class1 = {'Truth:yes': pd.Series([10, 10],
                              index=['classifier:yes', 'classifier:no']),
             'Truth:no': pd.Series([10, 970],
                              index=['classifier:yes', 'classifier:no'])
                }

        class2 = {'Truth:yes': pd.Series([90, 10],
                              index=['classifier:yes', 'classifier:no']),
             'Truth:no': pd.Series([10, 890],
                              index=['classifier:yes', 'classifier:no'])
                }

        # Creates Dataframe
        df_class1 = pd.DataFrame(class1)
        df_class2 = pd.DataFrame(class2)

        print('Example for micro and macro averaging ---------- \n')
        print(df_class1, '\n')
        print(df_class2, '\n')

        micro_average = {'Truth:yes': pd.Series([10, 10],
                              index=['classifier:yes', 'classifier:no']),
                        'Truth:no': pd.Series([10, 970],
                              index=['classifier:yes', 'classifier:no'])
                        }

        df_micro_average = pd.DataFrame(micro_average)
        print(df_micro_average)

        # Class precision
        def calculate_precision(df):
            """
            Support function for precision,
            df is dataframe with data
            :return: precision
            """
            total = df['Truth:yes'].sum()
            n_true = df['Truth:yes']['classifier:yes']

            return  round(n_true / total, 2)

        # Out of what was classified as Yes, what is actually Yes
        class1_precision = calculate_precision(df_class1)
        class2_precision = calculate_precision(df_class2)

        print(f'Macro-averaged precision : {class1_precision} + {class2_precision} / 2 = {(class1_precision + class2_precision)/2}')

        # Micro-averaged
        total = df_class1['Truth:yes'].sum() + df_class2['Truth:yes'].sum()
        n_true = df_class1['Truth:yes']['classifier:yes'] + df_class2['Truth:yes']['classifier:yes']
        print(f'Micro-averaged precision : {n_true} / {total} = {round(n_true/total,2)}')

        print(f'Micro-averaged precision : {round(calculate_precision(df_class2.add(df_class1)),2)}')
