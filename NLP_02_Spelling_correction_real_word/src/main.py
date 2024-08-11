from src.confusion_matrix import EditConfusionMatrix
from src.spelling_correction_STM_corpora import SpellingCorrectionCorpora
from spelling_correction_x_corpus import Tweets
import pandas as pd

# Initiate confusion matrix
confusionMatrix = EditConfusionMatrix()

# Load STM and twitter corpora
stm_corpus = SpellingCorrectionCorpora('S*')
twitter_corpus = Tweets()

# Channel model
confusionMatrix.calculate_channel_model(stm_corpus.words)

confusionMatrix.get_probability()

# Word model
confusionMatrix.calculate_word_model(stm_corpus.transform_data_for_lm(), 'STM')
confusionMatrix.calculate_word_model(twitter_corpus.transform_data_for_lm(), 'Twitter')

# Probability
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(confusionMatrix.df_subset)

# End -------------------------------

