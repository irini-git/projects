from confusion_matrix import EditConfusionMatrix
from spelling_correction_corpora import SpellingCorrectionCorpora
import pandas as pd

# Initiate confusion matrix
confusionMatrix = EditConfusionMatrix()

# Load STM corpora
stm_corpus = SpellingCorrectionCorpora('S*')
abdication_corpus = SpellingCorrectionCorpora('abdication*')

# Calculate probability
# channel model
confusionMatrix.calculate_channel_model(stm_corpus.words)

confusionMatrix.get_probability()

# word model
confusionMatrix.calculate_word_model(stm_corpus.transform_data_for_lm(), 'STM')

confusionMatrix.calculate_word_model(abdication_corpus.transform_data_for_lm(), 'Abdication')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(confusionMatrix.df_subset)

confusionMatrix.get_correct_spelling()

# End -------------------------------

