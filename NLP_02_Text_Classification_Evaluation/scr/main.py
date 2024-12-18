from explore_data import ClassificationDataset
from classify_data import ClassificationModel
from explore_results import ClassificationResults

# Load, explore and preprocess data
# data_explore = ClassificationDataset()

# Create a multi-label model
# data_classify = ClassificationModel(data_explore.df)

# Review results
# data_results = ClassificationResults()
ClassificationResults().perform_micro_macro_averaging()