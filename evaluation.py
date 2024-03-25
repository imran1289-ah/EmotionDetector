from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd

# Dummy predictions and true labels for 3 models (Main, Variant 1, Variant 2)
np.random.seed(42)  # For reproducibility

# Assuming a 3-class classification problem (classes 0, 1, 2)
n_samples = 100  # Number of test samples
true_labels = np.random.randint(0, 3, size=n_samples)

# Dummy predictions by the models
predictions_main = np.random.randint(0, 3, size=n_samples)
predictions_variant1 = np.random.randint(0, 3, size=n_samples)
predictions_variant2 = np.random.randint(0, 3, size=n_samples)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    micro_precision, micro_recall, micro_fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, fscore, micro_precision, micro_recall, micro_fscore, accuracy

# Calculating metrics for each model
metrics_main = calculate_metrics(true_labels, predictions_main)
metrics_variant1 = calculate_metrics(true_labels, predictions_variant1)
metrics_variant2 = calculate_metrics(true_labels, predictions_variant2)

cm = confusion_matrix(true_labels, predictions_main)
cm_df = pd.DataFrame(cm, 
                     index=['True Class 0', 'True Class 1', 'True Class 2'],
                     columns=['Predicted Class 0', 'Predicted Class 1', 'Predicted Class 2'])



print("\nConfusion Matrix:")
print(cm_df)


# Define the metric names as multi-level columns
metrics_columns = pd.MultiIndex.from_tuples([
    ('Macro', 'P'),
    ('Macro', 'R'),
    ('Macro', 'F'),
    ('Micro', 'P'),
    ('Micro', 'R'),
    ('Micro', 'F'),
    ('Accuracy', '')  # Accuracy does not have a sub-level
], names=['', ''])

# Define the index with model names
model_index = ['Main Model', 'Variant 1', 'Variant 2']

# Create an empty DataFrame with the specified multi-level columns and model index
formatted_metrics_table = pd.DataFrame('-', index=model_index, columns=metrics_columns)

# Now populate the empty DataFrame with actual metrics
for model, metrics in zip(model_index, [metrics_main, metrics_variant1, metrics_variant2]):
    formatted_metrics_table.loc[model, ('Macro', 'P')] = metrics[0]
    formatted_metrics_table.loc[model, ('Macro', 'R')] = metrics[1]
    formatted_metrics_table.loc[model, ('Macro', 'F')] = metrics[2]
    formatted_metrics_table.loc[model, ('Micro', 'P')] = metrics[3]
    formatted_metrics_table.loc[model, ('Micro', 'R')] = metrics[4]
    formatted_metrics_table.loc[model, ('Micro', 'F')] = metrics[5]
    formatted_metrics_table.loc[model, ('Accuracy', '')] = metrics[6]

print("\nFormatted Metrics Summary:")
print(formatted_metrics_table)