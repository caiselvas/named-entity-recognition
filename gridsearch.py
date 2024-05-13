import numpy as np
import spacy
import nltk
import svgling
import seaborn as sns
import matplotlib.pyplot as plt
from complete_class import CompleteNER

nltk.download('conll2002')
from nltk.corpus import conll2002

# Spanish
train_esp = conll2002.iob_sents('esp.train') # Train
val_esp = conll2002.iob_sents('esp.testa') # Val
test_esp = conll2002.iob_sents('esp.testb') # Test
# Dutch
train_ned = conll2002.iob_sents('ned.train') # Train
val_ned = conll2002.iob_sents('ned.testa') # Val
test_ned = conll2002.iob_sents('ned.testb') # Test


spanish = CompleteNER(train_esp, val_esp, test_esp, language="esp")

from itertools import product
import numpy as np
import pandas as pd

# Define your feature configurations
feature_configs = {
    "capitalization": [True, False],
    "has_upper": [True, False],
    "has_num": [True, False],
    "punctuation": [True, False],
    "suffix": [True, False],
    "word": [True, False],
    "length": [True, False],
    "prefix": [True, False],
    "prev_word": [True, False],
    "next_word": [True, False],
    "pos_tag": [True, False],
    "lemma": [True, False],
    "morph": [True, False],
    "title": [True, False],
    "names": [True, False],
    "surnames": [True, False],
    "cities": [True, False],
    "celebrities": [True, False],
    "companies": [True, False],
    "research": [True, False],
    "comilles": [True, False]
}

# Define combinations to avoid (you can adjust this based on your domain knowledge)
avoid_combinations = [
    {"capitalization": False, "has_upper": True},  # Capitalization usually implies having uppercase letters
    {"capitalization": False, "word": False},  # Capitalization is relevant only if words are included
    # Add more combinations to avoid if necessary
]

# Filter out combinations to avoid
valid_configs = []
for config_values in product(*feature_configs.values()):
    feature_config = dict(zip(feature_configs.keys(), config_values))
    if feature_config not in avoid_combinations:
        valid_configs.append(feature_config)

# Define function to evaluate a feature configuration
def evaluate_feature_config(feature_config):
    # Train model with given feature configuration
    model_name = "_".join([f"{key}_{str(value)}" for key, value in feature_config.items()])
    spanish.set_feature_config(feature_config)
    spanish.train(file=f"{model_name}.mdl")
    
    # Evaluate performance using validation data
    precision, recall, f1, err, default_acc, matrix = spanish.validation()
    
    # Return F1 score as the metric to optimize
    return precision, recall, f1, err, default_acc, matrix,  model_name
    
from multiprocessing import Pool

# Define function to evaluate a single feature configuration
def evaluate_single_feature_config(feature_config):
    precision, recall, f1, err, default_acc, matrix, model_name = evaluate_feature_config(feature_config)
    return [precision, recall, f1, err, default_acc, matrix, model_name]

# Define function to evaluate all feature configurations in parallel
def evaluate_all_feature_configs(feature_configs):
    with Pool() as pool:
        results = pool.map(evaluate_single_feature_config, feature_configs)
    return results

# Perform grid search in parallel
results = evaluate_all_feature_configs(valid_configs)

# Convert results to a DataFrame
results_df = pd.DataFrame(results, columns=["Precision", "Recall", "F1", "Error", "Default Accuracy", "Confusion Matrix", "Model Name"])

# Sort results by F1 score
results_df = results_df.sort_values(by="F1", ascending=False)

# Print best results
best_model_name = results_df.iloc[0]["Model Name"]
best_score = results_df.iloc[0]["F1"]
print("Best Feature Configuration:", best_model_name)
print("Best F1 Score:", best_score)

# Save results to a CSV file
results_df.to_csv("grid_search_results.csv", index=False)