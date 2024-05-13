from multiprocessing import Pool
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
nederlands = CompleteNER(train_ned, val_ned, test_ned, language="ned")

from itertools import product
import numpy as np
import pandas as pd

# Define your feature configurations
feature_configs = {
			'CAPITALIZATION': [True, False],
			'HAS_UPPER': [True, False],
			'HAS_NUM': [True, False],
			'PUNCTUATION': [True, False],
			'SUF': [True, False],
			'WORD': [True, False],
			'LEN': [True, False],
			'NEXT': [True, False],
            '2PREV': [True, False],
            '2NEXT': [True, False],
			'POS': [True, False],
			'LEMMA': [True, False],
			'CITY': [True, False],
			'COMPANY': [True, False],
			'CELEBRITY': [True, False],
			'RESEARCH_ORGANIZATION': [True, False],
			'NAME': [True, False],
			'SURNAME': [True, False],
			'PREV': [True, False],
			'NEXT': [True, False],
			'NUMBER': [True, False],
			'GENDER': [True, False],
			'PERSON': [True, False],
			'PRONTYPE': [True, False],
			'DEP':[True, False],
			'HEAD_DISTANCE': [True, False],
			'HEAD': [True, False],
		}


# Define function to evaluate a feature configuration
def evaluate_feature_config(feature_config):
	# Train model with given feature configuration
	model_name_esp = "_".join([f"{key}_{str(value)}_esp" for key, value in feature_config.items()])
	spanish.train(file=f"spanish.mdl", feature_opt=feature_config)

	# Evaluate performance using validation data
	precision_esp, recall_esp, f1_esp, err_esp, default_acc_esp, matrix_esp = spanish.validation()

	model_name_ned = "_".join([f"{key}_{str(value)}_ned" for key, value in feature_config.items()])
	nederlands.train(file=f"nederlands.mdl", feature_opt=feature_config)

	# Evaluate performance using validation data
	precision_ned, recall_ned, f1_ned, err_ned, default_acc_ned, matrix_ned = nederlands.validation()

	# Return F1 score as the metric to optimize
	return (precision_esp, recall_esp, f1_esp, err_esp, default_acc_esp, matrix_esp, model_name_esp),(precision_ned, recall_ned, f1_ned, err_ned, default_acc_ned, matrix_ned, model_name_ned)
    

from multiprocessing import Pool


def generate_feature_configs(feature_groups):
    feature_configs = []
    for config in product(*[[True, False] for _ in range(len(feature_groups))]):
        feature_config = {}
        for i, group in enumerate(feature_groups):
            for feature in group:
                feature_config[feature] = config[i]
        feature_configs.append(feature_config)
    return feature_configs

feature_groups = [
        ['CAPITALIZATION', 'HAS_UPPER', 'HAS_NUM', 'PUNCTUATION','SUF', 'PRE', 'WORD', 'LEN'],
        ['PREV', 'NEXT', '2PREV', '2NEXT', 'POS', 'LEMMA'],
        ['CITY', 'COMPANY', 'CELEBRITY', 'RESEARCH_ORGANIZATION', 'NAME', 'SURNAME'],
        ['NUMBER','GENDER', 'PERSON', 'PRONTYPE', ],
        ['DEP', 'HEAD', 'HEAD_DISTANCE']
    ]

all_feature_configs = generate_feature_configs(feature_groups)

def evaluate_feature_config_parallel(feature_config):
    esp_result, ned_result = evaluate_feature_config(feature_config)
    return esp_result, ned_result

# Define the number of processes to use
num_processes = 4  # You can adjust this based on your CPU cores

# Use Pool to parallelize feature configuration evaluation
with Pool(num_processes) as pool:
    results_parallel = pool.map(evaluate_feature_config_parallel, all_feature_configs)

# Unpack results
results_esp_parallel, results_ned_parallel = zip(*results_parallel)

# Convert results to DataFrames
results_df_esp_parallel = pd.DataFrame(results_esp_parallel, columns=["Precision", "Recall", "F1", "Error", "Default Accuracy", "Confusion Matrix", "Model Name"])
results_df_ned_parallel = pd.DataFrame(results_ned_parallel, columns=["Precision", "Recall", "F1", "Error", "Default Accuracy", "Confusion Matrix", "Model Name"])

# Sort and print best results for Spanish
results_df_esp_parallel_sorted = results_df_esp_parallel.sort_values(by="F1", ascending=False)
best_model_name_esp = results_df_esp_parallel_sorted.iloc[0]["Model Name"]
best_score_esp = results_df_esp_parallel_sorted.iloc[0]["F1"]
print("Best Feature Configuration for Spanish:", best_model_name_esp)
print("Best F1 Score for Spanish:", best_score_esp)

# Sort and print best results for Dutch
results_df_ned_parallel_sorted = results_df_ned_parallel.sort_values(by="F1", ascending=False)
best_model_name_ned = results_df_ned_parallel_sorted.iloc[0]["Model Name"]
best_score_ned = results_df_ned_parallel_sorted.iloc[0]["F1"]
print("Best Feature Configuration for Dutch:", best_model_name_ned)
print("Best F1 Score for Dutch:", best_score_ned)

# Save results to CSV files
results_df_esp_parallel.to_csv("grid_search_results_esp_parallel.csv", index=False)
results_df_ned_parallel.to_csv("grid_search_results_ned_parallel.csv", index=False)