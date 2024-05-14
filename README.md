# Named Entity Recognition (NER) Project

This project implements a Named Entity Recognition (NER) system for identifying and classifying entities in text data. 
It includes custom classes and methods for data processing, model training, and evaluation.
The default corpus has PER, ORG, LOC, MISC and O tags, but it can be used with others also (like CADEC)

## Installation

To run the code, ensure you have Python installed, along with the necessary dependencies listed in `requirements.txt`.

```
pip install -r requirements.txt
```

## Usage

The usage for each of the classes is shown in the notebooks. There's a notebook for each task: just use the tagger,
compare tagset methods, base models, validation...

### Instantiate classes
pos_tagger = CustomPOSTagger()
crf_tagger = MyCRFTagger()
ner = CompleteNER(train_ned, val_ned, test_ned, language="ned")

### Train model
ner.train(verbose=True, file="./models/nederlands.mdl")

### Evaluate model
metrics = ner.validation(plot=True)
print(metrics)

## File Structure

The project directory is organized as follows:

- `data/`: Contains data files, including training, validation, and test datasets.
  - `/CADEC/`: Contains CADEC data
  - `/regex/`: Contains gazzeters to use with regex
  - `/results/`: Contains validation results
  - `/token/`: Contains gazzeters to use with token
  - Gazzeter files (names, celebrities...)
- `models/`: Stores trained models.
- `other/`: Additional files for testing or miscellaneous purposes.
- `base_models.ipynb`: Notebook for executing the basic NLTK model
- `just_tagger.ipynb`: Notebook for executing the tagger alone
- `main_NER.ipynb`: Main notebook to execute easily the NER model
- `tagset_search.ipynb`: Notebook to test different tagsets (BIO, IO, BIOW)
- `validation_gridsearch`: Notebook to perform gridsearch
- `CADEC.ipynb`: Notebook to do NER with CADEC tags
- `complete_class.py`: Complete class implementation
- `mycrftagger_class.py`: Tagger class implementation
- `custom_pos_class.py`: POS class implementation

## Features

The model incorporates various features for entity recognition, including default features, additional features, and gazetteers. Refer to the `MyCRFTagger` class for detailed feature descriptions. You can select the features you want using a dict like:

```
features = {
			'CAPITALIZATION': True,
			'HAS_UPPER': True,
			'HAS_NUM': True,
			'PUNCTUATION': True,
			'SUF': True,
            'PRE': True,
            '2NEXT': True,
            '2PREV': True,
			'WORD': True,}
```

## Training and Evaluation

1. Prepare training data using the `CompleteNER` class.
2. Train the model using the `train()` method.
3. Evaluate model performance using the `validation()` method and the `test()` method.

You can easily do this in the main_NER.ipynb. The results using test and all the feature functions are:

| Language | Precision | Recall | F1-score | Total errors | Accuracy |
|----------|-----------|--------|----------|--------------|----------|
| ESP (OUR)     | 0.796     | 0.785  | 0.791    | 1460         | 0.972    |
| ESP (DEFAULT)| 0.741     | 0.708  | 0.724    | 1935         | 0.962    |
| Dutch (OUR)   | 0.7888    | 0.7619 | 0.7751   | 1520         | 0.9779   |
| Dutch (DEFAULT)| 0.701    | 0.621  | 0.659    | 2343         | 0.966    |


## Notes

- Customizable features allow for experimentation and fine-tuning of the model. The gridsearch done is very small
- The project is primarily focused on Spanish and Dutch, but it may support other languages
- The complete model takes about 6-8 minutes to train. If you don't use the gazzetters, regex and the morph and dependency features, it takes a lot less (< 2 min)
- There's things that need to be improved, like the evaluation metrics, and the execution time
