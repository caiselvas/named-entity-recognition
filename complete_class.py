from mycrftagger_class import MyCRFTagger
from typing import List, Set, Any, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class CompleteNER():
    def __init__(self, train_data, validation_data, test_data, language, method = "bio", custom = False, postag=True):
        assert method.lower() in ['bio', 'biow', 'io'], "Method not valid, options: 'bio', 'biow', 'io'"
        
        self.language = language
        self.method = method.lower()
        self.postag = postag
        self.train_data = self.get_tuples(train_data) if not custom else train_data
        self.validation_data = self.get_tuples(validation_data) if not custom else validation_data
        self.test_data = self.get_tuples(test_data) if not custom else test_data
        

    def get_tuples(self, X : list) -> list[tuple[str, str]]:
        """
        Get tuples from the dataset.

        Parameters
        ----------
        X : list
            Dataset.
        method : str
            Method to get tuples. Options: 'bio', 'biow', 'io'. Default: 'bio'.

        Returns
        -------
        list
            List of tuples.
        """
        new_X = []
        postags = {} if self.postag in [True, False] else self.postag
        for sentence in X:
            tuple_sentence = []
            postags[tuple(a[0] for a in sentence)] = []
            for idx, word in enumerate(sentence):
                if self.method == 'bio':
                    tuple_sentence.append((word[0], word[2]))
                    postags[tuple(a[0] for a in sentence)].append((word[0], word[1]))
                elif self.method == 'biow':
                    # When there is a B-TAG and the next word is an O-TAG, the B-TAG is changed to an W-TAG (length 1)
                    if word[2].startswith('B') and (idx+1) < len(sentence) and sentence[idx+1][2].startswith('O'):
                        tuple_sentence.append((word[0], f'W-{word[2][2:]}'))
                        postags[tuple(a[0] for a in sentence)].append((word[0], word[1]))
                    elif word[2].startswith('B') and (idx+1) == len(sentence):
                        tuple_sentence.append((word[0], f'W-{word[2][2:]}'))
                        postags[tuple(a[0] for a in sentence)].append((word[0], word[1]))
                    else:
                        tuple_sentence.append((word[0], word[2]))
                        postags[tuple(a[0] for a in sentence)].append((word[0], word[1]))
                elif self.method == 'io':
                    if word[2].startswith('B'):
                        tuple_sentence.append((word[0], f'I-{word[2][2:]}'))
                        postags[tuple(a[0] for a in sentence)].append((word[0], word[1]))
                    else:
                        tuple_sentence.append((word[0], word[2]))
                        postags[tuple(a[0] for a in sentence)].append((word[0], word[1]))
            new_X.append(tuple_sentence)
        self.postag = postags if self.postag else self.postag
        return new_X
    
    def load_from_file(self, file, feature_opt = {}, use_regex = False, custom_postag = False):
        """
        Load a model from a file and set it to the tagger.

        Parameters
        ----------
        file : str
            File path.

        feature_opt : dict
            Feature options.

        use_regex : bool
            Use regex for gazetteers.

        custom_postag : bool
            Use custom postags instead of automatically generating them with Spacy.
        """
        self.tagger = MyCRFTagger(language=self.language)
        self.tagger.set_model_file(file)
    
    def train(self, verbose = False, training_opt = {}, feature_opt = {}, use_regex = False, file = "model.mdl", custom_postag = False):
        """
        Train the CRF tagger.

        Parameters
        ----------
        verbose : bool
            Print information about the training process.

        training_opt : dict
            Training options.

        feature_opt : dict
            Feature options.

        use_regex : bool
            Use regex for gazetteers.

        file : str
            File path to save the model.
        """
        self.tagger = MyCRFTagger(verbose=verbose, language=self.language, training_opt=training_opt, feature_opt=feature_opt, use_regex=use_regex, custom_postag=self.postag if custom_postag else False)
        self.tagger.train(self.train_data, file)

    def validation(self):
        """
        Test the model with the validation data.

        Returns
        -------
        tuple
            Precision, recall, f1-score, total errors, default accuracy, confusion matrix.
        """
        precision, recall, f1, err, default_acc, matrix = self.test(self.validation_data)
        return precision, recall, f1, err, default_acc, matrix

    def test(self, data = None, plot = False, verb = False):
        """
        Test the model with the test data.

        Parameters
        ----------
        data : list
            Data to test.

        plot : bool
            Plot the confusion matrix.

        verb : bool
            Print information about the test process.

        Returns
        -------
        tuple
            Precision, recall, f1-score, total errors, default accuracy, confusion matrix.
        """
        data = self.test_data if data == None else data
        data_to_list =  [[token for token, _ in sentence] for sentence in data]
        tagged = []
        
        for sentence in data_to_list:
            tagged_sentence = self.tagger.tag(sentence)
            tagged.append(self.to_bio(tagged_sentence))
        tp, fn, fp, tot, err = self.evaluate([self.to_bio(d) for d in data], tagged)

        precision, recall, f1 = self.precision_recall_f1(tp, fn, fp)
        default_acc = self.default_accuracy(data)
        
        if verb:
            print(f"Language {self.language}")
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1-score:", f1)
            print("Total errors:", err)
            print("Default accuracy:", default_acc)
        
        matrix = self.plot_confusion_matrix(plot)
        
        return precision, recall, f1, err, default_acc, matrix

    def evaluate(self, data, predicted):
        """
        Evaluate the model based on the predicted data.

        Parameters
        ----------
        data : list
            Gold data.

        predicted : list
            Predicted data.

        Returns
        -------
        tuple
            True positives, false negatives, false positives, total errors, total tokens.
        """
        tp = 0
        fn = 0
        fp = 0
        tot = 0
        err = 0
        self.matrix = defaultdict(int)
        
        for idx, (gold_sentence, predicted_sentence) in enumerate(zip(data, predicted)):
            gold_entities = self.decode_entities(gold_sentence)
            predicted_entities = self.decode_entities(predicted_sentence)
            tp += len(gold_entities.intersection(predicted_entities))
            fn += len(gold_entities.difference(predicted_entities))
            fp += len(predicted_entities.difference(gold_entities))
            
            if gold_entities != predicted_entities:
                print("Sentence index:", idx)
                print("GOLD sentence: ", gold_sentence)
                print("PRED sentence: ", predicted_sentence)
                for i in range(len(gold_sentence)):
                    if gold_sentence[i][1] != predicted_sentence[i][1]:
                        print(f"ERROR {i} --- Gold: {gold_sentence[i]} Predicted: {predicted_sentence[i]}")
                        err += 1
                    tot += 1
                print()

            for gold_token, predicted_token in zip(gold_sentence, predicted_sentence):
                if gold_token[1] != 'O':
                    self.matrix[(gold_token[1][2:], predicted_token[1][2:])] += 1
        
        return tp, fn, fp, tot, err

    def decode_entities(self, phrase: List[Tuple[Any, str]]) -> Set[Tuple[int, int, str]]:
        """
        Decode entities from a phrase.

        Parameters
        ----------
        phrase : list
            Phrase.

        Returns
        -------
        set
            Set of entities.
        """
        first_idx = -1
        current_entity = None
        
        result = set()
        for i, (token, label) in enumerate(phrase):
            if label[0] == "B" or label == "O":
                if current_entity:
                    result.add((first_idx, i, current_entity))
                    current_entity = None
                if label[0] == "B":
                    first_idx = i
                    current_entity = label[2:]
        
        if current_entity:
            result.add((first_idx, len(phrase), current_entity))
    
        return result

    def to_bio(self, phrase: List[Tuple[Any, str]]) -> List[Tuple[Any, str]]:
        """
        Convert a phrase to BIO format.

        Parameters
        ----------
        phrase : list
            Phrase.

        Returns
        -------
        list
            Phrase in BIO format.
        """
        new_list = []
        if self.method == "biow":
            for i, (token, label) in enumerate(phrase):
                if label.startswith("W"):
                    new_list.append((token, "B-"+label[2:]))
                else:
                    new_list.append((token, label))
    
        elif self.method == "io":
            for i, (token, label) in enumerate(phrase):
                if i > 0 and label.startswith("I") and phrase[i-1][1] == "O":
                    new_list.append((token, "B-"+label[2:]))
                else:
                    new_list.append((token, label))
    
        else:
            new_list = phrase
    
        return new_list

    def default_accuracy(self, data) -> float:
        """
        Get the default accuracy of the data from the CRF tagger.

        Parameters
        ----------
        data : list
            Data.

        Returns
        -------
        float
            Default accuracy.
        """
        return self.tagger.accuracy(data)
    
    def precision_recall_f1(self, tp, fn, fp):
        """
        Calculate the precision, recall and f1-score.

        Parameters
        ----------
        tp : int
            True positives.

        fn : int
            False negatives.

        fp : int
            False positives.

        Returns
        -------
        tuple
            Precision, recall, f1-score.
        """
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if (tp+fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        return precision, recall, f1_score
    
    def plot_confusion_matrix(self, plot: bool):
        """
        Create a confusion matrix.

        Parameters
        ----------
        plot : bool
            Whether to plot the confusion matrix.

        Returns
        -------
        np.array
            Confusion matrix.
        """
        # Extract unique entity types
        matrix_dict = self.matrix
        unique_labels = sorted(set(label for pair in matrix_dict.keys() for label in pair))

        # Create an empty matrix
        matrix = np.zeros((len(unique_labels), len(unique_labels)))

        # Fill the matrix with counts
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                matrix[i, j] = matrix_dict.get((label1, label2), 0)

        # Create a heatmap
        if plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.show()
    
        return matrix
    def inference(self, phrase):
        phrase = phrase.split(" ")
        return self.tagger.tag(phrase)
