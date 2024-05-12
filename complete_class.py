from mycrftagger_class import MyCRFTagger
from typing import List, Set, Any, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class CompleteNER():
    def __init__(self, train_data, validation_data, test_data, language, method = "bio", custom = False):
        assert method.lower() in ['bio', 'biow', 'io'], "Method not valid, options: 'bio', 'biow', 'io'"
        self.language = language
        self.method = method.lower()
        self.train_data = self.get_tuples(train_data) if not custom else train_data
        self.validation_data = self.get_tuples(validation_data) if not custom else validation_data
        self.test_data = self.get_tuples(test_data) if not custom else test_data
        pass

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
        for sentence in X:
            tuple_sentence = []
            for idx, word in enumerate(sentence):
                if self.method == 'bio':
                    tuple_sentence.append((word[0], word[2]))
                elif self.method == 'biow':
                    # When there is a B-TAG and the next word is an O-TAG, the B-TAG is changed to an W-TAG (length 1)
                    if word[2].startswith('B') and (idx+1) < len(sentence) and sentence[idx+1][2].startswith('O'):
                        tuple_sentence.append((word[0], f'W-{word[2][2:]}'))
                    elif word[2].startswith('B') and (idx+1) == len(sentence):
                        tuple_sentence.append((word[0], f'W-{word[2][2:]}'))
                    else:
                        tuple_sentence.append((word[0], word[2]))
                elif self.method == 'io':
                    if word[2].startswith('B'):
                        tuple_sentence.append((word[0], f'I-{word[2][2:]}'))
                    else:
                        tuple_sentence.append((word[0], word[2]))
            new_X.append(tuple_sentence)
        return new_X
    def load_from_file(self, file):
        self.tagger = MyCRFTagger(language=self.language)
        self.tagger.set_model_file(file)
    def train(self, verbose = False, training_opt = {}, file = "model.mdl"):
        self.tagger = MyCRFTagger(verbose=verbose, language=self.language, training_opt=training_opt)
        self.tagger.train(self.train_data, file)
    def validation(self):
        self.test(self.validation_data)
    def test(self, data = None, plot = False):
        data = self.test_data if data == None else data
        data_to_list =  [[token for token, _ in sentence] for sentence in data]
        tagged = []
        for sentence in data_to_list:
            tagged_sentence = self.tagger.tag(sentence)
            tagged.append(self.to_bio(tagged_sentence))
        tp, fn, fp, tot, err = self.evaluate(data, tagged)

        precision, recall, f1 = self.precision_recall_f1(tp, fn, fp)
        default_acc = self.default_accuracy(data)
        print(f"Language {self.language}")
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)
        print("Total errors:", err)
        print("Default accuracy:", default_acc)
        self.plot_confusion_matrix() if plot else None


    def evaluate(self, data, predicted):
        tp = 0
        fn = 0
        fp = 0
        tot = 0
        err = 0
        self.matrix = defaultdict(int)
        for idx, (gold_sentence, predicted_sentence) in enumerate(zip(data, predicted)):
            gold_entities = self.decode_entities(gold_sentence)
            predicted_entities = self.decode_entities(predicted_sentence)
            tp += len(gold_entities.itersection(predicted_entities))
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

    def default_accuracy(self, data):
        return self.tagger.accuracy(data)
    
    def precision_recall_f1(self, tp, fn, fp):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1_score
    
    def plot_confusion_matrix(self):
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
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
