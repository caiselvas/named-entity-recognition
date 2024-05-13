# Natural Language Toolkit: Interface to the CRFSuite Tagger
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Long Duong <longdt219@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A module for POS tagging using CRFSuite
"""

import re
import unicodedata
from typing import Any

from nltk.tag.api import TaggerI
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from custom_pos import CustomPOSTagger

try:
	import pycrfsuite
except ImportError:
	pass
from functools import cache

class MyCRFTagger(TaggerI):
	"""
	A module for POS tagging using CRFSuite https://pypi.python.org/pypi/python-crfsuite

	>>> from nltk.tag import CRFTagger
	>>> ct = CRFTagger()  # doctest: +SKIP

	>>> train_data = [[('University','Noun'), ('is','Verb'), ('a','Det'), ('good','Adj'), ('place','Noun')],
	... [('dog','Noun'),('eat','Verb'),('meat','Noun')]]

	>>> ct.train(train_data,'model.crf.tagger')  # doctest: +SKIP
	>>> ct.tag_sents([['dog','is','good'], ['Cat','eat','meat']])  # doctest: +SKIP
	[[('dog', 'Noun'), ('is', 'Verb'), ('good', 'Adj')], [('Cat', 'Noun'), ('eat', 'Verb'), ('meat', 'Noun')]]

	>>> gold_sentences = [[('dog','Noun'),('is','Verb'),('good','Adj')] , [('Cat','Noun'),('eat','Verb'), ('meat','Noun')]]
	>>> ct.accuracy(gold_sentences)  # doctest: +SKIP
	1.0

	Setting learned model file
	>>> ct = CRFTagger()  # doctest: +SKIP
	>>> ct.set_model_file('model.crf.tagger')  # doctest: +SKIP
	>>> ct.accuracy(gold_sentences)  # doctest: +SKIP
	1.0
	"""

	def __init__(self, language: str, feature_func=None, verbose=False, training_opt={}, feature_opt = {}):
		"""
		Initialize the CRFSuite tagger

		:param feature_func: The function that extracts features for each token of a sentence. This function should take
			2 parameters: tokens and index which extract features at index position from tokens list. See the build in
			_get_features function for more detail.
		:param verbose: output the debugging messages during training.
		:type verbose: boolean
		:param training_opt: python-crfsuite training options
		:type training_opt: dictionary

		Set of possible training options (using LBFGS training algorithm).
			:'feature.minfreq': The minimum frequency of features.
			:'feature.possible_states': Force to generate possible state features.
			:'feature.possible_transitions': Force to generate possible transition features.
			:'c1': Coefficient for L1 regularization.
			:'c2': Coefficient for L2 regularization.
			:'max_iterations': The maximum number of iterations for L-BFGS optimization.
			:'num_memories': The number of limited memories for approximating the inverse hessian matrix.
			:'epsilon': Epsilon for testing the convergence of the objective.
			:'period': The duration of iterations to test the stopping criterion.
			:'delta': The threshold for the stopping criterion; an L-BFGS iteration stops when the
				improvement of the log likelihood over the last ${period} iterations is no greater than this threshold.
			:'linesearch': The line search algorithm used in L-BFGS updates:

				- 'MoreThuente': More and Thuente's method,
				- 'Backtracking': Backtracking method with regular Wolfe condition,
				- 'StrongBacktracking': Backtracking method with strong Wolfe condition
			:'max_linesearch':  The maximum number of trials for the line search algorithm.
		"""

		self._model_file = ""
		self._tagger = pycrfsuite.Tagger()

		if feature_func is None:
			self._feature_func = self._get_features
		else:
			self._feature_func = feature_func

		self._verbose = verbose
		self._training_options = training_opt
		self._pattern = re.compile(r"\d")
		

		self._lemmatizer = WordNetLemmatizer()

		assert language in ["ned", "esp"], "Language should be either 'ned' or 'esp'"
		if language == "ned":
			self._pos_tagger = CustomPOSTagger("nl_core_news_sm")
		else:
			self._pos_tagger = CustomPOSTagger("es_core_news_sm")
		self._language = language
		
		if self._language == "ned":
			self._names = set(open("data/names_ned.txt", encoding="utf-8").readlines())
			self._surnames = set(open("data/surnames_ned.txt", encoding="utf-8").readlines())
		elif self._language == "esp":
			self._names = set(open("data/names_esp.txt", encoding="utf-8").readlines())
			self._surnames = set(open("data/surnames_esp.txt", encoding="utf-8").readlines())
		self._cities = set(open("data/cities.txt", encoding="utf-8").readlines())
		self._companies = set(open("data/companies.txt", encoding="utf-8").readlines())
		self._celebrities = set(open("data/celebrities.txt", encoding="utf-8").readlines())
		self._research_organizations = set(open("data/research_organizations.txt", encoding="utf-8").readlines())
		self.feature_config = feature_opt

	def set_model_file(self, model_file):
		self._model_file = model_file
		self._tagger.open(self._model_file)


	def __call__(self, tokens, idx) -> Any:
		return self._get_features(tokens, idx)


	def update_feature_getter_params(self, params: list) -> None:
		self.__params = params


	def get_feature_getter_params(self) -> list:
		return self.__params
	
	@cache
	def get_postag(self, tokens) -> tuple:
		return self._pos_tagger.get_postag(tokens)
	
	@cache
	def get_morph(self, tokens) -> tuple:
		return self._pos_tagger.get_morph(tokens)
	
	@cache
	def get_dep(self, tokens) -> tuple:
		return self._pos_tagger.get_dep(tokens)
	
	@cache
	def _in_names(self, token) -> bool:
		return token in self._names
	
	@cache
	def _in_surnames(self, token) -> bool:
		return token in self._surnames
	
	@cache
	def _in_cities(self, token) -> bool:
		return token in self._cities
	
	@cache
	def _in_companies(self, token) -> bool:
		return token in self._companies
	
	@cache
	def _in_celebrities(self, token) -> bool:
		return token in self._celebrities
	
	@cache
	def _in_research_organizations(self, token) -> bool:
		return token in self._research_organizations
	
	def set_feature_config(self, feature_config):
		self._feature_config = feature_config

	def _get_features(self, tokens, idx, ):
		"""
		Extract basic features about this word including
			- Current word
			- is it capitalized?
			- Does it have punctuation?
			- Does it have a number?
			- Suffixes up to length 3

		Note that : we might include feature over previous word, next word etc.

		:param feature_config: Dictionary specifying which features to include
		:type feature_config: dict, optional
		:return: a list which contains the features
		:rtype: list(str)
		"""

		token = tokens[idx]

		feature_list = []

		if not token:
			return feature_list

		# Capitalization
		if self.feature_config.get("capitalization", True):
			if token[0].isupper():
				feature_list.append("CAPITALIZATION")

		if self.feature_config.get("has_upper", True):
			if any(map(str.isupper, token)):
				feature_list.append("HAS_UPPER")
		# Number
		if self.feature_config.get("has_num", True):
			if re.search(self._pattern, token) is not None:
				feature_list.append("HAS_NUM")

		# Punctuation
		if self.feature_config.get("punctuation", True):
			punc_cat = {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"}
			if all(unicodedata.category(x) in punc_cat for x in token):
				feature_list.append("PUNCTUATION")

		# Suffix up to length 3
		if self.feature_config.get("suffix", True):
			for i in range(1, min(len(token), 4)):
				feature_list.append("SUF_" + token[-i:])

		# Word
		if self.feature_config.get("word", True):
			feature_list.append("WORD_" + token)

		# Length of the word
		if self.feature_config.get("length", True):
			feature_list.append("LEN_" + str(len(token)))

		# Prefix up to length 3
		if self.feature_config.get("prefix", True):
			for i in range(1, min(len(token), 4)):
				feature_list.append("PRE_" + token[:i])

		# Previous word
		if self.feature_config.get("prev_word", True):
			if idx > 0:
				feature_list.append("PREV_" + tokens[idx - 1])

		# Next word
		if self.feature_config.get("next_word", True):
			if idx < len(tokens) - 1:
				feature_list.append("NEXT_" + tokens[idx + 1])

		# POS tag the sentence
		if self.feature_config.get("pos_tag", True):
			pos_tags = self.get_postag(tuple(tokens))
			feature_list.append("POS_" + pos_tags[idx][1])
			if idx > 0:
				feature_list.append("PREVPOS_" + pos_tags[idx - 1][1])
			if idx < len(tokens) - 1:
				feature_list.append("POSTPOS_" + pos_tags[idx + 1][1])

		# Lemma
		if self.feature_config.get("lemma", True):
			lemma = self._lemmatizer.lemmatize(token)
			feature_list.append("LEMMA_" + lemma)

		# Morphological features
		if self.feature_config.get("morph", True):
			morph = self.get_morph(tuple(tokens))			
			# Plural or singular
			if morph[idx][1].get("Number", None):
				feature_list.append("NUMBER_" + morph[idx][1].get("Number")[0])
			if idx > 0:
				if morph[idx-1][1].get("Number", None):
					feature_list.append("PREV_NUMBER_" + morph[idx-1][1].get("Number")[0])
			if idx < len(tokens) - 1:
				if morph[idx+1][1].get("Number", None):
					feature_list.append("NEXT_NUMBER_" + morph[idx+1][1].get("Number")[0])

			# Gender
			if morph[idx][1].get("Gender", None):
					feature_list.append("GENDER_" + morph[idx][1].get("Gender")[0])	
			if idx > 0:
				if morph[idx-1][1].get("Gender", None):
					feature_list.append("PREV_GENDER_" + morph[idx-1][1].get("Gender")[0])
			if idx < len(tokens) - 1:
				if morph[idx+1][1].get("Gender", None):
					feature_list.append("NEXT_GENDER_" + morph[idx+1][1].get("Gender")[0])

			# Person
			if morph[idx][1].get("Person", None):
				feature_list.append("PERSON_" + morph[idx][1].get("Person")[0])
			if idx > 0:
				if morph[idx-1][1].get("Person", None):
					feature_list.append("PREV_PERSON_" + morph[idx-1][1].get("Person")[0])
			if idx < len(tokens) - 1:
				if morph[idx+1][1].get("Person", None):
					feature_list.append("NEXT_PERSON_" + morph[idx+1][1].get("Person")[0])
			
			# PronType
			if morph[idx][1].get("PronType", None):
				feature_list.append("PRONTYPE_" + morph[idx][1].get("PronType")[0])
			if idx > 0:
				if morph[idx-1][1].get("PronType", None):
					feature_list.append("PREV_PRONTYPE_" + morph[idx-1][1].get("PronType")[0])
			if idx < len(tokens) - 1:
				if morph[idx+1][1].get("PronType", None):
					feature_list.append("NEXT_PRONTYPE_" + morph[idx+1][1].get("PronType")[0])

		# Dependencies
		if self.feature_config.get("dependencies", True):
			dep = self.get_dep(tuple(tokens))
			feature_list.append("DEP_" + dep[idx][1])
			if idx > 0:
				feature_list.append("PREV_DEP_" + dep[idx-1][1])
			if idx < len(tokens) - 1:
				feature_list.append("NEXT_DEP_" + dep[idx+1][1])

		# Title
		if self.feature_config.get("title", True):
			if token.istitle():
				feature_list.append("TITLE")
			if idx > 0:
				if tokens[idx - 1].istitle():
					feature_list.append("PREV_TITLE")
			if idx < len(tokens) - 1:
				if tokens[idx + 1].istitle():
					feature_list.append("NEXT_TITLE")
		
		# Gazetteer
		# Names
		if self.feature_config.get("names", True):
			if self._in_names(token):
				feature_list.append("NAME")

				# Previous and next name
				if idx > 0 and self._in_names(tokens[idx - 1]):
					feature_list.append("PREV_NAME")
				if idx < len(tokens) - 1 and self._in_names(tokens[idx + 1]):
					feature_list.append("NEXT_NAME")

		# Surnames
		if sef.feature_config.get("surnames", True):
			if self._in_surnames(token):
				feature_list.append("SURNAME")

				# Previous and next surname
				if idx > 0 and self._in_surnames(tokens[idx - 1]):
					feature_list.append("PREV_SURNAME")
				if idx < len(tokens) - 1 and self._in_surnames(tokens[idx + 1]):
					feature_list.append("NEXT_SURNAME")

		# Cities
		if self.feature_config.get("cities", True):
			if self._in_cities(token):
				feature_list.append("CITY")

				# Previous and next city
				if idx > 0 and self._in_cities(tokens[idx - 1]):
					feature_list.append("PREV_CITY")
				if idx < len(tokens) - 1 and self._in_cities(tokens[idx + 1]):
					feature_list.append("NEXT_CITY")

		# Celebrities
		if self.feature_config.get("celebrities", True):
			if self._in_celebrities(token):
				feature_list.append("CELEBRITY")

				# Previous and next celebrity
				if idx > 0 and self._in_celebrities(tokens[idx - 1]):
					feature_list.append("PREV_CELEBRITY")
				if idx < len(tokens) - 1 and self._in_celebrities(tokens[idx + 1]):
					feature_list.append("NEXT_CELEBRITY")

		# Companies
		if self.feature_config.get("companies", True):
			if self._in_companies(token):
				feature_list.append("COMPANY")

				# Previous and next company
				if idx > 0 and self._in_companies(tokens[idx - 1]):
					feature_list.append("PREV_COMPANY")
				if idx < len(tokens) - 1 and self._in_companies(tokens[idx + 1]):
					feature_list.append("NEXT_COMPANY")

		# Research organizations
		if self.feature_config.get("research", True):
			if self._in_research_organizations(token):
				feature_list.append("RESEARCH_ORGANIZATION")

				# Previous and next research organization
				if idx > 0 and self._in_research_organizations(tokens[idx - 1]):
					feature_list.append("PREV_RESEARCH_ORGANIZATION")
				if idx < len(tokens) - 1 and self._in_research_organizations(tokens[idx + 1]):
					feature_list.append("NEXT_RESEARCH_ORGANIZATION")

		if self.feature_config.get("comilles", True):
			if token == '"':
				feature_list.append("COMILLES")
		# # Previous tag prediction
		# if idx > 0:
		# 	feature_list.append("PREV_TAG_" + self._tagger.tag([self._get_features(tokens, idx - 1)])[0])

		return feature_list
	
	def tag_sents(self, sents):
		"""
		Tag a list of sentences. NB before using this function, user should specify the mode_file either by

		- Train a new model using ``train`` function
		- Use the pre-trained model which is set via ``set_model_file`` function

		:params sentences: list of sentences needed to tag.
		:type sentences: list(list(str))
		:return: list of tagged sentences.
		:rtype: list(list(tuple(str,str)))
		"""
		if self._model_file == "":
			raise Exception(
				" No model file is found !! Please use train or set_model_file function"
			)

		# We need the list of sentences instead of the list generator for matching the input and output
		result = []
		for tokens in sents:
			features = [self._feature_func(tokens, i) for i in range(len(tokens))]
			labels = self._tagger.tag(features)

			if len(labels) != len(tokens):
				raise Exception(" Predicted Length Not Matched, Expect Errors !")

			tagged_sent = list(zip(tokens, labels))
			result.append(tagged_sent)

		return result


	def train(self, train_data, model_file):
		"""
		Train the CRF tagger using CRFSuite
		:params train_data : is the list of annotated sentences.
		:type train_data : list (list(tuple(str,str)))
		:params model_file : the model will be saved to this file.

		"""
		trainer = pycrfsuite.Trainer(verbose=self._verbose)
		trainer.set_params(self._training_options)

		for sent in train_data:
			tokens, labels = zip(*sent)
			features = [self._feature_func(tokens, i) for i in range(len(tokens))]
			trainer.append(features, labels)

		# Now train the model, the output should be model_file
		trainer.train(model_file)
		# Save the model file
		self.set_model_file(model_file)


	def tag(self, tokens):
		"""
		Tag a sentence using Python CRFSuite Tagger. NB before using this function, user should specify the mode_file either by

		- Train a new model using ``train`` function
		- Use the pre-trained model which is set via ``set_model_file`` function

		:params tokens: list of tokens needed to tag.
		:type tokens: list(str)
		:return: list of tagged tokens.
		:rtype: list(tuple(str,str))
		"""

		return self.tag_sents([tokens])[0]