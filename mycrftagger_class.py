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

	def __init__(self, language: str, feature_func=None, verbose=False, training_opt={}):
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
			self._research_organizations = set(open("data/research_organizations_ned.txt", encoding="utf-8").readlines())
		elif self._language == "esp":
			self._names = set(open("data/names100_esp.txt", encoding="utf-8").readlines())
			self._surnames = set(open("data/surnames_esp.txt", encoding="utf-8").readlines())
			self._research_organizations = set(open("data/research_organizations_esp.txt", encoding="utf-8").readlines())

		self._cities = set(open("data/cities15000.txt", encoding="utf-8").readlines())
		self._companies = set(open("data/companies.txt", encoding="utf-8").readlines())
		self._celebrities = set(open("data/celebrities.txt", encoding="utf-8").readlines())

		print('Readed all files')

		# Create regex pattern for names, surnames, cities, companies, celebrities, and research organizations
		companies_pattern = r'\b(?:' + '|'.join(re.escape(company) for company in self._companies) + r')\b'
		self._companies_regex = re.compile(companies_pattern, re.IGNORECASE)

		celebrities_pattern = r'\b(?:' + '|'.join(re.escape(celebrity) for celebrity in self._celebrities) + r')\b'
		self._celebrities_regex = re.compile(celebrities_pattern, re.IGNORECASE)

		research_organizations_pattern = r'\b(?:' + '|'.join(re.escape(research_organization) for research_organization in self._research_organizations) + r')\b'
		self._research_organizations_regex = re.compile(research_organizations_pattern, re.IGNORECASE)

		cities_pattern = r'\b(?:' + '|'.join(re.escape(city) for city in self._cities) + r')\b'
		self._cities_regex = re.compile(cities_pattern, re.IGNORECASE)

		names_pattern = r'\b(?:' + '|'.join(re.escape(name) for name in self._names) + r')\b'
		self._names_regex = re.compile(names_pattern, re.IGNORECASE)

		surnames_pattern = r'\b(?:' + '|'.join(re.escape(surname) for surname in self._surnames) + r')\b'
		self._surnames_regex = re.compile(surnames_pattern, re.IGNORECASE)

		print('Compiled all regexes')

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
	def get_head(self, tokens) -> tuple:
		return self._pos_tagger.get_head(tokens)
	
	@cache
	def get_head_distance(self, tokens) -> tuple:
		return self._pos_tagger.get_head_distance(tokens)
	
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
	
	@cache
	def _get_company_indices(self, tokens) -> list[tuple[str, tuple[int]]]:
		sentence = ' '.join(tokens)
		
		# Preparamos para recolectar los índices
		indices = []
		# Buscamos todas las coincidencias
		for match in self._companies_regex.finditer(sentence):
			start_index, end_index = match.span()
			
			# Convert string indices to word indices
			start_word_index = len(re.findall(r'\S+', sentence[:start_index]))
			end_word_index = len(re.findall(r'\S+', sentence[:end_index])) - 1
			
			indices.append((match.group(), tuple(range(start_word_index, end_word_index + 1))))
		
		return indices
	
	@cache
	def _get_celebrity_indices(self, tokens) -> list[tuple[str, tuple[int]]]:
		sentence = ' '.join(tokens)
		
		# Preparamos para recolectar los índices
		indices = []
		# Buscamos todas las coincidencias
		for match in self._celebrities_regex.finditer(sentence):
			start_index, end_index = match.span()
			
			# Convert string indices to word indices
			start_word_index = len(re.findall(r'\S+', sentence[:start_index]))
			end_word_index = len(re.findall(r'\S+', sentence[:end_index])) - 1
			
			indices.append((match.group(), tuple(range(start_word_index, end_word_index + 1))))
		
		return indices
	
	@cache
	def _get_research_organization_indices(self, tokens) -> list[tuple[str, tuple[int]]]:
		sentence = ' '.join(tokens)
		
		# Preparamos para recolectar los índices
		indices = []
		# Buscamos todas las coincidencias
		for match in self._research_organizations_regex.finditer(sentence):
			start_index, end_index = match.span()
			
			# Convert string indices to word indices
			start_word_index = len(re.findall(r'\S+', sentence[:start_index]))
			end_word_index = len(re.findall(r'\S+', sentence[:end_index])) - 1
			
			indices.append((match.group(), tuple(range(start_word_index, end_word_index + 1))))
		
		return indices
	
	@cache
	def _get_city_indices(self, tokens) -> list[tuple[str, tuple[int]]]:
		sentence = ' '.join(tokens)
		
		# Preparamos para recolectar los índices
		indices = []
		# Buscamos todas las coincidencias
		for match in self._cities_regex.finditer(sentence):
			start_index, end_index = match.span()
			
			# Convert string indices to word indices
			start_word_index = len(re.findall(r'\S+', sentence[:start_index]))
			end_word_index = len(re.findall(r'\S+', sentence[:end_index])) - 1
			
			indices.append((match.group(), tuple(range(start_word_index, end_word_index + 1))))
		
		return indices
	
	@cache
	def _get_name_indices(self, tokens) -> list[tuple[str, tuple[int]]]:
		sentence = ' '.join(tokens)
		
		# Preparamos para recolectar los índices
		indices = []
		# Buscamos todas las coincidencias
		for match in self._names_regex.finditer(sentence):
			start_index, end_index = match.span()
			
			# Convert string indices to word indices
			start_word_index = len(re.findall(r'\S+', sentence[:start_index]))
			end_word_index = len(re.findall(r'\S+', sentence[:end_index])) - 1
			
			indices.append((match.group(), tuple(range(start_word_index, end_word_index + 1))))
		
		return indices
	
	@cache
	def _get_surname_indices(self, tokens) -> list[tuple[str, tuple[int]]]:
		sentence = ' '.join(tokens)
		
		# Preparamos para recolectar los índices
		indices = []
		# Buscamos todas las coincidencias
		for match in self._surnames_regex.finditer(sentence):
			start_index, end_index = match.span()
			
			# Convert string indices to word indices
			start_word_index = len(re.findall(r'\S+', sentence[:start_index]))
			end_word_index = len(re.findall(r'\S+', sentence[:end_index])) - 1
			
			indices.append((match.group(), tuple(range(start_word_index, end_word_index + 1))))
		
		return indices

	def _get_features(self, tokens, idx):
		"""
		Extract basic features about this word including
			- Current word
			- is it capitalized?
			- Does it have punctuation?
			- Does it have a number?
			- Suffixes up to length 3

		Note that : we might include feature over previous word, next word etc.

		:return: a list which contains the features
		:rtype: list(str)
		"""
		self._iterations_count += 1
		print(f'Iteration {self._iterations_count}/{self._total_iterations}', end='\r')
		tokens = tuple(tokens)
		token = tokens[idx]

		feature_list = []

		if not token:
			return feature_list

		# Capitalization
		if token[0].isupper():
			feature_list.append("CAPITALIZATION")
		
		if any(map(str.isupper, token)):
			feature_list.append("HAS_UPPER")
		# Number
		if re.search(self._pattern, token) is not None:
			feature_list.append("HAS_NUM")

		# Punctuation
		punc_cat = {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"}
		if all(unicodedata.category(x) in punc_cat for x in token):
			feature_list.append("PUNCTUATION")

		# Suffix up to length 3
		if len(token) > 1:
			feature_list.append("SUF_" + token[-1:])
		if len(token) > 2:
			feature_list.append("SUF_" + token[-2:])
		if len(token) > 3:
			feature_list.append("SUF_" + token[-3:])

		# Word
		feature_list.append("WORD_" + token)

		# Length of the word
		feature_list.append("LEN_" + str(len(token)))

		# Prefix up to length 3
		if len(token) > 1:
			feature_list.append("PRE_" + token[:1])
		if len(token) > 2:
			feature_list.append("PRE_" + token[:2])
		if len(token) > 3:
			feature_list.append("PRE_" + token[:3])

		# Previous word
		if idx > 0:
			feature_list.append("PREV_" + tokens[idx - 1])

		# Next word
		if idx < len(tokens) - 1:
			feature_list.append("NEXT_" + tokens[idx + 1])

		# POS tag the sentence
		pos_tags = self.get_postag(tokens)
		feature_list.append("POS_" + pos_tags[idx][1])
		if idx > 0:
			feature_list.append("PREVPOS_" + pos_tags[idx-1][1])
		if idx < len(tokens) - 1:
			feature_list.append("POSTPOS_" + pos_tags[idx+1][1])
		
		# Lemma
		lemma = self._lemmatizer.lemmatize(token)
		feature_list.append("LEMMA_" + lemma)

		# Morphological features
		morph = self.get_morph(tokens)
		
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
		dep = self.get_dep(tokens)
		feature_list.append("DEP_" + dep[idx][1])
		if idx > 0:
			feature_list.append("PREV_DEP_" + dep[idx-1][1])
		if idx < len(tokens) - 1:
			feature_list.append("NEXT_DEP_" + dep[idx+1][1])

		# # Distance to head
		# distances = self.get_head_distance(tokens)
		# feature_list.append("DIST_" + str(distances[idx]))
		# if idx > 0:
		# 	feature_list.append("PREV_DIST_" + str(distances[idx-1]))
		# if idx < len(tokens) - 1:
		# 	feature_list.append("NEXT_DIST_" + str(distances[idx+1]))
		
		# Gazetteer
		
		# # Names
		# if self._in_names(token):
		# 	feature_list.append("NAME")

		# 	# Previous and next name
		# 	if idx > 0 and self._in_names(tokens[idx - 1]):
		# 		feature_list.append("PREV_NAME")
		# 	if idx < len(tokens) - 1 and self._in_names(tokens[idx + 1]):
		# 		feature_list.append("NEXT_NAME")

		# # Surnames
		# if self._in_surnames(token):
		# 	feature_list.append("SURNAME")

		# 	# Previous and next surname
		# 	if idx > 0 and self._in_surnames(tokens[idx - 1]):
		# 		feature_list.append("PREV_SURNAME")
		# 	if idx < len(tokens) - 1 and self._in_surnames(tokens[idx + 1]):
		# 		feature_list.append("NEXT_SURNAME")

		# # Cities
		# if self._in_cities(token):
		# 	feature_list.append("CITY")

		# 	# Previous and next city
		# 	if idx > 0 and self._in_cities(tokens[idx - 1]):
		# 		feature_list.append("PREV_CITY")
		# 	if idx < len(tokens) - 1 and self._in_cities(tokens[idx + 1]):
		# 		feature_list.append("NEXT_CITY")

		# # Celebrities
		# if self._in_celebrities(token):
		# 	feature_list.append("CELEBRITY")

		# 	# Previous and next celebrity
		# 	if idx > 0 and self._in_celebrities(tokens[idx - 1]):
		# 		feature_list.append("PREV_CELEBRITY")
		# 	if idx < len(tokens) - 1 and self._in_celebrities(tokens[idx + 1]):
		# 		feature_list.append("NEXT_CELEBRITY")

		# # Companies
		# if self._in_companies(token):
		# 	feature_list.append("COMPANY")

		# 	# Previous and next company
		# 	if idx > 0 and self._in_companies(tokens[idx - 1]):
		# 		feature_list.append("PREV_COMPANY")
		# 	if idx < len(tokens) - 1 and self._in_companies(tokens[idx + 1]):
		# 		feature_list.append("NEXT_COMPANY")

		# # Research organizations
		# if self._in_research_organizations(token):
		# 	feature_list.append("RESEARCH_ORGANIZATION")

		# 	# Previous and next research organization
		# 	if idx > 0 and self._in_research_organizations(tokens[idx - 1]):
		# 		feature_list.append("PREV_RESEARCH_ORGANIZATION")
		# 	if idx < len(tokens) - 1 and self._in_research_organizations(tokens[idx + 1]):
		# 		feature_list.append("NEXT_RESEARCH_ORGANIZATION")

		# # Previous tag prediction
		# if idx > 0:
		# 	feature_list.append("PREV_TAG_" + self._tagger.tag([self._get_features(tokens, idx - 1)])[0])
			
		# New gazetteers
		
		# Names
		name_indices = self._get_name_indices(tokens)
		for name, indices in name_indices:
			if idx in indices:
				feature_list.append("NAME")
				break
		if idx > 0:
			for name, indices in name_indices:
				if idx - 1 in indices:
					feature_list.append("PREV_NAME")
					break
		if idx < len(tokens) - 1:
			for name, indices in name_indices:
				if idx + 1 in indices:
					feature_list.append("NEXT_NAME")
					break

		# Surnames
		surname_indices = self._get_surname_indices(tokens)
		for surname, indices in surname_indices:
			if idx in indices:
				feature_list.append("SURNAME")
				break
		if idx > 0:
			for surname, indices in surname_indices:
				if idx - 1 in indices:
					feature_list.append("PREV_SURNAME")
					break
		if idx < len(tokens) - 1:
			for surname, indices in surname_indices:
				if idx + 1 in indices:
					feature_list.append("NEXT_SURNAME")
					break

		# Cities
		city_indices = self._get_city_indices(tokens)
		for city, indices in city_indices:
			if idx in indices:
				feature_list.append("CITY")
				break
		if idx > 0:
			for city, indices in city_indices:
				if idx - 1 in indices:
					feature_list.append("PREV_CITY")
					break
		if idx < len(tokens) - 1:
			for city, indices in city_indices:
				if idx + 1 in indices:
					feature_list.append("NEXT_CITY")
					break

		# Companies
		company_indices = self._get_company_indices(tokens)
		for company, indices in company_indices:
			if idx in indices:
				feature_list.append("COMPANY")
				break
		if idx > 0:
			for company, indices in company_indices:
				if idx - 1 in indices:
					feature_list.append("PREV_COMPANY")
					break
		if idx < len(tokens) - 1:
			for company, indices in company_indices:
				if idx + 1 in indices:
					feature_list.append("NEXT_COMPANY")
					break

		# Celebrities
		celebrity_indices = self._get_celebrity_indices(tokens)
		for celebrity, indices in celebrity_indices:
			if idx in indices:
				feature_list.append("CELEBRITY")
				break
		if idx > 0:
			for celebrity, indices in celebrity_indices:
				if idx - 1 in indices:
					feature_list.append("PREV_CELEBRITY")
					break
		if idx < len(tokens) - 1:
			for celebrity, indices in celebrity_indices:
				if idx + 1 in indices:
					feature_list.append("NEXT_CELEBRITY")
					break

		# Research organizations
		research_organization_indices = self._get_research_organization_indices(tokens)
		for research_organization, indices in research_organization_indices:
			if idx in indices:
				feature_list.append("RESEARCH_ORGANIZATION")
				break
		if idx > 0:
			for research_organization, indices in research_organization_indices:
				if idx - 1 in indices:
					feature_list.append("PREV_RESEARCH_ORGANIZATION")
					break
		if idx < len(tokens) - 1:
			for research_organization, indices in research_organization_indices:
				if idx + 1 in indices:
					feature_list.append("NEXT_RESEARCH_ORGANIZATION")
					break

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
		self._iterations_count = 0
		self._total_iterations = 0
		for sent in train_data:
			self._total_iterations += len(sent)
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