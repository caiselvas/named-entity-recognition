import spacy

# Cargar el modelo en español
nlp = spacy.load('es_core_news_sm')

# Procesar texto
texto = "Ellos saltan la cuerda."
doc = nlp(texto)

morph = tuple((token.text, token.morph) for token in doc)
dep = tuple((token.text, token.dep_) for token in doc)
head = tuple((token.text, token.head.text) for token in doc)
child = tuple((token.text, [child.text for child in token.children]) for token in doc)
indexs = tuple(token.i for token in doc)
distances = tuple(token.head.i - token.i for token in doc)

import re

def _get_company_indices(tokens, companies) -> list[tuple[str, tuple[int]]]:
	# Unimos los nombres de las compañías en una expresión regular
	# Escapamos los caracteres especiales en los nombres de las compañías
	pattern = '|'.join(re.escape(company) for company in companies)
	regex = re.compile(pattern)
	sentence = ' '.join(tokens)
	
	# Preparamos para recolectar los índices
	indices = []
	# Buscamos todas las coincidencias
	for match in regex.finditer(sentence):
		start_index, end_index = match.span()
		
		# Convert string indices to word indices
		start_word_index = len(re.findall(r'\S+', sentence[:start_index]))
		end_word_index = len(re.findall(r'\S+', sentence[:end_index])) - 1
		
		indices.append((match.group(), tuple(range(start_word_index, end_word_index + 1))))
	
	return indices
# Uso de la función
sentence = "Hola mi nombre es Carlos y trabajo en Google Maps desde hace 5 años."
companies = ["Google Maps"]
print(_get_company_indices(sentence.split(), companies))

	

