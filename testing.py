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
print(child)
	

