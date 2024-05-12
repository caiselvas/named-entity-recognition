import spacy

# Cargar el modelo en español
nlp = spacy.load('es_core_news_sm')

# Procesar texto
texto = "Ellos saltan la cuerda."
doc = nlp(texto)

morph = tuple((token.text, token.morph) for token in doc)
print(morph)
    
feature_list = []

for idx, token in enumerate(doc):
	pass
	# Plural or singular
	

print(feature_list)