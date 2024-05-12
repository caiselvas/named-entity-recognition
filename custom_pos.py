import spacy
class CustomPOSTagger:
    def __init__(self, language):
        self.nlp = spacy.load(language)

    def get_postag(self, tokens):
        doc = self.nlp(" ".join(tokens))
        return [(token.text, token.pos_) for token in doc]