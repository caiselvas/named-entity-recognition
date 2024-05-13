import spacy
class CustomPOSTagger:
    def __init__(self, language):
        self.nlp = spacy.load(language)

    def get_postag(self, tokens):
        doc = self.nlp(" ".join(tokens))
        return tuple((token.text, token.pos_) for token in doc)
    
    def get_morph(self, tokens):
        doc = self.nlp(" ".join(tokens))
        return tuple((token.text, token.morph) for token in doc)
    
    def get_dep(self, tokens):
        doc = self.nlp(" ".join(tokens))
        return tuple((token.text, token.dep_) for token in doc)
    
    def get_head(self, tokens):
        doc = self.nlp(" ".join(tokens))
        return tuple((token.text, token.head.text) for token in doc)
    
    def get_head_distance(self, tokens):
        doc = self.nlp(" ".join(tokens))
        return tuple(token.head.i - token.i for token in doc)
