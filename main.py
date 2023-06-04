import re
import nltk
import numpy as np


def preprocess(text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A).casefold()

    text = ' '.join([c for c in text.split(' ') if c not in nltk.corpus.stopwords.words('english')])

    return text


def sentence_embedding(model, sentence: str):
    try:
        words = sentence.split(' ')
        vectors = [model.get_word_vector(word) for word in words]
        return np.mean(vectors, axis=0)
    except:
        print(sentence)
        raise Exception
