
import xmltodict
import re
import os
import re
import pickle

import gensim
import numpy as np
import pymorphy2
from collections import Counter

morph = pymorphy2.MorphAnalyzer()


def splitter(text):
    exp = re.compile(r',|\.|!|:|\?|-| |\(|\)|<|>|"|&|\n|\t', re.IGNORECASE)
    return [word for word in re.split(exp, text) if len(word) > 1]


def normalizer(morph, word):
    return morph.parse(word)[0].normal_form


def load_text(path):
    texts = []
    aspects_list = []

    with open(path, 'r') as f:
        xml_repr = xmltodict.parse(f.read())
        tp = xml_repr['reviews']['review']
        for review in tp:
            texts.append(review['text'])
            aspects = []
            for aspect in review['aspects']['aspect']:
                vocab = {}
                if isinstance(aspect, str):
                    print()
                    print (aspect)
                    print(review['aspects'])
                    print(review['text'])

                vocab['category'] = aspect['@category']
                vocab['from'] = int(aspect['@from'])
                vocab['mark'] = aspect['@mark']
                vocab['sentiment'] = aspect['@sentiment']
                vocab['term'] = normalizer(morph, aspect['@term'].lower())
                vocab['to'] = int(aspect['@to'])
                vocab['type'] = aspect['@type']

                if vocab['type'] == 'explicit':
                    aspects.append(vocab)

            aspects_list.extend(aspects)

    print(aspects_list[0])


    X_p = []

    X_sentences = []
    X_pure = []
    id_text = 0
    for text in texts:
        x_pr = []
        sents = []
        print('%s/%s' % (id_text, len(texts)))
        text = text.lower()
        words = splitter(text)

        words_norm = [normalizer(morph, x) for x in words]
        X_sentences.append(words_norm)
        l = []
        i = 0
        for j in range(len(words)):
            vocab = {}
            k = text.find(words[j])
            vocab['word'] = normalizer(morph, words[j].lower())
            vocab['from'] = i + k
            vocab['to'] = i + k + len(words[j])
            i += k + len(words[j])
            text = text[k + len(words[j]):]
            X_pure.append(vocab)
            x_pr.append(vocab)
        X_p.append(x_pr)
        id_text += 1

    return X_pure, X_sentences, aspects_list, X_p


if __name__ == '__main__':
    pass