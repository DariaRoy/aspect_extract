from collections import Counter

import pymorphy2

morph = pymorphy2.MorphAnalyzer()


def pos(x):
    return morph.parse(x)[0].tag.POS


def freq(X_pure_train, X_pure_test):
    words = []
    for word in X_pure_train:
        words.append(word['word'])

    words = Counter(words)

    aspects = []
    for word in words:
        if words[word] >= 5 and pos(word) == 'NOUN':
            aspects.append(word)

    y_result = []

    for word in X_pure_test:
        if word['word'] in aspects:
            y_result.append(1)
        else:
            y_result.append(0)

    return y_result