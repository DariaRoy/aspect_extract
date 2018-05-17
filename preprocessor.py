# -*- coding: utf-8 -*-

import pymorphy2
import re
import logging


def splitter(text):
    exp = re.compile(r',|\.|!|:|\?|-| |\(|\)|<|>|"|&|\n|\t', re.IGNORECASE)
    return [word for word in re.split(exp, text) if len(word) > 1]


def normalizer(morph, word):
    return morph.parse(word)[0].normal_form


def preprocessor(text_list):

    stop_words = set()
    new_text_list = []
    morph = pymorphy2.MorphAnalyzer()

    with open('stop_words.txt', 'r') as f:
        stop_words = set(f.read().decode('utf-8').split())

    iter_num = 0
    for text in text_list:
        text = text.lower()
        words = splitter(text)
        new_text = []

        for w in words:
            if w not in stop_words or not w.isnumeric():
                new_word = normalizer(morph, w)
            if new_word not in stop_words:
                if len(new_text) and new_text[-1] == u'не':
                    new_text[-1] += new_word
                else:
                    new_text.append(new_word)

        new_text_list.append(' '.join(new_text))
        if iter_num % 10 == 0:
            print '%s / %s' % (iter_num, len(text_list))
        iter_num += 1

    return new_text_list

if __name__ == '__main__':
    s = preprocessor(['тупо'.decode('utf-8')])
    for i in s:
        print i