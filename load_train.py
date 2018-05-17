# coding: utf-8
__author__ = 'air'

import xmltodict
import re
import os
import re
import pickle

import gensim
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
#from rep.estimators import XGBoostClassifier

import pymorphy2
from preprocessor import normalizer

CATEGORIES = {'Whole': 0, 'Service': 1, 'Food': 2, 'Interior': 3, 'Price': 4}
CATEGORIES_SENTIMENT = {'positive': 0, 'negative': 1, 'neutral': 2, 'both': 3}

morph = pymorphy2.MorphAnalyzer()

def load_train_text():

    if os.path.exists('xtrain.bin') and os.path.exists('ytrain.bin'):
        with open('xtrain.bin', 'r') as f:
            X_train = pickle.load(f)
        with open('ytrain.bin', 'r') as f:
            y_train = pickle.load(f)
        with open('xpuretrain.bin') as f:
            X_pure = pickle.load(f)
        return X_pure, X_train, y_train


    texts = []
    mark = []

    with open('SentiRuEval_rest_markup_train.xml', 'r') as f:
        xml_repr = xmltodict.parse(f.read())
        tp = xml_repr['reviews']['review']
        for review in tp:
            texts.append(review['text'])

            aspects = []

            for aspect in review['aspects']['aspect']:

                vocab = {}
                vocab['category'] = aspect['@category']
                vocab['from'] = int(aspect['@from'])
                vocab['mark'] = aspect['@mark']
                vocab['sentiment'] = aspect['@sentiment']
                vocab['term'] = normalizer(morph, aspect['@term'].lower())
                vocab['to'] = int(aspect['@to'])
                vocab['type'] = aspect['@type']

                aspects.append(vocab)
            mark.append(aspects)

    print mark[0]

    w2v = load_vw()
    X_train = []
    y_train = []
    X_pure = []
    id_text = 0
    for text in texts:
        print '%s/%s' % (id_text, len(texts))
        text = text.lower()
        words = splitter(text)
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

            # distances = [0, 0, 0, 0, 0]
            distances = [-1, -1, -1, -1, -1]
            is_aspect = 0
            for aspects in mark:
                for aspect in aspects:
                    try:
                        # distances[CATEGORIES[aspect['category']]] += \
                        #     w2v.similarity(words[j].encode('utf-8'),
                        #                    aspect['term'].encode('utf-8'))
                        distances[CATEGORIES[aspect['category']]] = max(
                            distances[CATEGORIES[aspect['category']]],
                            w2v.similarity(vocab['word'].encode('utf-8'),
                                           aspect['term'].encode('utf-8')))
                    except:
                        pass
                    # if aspect['from'] == vocab['from'] and aspect['to'] == vocab['to']:
                    #     is_aspect = 1

            for aspect in mark[id_text]:
                if aspect['from'] == vocab['from'] and aspect['to'] == vocab['to'] and aspect['type'] == 'explicit':
                    is_aspect = 1

            X_train.append(distances)
            y_train.append(is_aspect)

        # break
        id_text += 1

    with open('xtrain.bin', 'w') as f:
        pickle.dump(X_train, f)
    with open('ytrain.bin', 'w') as f:
        pickle.dump(y_train, f)
    with open('xpuretrain.bin', 'w') as f:
        pickle.dump(X_pure, f)

    return X_pure, X_train, y_train


def load_train_2():

    if os.path.exists('xtrain2.bin') and os.path.exists('ytrain2.bin'):
        with open('xtrain2.bin', 'r') as f:
            X_train = pickle.load(f)
        with open('ytrain2.bin', 'r') as f:
            y_train = pickle.load(f)
        with open('xpuretrain2.bin') as f:
            X_pure = pickle.load(f)
        return X_pure, X_train, y_train


    texts = []
    mark = []

    with open('SentiRuEval_rest_markup_train.xml', 'r') as f:
        xml_repr = xmltodict.parse(f.read())
        tp = xml_repr['reviews']['review']
        for review in tp:
            texts.append(review['text'])

            aspects = []

            for aspect in review['aspects']['aspect']:

                vocab = {}
                vocab['category'] = aspect['@category']
                vocab['from'] = int(aspect['@from'])
                vocab['mark'] = aspect['@mark']
                vocab['sentiment'] = aspect['@sentiment']
                vocab['term'] = aspect['@term'].lower()
                vocab['to'] = int(aspect['@to'])
                vocab['type'] = aspect['@type']

                aspects.append(vocab)
            mark.append(aspects)

    print mark[0]

    w2v = load_vw()
    X_train = []
    y_train = []
    X_pure = []
    id_text = 0
    for text in texts:
        print '%s/%s' % (id_text, len(texts))
        text = text.lower()
        words = splitter(text)
        l = []
        i = 0
        for j in range(len(words)):
            vocab = {}
            k = text.find(words[j])
            vocab['word'] = words[j]
            vocab['from'] = i + k
            vocab['to'] = i + k + len(words[j])
            i += k + len(words[j])
            text = text[k + len(words[j]):]
            X_pure.append(vocab)

            # distances = [0, 0, 0, 0, 0]
            distances = [0, 0, 0, 0, 0]
            distances2 = [0, 0, 0, 0]
            is_aspect = 0
            for aspects in mark:
                for aspect in aspects:
                    try:
                        # distances[CATEGORIES[aspect['category']]] += \
                        #     w2v.similarity(words[j].encode('utf-8'),
                        #                    aspect['term'].encode('utf-8'))
                        distances[CATEGORIES[aspect['category']]] += \
                            w2v.similarity(words[j].encode('utf-8'),
                                           aspect['term'].encode('utf-8'))
                        distances2[CATEGORIES_SENTIMENT[aspect['sentiment']]] += \
                            w2v.similarity(words[j].encode('utf-8'),
                                           aspect['term'].encode('utf-8'))
                    except:
                        pass
                    # if aspect['from'] == vocab['from'] and aspect['to'] == vocab['to']:
                    #     is_aspect = 1

            # print distances2, vocab['word']
            l.append((distances, distances2, vocab))

        for j in range(len(words)):
            for aspect in mark[id_text]:
                if aspect['from'] == l[j][2]['from'] and aspect['to'] == l[j][2]['to'] and aspect['type'] == 'explicit':
                    feat = []
                    feat.extend(l[j][0])
                    feat.extend(l[j][1])
                    try:
                        feat.extend(l[j - 1][0])
                        feat.extend(l[j - 1][1])
                    except:
                        feat.extend(l[j][0])
                        feat.extend(l[j][1])

                    try:
                        feat.extend(l[j + 1][0])
                        feat.extend(l[j + 1][1])
                    except:
                        feat.extend(l[j][0])
                        feat.extend(l[j][1])
                    # print feat, l[j][2]['word']
                    if CATEGORIES_SENTIMENT[aspect['sentiment']] != 2:
                        X_train.append(feat)
                        y_train.append(CATEGORIES_SENTIMENT[aspect['sentiment']])

            # X_train.append(distances)
            # y_train.append(is_aspect)

        # break
        id_text += 1

    with open('xtrain2.bin', 'w') as f:
        pickle.dump(X_train, f)
    with open('ytrain2.bin', 'w') as f:
        pickle.dump(y_train, f)
    with open('xpuretrain2.bin', 'w') as f:
        pickle.dump(X_pure, f)

    return X_pure, X_train, y_train


def load_test_text():

    if os.path.exists('xtest.bin') and os.path.exists('ytest.bin') and os.path.exists('xpuretest.bin'):
        with open('xtest.bin', 'r') as f:
            X_train = pickle.load(f)
        with open('ytest.bin', 'r') as f:
            y_train = pickle.load(f)
        with open('xpuretest.bin') as f:
            X_pure = pickle.load(f)
        return X_pure, X_train, y_train


    texts = []
    mark = []

    with open('SentiRuEval_rest_markup_test.xml', 'r') as f:
        xml_repr = xmltodict.parse(f.read())
        tp = xml_repr['reviews']['review']
        for review in tp:
            texts.append(review['text'])

            aspects = []

            for aspect in review['aspects']['aspect']:

                vocab = {}
                vocab['category'] = aspect['@category']
                vocab['from'] = int(aspect['@from'])
                vocab['mark'] = aspect['@mark']
                vocab['sentiment'] = aspect['@sentiment']
                vocab['term'] = normalizer(morph, aspect['@term'].lower())
                vocab['to'] = int(aspect['@to'])
                vocab['type'] = aspect['@type']

                aspects.append(vocab)
            mark.append(aspects)

    print mark[0]

    w2v = load_vw()
    X_test = []
    y_test = []
    X_pure = []
    id_text = 0
    for text in texts:
        print '%s/%s' % (id_text, len(texts))
        text = text.lower()
        words = splitter(text)
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

            distances = [-1, -1, -1, -1, -1]
            # distances = [0, 0, 0, 0, 0]
            is_aspect = 0
            for aspects in mark:
                for aspect in aspects:
                    try:
                        # distances[CATEGORIES[aspect['category']]] += \
                        #     w2v.similarity(words[j].encode('utf-8'),
                        #                    aspect['term'].encode('utf-8'))
                        distances[CATEGORIES[aspect['category']]] = max(
                            distances[CATEGORIES[aspect['category']]],
                            w2v.similarity(vocab['word'].encode('utf-8'),
                                           aspect['term'].encode('utf-8')))
                    except:
                        pass
                    # if aspect['from'] == vocab['from'] and aspect['to'] == vocab['to']:
                    #     is_aspect = 1

            for aspect in mark[id_text]:
                if aspect['from'] == vocab['from'] and aspect['to'] == vocab['to'] and aspect['type'] == 'explicit':
                    is_aspect = 1

            X_test.append(distances)
            y_test.append(is_aspect)

        # break
        id_text += 1

    with open('xpuretest.bin', 'w') as f:
        pickle.dump(X_pure, f)
    with open('xtest.bin', 'w') as f:
        pickle.dump(X_test, f)
    with open('ytest.bin', 'w') as f:
        pickle.dump(y_test, f)

    return mark, X_test, y_test


def load_test_text_rules():

    texts = []
    mark = []

    with open('SentiRuEval_rest_markup_test.xml', 'r') as f:
        xml_repr = xmltodict.parse(f.read())
        tp = xml_repr['reviews']['review']
        for review in tp:
            texts.append(review['text'])

            aspects = []

            for aspect in review['aspects']['aspect']:

                vocab = {}
                vocab['category'] = aspect['@category']
                vocab['from'] = int(aspect['@from'])
                vocab['mark'] = aspect['@mark']
                vocab['sentiment'] = aspect['@sentiment']
                vocab['term'] = normalizer(morph, aspect['@term'].lower())
                vocab['to'] = int(aspect['@to'])
                vocab['type'] = aspect['@type']

                aspects.append(vocab)
            mark.append(aspects)

    print mark[0]

    mark_word = set()
    with open('mark_word.txt', 'r') as f:
        for i in f:
            word, val = i.split()
            mark_word.add(word.decode('utf-8').lower())

    # w2v = load_vw()
    X_test = []
    y_test = []
    X_pure = []
    id_text = 0
    for text in texts:
        print '%s/%s' % (id_text, len(texts))
        text = text.lower()
        words = splitter(text)
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

            l.append(vocab)
            is_aspect = 0



            # X_test.append(distances)
        for j in range(len(words)):
            try:
                # print l[j]['word']

                if l[j + 1]['word'] == u'ресторан':
                    y_test.append(1)
                    # print l[j + 1]['word']
                elif l[j + 1]['word'] in mark_word:
                    y_test.append(1)
                    # print l[j]['word']
                elif l[j - 1]['word'] in mark_word:
                    y_test.append(1)
                else:
                    y_test.append(0)
            except:
                y_test.append(0)

        # break
        id_text += 1

        # break

    # with open('xpuretest.bin', 'w') as f:
    #     pickle.dump(X_pure, f)
    # with open('xtest.bin', 'w') as f:
    #     pickle.dump(X_test, f)
    # with open('ytest.bin', 'w') as f:
    #     pickle.dump(y_test, f)

    return X_pure, X_test, y_test


def load_test_2(prev_mark):

    if os.path.exists('xtrain2.bin') and os.path.exists('ytest2.bin'):
        with open('xtest2.bin', 'r') as f:
            X_test = pickle.load(f)
        with open('ytest2.bin', 'r') as f:
            y_test = pickle.load(f)
        with open('xpuretest2.bin') as f:
            X_pure = pickle.load(f)
        return X_pure, X_test, y_test


    texts = []
    mark = []

    with open('SentiRuEval_rest_markup_test.xml', 'r') as f:
        xml_repr = xmltodict.parse(f.read())
        tp = xml_repr['reviews']['review']
        for review in tp:
            texts.append(review['text'])

            aspects = []

            for aspect in review['aspects']['aspect']:

                vocab = {}
                vocab['category'] = aspect['@category']
                vocab['from'] = int(aspect['@from'])
                vocab['mark'] = aspect['@mark']
                vocab['sentiment'] = aspect['@sentiment']
                vocab['term'] = aspect['@term'].lower()
                vocab['to'] = int(aspect['@to'])
                vocab['type'] = aspect['@type']

                aspects.append(vocab)
            mark.append(aspects)

    print mark[0]

    w2v = load_vw()
    X_test = []
    y_test = []
    X_pure = []
    id_text = 0
    for text in texts:
        print '%s/%s' % (id_text, len(texts))
        text = text.lower()
        words = splitter(text)
        l = []
        i = 0
        for j in range(len(words)):
            vocab = {}
            k = text.find(words[j])
            vocab['word'] = words[j]
            vocab['from'] = i + k
            vocab['to'] = i + k + len(words[j])
            i += k + len(words[j])
            text = text[k + len(words[j]):]
            X_pure.append(vocab)

            # distances = [0, 0, 0, 0, 0]
            distances = [0, 0, 0, 0, 0]
            distances2 = [0, 0, 0, 0]
            is_aspect = 0
            for aspects in prev_mark:
                for aspect in aspects:
                    try:
                        # distances[CATEGORIES[aspect['category']]] += \
                        #     w2v.similarity(words[j].encode('utf-8'),
                        #                    aspect['term'].encode('utf-8'))
                        distances[CATEGORIES[aspect['category']]] += \
                            w2v.similarity(words[j].encode('utf-8'),
                                           aspect['term'].encode('utf-8'))
                        distances2[CATEGORIES_SENTIMENT[aspect['sentiment']]] += \
                            w2v.similarity(words[j].encode('utf-8'),
                                           aspect['term'].encode('utf-8'))
                    except:
                        pass
                    # if aspect['from'] == vocab['from'] and aspect['to'] == vocab['to']:
                    #     is_aspect = 1

            # print distances2, vocab['word']
            l.append((distances, distances2, vocab))

        for j in range(len(words)):
            for aspect in mark[id_text]:
                if aspect['from'] == l[j][2]['from'] and aspect['to'] == l[j][2]['to'] and aspect['type'] == 'explicit':
                    feat = []
                    feat.extend(l[j][0])
                    feat.extend(l[j][1])
                    try:
                        feat.extend(l[j - 1][0])
                        feat.extend(l[j - 1][1])
                    except:
                        feat.extend(l[j][0])
                        feat.extend(l[j][1])

                    try:
                        feat.extend(l[j + 1][0])
                        feat.extend(l[j + 1][1])
                    except:
                        feat.extend(l[j][0])
                        feat.extend(l[j][1])
                    # print feat, l[j][2]['word']
                    if CATEGORIES_SENTIMENT[aspect['sentiment']] != 2:
                        X_test.append(feat)
                        y_test.append(CATEGORIES_SENTIMENT[aspect['sentiment']])

            # X_train.append(distances)
            # y_train.append(is_aspect)

        # break
        id_text += 1

    with open('xtest2.bin', 'w') as f:
        pickle.dump(X_test, f)
    with open('ytest2.bin', 'w') as f:
        pickle.dump(y_test, f)
    with open('xpuretest2.bin', 'w') as f:
        pickle.dump(X_pure, f)

    return X_pure, X_test, y_test


def save_result(X_pure, y_result):

    result_file = open('SentiRuEval_result_test.xml', 'w')
    id = 0
    prev = -1
    with open('SentiRuEval_rest_markup_test.xml', 'r') as f:
        for line in f:
            if '<aspects>' in line:
                print >>result_file, '\t\t<aspects1>'
                while True:
                    if len(X_pure) <= id or X_pure[id]['from'] < prev:
                        prev = -1
                        break

                    label = y_result[id]
                    # print id
                    if label:
                        X_pure[id]['word'] = X_pure[id]['word'].replace('"', '&quot;')
                        print >>result_file, ('\t\t\t<aspect category="Whole" from="%s" mark="Rel" sentiment="neutral" term="%s" to="%s" type="explicit"/>' % (X_pure[id]['from'], X_pure[id]['word'], X_pure[id]['to'])).encode('utf-8')
                    prev = X_pure[id]['to']
                    id += 1
            elif '</aspects>' in line:
                print >>result_file, '\t\t</aspects1>'
            elif 'aspect' in line:
                pass
            else:
                print >>result_file, line.rstrip()

    result_file.close()
    print 'Results have been writen at file SentiRuEval_result_test.xml'


def save_result2(y_result):

    # texts = []
    # mark = []

    result_file = open('SentiRuEval_result_test2.xml', 'w')
    id = 0
    prev = -1
    with open('SentiRuEval_rest_markup_test.xml', 'r') as f:
        for line in f:
            if '<aspects>' in line:
                print >>result_file, '\t\t<aspects>'
            elif '<aspect category' in line:
                #while True:
                    # if len(X_pure) <= id or X_pure[id]['from'] < prev:
                    #     prev = -1
                    #     break

                left_pos = line.find('term=')
                right_pos = line.find('to=')
                tmp_line = str(line)
                if ' ' in line[left_pos:right_pos - 1]:
                    if 'sentiment="neutral"' in tmp_line:
                        continue
                    elif 'sentiment="positive"' in tmp_line:
                        tmp_line = tmp_line.replace('sentiment="positive"', '$$$')
                    elif 'sentiment="negative"' in tmp_line:
                        tmp_line = tmp_line.replace('sentiment="negative"', '$$$')
                    elif 'sentiment="both"' in tmp_line:
                        tmp_line = tmp_line.replace('sentiment="both"', '$$$')
                    tmp_line = tmp_line.replace('$$$', 'sentiment="positive"')
                    print >>result_file, tmp_line
                    continue

                if 'sentiment="neutral"' in tmp_line:
                    continue
                elif 'sentiment="positive"' in tmp_line:
                    tmp_line = tmp_line.replace('sentiment="positive"', '$$$')
                elif 'sentiment="negative"' in tmp_line:
                    tmp_line = tmp_line.replace('sentiment="negative"', '$$$')
                elif 'sentiment="both"' in tmp_line:
                    tmp_line = tmp_line.replace('sentiment="both"', '$$$')

                try:
                    label = y_result[id]
                except:
                    label = 0
                # print tmp_line
                # print id
                # tmp_line = str(line)


                if label == 0:
                    # X_pure[id]['word'] = X_pure[id]['word'].replace('"', '&quot;')
                    tmp_line = tmp_line.replace('$$$', 'sentiment="positive"')
                    # print >>result_file, ('\t\t\t<aspect category="Whole" from="%s" mark="Rel" sentiment="%s" term="orara" to="%s" type="explicit"/>' % (X_pure[id]['from'], 'positive', X_pure[id]['to'])).encode('utf-8')
                if label == 1:
                    # X_pure[id]['word'] = X_pure[id]['word'].replace('"', '&quot;')
                    tmp_line = tmp_line.replace('$$$', 'sentiment="negative"')
                    # print >>result_file, ('\t\t\t<aspect category="Whole" from="%s" mark="Rel" sentiment="%s" term="orara" to="%s" type="explicit"/>' % (X_pure[id]['from'], 'negative', X_pure[id]['to'])).encode('utf-8')
                if label == 3:
                    # X_pure[id]['word'] = X_pure[id]['word'].replace('"', '&quot;')
                    tmp_line = tmp_line.replace('$$$', 'sentiment="both"')
                    # print >>result_file, ('\t\t\t<aspect category="Whole" from="%s" mark="Rel" sentiment="%s" term="orara" to="%s" type="explicit"/>' % (X_pure[id]['from'], 'both', X_pure[id]['to'])).encode('utf-8')
                print >>result_file, tmp_line
                # prev = X_pure[id]['to']
                id += 1
            elif '</aspects>' in line:
                print >>result_file, '\t\t</aspects>'
            else:
                print >>result_file, line.rstrip()

    result_file.close()
    print 'Results have been writen at file SentiRuEval_result_test2.xml'


def splitter(text):
    exp = re.compile(r',|\.|!|:|\?|-| |', re.IGNORECASE)
    return [word.lower() for word in re.split(exp, text) if len(word) > 2]


def splitter2(text):
    exp = re.compile(r',|\.|!|:|\?|-| |\(|\)|<|>|&|\n|\t', re.IGNORECASE)
    return [word for word in re.split(exp, text) if len(word) > 1]


def load_vw():
    if os.path.exists('w2v_model3'):
        w2v = gensim.models.Word2Vec.load('w2v_model3')
    else:
        with open('whole_w2v_train4') as f:
            sentences = [[i for i in splitter2(s) if len(i) > 0] for s in sentences][0]

            sent2 = []
            for i in xrange(0, len(sentences), 30):
                sent2.append(sentences[i:i+20])


            w2v = gensim.models.Word2Vec(sentences=sent2, size=200, window=8, min_count=5, workers=4)

            w2v.save('w2v_model3')

    return w2v


def main():
    # clf = SVC(C=0.1, class_weight={0: 1.0, 1: 2.0})
    # clf = LinearSVC(C=0.01, class_weight={0: 1.0, 1: 3.0})
    clf = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    # clf = RandomForestClassifier(n_estimators=10)
    # clf = XGBoostClassifier(n_estimators=100, scale_pos_weight=10.0)
    _, X_train, y_train = load_train_text()
    X_test_pure, X_test, y_test = load_test_text()

    print X_test_pure

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=13)
    clf.fit(np.array(X_train), np.array(y_train))

    y_test_predict = clf.predict(np.array(X_test))
    print len(y_test_predict)
    print y_test
    y_test_predict_values = []
    for i in y_test_predict:
        print i
        if i > 0.15:
            y_test_predict_values.append(1)
        else:
            y_test_predict_values.append(0)
    print classification_report(y_test, y_test_predict_values)

    save_result(X_test_pure, y_test_predict_values)


def main3():
    # clf = SVC(C=0.1, class_weight={0: 1.0, 1: 2.0})
    # clf = LinearSVC(C=0.01, class_weight={0: 1.0, 1: 3.0})
    # clf = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    # clf = RandomForestClassifier(n_estimators=10)
    # clf = XGBoostClassifier(n_estimators=100, scale_pos_weight=10.0)
    # _, X_train, y_train = load_train_text()
    X_test_pure1, X_test1, y_test1 = load_test_text_rules()

    # print X_test_pure

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=13)
    # clf.fit(np.array(X_train), np.array(y_train))

    # y_test_predict = clf.predict(np.array(X_test))
    # print len(y_test_predict)
    # print y_test
    # y_test_predict_values = []
    # for i in y_test_predict:
        # print i
        # if i > 0.15:
            # y_test_predict_values.append(1)
        # else:
            # y_test_predict_values.append(0)
    # print classification_report(y_test, y_test_predict_values)

    # save_result(X_test_pure, y_test)

    clf = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    # clf = RandomForestClassifier(n_estimators=10)
    # clf = XGBoostClassifier(n_estimators=100, scale_pos_weight=10.0)
    _, X_train, y_train = load_train_text()
    X_test_pure, X_test, y_test = load_test_text()

    print X_test_pure

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=13)
    clf.fit(np.array(X_train), np.array(y_train))

    y_test_predict = clf.predict(np.array(X_test))
    print len(y_test_predict)
    print y_test
    y_test_predict_values = []
    for i in y_test_predict:
        print i
        if i > 0.13:
            y_test_predict_values.append(1)
        else:
            y_test_predict_values.append(0)
    print classification_report(y_test, y_test_predict_values)

    # for i in xrange(len(y_test_predict_values)):
    #     y_test_predict_values[i] = min(y_test_predict_values[i], y_test1)

    save_result(X_test_pure, y_test_predict_values)


def main2():
    mark, X_train, y_train = load_train_2()
    X_pure, X_test, y_test = load_test_2(mark)

    print len(X_pure), len(y_test)

    clf = LinearSVC(C=0.1, class_weight={0: 10, 1: 100, 3: 1})
    clf.fit(X_train, y_train)
    y_test_predict = clf.predict(X_test)

    print classification_report(y_test, y_test_predict)

    save_result2(y_test_predict)


if __name__ == '__main__':

    # main2()
    main3()
    # load_train_text()
    # X_pure, X_test, y_test = load_test_text()
    # save_result(X_pure, y_test)
    # w2v = load_vw()
    # print w2v.vocab
    # print w2v.similarity('плохой', 'хороший')
    # for w, score in w2v.most_similar(positive=['плохой']):
    #     print w, score