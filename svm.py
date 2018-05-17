from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.svm import LinearSVC
from tfidf import compute_tfidf
import xgboost as xgb
import pymorphy2
from sklearn.preprocessing import normalize
from parse_SentiRuLex import sentiRuLex
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from w2v import W2V
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier


morph = pymorphy2.MorphAnalyzer()

dict_pos = {
    'NOUN': 1,
    'ADJF': 2,
    'ADJS': 2,
    'COMP': 2,
    'VERB': 3,
    'INFN': 3,
    'PRTF': 4,
    'PRTS': 4,
    'GRND': 4,
}


def pos(x):
    pos = morph.parse(x)[0].tag.POS
    tag = 0
    if pos in dict_pos:
        tag = dict_pos[pos]
    l = [0] * 5
    l[tag] = 1
    return l


def pos_one(x):
    return morph.parse(x)[0].tag.POS


def make_x(X_pure, list_of_tfidf, sent_word, w2v, domain):

    words_count = []
    for word in X_pure:
        if pos_one(word['word']) == 'NOUN':
            words_count.append(word['word'])

    words_count = Counter(words_count)
    freq_noun = words_count.most_common(5)
    freq_noun = [x[0] for x in freq_noun]

    print(freq_noun)

    k = 0
    x = []
    for i in range(len(X_pure)):
        word = X_pure[i]['word']
        tmp = []

        # x_train
        if i != 0 and X_pure[i]['from'] <= 2:
            k += 1

        tf_idf = 0
        if word in list_of_tfidf[k]:
            tf_idf = list_of_tfidf[k][word]


        #add distance between word and domain name
        #tmp.extend(w2v.distance(X_pure[i]['word'], [domain]))


        #add distance between word and 5 most frequence nouns
        tmp.extend(w2v.distance(X_pure[i]['word'], freq_noun))

        # add tf-idf
        tmp.append(tf_idf)

        #add sentiment word in +-2 word
        if 1 < i < len(X_pure) - 3 and (X_pure[i - 1]['word'] in sent_word.keys() or X_pure[i + 1]['word'] in sent_word.keys()
                                              or X_pure[i - 2]['word'] in sent_word.keys() or X_pure[i + 2]['word'] in sent_word.keys()):
            tmp.append(1)
            #print(X_pure_train[i]['word'])
        else:
            tmp.append(0)

        # add category of part of speach
        tmp.extend(pos(X_pure[i]['word']))

        #add w2v
        tmp.extend(list(w2v.get_w2v(X_pure[i]['word'])))

        x.append(tmp)

    return x


def make_y_train(X_pure_train, aspects_list_train):
    y_train = []
    aspect_word = [x['term'] for x in aspects_list_train]
    for i in range(len(X_pure_train)):
        word = X_pure_train[i]['word']

        if word in aspect_word:
            y_train.append(1)
        else:
            y_train.append(0)
    return y_train


def make_data(X_pure_train, X_sentences_train, aspects_list_train, X_pure_test, X_sentences_test):

    sent_word = sentiRuLex()

    w2v = W2V()
    #w2v = None

    list_of_tfidf_train = compute_tfidf(X_sentences_train)
    list_of_tfidf_test = compute_tfidf(X_sentences_test)

    y_train = make_y_train(X_pure_train, aspects_list_train)
    x_train = make_x(X_pure_train, list_of_tfidf_train, sent_word, w2v, 'ресторан')
    x_test = make_x(X_pure_test,list_of_tfidf_test, sent_word, w2v, 'автомобиль')

    return x_train, y_train, x_test


def svm(X_pure_train, X_sentences_train, aspects_list_train, X_pure_test, X_sentences_test):

    x_train, y_train, x_test = make_data(X_pure_train, X_sentences_train, aspects_list_train, X_pure_test, X_sentences_test)

    print(x_train[0])
    for i in x_train:
        print(len(i))
    print(type(x_train[0]))

    x_train = normalize(x_train)
    x_test = normalize(x_test)

    clf = LinearSVC()
    clf.fit(x_train, y_train)
    pred1 = clf.predict(x_test)

    clf1 = RandomForestClassifier()
    clf1.fit(x_train, y_train)
    pred2 = clf1.predict(x_test)


    clf2 = GaussianNB()
    clf2.fit(x_train, y_train)
    pred3 = clf2.predict(x_test)

    #print(clf.coef_)

    return pred1, pred2, pred3
