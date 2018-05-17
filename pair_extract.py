import pymorphy2
from collections import Counter
morph = pymorphy2.MorphAnalyzer()
from math import log
import numpy

def pos_one(x):
    return morph.parse(x)[0].tag.POS


stop_words = ['этот']

def pair(X_pure, y):
    y_pure = []
    y_t = []
    k = 0
    sentences = []
    sent = ''
    for i in range(len(X_pure)):
        sent = ''
        for j in range(len(X_pure[i])):
            sent += X_pure[i][j]['word'] + ' '

        y_t = y[k:k + len(X_pure[i])]
        k += len(X_pure[i])
        y_pure.append(y_t)
        y_t = []

        sentences.append(sent)

    coll = []

    aspect_term = set()

    for i in range(len(X_pure)):
        for j in range(len(X_pure[i])):
            if y_pure[i][j]:
                aspect_term.add(X_pure[i][j]['word'])
                if pos_one(X_pure[i][j]['word']) == "NOUN":

                    if j and ((pos_one(X_pure[i][j-1]['word']) == 'ADJF' or pos_one(X_pure[i][j-1]['word']) == 'ADJS' or pos_one(X_pure[i][j-1]['word']) == 'NOUN') and X_pure[i][j-1]['word'] not in stop_words ) :
                        #print('BEFORE ',X_pure[i][j-1]['word'], X_pure[i][j]['word'])
                        coll.append(X_pure[i][j-1]['word'] + ' ' + X_pure[i][j]['word'])

                        if j < (len(X_pure[i])-2) and pos_one(X_pure[i][j+1]['word']) == 'NOUN':
                            #print('BEFORE AND NOUN ',X_pure[i][j-1]['word'], X_pure[i][j]['word'], X_pure[i][j+1]['word'])
                            coll.append(X_pure[i][j-1]['word'] + ' ' + X_pure[i][j]['word']+' '+ X_pure[i][j+1]['word'])

                    if j < (len(X_pure[i])-2) and ((pos_one(X_pure[i][j+1]['word']) == 'ADJF' or pos_one(X_pure[i][j+1]['word']) == 'ADJS' or pos_one(X_pure[i][j+1]['word']) == 'NOUN') and X_pure[i][j+1]['word'] not in stop_words):
                        #print('AFTER ',X_pure[i][j]['word'], X_pure[i][j+1]['word'])
                        coll.append(X_pure[i][j]['word'] + ' ' + X_pure[i][j+1]['word'])

    print('===============aspect term start=================')
    print(aspect_term)
    print('===============aspect term end=================')


    print(len(coll))
    count_coll = Counter(coll)
    print(len(count_coll))
    print(count_coll.most_common(30))
    coll_without_repeat = list(count_coll.keys())


    ############ C-Value ############
    c_val_dict = {}
    for phrase in coll_without_repeat:
        for term in coll_without_repeat:
            if phrase in term:
                if phrase in c_val_dict.keys():
                    c_val_dict[phrase].append(term)
                else:
                    c_val_dict[phrase] = [term]


    phrase_c_value = {}

    for phrase in c_val_dict:
        if len(c_val_dict[phrase]) > 1:
            count_of_frq_s = 0
            for s in c_val_dict[phrase]:
                count_of_frq_s += count_coll[s]
            phrase_c_value[phrase] = log(len(phrase.split())) * (count_coll[phrase] - count_of_frq_s / len(c_val_dict[phrase]))
        else:
            phrase_c_value[phrase] = log(len(phrase.split())) * count_coll[phrase]



    tmp_list = []
    for s in phrase_c_value:
        tmp_list.append((phrase_c_value[s], s))



    print('============C-value sorted=============')

    print(sorted(tmp_list, reverse=True))
    tmp_list.sort(reverse=True)

    coll_after_c_value = set()
    for phrase in tmp_list:
        tmp = []
        for s in c_val_dict[phrase[1]]:
            tmp.append(phrase_c_value[s])
        k = numpy.argmax(tmp)
        coll_after_c_value.add(c_val_dict[phrase[1]][k])

    print('============phrase after c-value=============')

    print(len(coll_after_c_value))
    print(coll_after_c_value)



    coll_domain_cons = {}
    for phrase in coll_after_c_value:
        k = 0
        for sent in sentences:
            d = 0
            if phrase in sent:
                d = sent.count(phrase) / len(sent.strip())
                d *= log(d)
            k += d
        coll_domain_cons[phrase] = -k

    print('============domain consensus=============')
    print(coll_domain_cons)

    print('============domain consensus sorted=============')

    domain_cons_sort = []
    for s in coll_domain_cons:
        domain_cons_sort.append((coll_domain_cons[s], s))

    domain_cons_sort.sort(reverse=True)
    print(domain_cons_sort)


    ########### tvq #############

    tvq_dict ={}

    for phrase in coll_without_repeat:
        k1 = 0
        k2 = 0
        for sent in sentences:
            if phrase in sent:
                k1 += sent.count(phrase)**2
                k2 += sent.count(phrase)
        tvq = k1 - (k2**2)/len(sentences)
        tvq_dict[phrase] = tvq

    print('============tvq sorted=============')

    tvq_sort = []
    for s in tvq_dict:
        tvq_sort.append((tvq_dict[s], s))

    tvq_sort.sort(reverse=True)
    print(tvq_sort)


    ############ term contribution ##############

    term_cont_dict = {}

    len_sent = len(sentences)
    for phrase in coll_without_repeat:
        l_mult = []

        df_phrase = 0
        for sent in sentences:
            if phrase in sent:
                df_phrase += 1

        for sent in sentences:
            tf = sent.count(phrase)
            lg = log((len_sent / df_phrase))
            l_mult.append(tf*lg)

        tc_phrase = 0

        for i in range(len(l_mult)):
            k = i + 1
            while k < len(l_mult):
                tc_phrase += l_mult[i]*l_mult[k]
                k += 1

        term_cont_dict[phrase] = tc_phrase
        #print(phrase, tc_phrase)

    print('============term contr sorted=============')

    tc_sort = []
    for s in term_cont_dict:
        tc_sort.append((term_cont_dict[s], s))

    tc_sort.sort(reverse=True)
    print(tc_sort)

    extract_character = set()
    extract_character.update(tc_sort[:10])
    extract_character.update(tvq_sort[:10])
    extract_character.update(domain_cons_sort[:10])

    print(extract_character)

