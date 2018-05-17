import pymorphy2

morph = pymorphy2.MorphAnalyzer()


def pos(x):
    return morph.parse(x)[0].tag.POS

def case(x):
    return morph.parse(x)[0].tag.case

def norm(x):
    return morph.parse(x)[0].normal_form


def pos_sequence(X_pure):
    seq = []
    for word in X_pure:
        vocab = {}
        vocab['word'] = word['word']
        vocab['from'] = word['from']
        vocab['to'] = word['to']
        vocab['pos'] = pos(word['word'])
        vocab['case'] = case(word['word'])
        vocab['norm'] = norm(word['word'])
        seq.append(vocab)
    return seq

def lingvistic(X_pure):
    word_sequence = pos_sequence(X_pure)
    y_pred = []
    for i in range(len(word_sequence)):
        if word_sequence[i]['pos'] == 'NOUN':
            y_pred.append(1)
        elif (word_sequence[i]['pos'] == 'ADJF' or word_sequence[i]['pos'] == 'PRTF'
              or word_sequence[i]['pos'] == 'NUMR') and i + 1 < len(word_sequence) \
                and word_sequence[i + 1]['pos'] == 'NOUN':
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred