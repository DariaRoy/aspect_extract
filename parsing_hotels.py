# coding: utf-8
__author__ = 'air'
import json


def normalize_text(review):
    review = review.lower()
    new_review = ''

    for c in review:
        if c in CYR_SMALL or c == ' ' or c == '\t':
            new_review += c

    new_review = ' '.join(new_review.split())
    return new_review


TRAIN_NUM = 30
CYR_SMALL = set(u'абвгдеёжзийклмнопрстуфхцчшщъыьэюя')

def load_train_test():
    """
    Загружаем файл json с отзывами
    Разбиваем его на обучающую/тестовую части

    Возвращаем два списка
    :return:
    """
    with open('отзывы.json') as f:
        hotel_records = json.loads(f.read())


    print ('TRAIN/TEST: %s/' % (len(hotel_records)))

    return hotel_records


def parse(review):
    review = review.lower()
    new_review = ''

    for c in review:
        if c in CYR_SMALL or c == ' ' or c == '\t':
            new_review += c

    new_review = ' '.join(new_review.split())
    return new_review


def parse_new(name):
    print('dasha')
    W_test = []
    f = open(name)
    for line in f:
        review = ' '.join(line.split('\n'))
        review = parse(review.decode('utf-8'))
        word = ''
        for i in review:
            if i == ' ':
                W_test.append(word)
                word = ''
            else:
                word += i
    print('dasha')
    print (W_test[0])
    return W_test



def prepare_task1(records):
    """
    Задача 1 - выделение аспектов в тексте
    Данная функция принимает на вход список записей, извлекает слова из термов
    Маркирует слова 1 - слово является аспектным (есть среди явных аспектов)
                    0 - слово не является аспектным
    Возвращает пару списков: X - список фраз, y - список меток, z- метки тональности 1 или -1 для аспектов и 0 для невспектов
    c- категории аспектов
    :param records: list of hotel's records
    :return: X, y, z, c
    """

    X = []
    y = []
    z = []
    c = []
    with open('out.txt','w') as f:
        for record in records:
            text = record['text'].lower()

            f.write(text.encode('utf-8') + '\n')

            # перебираем аспекты, сохраняем явные в explicit_aspects
            explicit_aspects = dict()
            category = dict()
            explicit_aspects_set = set()
            for a in record['aspects']:
                if a['type'] != u'явный':
                    continue
                term = a['term'].lower()
                explicit_aspects_set.add(term)
                if a['sentiment'] == u'позитивный':
                    explicit_aspects[term] = 1
                elif a['sentiment'] == u'негативный':
                    explicit_aspects[term] = -1
                else:
                    explicit_aspects[term] = 0

                if a['category'] == u'Сущность':
                    category[term] = 0
                elif a['category'] == u'Номер':
                    category[term] = 1
                elif a['category'] == u'Сервис':
                    category[term] = 2
                elif a['category'] == u'Территория':
                    category[term] = 3
                elif a['category'] == u'Питание':
                    category[term] = 4
                elif a['category'] == u'Расположение':
                    category[term] = 5
                elif a['category'] == u'Цена':
                    category[term] = 6
                else:
                    print(a['category'])
                    print(term)

            # перебираем слова и добавляем соответствующие
            # значения в итоговые списки
            for word in normalize_text(text).split(' '):
                if word in explicit_aspects_set:
                    X.append(word)
                    y.append(1)
                    z.append(explicit_aspects[word])
                    c.append(category[word])
                else:
                    X.append(word)
                    y.append(0)
                    z.append(0)

    return X, y, z, c


