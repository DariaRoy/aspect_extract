import numpy as np
from scipy.spatial.distance import cosine
import gensim
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors


class W2V:
    def __init__(self, alg='w2v', arc='skip-gram'):
        self.alg = alg
        self.arc = arc

        if self.alg == 'w2v':
            self.download_vectors_wiki_facebook()
        else:
            pass
            #self.download_vectors_fastText()


    def distance(self, word1, list_word):
        ans = []
        embed_vector_1 = self.embeddings_index.get(word1)
        for word in list_word:
            embed_vector_2 = self.embeddings_index.get(word)
            if embed_vector_1 is not None and embed_vector_2 is not None:
                ans.append(cosine(embed_vector_1, embed_vector_2))
            else:
                ans.append(1)
        return ans

    def get_w2v(self, word):

        embed_vector = self.embeddings_index.get(word)
        if embed_vector is not None:
            return embed_vector
        else:
            return [0]*300


    def download_vectors_w2v(self):
        print('Indexing word vectors.')
        self.embeddings_index = {}
        #w2v = gensim.models.Word2Vec.load('araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model')
        #w2v = KeyedVectors.load_word2vec_format('ruwikiruscorpora_0_300_20.bin', binary=True)
        w2v = Word2Vec.load('model_w2v.bin')
        print(list(w2v.wv.vocab.keys())[:20])

        for word in w2v.wv.vocab.keys():
            #s = word[:word.find('_')]
            s = word
            self.embeddings_index[s] = np.asarray(w2v[word], dtype='float32')

        print(list(self.embeddings_index.keys())[:20])

        print('Found %s word vectors.' % len(self.embeddings_index))



    def download_vectors_with_tag(self):

        print('Indexing word vectors.')

        self.embeddings_index = {}
        f = open('ruwikiruscorpora-nobigrams_upos_skipgram_300_5_2018.vec')
        for line in f:
            values = line.split()
            word = values[0]
            word = word[:word.find('_')]
            #print(word)
            #print(word, values[1:])
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except:
                continue
            self.embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(self.embeddings_index))


    def download_vectors_wiki_facebook(self):

        print('Indexing word vectors.')

        self.embeddings_index = {}
        f = open('/Users/dararoj/Documents/Models/wiki.ru.vec')
        for line in f:
            values = line.split()
            word = values[0]
            #print(word, values[1:])
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except:
                continue
            self.embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(self.embeddings_index))


if __name__ == '__main__':
    w2v = W2V()
    print('stop')