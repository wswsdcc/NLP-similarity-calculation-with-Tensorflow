# -*- coding: utf-8 -*-
from gensim.models import KeyedVectors
import jieba
import sys

reload(sys)

sys.setdefaultencoding('utf-8')

class Preprocess(object):

    in_file_name = "../atec_nlp_sim_train.csv"
    out_file_name = "../train_by_character.txt"
    trained_word2vec_model_path = "../../sgns.zhihu.word"
    wv_save_path = "../word_vector_0116.kv"


    def process(self, inpath, outpath):
        with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
            for line in fin:
                lineno, sen1, sen2, label = line.strip().split('\t')
                fout.write(lineno + '\t')
                for w in sen1.decode('utf-8'):
                    if w.strip() and w!=('*'):
                        fout.write(w + ' ')
                fout.write('\t')
                for w in sen2.decode('utf-8'):
                    if w.strip() and w!=('*'):
                        fout.write(w + ' ')
                fout.write('\t'+label+'\n')
    
    def model2WvKeyedVector(self, model_path, wv_path):
        word_vectors = KeyedVectors.load_word2vec_format(model_path,binary=False)
        word_vectors.wv.save(wv_path)
    
    def testWv(self, wv_path):
        wv = KeyedVectors.load(wv_path,mmap='r')
        # print wv[u'花呗']
        # print wv[u'借呗']
        sim = wv.most_similar(u'借',topn=5)
        print '\n借-top5:'
        for item in sim:
            print item[0],item[1]
            
pre = Preprocess()
pre.process(pre.in_file_name, pre.out_file_name)
#pre.model2WvKeyedVector(pre.trained_word2vec_model_path, pre.wv_save_path)
#pre.testWv(pre.wv_save_path)
