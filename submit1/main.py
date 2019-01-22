#/usr/bin/env python
#coding=utf-8
import jieba
import sys
import tensorflow as tf
from gensim.models import KeyedVectors

word_dim = 300
max_sen_len = 50

def align2MaxSenLen(sens_vector1 ,sens_vector2):
    zero_vector = [0] * word_dim
    sen_len1 = len(sens_vector1)
    add_dim1 = max_sen_len - sen_len1
    if add_dim1 > 0:
        filled_sen1 = sens_vector1 + ([zero_vector] * add_dim1)
    else:
        filled_sen1 = sens_vector1[:max_sen_len]
    
    sen_len2 = len(sens_vector2)
    add_dim2 = max_sen_len - sen_len2
    if add_dim2 > 0:
        filled_sen2 = sens_vector2 + ([zero_vector] * add_dim2)
    else:
        filled_sen2 = sens_vector2[:max_sen_len]
        
    return filled_sen1, filled_sen2

def process(inpath, outpath):
    wv = KeyedVectors.load('./word_vector.kv',mmap='r')
    sess=tf.Session()
    new_saver = tf.train.import_meta_graph('./checkpoint_dir/model20190115.model-2.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
    graph = tf.get_default_graph()
    inputx1 = graph.get_tensor_by_name("input_x1:0")
    inputx2 = graph.get_tensor_by_name("input_x2:0")
    logity = graph.get_tensor_by_name("score/logits:0")
    outputy = graph.get_tensor_by_name("score/output_y:0")
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            lineno, sen1, sen2 = line.strip().split('\t')
            vector1= []
            vector2= []
            for w in jieba.cut(sen1):
                if w.strip() and w!=('*'):
                    try:
                        vector1.append(wv[w])
                    except KeyError:
                        pass
            for w in jieba.cut(sen2):
                if w.strip() and w!=('*'):
                    try:
                        vector2.append(wv[w])
                    except KeyError:
                        pass
            fea1, fea2 = align2MaxSenLen(vector1, vector2)
            l = sess.run(logity, feed_dict={inputx1: [fea1], inputx2: [fea2]})
            y = sess.run(outputy, feed_dict={inputx1: [fea1], inputx2: [fea2]})
            fout.write(lineno + '\t' + str(y) + '\n')
            

if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])