# -*- coding: utf-8 -*-
from gensim.models import KeyedVectors
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import sys
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

reload(sys)
sys.setdefaultencoding('utf-8')

train_file_name = "../train_by_character.txt"
wv_save_path = "../word_vector_0116.kv"
err_log_path = "error_log.txt"
word_dim = 300
max_sen_len = 50
#num_classes = 2
batch_sizes = 500

def makeData(data_path, word_vector):
    y = []
    x1 = []
    x2 = []
    with open(data_path,'r') as fin, open(err_log_path,'w') as fout:
        for line in fin:
            lineno, sen1, sen2, label = line.strip().split('\t')
            sen1_words = []
            sen2_words = []
            #sen1_words = [wv[w.decode('utf-8')] for w in sen1.strip().split(' ')]
            #sen2_words = [wv[w.decode('utf-8')] for w in sen2.strip().split(' ')]
            for w in sen1.strip().split(' '):
                try:
                    sen1_words.append(wv[w.decode('utf-8')])
                except KeyError:
                    fout.write(str(lineno) +" : " + w + " " + str(KeyError) + '\n')
            
            for w in sen2.strip().split(' '):
                try:
                    sen2_words.append(wv[w.decode('utf-8')])
                except KeyError:
                    fout.write(str(lineno) +" : " + w + " " + str(KeyError) + '\n')
            y.append(label)
            x1.append(sen1_words)
            x2.append(sen2_words)  
    return y,x1,x2

def align2MaxSenLen(sens_vector1 ,sens_vector2):
    ret_sens1 = []
    ret_sens2 = []
    zero_vector = [0] * word_dim

    for sen in sens_vector1:
        sen_len = len(sen)
        add_dim = max_sen_len - sen_len
        if add_dim > 0:
            filled_sen = sen + ([zero_vector] * add_dim)
        else:
            filled_sen = sen[:max_sen_len]
        ret_sens1.append(filled_sen)
    
    for sen in sens_vector2:
        sen_len = len(sen)
        add_dim = max_sen_len - sen_len
        if add_dim > 0:
            filled_sen = sen + ([zero_vector] * add_dim)
        else:
            filled_sen = sen[:max_sen_len]
        ret_sens2.append(filled_sen)

    return ret_sens1,ret_sens2

def cosine_sim(a,b):
    a_norm = tf.sqrt(tf.reduce_sum(a*a, 1))
    b_norm = tf.sqrt(tf.reduce_sum(b*b, 1))
    a_b = tf.reduce_sum(a*b, 1)
    cosine = tf.div(a_b, a_norm * b_norm +1e-8, name="cosine_sim")
    return cosine

def get_batch(train1, train2, label, batch_size, now_batch, total_batch):
    if now_batch < total_batch-1:
        train1_batch = train1[now_batch*batch_size:(now_batch+1)*batch_size]
        train2_batch = train2[now_batch*batch_size:(now_batch+1)*batch_size]
        label_batch = label[now_batch*batch_size:(now_batch+1)*batch_size]
    else:
        train1_batch = train1[now_batch*batch_size:]
        train2_batch = train2[now_batch*batch_size:]
        label_batch = label[now_batch*batch_size:]
    return train1_batch, train2_batch, label_batch

wv = KeyedVectors.load(wv_save_path,mmap='r')

print "begin to make train data"
labels, sens1, sens2 = makeData(train_file_name, wv)
fea1, fea2 = align2MaxSenLen(sens1, sens2)

input_x1 = tf.placeholder(tf.float32, [None, max_sen_len, word_dim], name='input_x1')
input_x2 = tf.placeholder(tf.float32, [None, max_sen_len, word_dim], name='input_x2')
input_y = tf.placeholder(tf.int32, [None], name='input_y')


with tf.variable_scope('sentence1'):
    lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(num_units=256) 
    lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(num_units=128) 
    lstm_cell_3 = tf.nn.rnn_cell.LSTMCell(num_units=64)
    lstm_cell_x1 = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell_1,lstm_cell_2, lstm_cell_3])
    output_x1, state_x1 = tf.nn.dynamic_rnn(cell=lstm_cell_x1, inputs=input_x1, dtype=tf.float32)
    last_x1 = output_x1[:, -1, :]

with tf.variable_scope('sentence2'):
    lstm_cell_4 = tf.nn.rnn_cell.LSTMCell(num_units=256) 
    lstm_cell_5 = tf.nn.rnn_cell.LSTMCell(num_units=128) 
    lstm_cell_6 = tf.nn.rnn_cell.LSTMCell(num_units=64) 
    lstm_cell_x2 = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell_4,lstm_cell_5, lstm_cell_6])
    output_x2, state_x2 = tf.nn.dynamic_rnn(cell=lstm_cell_x2, inputs=input_x2, dtype=tf.float32)
    last_x2 = output_x2[:, -1, :]

with tf.name_scope("score"):
    #fc = tf.layers.dense(inputs=last, units=128, activation=tf.nn.tanh, name='fc1')
    cosine = cosine_sim(last_x1, last_x2)
    logits = tf.sigmoid(cosine, name="logits")
    predict = tf.cast(tf.math.rint(logits),tf.int32, name="output_y")
    predict_sum = tf.reduce_sum(predict,0)
    
with tf.name_scope("optimize"):
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=tf.cast(input_y,tf.float32), logits=logits,pos_weight=4)
    optim = tf.train.GradientDescentOptimizer(0.0000000000001).minimize(loss)
    batch_mean_loss = tf.reduce_sum(loss,0)

correct_prediction = tf.equal(input_y, predict)
acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)
    max_acc = 0
    kf = KFold(n_splits=5,shuffle=True)
    num_batch = 20
    for batch in range(num_batch):
        batch_x1, batch_x2, batch_y = get_batch(fea1, fea2, labels, batch_sizes, batch, num_batch)
        iteration = 0
        for train_indices, test_indices in kf.split(range(len(batch_y))):
            x1_train = np.array(batch_x1)[train_indices]
            x1_test = np.array(batch_x1)[test_indices]
            x2_train = np.array(batch_x2)[train_indices]
            x2_test = np.array(batch_x2)[test_indices]
            y_train = np.array(batch_y)[train_indices]           
            y_test = np.array(batch_y)[test_indices]
            sess.run(optim, feed_dict={input_x1: x1_train, input_x2: x2_train, input_y: y_train})
            train_loss = sess.run(batch_mean_loss/400,feed_dict={input_x1: x1_train, input_x2: x2_train, input_y: y_train})
            test_loss = sess.run(batch_mean_loss/100,feed_dict={input_x1: x1_test, input_x2: x2_test, input_y: y_test})
            val_acc = sess.run(acc,feed_dict={input_x1: x1_test, input_x2: x2_test, input_y: y_test})
            p_sum = sess.run(predict_sum,feed_dict={input_x1: x1_test, input_x2: x2_test, input_y: y_test})            
            print "batch    " + str(batch) + ", iteration    " + str(iteration) + ": acc   " + str(val_acc) + ", train loss    " + str(train_loss) + ", test loss    " + str(test_loss)
            iteration += 1
            print "p_sum:" + str(p_sum)
        saver.save(sess, './checkpoint_dir/model20190116.model')
    sess.close()