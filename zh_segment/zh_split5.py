#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import re
import jieba 
import os
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import io
import optparse
import sys
import time
import datetime
import random


class SentenceClasscify:
    __maxSeqLength = 32
    __lstmUnits = 48
    __lstmUnits2 = 24
    
    __labelSize = 0

    __wv_dim = 0
    __wv_matrix = []
    __wordlist = []
    __wordmap = {}
    __max_word_num = 0

    
    __trainX = []
    __trainY = []
    __testX = []
    __testY = []
    
    __loss = None
    __optimizer = None
    __acc = None
    __prediction = None
    
    __labels = None
    __input_data = None
    
    __name = ""
    
    __y_map = {'s':[1, 0, 0, 0, 0],
               'b':[0, 1, 0, 0, 0],
               'm':[0, 0, 1, 0, 0],
               'e':[0, 0, 0, 1, 0],
               'x':[0, 0, 0, 0, 1]
             }

    __fileLineNum = 0

    def __init__(self, name=""):
        self.__name = name 
        self.__labelSize = self.__maxSeqLength * 5
    
    def index2word(self, idx):
        return self.__wordlist[idx]
        
    def word2index(self, word):
        return self.__wordmap[word]
        
    def word2vector(self, word):
        return self.__wv[word2index(word)]
    
    def index2vector(self, idx):
        return self.__wv[idx]
        
    def tokens2sentence(self, tokens):
        return ''.join([self.index2word(i) for i in tokens])
        
    def sentence2tokens(self, sentence):
        #cut = jieba.cut(sentence)
        #cut= [e for e in cut]
        tokens = []
        for w in sentence:
            try:
                tokens.append(self.__wordmap[w])
            except KeyError as e:
                tokens.append(0)
        return tokens
    
    def loadCorpus(self, filename):
        X, Y= self.__loadCorpus(filename)
        self.__testX = X[0:128]
        self.__testY = Y[0:128]
        self.__trainX = X[128:]
        self.__trainY = Y[128:]
        
    def loadTrainCorpus(self, filename):
        self.__trainX, self.__trainY = self.__loadCorpus(filename)
        
    def loadTestCorpus(self, filename):
        self.__testX, self.__testY = self.__loadCorpus(filename)
        
    def loadWV(self, filename, max_word_num = 0):
        try:
            wv_model = KeyedVectors.load(filename, mmap='r')
        except:
            try:
                wv_model = KeyedVectors.load_word2vec_format(filename,binary=True)
            except:
                wv_model = KeyedVectors.load_word2vec_format(filename,binary=False)
                
        wv = wv_model.wv
        print "wv dict len is %d" % len(wv.vocab)

        self.__max_word_num = max_word_num
        if self.__max_word_num == 0 or self.__max_word_num > len(wv.vocab):
            self.__max_word_num = len(wv.vocab)
  
         
        print "maxk word num is %d" % self.__max_word_num

        print "%s: %s" %(wv.index2word[1], str(wv[wv.index2word[1]]))
  
        self.__wv_dim = wv[wv.index2word[0]].shape[0]
        self.__wv_matrix = np.zeros((self.__max_word_num + 1, self.__wv_dim))
        self.__wv_matrix[0,:] = [0] * self.__wv_dim
        self.__wordlist = []
        self.__wordlist.append("*") # unknown
        self.__wordmap = {"*":0}
        for i in range(self.__max_word_num):
            self.__wordlist.append(wv.index2word[i])
            self.__wordmap[wv.index2word[i]] = i + 1
            self.__wv_matrix[i+1,:] = wv[wv.index2word[i]]
            
        self.__wv_matrix = self.__wv_matrix.astype('float32')
        
        print np.sum(wv[wv.index2word[22]] == self.__wv_matrix[22] )
        print self.__wv_matrix.shape
        
        print "新：" + str(wv[u'新'])
        print self.__wordmap[u'新']   
        print [u'新']
     
    def get_train_batch(self, batchSize) :
        train_size = len(self.__trainX)
            
        indexs = [i for i in range(train_size)]
        random.shuffle(indexs)
        X = []
        Y = []
        for i in range(batchSize):
            X.append(self.__trainX[indexs[i]])
            Y.append(self.__trainY[indexs[i]])
        return X, Y
        
        
    __last_batch_index = 0    
    def get_train_batch2(self, batchSize) :
        batchSize = int(batchSize)
        train_size = len(self.__trainX)
        X = [a for a in self.__trainX[self.__last_batch_index: self.__last_batch_index+batchSize] ]
        Y = [a for a in self.__trainY[self.__last_batch_index: self.__last_batch_index+batchSize] ]
        self.__last_batch_index = self.__last_batch_index + batchSize
        if len(X) < batchSize:
            X.extend(self.__trainX[0: batchSize - len(X)])
            Y.extend(self.__trainY[0: batchSize - len(Y)])
            self.__last_batch_index =   batchSize - len(X)

        return X, Y
        
    def creataModel(self, batchSize):
        #tf.reset_default_graph()
        print "self.__lableSize %d" %self.__lableSize 
        self.__labels = tf.placeholder(tf.float32, [None, self.__lableSize], name='labels')
        self.__input_data = tf.placeholder(tf.int32, [None, self.__maxSeqLength], name='input')
        
        #data = tf.Variable(tf.zeros([tf.shape(self.__input_data)[0], self.__maxSeqLength, self.__wv_dim]), dtype=tf.float32)
        data = tf.zeros([tf.shape(self.__input_data)[0], self.__maxSeqLength, self.__wv_dim])
        data = tf.nn.embedding_lookup(self.__wv_matrix, self.__input_data)

        #调用现成的BasicLSTMCell，建立两条完全一样，又独立的LSTM结构
        lstm_qx = tf.contrib.rnn.BasicLSTMCell(self.__lstmUnits, forget_bias = 1.0)
        lstm_qx = tf.contrib.rnn.DropoutWrapper(cell=lstm_qx, output_keep_prob=0.75)
        lstm_hx = tf.contrib.rnn.BasicLSTMCell(self.__lstmUnits, forget_bias = 1.0)
        lstm_hx = tf.contrib.rnn.DropoutWrapper(cell=lstm_hx, output_keep_prob=0.75)
        #两个完全一样的LSTM结构输入到static_bidrectional_rnn中，由这个op来管理双向计算过程。
        #data = tf.unstack(data, n_steps, 1)
        output1, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_qx, lstm_hx, data, dtype = tf.float32)
        #weight1 = tf.Variable(tf.truncated_normal([2*self.__lstmUnits, self.__lstmUnits2]))
        #bias1 = tf.Variable(tf.constant(0.1, shape=[self.__lstmUnits2]))
        
        output1_fw = output1[0]
        output1_bw = output1[1]#原形状为[batch_size,max_len,hidden_num]

        #output1_fw = tf.transpose(output1_fw,[1,0,2])#现在形状为[max_len,batch_size,hidden_num]
        #output1_bw = tf.transpose(output1_bw,[1,0,2])
        #outputs1 = [output1_fw,output1_bw]
        #lstmoutputs = tf.concat(outputs1, axis=-1)#连接后形状为[max_len,batch_size,2*hidden_num]
        #last = lstmoutputs[-1]#最后一个time_step的输出，为[batch_size,2*hidden_num]
        #prediction = (tf.matmul(last, weight1) + bias1)

        outputs1 = [output1_fw,output1_bw]
        lstmoutputs = tf.concat(outputs1, axis=-1)
        print "print output1.shape " 
        print np.array(output1).shape
        bilstmlast = lstmoutputs


        '''
        lstm_qx2 = tf.contrib.rnn.BasicLSTMCell(self.__lstmUnits, forget_bias = 1.0)
        lstm_hx2 = tf.contrib.rnn.BasicLSTMCell(self.__lstmUnits, forget_bias = 1.0)
        output2, output_states2 = tf.nn.bidirectional_dynamic_rnn(lstm_qx2, lstm_hx2, lstmoutputs, dtype = tf.float32)
        output2_fw = output2[0]
        output2_bw = output2[1]#原形状为[batch_size,max_len,hidden_num]
        outputs2 = [output2_fw,output2_bw]
        lstmoutputs2 = tf.concat(outputs2, axis=-1)#连接后形状为[max_len,batch_size,2*hidden_num]
        bilstmlast = lstmoutputs2
        print "print output2.shape " 
        print np.array(output2).shape
        '''

        #lstms = []
        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.__lstmUnits2)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        #lstms.append(lstmCell)
        #lstmCell2 = tf.contrib.rnn.BasicLSTMCell(64)
        #lstmCell2 = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        #lstms.append(lstmCell2)
        #lstmCell3 = tf.contrib.rnn.BasicLSTMCell(16)
        #lstmCell3 = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        #lstms.append(lstmCell3)
        #mlstm_cell = tf.contrib.rnn.MultiRNNCell(lstms )
        
        mlstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.__lstmUnits2) for _ in range(5)] )
        value, _ = tf.nn.dynamic_rnn(mlstm_cell, bilstmlast, dtype=tf.float32)
        print "print value.shape " + str(value.shape)
        weight2 = tf.Variable(tf.truncated_normal([self.__lstmUnits2, self.__lableSize]))
        #bias2 = tf.Variable(tf.constant(0.1, shape=[self.__lableSize]))
        bias2 = tf.Variable(tf.random_normal([self.__lableSize]))
        value = tf.transpose(value, [1, 0, 2])
        #last = tf.gather(value, int(value.get_shape()[0]) - 1)
        last = value[-1]
        
        #last = tf.reduce_sum(value, 1)
        print "print last.shape " + str(last.shape)
        prediction = (tf.matmul(last, weight2) + bias2)
        
        print "prediction.shape:" + str(prediction.shape)

        '''
        #softmax   2 classcify
        # 定义正确的预测函数和正确率评估参数
        correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.__labels, 1))
        self.__acc = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        # 最后将标准的交叉熵损失函数定义为损失值，这里是以adam为优化函数
        self.__loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=self.__labels), name='loss')
        '''

        ys = tf.reshape(prediction, [-1, self.__maxSeqLength, 5], name='prediction')
        ys = tf.argmax(ys, 2)

        labels = tf.reshape(self.__labels, [-1, self.__maxSeqLength, 5])
        labels = tf.argmax(labels, 2)
  
        correctPred = tf.equal(ys, labels)
        self.__acc = tf.reduce_mean(tf.cast(correctPred, tf.float32), name='accuracy')
        

        # 定义正确的预测函数和正确率评估参数
        #correctPred = tf.cast(tf.sigmoid(prediction)>0.6, tf.float32)
        #self.__acc = tf.reduce_mean(tf.cast(tf.equal(correctPred, self.__labels),tf.float32), name='accuracy')
        # 最后将标准的交叉熵损失函数定义为损失值，这里是以adam为优化函数
        self.__loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=self.__labels), 
            name='loss')
        
        
        self.__optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.__loss, name='optimizer')
        tf.add_to_collection('optimizer', self.__optimizer)
        
        
    def restore(self, sess, model_path, model_name):
        print "restroe from " + model_path
        
        #saver = tf.train.Saver(max_to_keep = 1)
        saver = tf.train.import_meta_graph(model_path + '/' + model_name + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        self.__labels = graph.get_tensor_by_name("labels:0")
        self.__input_data = graph.get_tensor_by_name("input:0")
        self.__loss = graph.get_tensor_by_name("loss:0")
        self.__acc = graph.get_tensor_by_name("accuracy:0")
        self.__prediction = graph.get_tensor_by_name("prediction:0")
        self.__optimizer = tf.get_collection('optimizer')[0]
        return saver
        
    def evaluate(self, model_path):
        model_name = os.path.basename(model_path.rstrip('/'))
        tf.reset_default_graph()        
        sess = tf.InteractiveSession()
        
        if os.path.exists(model_path + '/checkpoint'):
            saver = self.restore(sess, model_path, model_name)
        else:
            print "model is not exit!"
            os.exit()
        
        accuracy_=(sess.run(self.__acc, {self.__input_data: self.__testX, self.__labels: self.__testY})) * 100
        print("accuracy:{}".format(accuracy_))    
        
    def prediction(self, model_path, text):
        model_name = os.path.basename(model_path.rstrip('/'))
        tf.reset_default_graph()        
        sess = tf.InteractiveSession()
        
        if os.path.exists(model_path + '/checkpoint'):
            saver = self.restore(sess, model_path, model_name)
        else:
            print "model is not exit!"
            os.exit()

        text = text.decode(sys.stdin.encoding)
        print [u'新华网']
        tag_map = ['s', 'b', 'm', 'e', 'x']
        while True:
            print "-------------------------------------------------------"
            x = [] 
            print [text]
            print text
            x.append(self.sentence2tokens(text))
            x = self.__padding(x, self.__maxSeqLength)
            y=sess.run(self.__prediction, {self.__input_data: x})
            y = tf.argmax(y, 2)
            y = y.eval()[0]
            print x[0]
            print [tag_map[i] for i in y ]
            cut = u""

            for i in range(len(text)):
                if tag_map[y[i]] == 'b':
                  if i > 0 and y[i-1] == 1:
                    y[i-1] = 0
                
            for i in range(len(text)):
                cut += text[i]
                cut += u'/'
                cut += tag_map[y[i]]
            print cut
            print cut.replace(u'/s', u' ').replace(u'/e', u' ').replace(u'/m', u'').replace(u'/b', u'').replace(u'/x', u'')  
            text = raw_input("input sentence:").decode(sys.stdin.encoding)

        
        
    def train(self, model_path, infile, batchSize=128):
    
        model_name = os.path.basename(model_path.rstrip('/'))
        tf.reset_default_graph()        
        sess = tf.InteractiveSession()
        
        if os.path.exists(model_path + '/checkpoint'):
            saver = self.restore(sess, model_path, model_name)
        else:
            self.creataModel(batchSize)
            tf.train.export_meta_graph(filename=model_path + '/' + model_name + '.meta')
            saver = tf.train.Saver(max_to_keep = 1)
            sess.run(tf.global_variables_initializer())
            
        
        
        
        tf.summary.scalar('loss', self.__loss)
        tf.summary.scalar('Accrar', self.__acc)
        #print tf.summary
        merged=tf.summary.merge_all()
        logdir='tensorboard/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"
        writer=tf.summary.FileWriter(logdir,sess.graph)

        #batchSize = self.__input_data.shape[0]
        iterations = 10000000
        print('..........')
        f = io.open(infile, 'r')
        #lineNum = self.getLineNum(f)
        #print "line num: " + str(lineNum)
        for i in range(iterations):
                
            #next_batch, next_batch_labels = self.get_train_batch2(batchSize)
            next_batch, next_batch_labels = self.loadBatch(f, batchSize)
            
            r = sess.run(self.__optimizer, {self.__input_data: next_batch, self.__labels: next_batch_labels}) 
 
            if(i%50==0):
                summary=sess.run(merged,{self.__input_data: next_batch, self.__labels: next_batch_labels})
                writer.add_summary(summary,i)

              
                          
            if (i%50==0):
                print ("training %d batch..." %i)
                #next_batch, next_batch_labels = self.get_train_batch2(batchSize)
                loss_ = sess.run(self.__loss, {self.__input_data: next_batch, self.__labels: next_batch_labels})
                accuracy_=(sess.run(self.__acc, {self.__input_data: next_batch, self.__labels: next_batch_labels})) * 100
                #loss_ = sess.run(self.__loss, {self.__input_data: self.__testX, self.__labels: self.__testY})
                #accuracy_=(sess.run(self.__acc, {self.__input_data: self.__testX, self.__labels: self.__testY})) * 100
                print("iteration:{}/{}".format(i+1, iterations),
                          "loss:{}".format(loss_),
                          "accuracy:{}".format(accuracy_))    
                print('..........')  

            if(i%100==0 and i!=0):
                save_path=saver.save(sess, model_path + "/" + model_name, global_step=i)
                print("saved to %s"% save_path)
        
        f.close()
        

    __corpus_file = None
    def __loadCorpusBig(self, filename):
        if os.path.isfile(filename):
           __loadCorpusBigFromFile(filename)
        elif os.path.isdir(filename):
           __loadCorpusBigFromDir(filename)
        else:
           sys.exit()
    
    def __loadCorpusBigFromFile(self, filename):
        print "big file"
    
    def getLineNum(self, f):
        count = 0
        for index, line in enumerate(f):
            count += 1
        f.seek(0,0)
        return count

    def readOne(self, f):
        labelMaxSize = self.__maxSeqLength * len(y_map['s'])
        label = []

        line = f.readline().strip()
        if not line:
            return None
        for s in line:
            label.extend(self.__y_map[s])
        left = self.__maxSeqLength - len(line)
        if left > 0:
            label.extend(self.__y_map['x'] * left)
        else :
            label = label[0: labelMaxSize]

        line = f.readline().strip()
        if not line:
            return None
               
        line = self.sentence2tokens(line)
             
        _x = self.__padding([line], self.__maxSeqLength)
        return _x[0], lable 
                    
    
    def loadBatch(self, f, num):
        
        _y = []
        _x = []
        y_map = {'s':[1, 0, 0, 0, 0],
                 'b':[0, 1, 0, 0, 0],
                 'm':[0, 0, 1, 0, 0],
                 'e':[0, 0, 0, 1, 0],
                 'x':[0, 0, 0, 0, 1]
             }
        labelMaxSize = self.__maxSeqLength * len(y_map['s'])
        i = 0
        line = f.readline().strip()
        if not line:
            f.seek(0,0)
            line = f.readline().strip()

        while line:
            label = []
            for s in line:
               #if i == 0:
                  #print s 
                  #print y_map[s]
               label.extend(y_map[s])
            #if i == 0:
               #print line
               #print "line size: " + str(len(line))
               #print "label size: %d" %len(label)
            left = self.__maxSeqLength - len(line)
            #if i == 0:
               #print "left:%d " %left
            if left > 0:
                label.extend(y_map['x'] * left)
                #if i == 0:
                #   print "lable size: %d" % len(label)
            else :
                label = label[0: labelMaxSize]
            _y.append(label)
            #if i == 0:
               #print "_y[0] size: " + str(len(_y[0]) ) 
               
            line = f.readline().strip()
            #if i == 0 or i == 10 or i == 100:
            #   print line
            line = self.sentence2tokens(line)
            #if i == 0 or i == 10 or i == 100:
            #   print line
            _x.append(line)
            i += 1
            if i >= num:
                break
            line = f.readline().strip()
            if not line:
                f.seek(0,0)
                line = f.readline().strip()
             

        _x = self.__padding(_x, self.__maxSeqLength)
                    
        #print _x[0]
        #print _y[0]
        #print _x[10]
        #print _x[100]
        #print len(_y[0])
        #print len(_y[10])
        #print len(_y[100])
        #self.__lableSize = len(_y[0])
        return _x, _y 
        
        
    def __loadCorpus(self, filename):
        f = io.open(filename, 'r')
        line = f.readline().strip()
        _y = []
        _x = []
        y_map = {'s':[1, 0, 0, 0, 0],
                 'b':[0, 1, 0, 0, 0],
                 'm':[0, 0, 1, 0, 0],
                 'e':[0, 0, 0, 1, 0],
                 'x':[0, 0, 0, 0, 1]
             }
        '''
        y_map = {'s':[1],
                 'b':[2],
                 'm':[3],
                 'e':[4],
                 'x':[5]
             }
        '''
        labelMaxSize = self.__maxSeqLength * len(y_map['s'])
        print "labelMaxSize: %d" % labelMaxSize
        i = 0
        print "---------------loading corpus----------------------"
        while line:
            label = []
            for s in line:
               if i == 0:
                  print s 
                  print y_map[s]
               label.extend(y_map[s])
            if i == 0:
               print line
               print "line size: " + str(len(line))
               print "label size: %d" %len(label)
            left = self.__maxSeqLength - len(line)
            if i == 0:
               print "left:%d " %left
            if left > 0:
                label.extend(y_map['x'] * left)
                if i == 0:
                   print "lable size: %d" % len(label)
            else :
                label = label[0: labelMaxSize]
            _y.append(label)
            if i == 0:
               print "_y[0] size: " + str(len(_y[0]) ) 
               
            line = f.readline().strip()
            if i == 0 or i == 10 or i == 100:
               print line
            line = self.sentence2tokens(line)
            if i == 0 or i == 10 or i == 100:
               print line
            _x.append(line)
            i += 1
            line = f.readline().strip()
             

        _x = self.__padding(_x, self.__maxSeqLength)
                    
        #print _x[0]
        #print _y[0]
        #print _x[10]
        #print _x[100]
        print len(_y[0])
        print len(_y[10])
        #print len(_y[100])
        self.__lableSize = len(_y[0])
        return _x, _y 
        
    def __padding(self, tokens, max_tokens):
        return pad_sequences(tokens, maxlen=max_tokens, padding='post', truncating='post')


def main():
    _OPT = optparse.OptionParser()
    _OPT.add_option('-m', '--model', action='store', dest='model', help='model file name, load or save')
    _OPT.add_option('-i', '--input', action='store', dest='input', help='input text')
    _OPT.add_option('-v', '--word2vector', action='store', dest='w2v', help='word to vector model')
    _OPT.add_option('-c', '--corpus', action='store', dest='corpus', help='corpus')
    _OPT.add_option('-e', '--epochs', action='store', dest='epochs', help='epochs')
    _OPT.add_option('-E', '--evaluate', action='store', dest='evaluate', help='evaluate')
    _OPT.add_option('-r', '--rate', action='store', dest='rate', help='rate')
    _OPT.add_option('-a', '--auto-classify', action='store', dest='auto_classify_inout', help='out classify infile and outfile. e.g. ./infile,./outfile')
    _OPT.add_option('-I', '--items', action='store', dest='items', help='classify items, e.g. xingjiabi,waiguan,xingneng,zhaoxiang')

    (opts,args) = _OPT.parse_args()
    print opts
    print args
    
    
    clas = SentenceClasscify()
    clas.loadWV(opts.w2v)

    if opts.evaluate:
        clas.loadTestCorpus(opts.evaluate)
        clas.evaluate(opts.model)
    elif opts.input:
        inp = opts.input
        clas.prediction(opts.model, inp)
    else:
        #clas.loadCorpus(opts.corpus)
        f = io.open(opts.corpus, 'r')
        clas.loadBatch(f, 20)
        f.close()
        clas.train(opts.model, opts.corpus)
    
   
if __name__ == '__main__':
    main()
