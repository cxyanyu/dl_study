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
    __tagNum = 0

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
    __sequence_length = None
    
    __name = ""
    '''
    __y_map = {'s':[1, 0, 0, 0, 0],
               'b':[0, 1, 0, 0, 0],
               'm':[0, 0, 1, 0, 0],
               'e':[0, 0, 0, 1, 0],
               'x':[0, 0, 0, 0, 1]
             }
    '''
    __tag_map = {'s':0,
                 'b':1,
                 'm':2,
                 'e':3,
                 'x':4
             }
    __tag_list = ['s', 'b', 'm', 'e', 'x']
    
    __fileLineNum = 0

    def __init__(self, name=""):
        self.__name = name 
        self.__tagNum = 5
        self.__labelSize = self.__maxSeqLength
    
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
        
    def creataModel(self, batchSize, is_train= False):
        #tf.reset_default_graph()
        print "self.__labelSize %d" %self.__labelSize 
        self.__labels = tf.placeholder(tf.int32, [None, self.__maxSeqLength], name='labels')
        self.__input_data = tf.placeholder(tf.int32, [None, self.__maxSeqLength], name='input')
        self.__sequence_length = tf.placeholder(tf.int32, [None], name='seq_length')

        data = tf.zeros([tf.shape(self.__input_data)[0], self.__maxSeqLength, self.__wv_dim])
        data = tf.nn.embedding_lookup(self.__wv_matrix, self.__input_data)

        lstm_qx = tf.contrib.rnn.BasicLSTMCell(self.__lstmUnits, forget_bias = 1.0)
        #lstm_qx = tf.contrib.rnn.DropoutWrapper(cell=lstm_qx, output_keep_prob=0.75)
        lstm_hx = tf.contrib.rnn.BasicLSTMCell(self.__lstmUnits, forget_bias = 1.0)
        #lstm_hx = tf.contrib.rnn.DropoutWrapper(cell=lstm_hx, output_keep_prob=0.75)

        output1, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_qx, lstm_hx, data, sequence_length = self.__sequence_length, dtype = tf.float32)
        outputs1 = [output1[0],output1[1]]
        
        bilstmoutputs = tf.concat(outputs1, axis=-1)
        print "print output1.shape " 
        print np.array(output1).shape


        #lstmCell = tf.contrib.rnn.BasicLSTMCell(self.__tagNum)
        #lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        #value, _ = tf.nn.dynamic_rnn(lstmCell, bilstmoutputs, dtype=tf.float32)


        lstmCell1 = tf.contrib.rnn.BasicLSTMCell(24)
        #lstmCell1 = tf.contrib.rnn.DropoutWrapper(cell=lstmCell1, output_keep_prob=0.75)
        lstmCell2 = tf.contrib.rnn.BasicLSTMCell(24)
        #lstmCell2 = tf.contrib.rnn.DropoutWrapper(cell=lstmCell2, output_keep_prob=0.75)
        lstmCell3 = tf.contrib.rnn.BasicLSTMCell(24)
        #lstmCell3 = tf.contrib.rnn.DropoutWrapper(cell=lstmCell3, output_keep_prob=0.75)
        lstmCell4 = tf.contrib.rnn.BasicLSTMCell(24)
        #lstmCell4 = tf.contrib.rnn.DropoutWrapper(cell=lstmCell4, output_keep_prob=0.75)
        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.__tagNum)
        #lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell2, output_keep_prob=0.75)
        mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstmCell1, lstmCell2, lstmCell3, lstmCell4, lstmCell] )
        value, _ = tf.nn.dynamic_rnn(mlstm_cell, bilstmoutputs, dtype=tf.float32)


        print value.shape
        
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                            value, 
                            self.__labels, 
                            self.__sequence_length)

        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
            value, transition_params, self.__sequence_length)

        self.__loss = tf.reduce_mean(-log_likelihood, name= "loss")
   
        self.__optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.__loss)
 
            
        #self.__prediction = tf.cast(viterbi_sequence, dtype=tf.int32, name='prediction')
        ys = tf.reshape(viterbi_sequence, [-1, self.__maxSeqLength], name='prediction')
        
        mask  = tf.sequence_mask(self.__sequence_length) 
        #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=value, labels=self.__labels)
        #losses = tf.boolean_mask(losses, mask)
        #self.__loss = tf.reduce_mean(losses, name='loss')
        
            
        #self.__loss = tf.reduce_mean(-log_likelihood, name='loss')
        #self.__optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(self.__loss, name='optimizer')
        
        #mask = (np.expand_dims(np.arange(self.__maxSeqLength), axis=0) <    
        #        np.expand_dims(self.__sequence_length, axis=1)) 
        #mask  = tf.sequence_mask(self.__sequence_length)   
        total_labels = tf.cast(tf.reduce_sum(self.__sequence_length), dtype=tf.float32)
        correct_labels = tf.equal(self.__labels, viterbi_sequence)
        correct_labels = tf.boolean_mask(correct_labels, mask)
        correct_labels = tf.cast(correct_labels, dtype=tf.float32)
        correct_labels = tf.reduce_sum(correct_labels)
        self.__acc = tf.div(correct_labels, total_labels, name='accuracy')
        
        #self.__optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.__loss, name='optimizer')
        tf.add_to_collection('optimizer', self.__optimizer)   

        
    def restore(self, sess, model_path, model_name):
        print "restroe from " + model_path

        saver = tf.train.import_meta_graph(model_path + '/' + model_name + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        self.__labels = graph.get_tensor_by_name("labels:0")
        self.__input_data = graph.get_tensor_by_name("input:0")
        self.__sequence_length = graph.get_tensor_by_name('seq_length:0')
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
            print x
            length=[len(e) for e in x]
            y=sess.run(self.__prediction, {self.__input_data: x, self.__sequence_length: length})
            #y = tf.argmax(y, 2)
            y = y.tolist()[0]
            print y
            print x[0]
            print [tag_map[i] for i in y ]
            cut = u""

            for i in range(len(text)):
                if tag_map[y[i]] == 'b':
                  if i > 0 and y[i-1] == 1:
                    print u"纠正..."
                    y[i-1] = 0
                
            for i in range(len(text)):
                cut += text[i]
                cut += u'/'
                cut += tag_map[y[i]]
            print cut
            print cut.replace(u'/s', u' ').replace(u'/e', u' ').replace(u'/m', u'').replace(u'/b', u'').replace(u'/x', u'')  
            text = raw_input("input sentence:").decode(sys.stdin.encoding)

        
        
    def train(self, model_path, infile, batchSize=256):
    
        model_name = os.path.basename(model_path.rstrip('/'))
        tf.reset_default_graph()        
        sess = tf.InteractiveSession()
        
        '''
        if os.path.exists(model_path + '/checkpoint'):
            saver = self.restore(sess, model_path, model_name)
        else:
            self.creataModel(batchSize)
            saver = tf.train.Saver(max_to_keep = 1)
            tf.train.export_meta_graph(filename=model_path + '/' + model_name + '.meta')
            sess.run(tf.global_variables_initializer())
        '''
        
        self.creataModel(batchSize)
        saver = tf.train.Saver(max_to_keep = 1)
        tf.train.export_meta_graph(filename=model_path + '/' + model_name + '.meta')
        if os.path.exists(model_path + '/checkpoint'):
            print "restroe from " + model_path
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        else :
            sess.run(tf.global_variables_initializer())
        #'''
        
        tf.summary.scalar('loss', self.__loss)
        tf.summary.scalar('Accrar', self.__acc)
        
        print tf.summary
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
            next_batch, next_batch_labels, next_seq_length = self.loadBatch(f, batchSize)
            #print next_batch[0]
            #print next_batch_labels[0]
            #print next_seq_length[0]
            #print next_seq_length

            #print np.array(next_batch).shape
            #print np.array(next_batch_labels).shape
            #print np.array(next_seq_length).shape
            #print next_seq_length
            r = sess.run(self.__optimizer, {self.__input_data: next_batch, self.__labels: next_batch_labels, self.__sequence_length: next_seq_length}) 
 
            if(i%50==0):
                summary=sess.run(merged,{self.__input_data: next_batch, self.__labels: next_batch_labels, self.__sequence_length: next_seq_length})
                writer.add_summary(summary,i)

              
                          
            if (i%20==0):
                print ("training %d batch..." %i)
                #next_batch, next_batch_labels = self.get_train_batch2(batchSize)
                loss_, accuracy_= sess.run([self.__loss, self.__acc], {self.__input_data: next_batch, self.__labels: next_batch_labels, self.__sequence_length: next_seq_length})
                #accuracy_=(sess.run(self.__acc, {self.__input_data: next_batch, self.__labels: next_batch_labels})) * 100
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
        _length = []

        labelMaxSize = self.__maxSeqLength
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
               label.extend([self.__tag_map[s]])
            #if i == 0:
               #print line
               #print "line size: " + str(len(line))
               #print "label size: %d" %len(label)
            left = self.__maxSeqLength - len(line)
            #if i == 0:
               #print "left:%d " %left
            if left > 0:
                label.extend([self.__tag_map['x']] * left)
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
        _length=[len(e) for e in _x]
        #print _x[0]
        #print _y[0]
        #print _x[10]
        #print _x[100]
        #print len(_y[0])
        #print len(_y[10])
        #print len(_y[100])
        #self.__labelSize = len(_y[0])
        #print _length
        #print _x
        #print _y
        return _x, _y, _length
        
        
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
        #self.__labelSize = len(_y[0])
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
