#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import re
import jieba 
import os
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import io
import optparse
import sys
import math


        
class Segment:
    def cut(self, sentence):
        return [e for e in jieba.cut(sentence)]
        
class WV:
    __wv = None
    __wv_dim = 0
    __wv_matrix = None
    __wordlist = None
    __wordmap = None
    
    def __init__(self):
        __wv_dim = 0
    
    def load(self, filename, max_word_num = 0):
        try:
            wv_model = KeyedVectors.load(filename, mmap='r')
        except:
            try:
                wv_model = KeyedVectors.load_word2vec_format(filename,binary=True)
            except:
                wv_model = KeyedVectors.load_word2vec_format(filename,binary=False)
                
        wv = wv_model.wv

        max_word_num = max_word_num
        if max_word_num == 0 or max_word_num > len(wv.vocab):
            max_word_num = len(wv.vocab)
  
  
        self.__wv_dim = wv[wv.index2word[0]].shape[0]
        self.__wv_matrix = np.zeros((max_word_num + 1, self.__wv_dim))
        self.__wv_matrix[0,:] = [0] * self.__wv_dim
        self.__wordlist = ["*"]# unknown
        self.__wordmap = {"*":0}
        for i in range(max_word_num):
            self.__wordlist.append(wv.index2word[i])
            self.__wordmap[wv.index2word[i]] = i + 1
            self.__wv_matrix[i+1,:] = wv[wv.index2word[i]]
            
        self.__wv_matrix = self.__wv_matrix.astype('float32')
        
        print np.sum(wv[wv.index2word[22]] == self.__wv_matrix[23] )
        
        print wv.index2word[10]
        print wv[wv.index2word[10]]
        print self.__wordmap[wv.index2word[10]]
        print self.__wv_matrix[10+1]
        print self.__wordlist[10+1]
        
        print self.__wv_matrix.shape
        
    def word2idx(self, word):
        return self.__wordmap[word]
        
    def idx2word(self, idx):
        return self.__wordlist[idx]
        
    def idx2vector(self, idx):
        return self.__wv_matrix[idx]
        
    def getVectrDim(self):
        return self.__wv_dim
        
    def getSize(self):
        return len(self.__wordlist)
        
    def getMarix(self):
        return self.__wv_matrix;  
    
    def tokens2sentence(self, tokens):
        text = "".join([self.idx2word(idx) for idx in tokens])
        return text

    def sentence2tokens(self, sentence):
        seg = Segment()
        cut_list = seg.cut(sentence)
        tokens = []
        for i, word in enumerate(cut_list):
            try:
                tokens.append(self.word2idx(word))
            except KeyError:
                tokens.append(0)
        return tokens

class Corpus:
    __x = None
    __y = None
    __wv = None
    __items = None
    __batchSize = 128
    __sampleNum = 0
    __seqMaxLen = 96
    __label_size = None
    __file = None
    
    def __init__(self, wv, filename = None, has_mark = False, batchSize = 128, seqMaxLen = 96):
        self.__wv = wv
        self.__batchSize = batchSize
        self.__seqMaxLen = seqMaxLen
        if filename:
            self.load(filename, has_mark)
        
    def load(self, filename, has_mark = False):
        f = io.open(filename, 'r')
        self.__file = f
        if has_mark:
            self.__sampleNum = (len([ "" for line in f]) - 1)/2
            f.seek(0,0)
            f.readline()
            x, y = self.__readOne()
            self.__label_size = len(y)
            f.seek(0,0)
            f.readline()
            print line
            self.__items = line.split(u',')
        else:
            self.__sampleNum = len([ "" for line in f])
            f.seek(0,0)
  
    def __readreset(self):
        self.__file.seek(0,0)
        self.__file.readline()
  
    def __readOne(self):
        line = self.__file.readline().strip()
        if not line:
            return None,None
            
        strs = line.split(u',')
        _y = [int(s) for s in strs]

        line = self.__file.readline().strip() 
        if not line:
            return None,None   
        _x = self.sentence2tokens(line)    
          
        return _x, _y
        
    def readall(self):
        self.__readreset()
        _x = []
        _y = []
        x, y = self.__readOne()
        while x and y:
            _x.append(x)
            _y.append(y)
            x, y = self.__readOne()
        return np.array(_x), np.array(_y)
        
        
    def read(self, num, loop = False):
        _x = []
        _y = []
        for i in xrange(num) :  
            x, y = self.__readOne()
            if not x:
                if not loop:
                    break
                else:
                    self.__readreset()
                    x, y = self.__readOne()
            if not x:
                raise RuntimeError('corpus maybe empty!')
                
            _x.append(x)
            _y.append(y)  
        return np.array(_x), np.array(_y)
        
    def getLabelSize(self) :
        return self.__label_size

    def sentence2tokens(self, texts):
        if type(texts) == type([]):
            return [self.__sentence2tokens(tx) for tx in texts]
        else :
            return self.__sentence2tokens(texts)

    def __iter__(self): 
       return self 
       
    def next(self):
        _x, _y = self.read(self.__batchSize)
        if len(_x) == 0:
            self.__readreset()
            _x, _y = self.read(self.__batchSize)
        return _x, _y
        
    def getSteps(self):
        return math.ceil(self.__sampleNum*1.0/self.__batchSize)
    
    def setBatchSize(self, size):
        self.__batchSize = size
    
    def padding(self, tokens):
        if len(tokens) > self.__seqMaxLen:
            tokens = tokens[0: self.__seqMaxLen]
        elif len(tokens) < self.__seqMaxLen:
            padnum = self.__seqMaxLen - len(tokens)
            tokens = [0] * padnum + tokens
        return tokens
        
    def __sentence2tokens(self, sentence):
        tokens = self.__wv.sentence2tokens(sentence)
        return self.padding(tokens)
        
        
class Model:
    __model = None 
    __wv = None
    __save_path = None
    __lr = 0.001
    __min_lr = 0.00001
    __train_cbk = None
    
    def __init__(self, wv, model_pathname, seqLength = 96, label_size = None, lr = 0.001, min_lr = 0.00001):
        self.__wv = wv
        self.__save_path = model_pathname
        self.__lr = lr
        self.__min_lr = lr
        
        if os.path.exists(model_pathname):
            try:
                self.__model = load_model(model_pathname)
                print "load model!"
            except :
                print "load model failed!"
        
        if not self.__model:
            self.__model = self.__create(seqLength, label_size)
        
    def __create(self, seqLength, label_size):
        print "create mode.........."
        print "lr %f, min_lr %f" %(self.__lr, self.__min_lr)
        
        print self.__wv.getSize()
        print self.__wv.getVectrDim()
        
        model = Sequential()
        print self.__wv.getSize()
        print self.__wv.getVectrDim()
        print self.__wv.getMarix()[10]
        
        model.add(Embedding(self.__wv.getSize(),
                            self.__wv.getVectrDim(),
                            weights=[self.__wv.getMarix()],
                            input_length=seqLength,
                            trainable=False, name="embedding_1"))
        model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
        model.add(LSTM(units=16, return_sequences=False))
        model.add(Dense(label_size, activation='sigmoid', name="output"))
        optimizer = Adam(lr=self.__lr)


        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
                    
        earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
  
                      
        lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.1, 
                                         min_lr= self.__min_lr, 
                                         patience=0,
                                         verbose=1)

        checkpoint = ModelCheckpoint(filepath=self.__save_path, monitor='val_loss',
                                              verbose=1, save_weights_only=False,
                                              save_best_only=True)

        self.__train_cbk = [
            earlystopping, 
            checkpoint,
            lr_reduction
        ]


        print model.summary()

        try:
            model.load_weights(self.__save_path)
        except Exception as e:
            print(e)

        return model
        
        
    def train(self, train_corpus, test_corpus = None, epochs = 20, batch_size = 128):

        '''
        X_train, Y_train = train_corpus.readall()
        self.__model.fit(X_train, Y_train,
                  validation_split=0.1, 
                  epochs=epochs,
                  batch_size=128,
                  shuffle=True,
                  callbacks=self.__train_cbk)

        '''
        
        validation_data = None
        if test_corpus:
            validation_data= test_corpus.readall()

        print "steps: " + str(train_corpus.getSteps()) + ", epochs: " + str(epochs)
        train_corpus.setBatchSize(batch_size)
        self.__model.fit_generator(train_corpus, 
                        steps_per_epoch=train_corpus.getSteps(),
                        epochs=epochs, 
                        callbacks=self.__train_cbk, 
                        validation_data=validation_data, 
                        shuffle=True)

                        
    def evaluate(self, corpus, batch_size = 128):
        corpus.setBatchSize(batch_size)
        result = self.__model.evaluate_generator(corpus, 
                            steps=corpus.getSteps())  
        print result
        
    def predict(self, corpus, batch_size = 128):
        corpus.setBatchSize(batch_size)
        result = self.__model.predict_generator( corpus, 
                                steps=corpus.getSteps()) 
        
        
    def predict_one(self, inp, items = None):
        cps = Corpus(self.__wv)
        tokens = cps.sentence2tokens(inp)
        print inp
        print tokens
        result = self.__model.predict(x=np.array([tokens]))
        print result
        coef = result[0]
        c_format = []
        for c in coef:
            c_format.append('{0:.2f}'.format(c))
        print 'output= %s' %c_format

        tags = np.array(c_format[0:-1])
        tags = tags.reshape(-1,3)
        
        for i in range(len(tags)):
            if items:
                print "%s: %s" %(items[i], tags[i])
            else :
                print "%s" %(tags[i])

        print "other: %s" % c_format[-1]
        
    
    def input_predict(self,items = None):
        while True:
            print "-------------------------------------------------------"
            text = raw_input("input sentence:").decode(sys.stdin.encoding)
            self.predict_one(text, items)




def main():
    _OPT = optparse.OptionParser()
    _OPT.add_option('-m', '--model-filename', action='store', dest='model_filename', help='model file name, load or save')
    _OPT.add_option('-i', '--input', action='store', dest='input', help='input text')
    _OPT.add_option('-v', '--word2vector', action='store', dest='w2v', help='word to vector model')
    _OPT.add_option('-t', '--train', action='store', dest='train', help='tain')
    _OPT.add_option('-V', '--validation', action='store', dest='validation', help='Validation')
    _OPT.add_option('-e', '--epochs', action='store', dest='epochs', help='epochs')
    _OPT.add_option('-E', '--evaluate', action='store', dest='evaluate', help='evaluate')
    _OPT.add_option('-r', '--rate', action='store', dest='rate', help='rate')
    _OPT.add_option('-p', '--predict', action='store', dest='predict', help='predict, infile and outfile. e.g. ./infile,./outfile')
    _OPT.add_option('-I', '--items', action='store', dest='items', help='classify items, e.g. xingjiabi,waiguan,xingneng,zhaoxiang')

    (opts,args) = _OPT.parse_args()
    print opts
    print args
    
    items = None
    if opts.items:
        items = opts.items.split(",")
        print items
        
    epochs = 20
    start_rate=1e-3
    min_rate=1e-5

    if opts.rate:
        rates = opts.rate.split(',')
        start_rate = float(rates[0])
        min_rate = float(rates[1])

    if opts.epochs:
        epochs = int(opts.epochs)        
    
    wv = WV()
    wv.load(opts.w2v)
    
    model = None
    
    if opts.train: 
        print "learn rate, " + str(start_rate) + ", " + str(min_rate)
        train_cps = Corpus(wv, opts.train, has_mark = True)
        label_size = train_cps.getLabelSize()
        print "label_size %d" %label_size
        
        val_cps = None
        if opts.validation:
            val_cps = Corpus(wv, opts.validation, has_mark = True)
            
        model = Model(wv, opts.model_filename, label_size = label_size, lr = start_rate, min_lr = min_rate)
        model.train(train_cps, test_corpus = val_cps)
    
    if opts.evaluate:
        eval_cps = Corpus(wv, opts.evaluate)
        if not model:
            model = Model(wv, opts.model_filename)
        model.evaluate(eval_cps)
        
    if opts.predict:
        files = opts.predict.split(u",")
        predict_cps = Corpus(wv, files[0])
        if not model:
            model = Model(wv, opts.model_filename)
        model.predict(predict_cps, files[1])
            
    if opts.input:
       inp = opts.input
       if not model:
            model = Model(wv, opts.model_filename)
       model.predict_one(inp, items)
       model.input_predict(items)
       
   
if __name__ == '__main__':
    main()
