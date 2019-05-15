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

class CorpusParse0:
    _file = None
    _sampleNum = None

    def __init__(self, filename, label_names = None):
        self._load(filename)
      
    def _load(self, filename):
        f = io.open(filename, 'r', encoding="utf-8")
        self._file = f
        self._countSampleNum()

    def getLabelSize(self) :
        return None

    def readOne(self):
        line = self._file.readline().strip()
        if not line:
            return None,None
        return None, line

    def readReset(self):
        self._file.seek(0,0)
 
    def getSampleNum(self):
        return self._sampleNum

    def _countSampleNum(self):
        self.readReset()
        self._sampleNum = len([ "" for line in self._file])
        print "sample num %d" %self._sampleNum

class CorpusParse1(object):
    _file = None
    _sampleNum = None
    _label_map = None
    _label_names = None
    _label_size = None

    def __init__(self, filename, label_names):
        self._label_names = label_names
        self._load(filename)
      
      
    def _load(self, filename):
        f = io.open(filename, 'r', encoding="utf-8")
        self._file = f
        self._build_label_map()
        #self._build_label_map(self._getLabelNames())
        self._countSampleNum()
        self.readReset()
        y, line= self.readOne()
        self._label_size = len(y)
        self.readReset()

    def _getLabelNames(self):
        self._file.seek(0,0)
	line1 = self._file.readline().strip()       
        return line1.split(u',')
        
    def _build_label_map(self):
        getLabelNames = getattr(self, '_getLabelNames')
        names = getLabelNames()
        #names = self._getLabelNames()
        print u" ".join(names)
        print u" ".join(self._label_names)
	self._label_map = [names.index(e) for e in self._label_names]
        print self._label_map
        return self._label_map

    def _countSampleNum(self):
        self.readReset()
        self._sampleNum = len([ "" for line in self._file])/2
        print "sample num %d" %self._sampleNum

    def getLabelSize(self) :
        return self._label_size

    def readReset(self):
        self._file.seek(0,0)
        self._file.readline()

    def getSampleNum(self):
        return self._sampleNum

    def readOne(self):
        line = self._file.readline().strip()
        if not line:
            return None,None
            
        strs = line.split(u',')
        _y = [int(s) for s in strs]
        line = self._file.readline().strip() 
           
        if not line:
            return None,None   
        return _y, line
  
class CorpusParse2(CorpusParse1):
    def readOne(self):
        line = self._file.readline().strip()
        if not line:
            return None,None
            
        strs = line.split(u',')
        m = [[0,0,0], [0,0,1], [0,1,0], [1,0,0]]
        _y = [m[int(e)] for e in strs]
        _y = np.array(_y).reshape(-1).tolist() 
        if sum(_y) > 0:
            _y = np.append(_y, 0)
        else :
            _y = np.append(_y, 1)

        line = self._file.readline().strip() 
           
        if not line:
            return None,None   

        return _y, line

class CorpusParse3(CorpusParse1):
    '''
    def __init__(self, filename, label_names):
        CorpusParse1._label_names = label_names
        self._load(filename)
    '''
    '''
    def _load(self, filename):
        f = io.open(filename, 'r', encoding="utf-8")
        CorpusParse1._file = f
        super(CorpusParse3, self)._build_label_map(self._getLabelNames())
        CorpusParse1._countSampleNum()
        CorpusParse1.readReset()
        y, line= CorpusParse1.readOne()
        CorpusParse1._label_size = len(y)
        CorpusParse1.readReset()
    '''
    def _getLabelNames(self):
        self._file.seek(0,0)
    	line1 = self._file.readline().strip()       
        return line1.split(u',')[2:-1]

    def _countSampleNum(self):
        self.readReset()
        self._sampleNum = len([ "" for line in self._file])
        print "sample num %d" %self._sampleNum

    def readOne(self):
        line = self._file.readline().strip()
        if not line:
            return None,None
            
        strs = line.split(u',')
        line = strs[1]
        labels = strs[2:-1]
        m = [[0,0,0], [0,0,1], [0,1,0], [1,0,0]]
        _y = []
        for l in labels:
            if l == '':
                l = '0'
            _y.append(m[int(l)])
        #_y = [m[int(e)] for e in labels]
        #print self._label_map
        #print _y
        #print len(_y)
        _y = [_y[self._label_map[i]] for i in range(len(_y))] 
        _y = np.array(_y).reshape(-1).tolist() 
        if sum(_y) > 0:
            _y = np.append(_y, 0)
        else :
            _y = np.append(_y, 1)
                
        return _y, line

class CorpusParse4:
    _file = None
    _sampleNum = None
    _label_map = None
    _label_names = None
    _label_size = None

    def __init__(self, filename, label_names):
        self._label_names = label_names
        self._load(filename)
      
    def _load(self, filename):
        f = io.open(filename, 'r', encoding="utf-8")
        self._file = f
        self._countSampleNum()
        self.readReset()
        y, line= self.readOne()
        self._label_size = len(y)
        self.readReset()

    def _countSampleNum(self):
        self.readReset()
        self._sampleNum = len([ "" for line in self._file])/2
        print "sample num %d" %self._sampleNum

    def getLabelSize(self) :
        return self._label_size

    def readReset(self):
        self._file.seek(0,0)

    def getSampleNum(self):
        return self._sampleNum

    def readOne(self):
        line = self._file.readline().strip()
        if not line:
            return None,None
            
        strs = line.split(u',')
        strs = [e.split(u':')[1] for e in strs]
        strs = u''.join(strs)
        _y = np.array([int(e) for e in strs])
        if sum(_y) > 0:
            _y = np.append(_y, 0)
        else :
            _y = np.append(_y, 1)
        line = self._file.readline().strip() 
        
        if not line:
            return None,None   
        return _y, line

class CorpusParser:
    __real_parser = None
    __mark_type = None
    
    def __init__(self, filename, label_names):
      
        f = io.open(filename, 'r', encoding="utf-8")
        self.__check_mark_type(f)
        print "mark type is " + str(self.__mark_type)
        f.close()
        tp = self.__mark_type
        if tp == 0:
           self.__real_parser = CorpusParse0(filename, label_names) 
        elif tp == 1:
           self.__real_parser = CorpusParse1(filename, label_names) 
        elif tp == 2:
           self.__real_parser = CorpusParse2(filename, label_names) 
        elif tp == 3:
           self.__real_parser = CorpusParse3(filename, label_names) 
        elif tp == 4:
           self.__real_parser = CorpusParse4(filename, label_names) 
        else:
           self.__real_parser = CorpusParse0(filename, label_names) 
         
    def __check_mark_type(self, f):
        f.seek(0,0)

	line1 = f.readline().strip()       
	line2 = f.readline().strip()       

        tag1 = np.array(line1.split(u','))
        tag2 = np.array(line2.split(u','))
        if (len(tag1) * 3) + 1  == len(tag2):
            try :
		tag2 = tag2.astype(int)
                chk = ((tag2 == 0) | (tag2 == 1)).astype(int)
                if sum(chk) == len(chk):
                    self.__mark_type = 1
                    return self.__mark_type
            except:
                print "0 line2 is not label line"

        tag1 = np.array(line1.split(u','))
        tag2 = np.array(line2.split(u','))
        if len(tag1) == len(tag2):
            try:
                tag2 = tag2.astype(int)
                chk = ((tag2 == 0) | (tag2==1) | (tag2==2)| (tag2==3)).astype(int)
                if sum(chk) == len(chk):
                    self.__mark_type = 2
                    return self.__mark_type
            except:
                print "1 line2 is not label line"

        tag1 = np.array(line1.split(u','))
        tag2 = np.array(line2.split(u','))
        chk = np.array([ len(e.split(u':')) == 2 for e in tag1]).astype(int)
        print chk
        if sum(chk) == len(chk):
            self.__mark_type = 4
            return self.__mark_type

        tag1 = np.array(line1.split(u','))
        tag2 = np.array(line2.split(u','))
        if len(tag1) == len(tag2):
            print (tag1[0], tag1[1], tag2[0])
            #if int(tag1[0]) == 0 and tag1[1] == u'' and tag2[0] == u'1':
            if tag1[1] == u'':
                self.__mark_type = 3
                return self.__mark_type
        
        self.__mark_type = 0
        return self.__mark_type

    def readReset(self):
        self.__real_parser.readReset()

    def getLabelSize(self) :
        return self.__real_parser.getLabelSize()

    def getSampleNum(self):
        return self.__real_parser.getSampleNum()

    def readOne(self):
        return self.__real_parser.readOne()

class Corpus:
    __x = None
    __y = None
    __wv = None
    __batchSize = 128
    __sampleNum = 0
    __seqMaxLen = None
    __label_size = None
    __label_names = None
    __parser = None
 
    def __init__(self, wv=None, filename = None, has_mark = False, label_names = None, batchSize = 128, seqMaxLen = 64):
        self.__wv = wv
        self.__batchSize = batchSize
        self.__seqMaxLen = seqMaxLen

        if filename:
            self.__parser = CorpusParser(filename, label_names)
            self.__label_size = self.__parser.getLabelSize()
            self.__sampleNum = self.__parser.getSampleNum()
 

        self.__label_names = label_names
    
  
    def readOne(self):
        return self.__parser.readOne()

    def __readreset(self):
        return self.__parser.readReset()

    def readOneWithTokens(self):
         _y, line = self.readOne()
         if line == None:
             return None, None, None
         _x = self.sentence2tokens(line)    
         return _x, _y, line 

    def formatOutput(self, outfile):
        f = io.open(outfile, 'w')
        f.write(u','.join(self.__label_names))
        f.write(u'\n')
        y, line = self.readOne()
        while line != None:
            y = [str(e) for e in y]
            f.write(u','.join(y))
            f.write(u'\n')
            f.write(line)
            f.write(u'\n')
            y, line = self.readOne()
        f.close()
    
    def readall(self):
        self.__readreset()
        _x = []
        _y = []
        _line = []
        x, y, line = self.readOneWithTokens()
        while x != None:
            _x.append(x)
            _y.append(y)
            _line.append(line)
            x, y, line = self.readOneWithTokens()
        return np.array(_x), np.array(_y), np.array(_line)
        
    def read(self, num, loop = False):
        _x = []
        _y = []
        _line = []
        for i in xrange(num) :  
            x, y, line  = self.readOneWithTokens()
            if not x:
                if not loop:
                    break
                else:
                    self.__readreset()
                    x, y, line = self.readOneWithTokens()
            if not x:
                raise RuntimeError('corpus maybe empty!')
                
            _x.append(x)
            _y.append(y)  
            _line.append(line)
        return np.array(_x), np.array(_y), np.array(_line)
        
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
        _x, _y, _line = self.read(self.__batchSize)
        if len(_x) == 0:
            self.__readreset()
            _x, _y, _line = self.read(self.__batchSize)
        return _x, _y, _line
        
    def getSteps(self):
        print "getSteps, batchsize: " + str(self.__batchSize)
        steps =  int(math.ceil(self.__sampleNum*1.0/self.__batchSize))
        print "steps: " + str(steps)
        return steps
 
    def setBatchSize(self, size):
        self.__batchSize = size
        print "setBatchSize, batchsize: " + str(self.__batchSize)
    
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

    def predictOutput(self, outfile, predict, label_names):
        f = io.open(outfile, 'w')
        steps = self.getSteps()        
        for i in range(steps):
            x, y, lines = self.next()
            result = predict.predict(x=x)
            y = []
            for res in result:
                #c_format = ['{0:.2f}'.format(c) for c in res]
                c_format = []
                for c in res:
                    if c > 0.5:
                        c_format.append('1')
                    else:
                        c_format.append('0')
                y.append(c_format)
                #print 'output= %s' %c_format
            for e in zip(y, lines):
                label = np.array(e[0])
                label = label[0:-1].reshape(-1,3)
                str_label = []
                for l in zip(label_names, label):
                    str_label.append(l[0] + u":" + u''.join(l[1]))
                f.write(u','.join(str_label))
                #f.write(u','.join(e[0]))
                
                f.write(u'\n')
                f.write(e[1])
                f.write(u'\n')
                f.write(u'----------------\n')
            print "-------------------------"
        f.close() 

        
        
def data_generate(corpus):
    while 1:
        x, y, lines = corpus.next()
        yield (x,y)
        
def data_generate_prediction(corpus):
    while 1:
        x, y, lines = corpus.next()
        yield x
        
class Model:
    __model = None 
    __wv = None
    __save_path = None
    __lr = 0.001
    __min_lr = 0.00001
    __train_cbk = None
    
    def __init__(self, wv, model_pathname, seqLength = None, label_size = None, lr = 0.001, min_lr = 0.00001):
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
        model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(LSTM(units=48, return_sequences=False))

        #model.add(Bidirectional(GRU(units=32, return_sequences=True)))
        #model.add(Bidirectional(GRU(units=64, return_sequences=True)))
        #model.add(GRU(units=64, return_sequences=True))
        #model.add(GRU(units=48, return_sequences=False))

        model.add(Dense(label_size, activation='sigmoid', name="output"))
        optimizer = Adam(lr=self.__lr)


        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
                    
        print model.summary()

        try:
            model.load_weights(self.__save_path)
            print "load weights"
        except Exception as e:
            print(e)

        return model
        
    def train(self, train_corpus, test_corpus = None, epochs = 20, batch_size = 128):

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

        validation_data = None
        if test_corpus:
            validation_data = test_corpus.readall()
            validation_data = validation_data[0:-1]

        print "steps: " + str(train_corpus.getSteps()) + ", epochs: " + str(epochs)
        train_corpus.setBatchSize(batch_size)
        self.__model.fit_generator(data_generate(train_corpus), 
                        steps_per_epoch=train_corpus.getSteps(),
                        epochs=epochs, 
                        callbacks=self.__train_cbk, 
                        validation_data=validation_data, 
                        shuffle=True)

                        
    def evaluate(self, corpus, batch_size = 128):
        corpus.setBatchSize(batch_size)
        result = self.__model.evaluate_generator(data_generate(corpus), 
                            steps=corpus.getSteps())  
        print result
        
    def predict(self, x):
        return self.__model.predict(x=x)
    '''
    def predict(self, corpus, outfile, batch_size = 128):
        corpus.setBatchSize(batch_size)
        
        steps = corpus.getSteps()        
        for i in range(steps):
            x, y, lines = corpus.next()
            result = self.__model.predict(x=x)
            y = []
            for res in result:
                c_format = ['{0:.2f}'.format(c) for c in res]
                print 'output= %s' %c_format
               
            print "-------------------------"

        #result = self.__model.predict_generator(data_generate(corpus), 
                                steps=corpus.getSteps()) 
        #print result 
        #for res in result:
        #    c_format = []
        #    for c in res:
        #        c_format.append('{0:.2f}'.format(c))
        #    print 'output= %s' %c_format
    '''

    def predict_one(self, inp, items = None):
        tokens = self.__wv.sentence2tokens(inp)
        print "tokens len: " + str(len(tokens))
        cps = Corpus(self.__wv, seqMaxLen=len(tokens))
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
    _OPT.add_option('-p', '--predict', action='store', dest='predict', help='predict, infile and outfile. e.g. -p infile,outfile')
    #_OPT.add_option('-I', '--items', action='store', dest='items', help='classify items, e.g. xingjiabi,waiguan,xingneng,zhaoxiang')
    _OPT.add_option('-l', '--seqMaxLen', action='store', dest='seqMaxLen', help='set max sequece length')
    _OPT.add_option('-f', '--format', action='store', dest='format', help='format mark corpus file. e.g.  -f infile,outfile')

    label_names = [u'材质做工', u'操作体验', u'充电体验', u'防水防尘',
                   u'降噪效果', u'蓝牙性能', u'佩戴体验', u'售后服务',
                   u'外观设计', u'物流服务', u'性价比', u'续航能力', u'音质效果']

    (opts,args) = _OPT.parse_args()
    print opts
    print args
    
    if opts.format:
        files = opts.format.split(',')
        infile = files[0]
        outfile = files[1]
        cps = Corpus(filename = infile, has_mark = True, label_names = label_names)
        cps.formatOutput(outfile)
        return

    
    epochs = 20
    start_rate=1e-3
    min_rate=1e-5
    seqMaxLen = 64
    if opts.rate:
        rates = opts.rate.split(',')
        start_rate = float(rates[0])
        min_rate = float(rates[1])

    if opts.epochs:
        epochs = int(opts.epochs)        
    
    if opts.seqMaxLen:
        seqMaxLen = int(opts.seqMaxLen)        

    wv = WV()
    wv.load(opts.w2v)
    
    model = None
    
    if opts.train: 
        print "learn rate, " + str(start_rate) + ", " + str(min_rate)
        train_cps = Corpus(wv, opts.train, has_mark = True, seqMaxLen=seqMaxLen, label_names = label_names)
        label_size = train_cps.getLabelSize()
        print "label_size %d" %label_size
        
        val_cps = None
        if opts.validation:
            val_cps = Corpus(wv, opts.validation, has_mark = True, seqMaxLen=seqMaxLen, label_names = label_names)
            
        model = Model(wv, opts.model_filename, label_size = label_size, lr = start_rate, min_lr = min_rate)
        model.train(train_cps, test_corpus = val_cps, epochs = epochs)
    
    if opts.evaluate:
        eval_cps = Corpus(wv, opts.evaluate, batchSize = 2, has_mark = True, seqMaxLen=seqMaxLen, label_names = label_names)
        if not model:
            model = Model(wv, opts.model_filename)
        model.evaluate(eval_cps)
        
    if opts.predict:
        files = opts.predict.split(u",")
        predict_cps = Corpus(wv, files[0], seqMaxLen=seqMaxLen)
        if not model:
            model = Model(wv, opts.model_filename)
        #model.predict(predict_cps, files[1])
        predict_cps.predictOutput(files[1], model, label_names)
            
    if opts.input:
       inp = opts.input
       if not model:
            model = Model(wv, opts.model_filename)
       model.predict_one(inp, label_names)
       model.input_predict(label_names)
       
   
if __name__ == '__main__':
    main()
