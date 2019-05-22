#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import re
import os
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from gensim.models import KeyedVectors
import io
import optparse
import sys
import math
import time
import datetime
import random


        
class Segment:
    def cut(self, sentence):
        return [e for e in sentence]
        
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
        
    def getMatrix(self):
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

class CorpusParser(object):
    _file = None
    _sampleNum = None
    _label_map = None
    _label_size = None

    __tag_map = {'s':0,
                 'b':1,
                 'm':2,
                 'e':3,
                 'x':4
             }
    def __init__(self, filename):
        self._load(filename)
      
      
    def _load(self, filename):
        f = io.open(filename, 'r', encoding="utf-8")
        self._file = f
        print "count sample Num..."
        self._countSampleNum()
        print "sample num is %d" % self._sampleNum
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
            
        label = [self.__tag_map[s] for s in line]

        line = self._file.readline().strip() 
           
        if not line:
            return None,None   
        return label, line
      
    def labelPadding(self, label, maxLen):
        if len(label) >= maxLen:
            label = label[0:maxLen]
        else:
            label.extend([self.__tag_map['x']] * (maxLen-len(label)))
        return label


class Corpus:
    __x = None
    __y = None
    __wv = None
    __batchSize = 128
    __sampleNum = 0
    __seqMaxLen = None
    __label_size = None
    __parser = None
 
    def __init__(self, wv=None, filename = None, has_mark = False, batchSize = 128, seqMaxLen = 32):
        self.__wv = wv
        self.__batchSize = batchSize
        self.__seqMaxLen = seqMaxLen

        if filename:
            self.__parser = CorpusParser(filename)
            self.__label_size = self.__parser.getLabelSize()
            self.__sampleNum = self.__parser.getSampleNum()
    
  
    def readOne(self):
        return self.__parser.readOne()

    def __readreset(self):
        return self.__parser.readReset()

    def readOneWithTokens(self):
         _y, line = self.readOne()
         if line == None:
             return None, None, None, None
         _len, _x = self.sentence2tokens(line)    
         
         return _len, _x, _y, line 

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
        '''
        _x = []
        _y = []
        _line = []
        x, y, line = self.readOneWithTokens()
        while x != None:
            _x.append(x)
            _y.append(y)
            _line.append(line)
            x, y, line = self.readOneWithTokens()
        lengths = [len(l) for l in _line]
        _y = [self.__parser.labelPadding(y, self.__seqMaxLen) for y in _y]
        return np.array(_x), np.array(_y), np.array(_line)
        '''
        return self.read(self.__parser.getSampleNum())
        
    def read(self, num, loop = False):
        _x = []
        _y = []
        _line = []
        _len = []
        for i in xrange(num) :  
            length, x, y, line  = self.readOneWithTokens()
            if not x:
                if not loop:
                    break
                else:
                    self.__readreset()
                    length, x, y, line = self.readOneWithTokens()
            if not x:
                raise RuntimeError('corpus maybe empty!')
                
            _x.append(x)
            _y.append(y)  
            _line.append(line)
            _len.append(length)

        #lengths = [min(len(l), self.__seqMaxLen) for l in _line]

        #lengths = [len(e) for e in _x]
        
        _y = [self.__parser.labelPadding(y, self.__seqMaxLen) for y in _y]
        return np.array(_x), np.array(_y), np.array(_line), np.array(_len)
        
    def getLabelSize(self) :
        return self.__label_size

    def __iter__(self): 
       return self 
       
    def next(self):
        _x, _y, _line, _len = self.read(self.__batchSize)
        if len(_x) == 0:
            self.__readreset()
            _x, _y, _line, _len = self.read(self.__batchSize)
        return _x, _y, _line, _len
        
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
            tokens = tokens + [0] * padnum
        return tokens
        
    def sentence2tokens(self, sentence):
        tokens = self.__wv.sentence2tokens(sentence)
        return min(len(tokens), self.__seqMaxLen), self.padding(tokens)

    def predictOutput(self, outfile, predict):
        f = io.open(outfile, 'w')
        steps = self.getSteps()        
        for i in range(steps):
            x, y, lines, lens = self.next()
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
                #for l in zip(label_names, label):
                #    str_label.append(l[0] + u":" + u''.join(l[1]))
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
    __maxSeqLength = 32
    __lstmUnits = 48
    __lstmUnits2 = 24

    __wv = None
    __save_path = None
    __lr = 0.001
    __min_lr = 0.00001
    __model_name = None

    __sess = None
    __saver = None
    __labels = None
    __input_data = None
    __sequence_length = None
    __loss = None 
    __acc = None
    __prediction = None
    __optimizer = None

    def __init__(self, wv, model_filename, seqLength = None, label_size = None, lr = 0.001, min_lr = 0.00001):
        self.__wv = wv
        self.__save_path = model_filename
        self.__lr = lr
        self.__min_lr = lr

        self.__model_name = os.path.basename(model_filename.rstrip('/'))
        tf.reset_default_graph()        
        self.__sess = tf.InteractiveSession()

        if self.is_model_file_exit(model_filename):
            #try:
                self.load_model(model_filename)
                print "load model!"
            #except :
            #    print "load model failed!"

        if self.__prediction == None:
            self.createModel(label_size)
            tf.train.export_meta_graph(filename=model_filename + '/' + self.__model_name + '.meta')
            self.__saver = tf.train.Saver(max_to_keep = 1)
        
        tf.summary.scalar('loss', self.__loss)
        tf.summary.scalar('Accrar', self.__acc)

    def is_model_file_exit(self, path):
        print path
        if os.path.exists(path + '/checkpoint'):
            print path + '/checkpoint'
            print "model file exit"
            return True
        else:
            print "model file not exit"
            return False
      

    def load_model(self, model_filename):
        print "restroe from " + model_filename
        self.__saver = tf.train.import_meta_graph(model_filename + '/' + self.__model_name + '.meta')
        self.__saver.restore(self.__sess, tf.train.latest_checkpoint(model_filename))
        graph = tf.get_default_graph()
        self.__labels = graph.get_tensor_by_name("labels:0")
        self.__input_data = graph.get_tensor_by_name("input:0")
        self.__sequence_length = graph.get_tensor_by_name('seq_length:0')
        self.__loss = graph.get_tensor_by_name("loss:0")
        self.__acc = graph.get_tensor_by_name("accuracy:0")
        self.__prediction = graph.get_tensor_by_name("prediction:0")
        self.__optimizer = tf.get_collection('optimizer')[0]
        graph_def = tf.GraphDef()
        for i,n in enumerate(graph_def.node):
	    print("Name of the node - %s" % n.name)

    def createModel(self, tagNum, is_train= False):
        print "create mode.........."
        print "lr %f, min_lr %f" %(self.__lr, self.__min_lr)
        print "word2vector size %d, vector dim %d" %(self.__wv.getSize(), self.__wv.getVectrDim())
        print "tagNum %d" %tagNum 

        self.__labels = tf.placeholder(tf.int32, [None, self.__maxSeqLength], name='labels')
        self.__input_data = tf.placeholder(tf.int32, [None, self.__maxSeqLength], name='input')
        self.__sequence_length = tf.placeholder(tf.int32, [None], name='seq_length')

        data = tf.zeros([tf.shape(self.__input_data)[0], self.__maxSeqLength, self.__wv.getVectrDim()])
        data = tf.nn.embedding_lookup(self.__wv.getMatrix(), self.__input_data)

        def lstm(units, output_keep_prob = 1.0, input_keep_prob = 1.0):
            cell = tf.contrib.rnn.BasicLSTMCell(units, forget_bias = 1.0)
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, 
                output_keep_prob=output_keep_prob,
                input_keep_prob=input_keep_prob)
            return cell
        
        fw1 = lstm(32)            
        fw2 = lstm(64)            
        bw1 = lstm(32)            
        bw2 = lstm(64)            
        fws = tf.contrib.rnn.MultiRNNCell([fw1, fw2])
        bws = tf.contrib.rnn.MultiRNNCell([bw1, bw2])

        output, output_states = tf.nn.bidirectional_dynamic_rnn(
                          fws, bws, data, sequence_length = self.__sequence_length, dtype = tf.float32)
        output = tf.concat([output[0], output[1]], axis=-1)

        cells = [lstm(64), lstm(64)]
        cells = tf.contrib.rnn.MultiRNNCell(cells)
        value, _ = tf.nn.dynamic_rnn(cells, output, dtype=tf.float32)

        print value.shape
        
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                            value, 
                            self.__labels, 
                            self.__sequence_length)

        self.__loss = tf.reduce_mean(-log_likelihood, name= "loss")
        self.__optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.__loss)
        tf.add_to_collection('optimizer', self.__optimizer)   
 
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
            value, transition_params, self.__sequence_length)

        self.__prediction = tf.reshape(viterbi_sequence, [-1, self.__maxSeqLength], name='prediction')
        
        mask  = tf.sequence_mask(self.__sequence_length) 
        total_labels = tf.cast(tf.reduce_sum(self.__sequence_length), dtype=tf.float32)
        correct_labels = tf.equal(self.__labels, viterbi_sequence)
        correct_labels = tf.boolean_mask(correct_labels, mask)
        correct_labels = tf.cast(correct_labels, dtype=tf.float32)
        correct_labels = tf.reduce_sum(correct_labels)
        self.__acc = tf.div(correct_labels, total_labels, name='accuracy')
        self.__sess.run(tf.global_variables_initializer())

        #for n in tf.get_default_graph().as_graph_def().node
        #    print n.name
        graph_def = tf.GraphDef()
        for i,n in enumerate(graph_def.node):
	    print("Name of the node - %s" % n.name)

    def train(self, corpus, test_corpus = None, epochs = 20, batch_size = 128):
        print tf.summary
        merged=tf.summary.merge_all()
        logdir='tensorboard/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"
        writer=tf.summary.FileWriter(logdir, self.__sess.graph)

        corpus.setBatchSize(batch_size)
        for i in xrange(epochs):
            self.__train(corpus, test_corpus, i, merged, writer)


    def __train(self, corpus, test_corpus, loop, merged, writer):
        steps = corpus.getSteps()
        print('..........')
        for i in range(steps):
            next_batch, next_batch_labels, next_lines, next_seq_length = corpus.next()
            loss_, accuracy_, r= self.__sess.run([self.__loss, self.__acc, self.__optimizer], 
                                                 {self.__input_data: next_batch, 
                                                  self.__labels: next_batch_labels, 
                                                  self.__sequence_length: next_seq_length})
            if(i%100==0):
                summary=self.__sess.run(merged, {self.__input_data: next_batch, 
                                                 self.__labels: next_batch_labels, 
                                                 self.__sequence_length: next_seq_length})
                writer.add_summary(summary,i)

            if (i%20==0):
                print ("training %d batch..." %i)
                print("epochs:{}, steps:{}/{}".format(loop, i, steps),
                          "loss:{}".format(loss_),
                          "accuracy:{}".format(accuracy_))    
                print('..........')  

            if(i%100==0 and i!=0):
                save_path = self.__saver.save(self.__sess, self.__save_path + "/" + self.__model_name, global_step=i)
                print("saved to %s"% save_path)
        
    def evaluate(self, corpus):
        X, Y, lines, lens = corpus.readall()
        accuracy_=(self.__sess.run(self.__acc, {self.__input_data: X, self.__labels:Y, self.__sequence_length: lens})) * 100
        print("accuracy:{}".format(accuracy_))    
        
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

    def predict_one(self, inp):
        tag_map = ['s', 'b', 'm', 'e', 'x']
        cps = Corpus(self.__wv)
        length, tokens = cps.sentence2tokens(inp)
        
        print "tokens len: " + str(length)
        print inp
        print tokens

        y = self.__sess.run(self.__prediction, {self.__input_data: [tokens], self.__sequence_length: [length]})
        y = y.tolist()[0]
        print y
        print tokens[0]
        print [tag_map[i] for i in y ]
        cut = u""

        for i in range(len(inp)):
            if tag_map[y[i]] == 'b':
                if i > 0 and y[i-1] == 1:
                    print u"纠正..."
                    y[i-1] = 0

        for i in range(len(inp)):
            cut += inp[i]
            cut += u'/'
            cut += tag_map[y[i]]
        print cut
        print cut.replace(u'/s', u' ').replace(u'/e', u' ').replace(u'/m', u'').replace(u'/b', u'').replace(u'/x', u'')

    
    def input_predict(self,items = None):
        while True:
            print "-------------------------------------------------------"
            text = raw_input("input sentence:").decode(sys.stdin.encoding)
            self.predict_one(text)


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

    (opts,args) = _OPT.parse_args()
    print opts
    print args
    
    if opts.format:
        files = opts.format.split(',')
        infile = files[0]
        outfile = files[1]
        cps = Corpus(filename = infile, has_mark = True)
        cps.formatOutput(outfile)
        return

    
    epochs = 20
    start_rate=1e-3
    min_rate=1e-5
    seqMaxLen = 32
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
        train_cps = Corpus(wv, opts.train, has_mark = True, seqMaxLen=seqMaxLen)
        label_size = train_cps.getLabelSize()
        print "label_size %d" %label_size
        
        val_cps = None
        if opts.validation:
            val_cps = Corpus(wv, opts.validation, has_mark = True, seqMaxLen=seqMaxLen)
            
        model = Model(wv, opts.model_filename, label_size = label_size, lr = start_rate, min_lr = min_rate)
        model.train(train_cps, test_corpus = val_cps, epochs = epochs)
    
    if opts.evaluate:
        eval_cps = Corpus(wv, opts.evaluate, batchSize = 2, has_mark = True, seqMaxLen=seqMaxLen)
        if not model:
            model = Model(wv, opts.model_filename)
        model.evaluate(eval_cps)
        
    if opts.predict:
        files = opts.predict.split(u",")
        predict_cps = Corpus(wv, files[0], seqMaxLen=seqMaxLen)
        if not model:
            model = Model(wv, opts.model_filename)
        #model.predict(predict_cps, files[1])
        predict_cps.predictOutput(files[1], model)
            
    if opts.input:
       inp = opts.input.decode(sys.stdin.encoding)
       if not model:
            model = Model(wv, opts.model_filename)
       model.predict_one(inp)
       model.input_predict()
       
   
if __name__ == '__main__':
    main()
