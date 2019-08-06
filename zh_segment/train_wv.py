#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import logging
import io
import sys
import os


def split_word(input_file, output_file): 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    texts_num = 0
    stop_word_set = set()
    output = io.open(output_file, 'w', encoding='utf-8')
    with io.open(input_file, 'r', encoding='utf-8') as content:
        for line in content:
            line = line.strip('\n').strip()
            #words = jieba.cut(line, cut_all=False)
            words = line
            for word in words:
                if word not in stop_word_set:
                    output.write(word + ' ')
            output.write(u'\n')
            texts_num += 1
            if texts_num % 10000 == 0:
                logging.info("已完成前 %d 行的分词" % texts_num)
    output.close()




from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence

def test_print(model):
    model_len = len(model.wv.vocab)
    print('cn_model len: ', model_len)
    embedding_dim = model[u'a'].shape[0]
    print('词向量的长度为{}'.format(embedding_dim))
    print_char(model, u'a')
    print_char(model, u',')
    print_char(model, u'。')
    print_char(model, u'，')
    print_char(model, u'.')
    print_char(model, u'无')
    print_char(model, u'理')
    print_char(model, u'长')

def print_char(model, c):
    print c
    print model.wv.vocab[c].index   
    print model[c]

def w2v(input_file, output_file):

    model = None
    if os.path.exists(output_file):
         #model = KeyedVectors.load(output_file)
         #model = Word2Vec.load(output_file,  mmap='r')
         model = Word2Vec.load(output_file)
         #wv = KeyedVectors.load_word2vec_format(output_file, binary=True)
         #test_print(model)
         
  
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if model:
        print "continue trainning..."
        
        f = io.open(input_file)
        txts = f.read().splitlines()
        f.close()
        txtlist = []
        for t in txts:
           if len(t) > 0:
             txtlist.append(t.split())
             #print t.split()

        model.build_vocab(LineSentence(input_file),update=True)
        #sentences = word2vec.Text8Corpus(input_file)
        #model.build_vocab(sentences)
        
    
        
        #model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        model.train(LineSentence(input_file), total_examples=model.corpus_count, epochs=model.epochs)
        
        #model.train(txtlist, total_examples=1, epochs=1)
        
        #model.train(txtlist, total_examples=1, epochs=1)
        
        #sentences = word2vec.Text8Corpus(input_file)
        #print model.corpus_count
        #model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        #model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        #model.train([['你好', '中国']], total_examples=1, epochs=1)
    else:
        print "start trainning..."
        sentences = word2vec.Text8Corpus(input_file)
        model = word2vec.Word2Vec(sentences, size=400, window=4, min_count = 1)

    model.save(output_file)
    test_print(model)


#split_word("./data/sentence_all", "./data/sentence_all_seg")
infile = sys.argv[1]
infile_seg = infile + "_seg"
outfile = sys.argv[2]
split_word(infile, infile_seg)
w2v(infile_seg, outfile)


def test_result():
    #cn_model = load_w2v("w2v.model")
    cn_model = KeyedVectors.load('w2v.model', mmap='r')
    cn_model_len = len(cn_model.wv.vocab)
    print('cn_model len: ', cn_model_len)
    embedding_dim = cn_model[u'酒店'].shape[0]
    print('词向量的长度为{}'.format(embedding_dim))
    print cn_model[u'酒店']
    print cn_model.wv.vocab[u'酒店'].index   

#test_result()
