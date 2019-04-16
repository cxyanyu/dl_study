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




def load_wv(wv_file):
    try:
      cn_w2v = KeyedVectors.load(wv_file, mmap='r')
    except:
      cn_w2v = KeyedVectors.load_word2vec_format(wv_file,binary=True)
    
    #embedding_dim = cn_model[u'酒店'].shape[0]
    #cn_model_len = len(cn_model.wv.vocab)
    return cn_w2v


def load_sample_dir(sample_dir):
    texts = []
    files = os.listdir(sample_dir)
    for i in range(len(files)):
        with open(sample_dir + "/" + files[i], 'r') as f:
            text = f.read().strip()
            texts.append(text)
            f.close()
    return texts

def build_index_tokens(texts, cn_w2v):
    tokens = []
    for tx in texts:
        cut = jieba.cut(tx)
        cut_list = [e for e in cut]
        for i, word in enumerate(cut_list):
            try: 
                cut_list[i]  = cn_w2v.wv.vocab[word].index + 1
            except KeyError:
                cut_list[i] = 0 
        tokens.append(cut_list)
    return tokens

def clac_max_token_nums(all_tokens):
    nums = []
    nums.extend([len(t) for t in all_tokens])
    nums = np.array(nums)
    real_max_num = np.max(nums)
    print "token len mean %d"%np.mean(nums)
    print "token len max %d"%np.max(nums)
    #plt.hist(np.log(nums), bins = 100)
    #plt.hist(np.log(nums))
    #plt.xlim((0,10))
    #plt.ylabel('number of tokens')
    #plt.xlabel('length of tokens')
    #plt.title('Distribution of tokens length')
    #plt.show()

    #max_num = np.mean(nums) + 2 * np.std(nums)
    #max_num = int(max_num)
    #print "max token nums: %d"%max_num
    #print np.sum( nums < max_num) / len(nums)
    #return max_num
    return real_max_num


# 用来将tokens转换为文本
def reverse_tokens(tokens, cn_w2v):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_w2v.wv.index2word[i-1]
        else:
            text = text + '*'
    return text



def build_dict(words_num, cn_w2v):
    embedding_dim = cn_w2v[cn_w2v.wv.index2word[0]].shape[0]
    embedding_matrix = np.zeros((words_num + 1, embedding_dim))
    embedding_matrix[0,:] = [0] * embedding_dim
    for i in range(words_num):
        embedding_matrix[i+1,:] = cn_w2v[cn_w2v.wv.index2word[i]]
    embedding_matrix = embedding_matrix.astype('float32')
    print np.sum(cn_w2v[cn_w2v.wv.index2word[22]] == embedding_matrix[22] )
    print embedding_matrix.shape
    return embedding_matrix

 
def pad_tokens(tokens, max_tokens):
    pads = pad_sequences(tokens, maxlen=max_tokens, padding='pre', truncating='pre')
    return pads


def make_xy(cn_w2v, sample_dir, y) :
    texts = load_sample_dir(sample_dir)
    x_ = build_index_tokens(texts, cn_w2v)
    y_ = []

    print "len(x_)%d" %len(x_)
    for i in range(len(x_)):
      y_.append(y)
    print "len(y_)%d" %len(y_)
    return x_, y_


def create_mode(embedding_matrix, output_size, max_tokens, model_filename, rate):
    # 用LSTM对样本进行分类
    print "create mode.........."
    model = Sequential()

    # 模型第一层为embedding
    print embedding_matrix.shape[0]
    print embedding_matrix[0].shape[0]
    model.add(Embedding(embedding_matrix.shape[0],
                        embedding_matrix[0].shape[0],
                        weights=[embedding_matrix],
                        input_length=max_tokens,
                        trainable=False, name="embedding_1"))

    model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
    model.add(LSTM(units=16, return_sequences=False))

    # GRU的代码
    # model.add(GRU(units=32, return_sequences=True))
    # model.add(GRU(units=16, return_sequences=True))
    # model.add(GRU(units=4, return_sequences=False))

    model.add(Dense(output_size, activation='sigmoid', name="output"))
    # 我们使用adam以0.001的learning rate进行优化
    optimizer = Adam(lr=rate)


    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print model.summary()


    
    #embedding_layer_output_model = Model(inputs=model.input,
    #                                outputs=model.get_layer('Dense_1').output)


    # 尝试加载已训练模型
    try:
        model.load_weights(model_filename)
    except Exception as e:
        print(e)

    #w = model.get_layer('embedding_1').get_weights()
    return model


def train(X_train, Y_train, model, save_path, eps, min_rate):
    
    earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    # 自动降低learning rate
    lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1, min_lr=min_rate, patience=0,
                                           verbose=1)


    # 建立一个权重的存储点
    checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss',
                                          verbose=1, save_weights_only=False,
                                          save_best_only=True)
    # 定义callback函数
    callbacks = [
        earlystopping, 
        checkpoint,
        lr_reduction
    ]

    # 开始训练
    print X_train.shape

    model.fit(X_train, Y_train,
              validation_split=0.1, 
              epochs=eps,
              batch_size=128,
              shuffle=True,
              callbacks=callbacks)

    #result = model.evaluate(X_test, Y_test)
    #w = model.get_layer('embedding_1').get_weights()
    #print('Accuracy:{0:.2%}'.format(result[1]))

def evaluate(model, X, Y):
    result = model.evaluate(X, Y)
    #w = model.get_layer('embedding_1').get_weights()
    print('Accuracy:{0:.2%}'.format(result[1]))


def predict_sentiment(text, model, w2v, max_tokens, items):
        num_words = len(w2v.wv.vocab)
        #print(text)
        # 去标点
        #text = text.decode("utf8")
        #text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"),text)
        #text = text.encode("utf8")
        print(text)
        # 分词
        #cut = jieba.cut(text)

        #cut_list = [ i for i in cut ]
        #print "cut_list:%s" %str(cut_list).replace("u\'","\'").decode('unicode_escape')
        # tokenize
        #print cut_list
        cut_list = build_index_tokens([text], w2v)
        print cut_list
        '''
        for i, word in enumerate(cut_list):
            try:
                cut_list[i] = w2v.wv.vocab[word].index + 1
                if cut_list[i] > num_words:
                   cut_list[i] = 0 
                #print('index', w2v.wv.vocab[word].index)
            except KeyError:
                cut_list[i] = 0
        '''
        #print "cut_list:%s" %str(cut_list).replace("u\'","\'").decode('unicode_escape')
        # padding
        tokens_pad = pad_sequences(cut_list, maxlen=max_tokens,
                               padding='pre', truncating='pre')
        print reverse_tokens(tokens_pad[0], w2v)
        # 预测
        result = model.predict(x=tokens_pad)
        coef = result[0]
        print 'output='
        #print coef
        c_format = []
        for c in coef:
            c_format.append('{0:.2f}'.format(c))
        print c_format

        i = 0
        for item in items:
             print "%s: %s, %s" %(item, c_format[2*i], c_format[2*i + 1])
             i += 1
        print "other: %s" % c_format[2*i]

        
        #coef = result[0][0]
        #if coef >= 0.5:
        #    print '%s'%'是一例正面评价' 'output=%.2f'%coef
        #else:
        #    print '%s'%'是一例负面评价' 'output=%.2f'%coef
        return coef



def evaluate_inp(model, w2v, max_len, inp, items):
    test_list = [
        '酒店设施不是新的，服务态度很不好',
        '酒店卫生条件非常不好',
        '床铺非常舒适',
        '房间很凉，不给开暖气',
        '房间很凉爽，空调冷气很足',
        '酒店环境不好，住宿体验很不好',
        '房间隔音不到位' ,
        '晚上回来发现没有打扫卫生',
        '因为过节所以要我临时加钱，比团购的价格贵',
        '卫生很差，床单没有换',
        '房间很大，采光很好',
        '手机速度很快，很赞',
        '玩游戏很不错',
        '手机到手了，物流太慢了！玩游戏很流畅，就是手机外观太丑，客服妹妹很漂亮',
        '给老公买的，性价比超低。反应一点都不卡!那天没抢到!相机垃圾;这个价格太黑了;使用体验很棒，相机很差.玩游戏很流畅！恩;给老公买的;'
    ]
    if len(inp) > 0:
        test_list.append(inp)

    for text in test_list:
        print "-------------------------------------------------------"
        predict_sentiment(text, model, w2v, max_len, items)

    while True:
        print "-------------------------------------------------------"
        text = raw_input("input sentence:").decode(sys.stdin.encoding).encode("utf8")
        predict_sentiment(text, model, w2v, max_len, items)

def load_corpus(filename, cn_w2v):
    f = io.open(filename, 'r')
    line = f.readline().strip(u'\n').strip()
    print line
    items = line.split(u',')
    line = f.readline().strip(u'\n').strip()
    
    _y = []
    _x = []
    while line:
        strs = line.split(u',')
        nums = []
        for s in strs:
           nums.append(int(s))
        _y.append(nums)
           
           
        line = f.readline().strip(u'\n').strip()
        _x.append(line)
        line = f.readline().strip(u'\n').strip()
    _x = build_index_tokens(_x, cn_w2v)
    return _x, _y 

def auto_classify(infile, outfile, model, w2v, max_len, items) :
    inf = io.open(infile, 'r')
    outf = io.open(outfile, 'w')
    outf.write(u','.join(items))
    outf.write(u'\n')
    line = inf.readline()

    while line:
        print "line:[%s]" %line

        line = line.strip('\n').strip()
        if len(line) == 0:
            line = inf.readline()
            continue

        res = predict_sentiment(line, model, w2v, max_len, items)
        int_res = [0] * len(res)
        s = 0
        item_num = (len(res) - 1)/2
        for i in range(item_num):           
           if res[2*i] > 0.6 and res[2*i] > res[2*i + 1]:
              int_res[2*i] = 1

           if res[2*i + 1] > 0.6 and res[2*i+1] > res[2*i]:
              int_res[2*i+1] = 1
           
           s += int_res[2*i] + int_res[2*i +1]

        if s == 0 :
            int_res[len(int_res) - 1] = 1       
         
        outf.write(u','.join([str(x) for x in int_res]))
        outf.write(u'\n')
        outf.write(line)
        outf.write(u'\n')

        line = inf.readline()

    inf.close()
    outf.close()


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

    cn_w2v = load_wv(opts.w2v)

    num_words = len(cn_w2v.wv.vocab)

    embedding_matrix = build_dict(num_words, cn_w2v)
    
    epochs = 20
    
    auto_classify_in = None
    auto_classify_out = None
    
    model = None
    def_max_tokens = 128

    items=[u'性价比', u'外观', u'性能', u'照相']
    start_rate=1e-3
    min_rate=1e-5

    if opts.rate:
        rates = opts.rate.split(',')
        print rates
        start_rate = float(rates[0])
        min_rate = float(rates[1])
        print start_rate
        print min_rate

    if opts.epochs:
        epochs = int(opts.epochs)        
  

    if opts.auto_classify_inout:
        files = opts.auto_classify_inout.split(u",")
        auto_classify_in = files[0]        
        auto_classify_out = files[1]        

    if opts.items:
       print opts.items
       print items
       items = opts.items.split(",")
       print items

    if os.path.exists(opts.model):
        try:
            model = load_model(opts.model)
        #ValueError
        except ValueError as e:
            print "load model failed!"
            print e

    if opts.corpus:

        _x, _y = load_corpus(opts.corpus, cn_w2v)

        print _x[10]
        print _y[10]
        #tokens_max_num = clac_max_token_nums(_x)
        tokens_max_num = def_max_tokens
        _x = pad_tokens(_x, tokens_max_num)
        print _x[30]
        print "len(_x) %d" %len(_x)
        print "len(_y) %d" %len(_y)
        X_train, X_test, Y_train, Y_test = train_test_split(_x,
                                                    _y,
                                                    test_size=0.1,
                                                    random_state=12)
        print X_train.shape

        print(reverse_tokens(X_train[30], cn_w2v))
        print('classify: ',Y_train[30])
     
        if model == None:
            model = create_mode(embedding_matrix, len(Y_train[0]), tokens_max_num, opts.model, start_rate)
         
        train(X_train, Y_train, model, opts.model, epochs, min_rate)
        evaluate(model, X_test, Y_test)


    if opts.evaluate:
        _x, _y = load_corpus(opts.evaluate, cn_w2v)
        tokens_max_num = def_max_tokens
        _x = pad_tokens(_x, tokens_max_num)
        evaluate(model, _x, _y)


    test_max_len = def_max_tokens

    inp = ""
    if opts.input:
       inp = opts.input
       evaluate_inp(model, cn_w2v, test_max_len, inp, items)
      
    if auto_classify_in and auto_classify_out :
       auto_classify(auto_classify_in, auto_classify_out, model, cn_w2v, test_max_len,items)
    
        
   
if __name__ == '__main__':
    main()
