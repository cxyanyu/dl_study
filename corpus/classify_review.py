#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import chardet
import time
import random
import optparse
import sys
import os
import io


def readfile(path):
    tags = []
    corpus = []
    with io.open(path, 'r') as f:
      tags = f.readline().strip('\n').strip().split(',')
      line = f.readline().strip('\n')
      while line:
        xy = {}
        y = line.split(',')
        xy['y'] = [int(a) for a in y]
        xy['x'] = f.readline().strip('\n')
        corpus.append(xy)
        line = f.readline().strip('\n')
    return tags, corpus  
        
def check_input(inp):
    if len(inp) == 0:
       return True
    if inp == 'y' or inp == 'Y':
       return True
    if inp == '1' or inp == '2' or inp == '3':
       return True
    return False


def output(filename, text):
   outf = io.open(filename, 'a')
   outf.write(text)
   outf.close()

def main():
    _OPT = optparse.OptionParser()
    _OPT.add_option('-i', '--input_file', action='store', dest='input_file', help='input file')
    _OPT.add_option('-o', '--output_file', action='store', dest='output_file', help='output file')

    (opts,args) = _OPT.parse_args()
    print opts
    print args
    
    tags = [] 
    file_index = -1
          
  
    reload(sys)
    sys.setdefaultencoding( "utf-8" )

    in_tags, in_corpus = readfile(opts.input_file)

    out_tags = []
    out_corpus = []
    try:
        out_tags, out_corpus = readfile(opts.output_file)
    except:
        print "outfile is not exsit, will created."

    tags = in_tags
    #print out_corpus
    index = len(out_corpus)
    
    if index > 0 and  out_corpus[index -1]['x'] != in_corpus[index-1]['x']:
        print "check failed!"
        print "output[%d] %s" %(index-1, out_corpus[index -1]['x'])
        print "input[%d] %s" %(index-1, in_corpus[index -1]['x'])
        return
    
    if index == 0:
        output(opts.output_file, ','.join(tags) + '\n')
        
    print "index=%d" %index
    for i, c in enumerate(in_corpus):
        if i < index:
           continue

        info = u""
        for j, t in enumerate(tags):
            info += t + u": " + str(c['y'][j*2]) + u", " + str(c['y'][j*2 + 1])
            info += u";  "
        print c['x']
        print info
        items_y = []
        for j, t in enumerate(tags):
            item_y_str = str(c['y'][j*2]) + u"," + str(c['y'][j*2 + 1])
            info = t + u"[" + item_y_str + "], 确认结果(Y/1/2/3)："
            #inp = raw_input(info.decode('utf8').encode(sys.stdin.encoding)).decode(sys.stdin.encoding).encode("utf8")
            inp = raw_input(info.decode('utf8')).decode(sys.stdin.encoding).encode("utf8")

            while not check_input(inp):
                print u"输入无效！请重新输入"
                inp = raw_input(info.decode('utf8')).decode(sys.stdin.encoding).encode("utf8")

            if len(inp) == 0:
                inp = 'y'
            
            if inp == 'y' or inp == 'Y':
                items_y.append(item_y_str)
            else:
                if inp == '1':
                  items_y.append('1,0') 
                elif inp == '2':
                  items_y.append('0,1') 
                else:
                  items_y.append('0,0') 

        y_str = ','.join(items_y)
        y = y_str.split(',')            
        y = [int(a) for a in y]
        s = sum(y)
        if s > 0:
            y_str += u',0'
        else:
            y_str += u',1'

        print "confirm result: " + y_str
        
        output(opts.output_file, y_str + '\n' + c['x'] + '\n')
        print '======================================================='





          
if __name__ == '__main__':
    main()
