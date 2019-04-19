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
    output = io.open(output_file, 'w', encoding='utf-8')
    with io.open(input_file, 'r', encoding='utf-8') as content:
        out_lines = ""
        for line in content:
            line = line.strip()
            if len(line) == 0:
               continue
            words = jieba.cut(line, cut_all=False)
            marks = u""
            for word in words:
               if len(word) == 1:
                   marks += u's'
               elif len(word) == 2:
                   marks += u'b'
                   marks += u'e'
               else :
                   marks += u'b'
                   marks += u'm' * (len(word) - 2)
                   marks += u'e'
                
            output.write(marks)
            output.write(u'\n')
            output.write(line)
            output.write(u'\n')
            #out_lines += marks
            #out_lines += u'\n'
            #out_lines += line
            #out_lines += u'\n'
            texts_num += 1
            if texts_num % 2000 == 0:
                #output.write(out_lines)
                #out_lines = ""
                logging.info("已完成前 %d 行的mark" % texts_num)
        if out_lines != "":
            output.write(out_lines)
            out_lines = ""
    output.close()



infile = sys.argv[1]
outfile = sys.argv[2]
split_word(infile, outfile)


