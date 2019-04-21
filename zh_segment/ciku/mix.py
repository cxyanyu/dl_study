#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import logging
import io
import sys
import os
import random
from random import shuffle

def split_word(input_file, output_file): 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    texts_num = 0
    output = io.open(output_file, 'w', encoding='utf-8')
    mix_words = [u"我", u"的", u"你", u"她", u"他", u"吗", u"啊", u"呢", u"嗯", u"为什么", u"什么", u"哪里", u"是", u"不是"]
    lines = []
    with io.open(input_file, 'r') as content:
        out_lines = ""
        for line in content:
            line = line.strip()
            if len(line) == 0 or len(line) > 5:
               continue
            lines.append(line)

    for w in lines:
        words = random.sample(mix_words, 3)
        words += [w]
        words += random.sample(lines, 2)

        for times in range(1):
            shuffle(words)
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
            output.write(u''.join(words))
            output.write(u'\n')
        
        texts_num += 1
        if texts_num % 10000 == 0:
            logging.info("已完成前 %d 行的mark" % texts_num)
    output.close()



infile = sys.argv[1]
outfile = sys.argv[2]
split_word(infile, outfile)


