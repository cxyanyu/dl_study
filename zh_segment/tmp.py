#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import sys
import os


def split_word(input_file, output_file): 
    texts_num = 0
    output = io.open(output_file, 'w', encoding='utf-8')
    with io.open(input_file, 'r', encoding='utf-8') as content:
        pair = []
        for line in content:
            line = line.strip()
            if len(line) == 0 :
                continue
            pair.append(line)
            if len(pair) == 2:
                output.write(pair[1])
                output.write(u'\n')
                output.write(pair[0])
                output.write(u'\n')
                pair = []
            texts_num += 1
            if texts_num % 10000 == 0:
                print "已完成前 %d 行的mark" % texts_num
    output.close()



infile = sys.argv[1]
outfile = sys.argv[2]
split_word(infile, outfile)


