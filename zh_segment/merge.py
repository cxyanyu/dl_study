#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import logging
import io
import sys
import os
import random
from random import shuffle
import numpy as np


def readlines(f, num) :
    lines = []
    for i in range(num):
        line = f.readline()
        if not line:
            break
        lines.append(line)
    return lines


def split_word(input_file1, input_file2, output_file): 
    texts_num = 0
    output = io.open(output_file, 'w', encoding='utf-8')
    inf1 = io.open(input_file1, 'r')
    inf2 = io.open(input_file2, 'r')

    lines1 = readlines(inf1, 60000)
    lines2 = readlines(inf2, 15000)
    print "lines1 len: %d" %len(lines1)
    while (lines1) > 0 and len(lines2) > 0:
        lines1 = np.array(lines1)
        lines2 = np.array(lines2)
        print lines1.shape
        lines1 = lines1.reshape(-1,2)
        lines2 = lines2.reshape(-1,2)
        print lines1.shape
        lines = np.concatenate((lines1, lines2),axis=0)
        print lines.shape
        lines = lines.tolist()
        shuffle(lines)
        lines = np.array(lines)
        lines = lines.reshape(-1)
        output.writelines(lines.tolist())
        lines1 = readlines(inf1, 60000)
        lines2 = readlines(inf2, 15000)
    output.close()

infile1 = sys.argv[1]
infile2 = sys.argv[2]
outfile = sys.argv[3]
split_word(infile1, infile2, outfile)


