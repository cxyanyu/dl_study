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


def main():
    _OPT = optparse.OptionParser()
    _OPT.add_option('-i', '--input', action='store', dest='input', help='input file')
    _OPT.add_option('-o', '--output', action='store', dest='output', help='output file')
    (opts,args) = _OPT.parse_args()
    print opts
    print args

    f = io.open(opts.input, 'r')
    line1 = f.readline()
    print line1

    line = f.readline()
    line += f.readline()
    texts = []
    while line:
        texts.append(line)
           
        line = f.readline()
        line += f.readline()
    f.close()
    
    random.shuffle(texts)
    
    f = io.open(opts.output, 'w')
    f.write(line1)
    for t in texts:
        f.write(t)
    f.close()
 

    

if __name__ == '__main__':
    main()
