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
    _OPT.add_option('-i', '--input_file', action='store', dest='infile', help='input dir')
    _OPT.add_option('-o', '--output_file', action='store', dest='output_file', help='output file')

    (opts,args) = _OPT.parse_args()
    print opts
    print args


    outf = io.open(opts.output_file, 'w')
    f = io.open(opts.infile, 'r')
    for s in f:
            s = s.strip()
            if len(s) == 0:
                continue
            s = s.replace(u"。", u"。\n").replace(u" ", '')
            s = s.replace(u"，", u"，\n")
            s = s.replace(u"、", u"、\n")
            s = s.replace(u"；", u"；\n")
            s = s.replace(u"？", u"？\n")
            s = s.replace(u"：", u"：\n")
            outf.write(s)
            if s[-1] != u'\n':
                outf.write(u'\n')

    f.close()
    outf.close()    

    



if __name__ == '__main__':
    main()
