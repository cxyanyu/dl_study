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
    _OPT.add_option('-i', '--input_dir', action='store', dest='input_dir', help='input dir')
    _OPT.add_option('-o', '--output_file', action='store', dest='output_file', help='output file')

    (opts,args) = _OPT.parse_args()
    print opts
    print args


    outf = io.open(opts.output_file, "w")
    infiles = os.listdir(opts.input_dir)
    for infile in infiles:
        print "file: " + infile
        f = io.open(opts.input_dir + "/" + infile, 'r',  encoding='gb18030')
        texts = f.read()
        f.close()
        slist = re.findall(r"<content>(.?|.+?)</content>", texts, re.S)
        for s in slist:
            s = s.replace(u"。", u"。\n").replace(u" ", '')
            outf.write(s)
            outf.write(u'\n')

    outf.close()    

    



if __name__ == '__main__':
    main()
