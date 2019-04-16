#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import urllib2
import re
import json
import chardet
import time
import random
import optparse
import os


 
def get_content(req_url):
    request = urllib2.Request(req_url)
    response = urllib2.urlopen(request, timeout = 20)
    content = response.read()
    return content

def get_comment_urls(search_page_url):
    request = urllib2.Request(search_page_url)
    response = urllib2.urlopen(request, timeout = 20)
    content = response.read()
    soup = BeautifulSoup(content, 'html.parser')
    tag = soup.title
    comment_tags = soup.find_all('a', id=re.compile("^J_comment"))
    print comment_tags

    hrefs = []
    for a_tag in comment_tags:
        a_soup = BeautifulSoup(str(a_tag))
        a_href = "https:" + a_soup.a['href']
        print a_href
        hrefs.append(a_href)
    return hrefs

def get_comments(comment_page_url):
    request = urllib2.Request(comment_page_url)
    response = urllib2.urlopen(request, timeout = 20)
    content = response.read()
    soup = BeautifulSoup(content, 'html.parser')
    comment_tags = soup.find_all('p', class_=re.compile('^comment-con'))
    print comment_tags
 



def get_all_productId(search_page_url):
    request = urllib2.Request(search_page_url)
    response = urllib2.urlopen(request, timeout = 20)
    content = response.read()
    soup = BeautifulSoup(content, 'html.parser')
    tag = soup.title
    comment_tags = soup.find_all('li', class_='gl-item')
    print comment_tags

    ids = []
    for cmt_tag in comment_tags:
        cmt_soup = BeautifulSoup(str(cmt_tag))
        pid = cmt_soup.li['data-sku']
        print pid
        ids.append(pid)
    return ids


import io



def get_comments(pid, save_path):
    url_pre="https://sclub.jd.com/comment/productPageComments.action?score=0&sortType=5&pageSize=10&productId=" + pid + "&page="
    npage = 0
    text_cmts = []
    product_url = "https://item.jd.com/" + pid + ".html"
    try:
        p_req = urllib2.Request(product_url)
        p_res = urllib2.urlopen(p_req, timeout = 10)
    except:
        print "get product home page fialed"

    idx = 0
    while (1):
        print '--------------------------------------------get page ' + bytes(npage)
        time.sleep(random.randint(1,5))
        query_url = url_pre + bytes(npage)
        npage += 1
        
        request = urllib2.Request(query_url)
        request.add_header('content-type', 'text/html;charset=gbk')

        req_failed = 1
        for try_times in range(3):
            try:
                response = urllib2.urlopen(request, timeout = 20)
                req_failed = 0
                break;
            except:
                print "query failed, try again"
        if req_failed == 1:
            print "query failed!!!!"
            print query_url
            continue

        res = response.read()
        #fencoding=chardet.detect(res)
        #print fencoding
        if len(res) == 0:
             print "response is empty!!!!!!!!"
             break
   
        try:
           res = res.decode('gbk').encode('utf8')
        except:
           print "decode error!"
           continue

        cmts = []
        try:
            res = json.loads(res, encoding='utf-8')
            cmts = res['comments']
        except:
            print res
            print query_url

        for cmt in cmts:
            #print cmt['content']
            text_cmts.append(cmt['content'])
            savefile(save_path + bytes(pid) + "_" + bytes(idx) + ".txt", cmt['content'])
            idx += 1

        if len(cmts) == 0:
            break;
    return text_cmts

def mkdirs(dir):
    if not os.path.exists(dir) :
        os.makedirs(dir)

def savefile(path, content):
    f = io.open(path, "w")
    f.write(content.strip('\n').strip())
    f.write(u'\n')
    f.close()

def main():
    _OPT = optparse.OptionParser()
    _OPT.add_option('-u', '--url', action='store', dest='url', help='production search page url')
    _OPT.add_option('-s', '--save-path', action='store', dest='save_dir', help='save directory')

    (opts,args) = _OPT.parse_args()
    print opts
    print args
    pids = get_all_productId(opts.url)

    opts.save_dir += '/'
   
    mkdirs(opts.save_dir)

    for pid in pids:
        print '====================================%s' %pid
        cmts = get_comments(pid, opts.save_dir)
        print "len: %d" % len(cmts)
        '''
        index = 0
        for cmt in cmts:
            output = io.open(opts.save_dir + bytes(pid) + "_" + bytes(index) + ".txt", "w")
            #output.write(u"===begin===\n")
            output.write(cmt)
            output.write('\n')
            #output.write(u"\n===end===\n")
            output.close()
            index += 1
       '''

if __name__ == '__main__':
    main()
