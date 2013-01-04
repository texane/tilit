#!/usr/bin/env python


import urllib2
import re
import hashlib


url_base = 'http://www.trekearth.com/browse/gallery/Asia/India'


def get_html(url):
    try:
        f = urllib2.urlopen(url)
        s = f.read()
    except:
        return None
    return s


def get_contents(url):
    return get_html(url)


scaled_jpg_re = re.compile('<img src="(http://i1\.trekearth\.com/photos/\d+/.*_s2_\.jpg)')

def get_page(year, page):

    url = url_base + '/' + str(year) + '/page' + str(page) + '.htm'
    html = get_html(url)
    if html == None: return False

    res = scaled_jpg_re.findall(html)
    if res == None or len(res) == 0: return False

    for s in res:
        unscaled_url = re.sub(r'_s2_', r'', s)

        image_data = get_contents(unscaled_url)
        if image_data == None: continue

        d = hashlib.md5()
        d.update(unscaled_url)
        path = './' + d.hexdigest() + '.jpg'

        f = open(path, 'w')
        f.write(image_data)
        f.close()
    
    return True


def get_year(year):

    page_url = url_base + '/' + str(year)
    page_html = get_html(page_url)
    if page_html == None: return False

    # fast way to retrieve the max year
    max_year_re = re.compile(str(year) + '/page(\d+)\.htm')
    s = max_year_re.findall(page_html)
    if s == None or len(s) == 0: return False

    # included bounds
    min_page = 1
    max_page = int(s[-1])

    for page in range(min_page, max_page + 1):
        print('get ' + str(year) + '/' + str(page) + ' on ' + str(max_page))
        get_page(year, page)
    return True


def main():
    # included bounds
    min_year = 2010
    max_year = 2012
    for year in range(min_year, max_year + 1): get_year(year)
    return


main()