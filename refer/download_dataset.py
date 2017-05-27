from __future__ import print_function
from multiprocessing import Pool
import urllib2
import sys
import os
import json

dataset = 'data/refcoco+'
image_path = 'data/images/mscoco/images/train2014'

def download_single_img(img):
    file_name = img['file_name']
    f = open(image_path + '/' + file_name, 'wb')

    url = img['coco_url']

    while True:
        try:
            u = urllib2.urlopen(url)
            d = u.read()
        except:
            pass
        else:
            break

    f.write(d)
    f.close()

def download():
    pool = Pool()

    imgs = d['images']
    cnt = 0
    try:
        for i in pool.imap(download_single_img, imgs):
            cnt += 1
            sys.stdout.write('Images downloaded:' + str(cnt) + '/' + str(len(imgs)) + '\r')
            sys.stdout.flush()
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print('terminating download process...')
        pool.terminate()

if __name__ == '__main__':
    if not os.path.isdir(image_path):
        acc_path = ''
        for path in image_path.split('/'):
            acc_path = path if acc_path == '' else '/'.join([acc_path, path])
            os.mkdir(acc_path)
    
    f = open(dataset + '/instances.json')
    d = json.load(f)
    download()
