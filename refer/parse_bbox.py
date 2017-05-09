import numpy as np
import scipy.io
import os
import sys

data_path = 'coco_bbox'
out_path = 'data/refcoco+_edgeboxes_top100'

if not os.path.isdir(out_path):
    os.mkdir(out_path)

# read all image names
img_list = open(data_path + '/coco_train2014.txt')
img_names = list()

while True:
    img_name = img_list.readline().split('/')[-1][:-1]
    if img_name == '':
        break
    else:
        img_names.append(img_name)

img_list.close()
    
# load refcoco+ image names
bbox = scipy.io.loadmat(data_path + '/bbox_train2014_0.3.mat')['BBOX']


img_list = open('data/refcoco+_all_imlist.txt')
target_name = img_list.readline()[:-1]
cnt = 0
for img_name in img_names:
    if img_name == target_name:
        box = bbox[cnt][0][0]
        data_shape = box.shape

        for i in range(data_shape[0]):
            for j in range(2, 4):
                box[i, j] -= 1

        while data_shape[0] < 100:
            box = np.concatenate((box, bbox[cnt][0][0]), axis=0)
            data_shape = box.shape
            
        np.savetxt(out_path + '/' + target_name[:-4] + '.txt', 
            box[:100,:4], fmt="%16.7e", delimiter='')

        target_name = img_list.readline()[:-1]
    cnt += 1
    sys.stdout.write('processing...' + str(int(100 * float(cnt) / len(img_names))) + '%\r')
    sys.stdout.flush()
        

print 'Total', cnt, 'images processed'
