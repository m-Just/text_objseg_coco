from refer import REFER
import scipy.io
import os, sys
import json
import numpy as np
import extract_vocabulary

data_set_name = 'refcoco+'
split_by = 'unc'

data_root = './data/'
mask_folder = './data/mask/'

def extract_imlist():
    refs = map(lambda id: refer.loadRefs(id)[0], ref_ids)
    split = set(map(lambda ref: ref['split'], refs))

    all_imlist = list()
    trainval_imlist = list()

    last_image_id = -1
    for s in split:
        splited_dict = dict()
        splited_refs = filter(lambda ref: ref['split'] == s, refs)
        imlist = list()
        for ref in splited_refs:
            img_file_name = '_'.join(ref['file_name'].split('_')[:-1]) + '.jpg'
            if ref['image_id'] == last_image_id:
                ref_no += 1
            else:
                ref_no = 1
                imlist.append(img_file_name)
                all_imlist.append(img_file_name)
                if s in ['train', 'val']:
                    trainval_imlist.append(img_file_name)

            last_image_id = ref['image_id']

        assert(len(imlist) == len(set(imlist)))
        imlist.sort()
        imlist_file = open(data_root + data_set_name + '_' + s + '_' + 'imlist.txt', 'w')
        for im in imlist:
            imlist_file.write(im + '\n')
        imlist_file.flush()
        imlist_file.close()
        print s + '_imlist extracted:', len(imlist), 'images'

    assert(len(trainval_imlist) == len(set(trainval_imlist)))
    trainval_imlist.sort()
    trainval_imlist_file = open(data_root + data_set_name + '_trainval_imlist.txt', 'w')
    for im in trainval_imlist:
        trainval_imlist_file.write(im + '\n')
    trainval_imlist_file.flush()
    trainval_imlist_file.close()
    print 'trainval_imlist extracted:', len(trainval_imlist), 'images'

    assert(len(all_imlist) == len(set(all_imlist)))
    all_imlist.sort()
    all_imlist_file = open(data_root + data_set_name + '_all_imlist.txt', 'w')
    for im in all_imlist:
        all_imlist_file.write(im + '\n')
    all_imlist_file.flush()
    all_imlist_file.close()
    print 'all_imlist extracted:', len(all_imlist), 'images'

    print '-------imlist extraction complete--------'

def extract_imsize():
    size_dict = dict()
    for img in instances['images']:
        size_dict[img['file_name']] = [img['width'], img['height']]
    f = open(data_root + data_set_name + '_imsize.json', 'w')
    json.dump(size_dict, f, indent=0)
    f.close()

    print 'imsize extracted:', len(size_dict.keys()), 'images'
    print '--------imsize extraction complete--------'
    return size_dict

def extract_imcrop():
    refs = map(lambda id: refer.loadRefs(id)[0], ref_ids)
    split = set(map(lambda ref: ref['split'], refs))

    imcrop_dict = dict()

    img_cnt = 0
    last_image_id = -1
    for s in split:
        splited_refs = filter(lambda ref: ref['split'] == s, refs)
        for ref in splited_refs:
            img_file_name = '_'.join(ref['file_name'].split('_')[:-1]) + '.jpg'
            if ref['image_id'] == last_image_id:
                ref_no += 1
            else:
                ref_no = 1
                img_cnt += 1
                imcrop_dict[img_file_name] = list()

            query_id = str(ref['image_id']) + '_' + str(ref_no)
            imcrop_dict[img_file_name].append(query_id)
            last_image_id = ref['image_id']

    f = open(data_root + data_set_name + '_imcrop.json', 'w')
    json.dump(imcrop_dict, f, indent=0)
    f.close()

    print 'imcrop extracted from', img_cnt, 'images'
    print '--------imcrop extraction complete--------'

def load_special_tokens():
    stk_file = open('data/special_tokens_refcoco+.txt')
    tokens = set()

    token = stk_file.readline()
    while not token == '':
        tokens.add(token[:-1])
        token = stk_file.readline()

    print 'total', len(tokens), 'special tokens loaded from file'
    print '--------token loading complete--------'
    return tokens

def extract_mask():
    if not os.path.isdir(mask_folder):
        os.mkdir(mask_folder)

    cnt = 0
    img_cnt = 0
    last_image_id = -1
    old_img_set = set()
    for ref_id in ref_ids:
        ref = refer.loadRefs(ref_id)[0]
        if ref['image_id'] == last_image_id:
            ref_no += 1
        else:
            ref_no = 1
            img_cnt += 1
            if ref['image_id'] in old_img_set:
                print 'Error: unexpeted image', ref['image_id']
            old_img_set.add(last_image_id)

        last_image_id = ref['image_id']

        mask = refer.getMask(ref)['mask'].astype(np.int16)-1    # 0 stands for mask, while -1 stands for background
        mask_path = mask_folder + str(ref['image_id']) + '_' + str(ref_no) + '.mat'

        scipy.io.savemat(mask_path, {'segimg_t':mask}, do_compression=True)
        sys.stdout.write('Saved %s                \r' % (mask_path))
        sys.stdout.flush()

        cnt += 1

    print 'total', cnt, 'masks extracted from', img_cnt, 'images'
    print '--------mask extraction complete--------'

def extract_bbox():
    bbox_dict = dict()
    img_cnt = 0
    last_image_id = -1
    for ref_id in ref_ids:
        ref = refer.loadRefs(ref_id)[0]
        if ref['image_id'] == last_image_id:
            ref_no += 1
        else:
            ref_no = 1
            img_cnt += 1
        last_image_id = ref['image_id']
        bbox = refer.getRefBox(ref_id)
        bbox = map(int, [bbox[0], bbox[1], bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1])
        size = size_dict['COCO_train2014_' + '{:012d}'.format(ref['image_id']) + '.jpg']
        if bbox[2] >= size[0] or bbox[3] >= size[1]:
            print "Error:", bbox[2:], ">=", size
        bbox_dict[str(ref['image_id']) + '_' + str(ref_no)] = bbox

    f = open(data_root + data_set_name + '_bbox.json', 'w')
    json.dump(bbox_dict, f, indent=0)
    f.close()

    print 'bbox extrated:', len(bbox_dict.keys()), 'from', img_cnt, 'images'
    print '--------bbox extraction complete--------'

def extract_query():
    refs = map(lambda id: refer.loadRefs(id)[0], ref_ids)
    split = set(map(lambda ref: ref['split'], refs))

    trainval_dict = dict()
    query_dict = dict()

    last_image_id = -1
    for s in split:
        splited_dict = dict()
        splited_refs = filter(lambda ref: ref['split'] == s, refs)
        for ref in splited_refs:
            img_file_name = '_'.join(ref['file_name'].split('_')[:-1]) + '.jpg'
            if ref['image_id'] == last_image_id:
                ref_no += 1
            else:
                ref_no = 1

            query_id = str(ref['image_id']) + '_' + str(ref_no)

            query = list()
            for sent in ref['sentences']:
                text = sent['sent']
                for token in sent['tokens']:
                    if token in stk:
                        text = text.replace(token, extract_vocabulary.special_token)
                query.append(text)

            query_dict[query_id] = query
            if s in ['testA', 'testB']:
                splited_dict[query_id] = [sent['sent'] for sent in ref['sentences']]
            else:
                splited_dict[query_id] = query
            if s in ['train', 'val']:
                trainval_dict[query_id] = query
            last_image_id = ref['image_id']

        f = open(data_root + data_set_name + '_query_' + s + '.json', 'w')
        json.dump(splited_dict, f, indent=0)
        f.close()

    f = open(data_root + data_set_name + '_query_trainval.json', 'w')
    json.dump(trainval_dict, f, indent=0)
    f.close()
    f = open(data_root + data_set_name + '_query_all.json', 'w')
    json.dump(query_dict, f, indent=0)
    f.close()

    print '--------query extraction complete--------'

if __name__ == '__main__':
    refer = REFER(data_root, dataset=data_set_name, splitBy=split_by)
    instances = json.load(open(data_root + data_set_name + '/instances.json'))
    ref_ids = refer.getRefIds()

    #extract_imlist()
    #extract_imcrop()
    size_dict = extract_imsize()
    stk = load_special_tokens()
    #extract_mask()
    extract_bbox()
    #extract_query()
