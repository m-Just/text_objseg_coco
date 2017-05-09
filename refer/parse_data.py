from refer import REFER
import scipy.io
import os, sys
import json
import numpy as np

data_set_name = 'refcoco+'
split_by = 'unc'

data_root = './data/'
mask_folder = './data/mask/'

def load_special_tokens():
    stk_file = open('data/special_tokens_refcoco+.txt')
    tokens = set()

    token = stk_file.readline()
    while not token == '':
        tokens.add(token[:-1])
        token = stk_file.readline()
    
    return tokens

def extract_mask():
    if not os.path.isdir(mask_folder):
        os.mkdir(mask_folder)

    last_image_id = -1
    for ref_id in ref_ids:
        ref = refer.loadRefs(ref_id)[0]
        if ref['image_id'] == last_image_id:
            ref_no += 1
        else:
            ref_no = 1
        last_image_id = ref['image_id']

        mask = refer.getMask(ref)['mask'].astype(np.int8)-1
        mask_path = mask_folder + str(ref['image_id']) + '_' + str(ref_no) + '.mat'

        scipy.io.savemat(mask_path, {'segimg_t':mask}, do_compression=True)
        sys.stdout.write('Saved %s                \r' % (mask_path))
        sys.stdout.flush()

    print 'Mask extraction complete'

# also extract imlist & imcrop alongwith query
def extract_query():
    refs = map(lambda id: refer.loadRefs(id)[0], ref_ids)
    split = set(map(lambda ref: ref['split'], refs))

    imcrop_dict = dict()
    trainval_dict = dict()
    query_dict = dict()

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
                imcrop_dict[img_file_name] = list()
                imlist.append(img_file_name)
                all_imlist.append(img_file_name)
                if s in ['train', 'val']:
                    trainval_imlist.append(img_file_name)

            query_id = str(ref['image_id']) + '_' + str(ref_no)
            imcrop_dict[img_file_name].append(query_id)

            query = list()
            for sent in ref['sentences']:
                text = sent['sent']
                for token in sent['tokens']:
                    if token in stk:
                        text = text.replace(token, '<unk>')
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
        f = open(data_root + data_set_name + '_imcrop.json', 'w')
        json.dump(imcrop_dict, f, indent=0)
        f.close()

        imlist.sort()
        imlist_file = open(data_root + data_set_name + '_' + s + '_' + 'imlist.txt', 'w')
        for im in imlist:
            imlist_file.write(im + '\n')
        imlist_file.flush()
        imlist_file.close()

    f = open(data_root + data_set_name + '_query_trainval.json', 'w')
    json.dump(trainval_dict, f, indent=0)
    f.close()
    f = open(data_root + data_set_name + '_query_all.json', 'w')
    json.dump(query_dict, f, indent=0)
    f.close()

    trainval_imlist.sort()
    trainval_imlist_file = open(data_root + data_set_name + '_trainval_imlist.txt', 'w')
    for im in trainval_imlist:
        trainval_imlist_file.write(im + '\n')
    trainval_imlist_file.flush()
    trainval_imlist_file.close()
    all_imlist.sort()
    all_imlist_file = open(data_root + data_set_name + '_all_imlist.txt', 'w')
    for im in all_imlist:
        all_imlist_file.write(im + '\n')
    all_imlist_file.flush()
    all_imlist_file.close()

    print 'Queries extraction complete'

def extract_bbox():
    bbox_dict = dict()
    last_image_id = -1
    for ref_id in ref_ids:
        ref = refer.loadRefs(ref_id)[0]
        if ref['image_id'] == last_image_id:
            ref_no += 1
        else:
            ref_no = 1
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

    print 'Bounding box extraction complete'

def extract_imsize():
    size_dict = dict()
    for img in instances['images']:
        size_dict[img['file_name']] = [img['width'], img['height']]
    f = open(data_root + data_set_name + '_imsize.json', 'w')
    json.dump(size_dict, f, indent=0)
    f.close()

    print 'Image size extraction complete'
    return size_dict

if __name__ == '__main__':
    refer = REFER(data_root, dataset=data_set_name, splitBy=split_by)
    instances = json.load(open(data_root + data_set_name + '/instances.json'))
    ref_ids = refer.getRefIds()

    stk = load_special_tokens()
    size_dict = extract_imsize()
    extract_mask()
    extract_query()
    extract_bbox()
