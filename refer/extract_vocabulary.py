import matplotlib.pyplot as plt
import numpy as np

import pickle
import re
import operator

threshold = 1
special_token = 'stk'

def extract():
    print "Threshold =", threshold

    out.write('<pad>' + '\n')
    out.write('<go>' + '\n')
    out.write('<eos>' + '\n')
    out.write('<unk>' + '\n')
    out.write('stk' + '\n')

    word_dict = dict()
    word_dict[special_token] = 0
    stk_set = set()

    for ref in data:
        if ref['split'] == 'train':
            for sentence in ref['sentences']:
                for token in sentence['tokens']:
                    if token in word_dict:
                        word_dict[token] += 1
                    else:
                        word_dict[token] = 1

    cnt = 0
    stk_cnt = 0
    remain_word_list = list()
    for word in word_dict:
        if word_dict[word] > threshold:
            out.write(word + '\n')
            remain_word_list.append(word)
            cnt += 1
        else:
            if word != special_token:
                stk_set.add(word)
                word_dict[special_token] += word_dict[word]
                stk_cnt += 1

    stat.write('Threshold = %d\n' % (threshold))
    stat.write('Total %d vocabularies is extracted\n' % (cnt))

    if word_dict[special_token] > 0:
        stat.write('Total %d special tokens found with occurence %d\n' % (stk_cnt, word_dict[special_token]))

    sorted_dict = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)


    total_cnt = sum(word_dict.values())
    stat.write("{:<10}{:>6}{:>10}\n".format("total", total_cnt, "%"))
    for i in range(20):
        stat.write("{:<10}{:>6}{:>10.2}\n".format(sorted_dict[i][0], sorted_dict[i][1], float(sorted_dict[i][1]) / total_cnt * 100))

    # Count the number of expressions containing <stk>
    exp_cnt = 0
    stk_exp_cnt = 0
    for ref in data:
        if ref['split'] == 'train':
            for sentence in ref['sentences']:
                exp_cnt += 1
                for token in sentence['tokens']:
                    if word_dict[token] <= threshold:
                        stk_exp_cnt += 1
                        break
    stat.write('Total %d special tokens occur in all %d expressions\n' % (stk_exp_cnt, exp_cnt))


    for tokens in stk_set:
        stk.write(tokens + '\n')


if __name__ == '__main__':
    data = pickle.load(open('data/refcoco+/refs(unc).p'))
    out = open('data/vocabulary_refcoco+.txt', 'w')
    stk = open('data/special_tokens_refcoco+.txt', 'w')

    stat = open('stat_thresh_' + str(threshold) + '.txt', 'w')
    extract()
    stat.flush()
    stat.close()

    stk.flush()
    stk.close()
    out.flush()
    out.close()
