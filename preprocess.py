# /usr/bin/env python
# -*- coding: utf-8 -*-
import json,time,os,re
import numpy as np
import torch
from functools import reduce
import itertools
from joblib import Parallel, delayed
from libs.Dict import Dict as Vocabulary
from config.config import *
import jieba

data_path = CORPUS_DIR
data_filenames = {
    'train': 'train.txt',
    'valid': 'valid.txt',
    'test': 'test.txt'
}
vocab_file = os.path.join(RUNTIME_DIR, 'vocab.json')
dict_file = os.path.join(RUNTIME_DIR, 'dict.pt')

def rm_sign(strs):
    return re.sub("[\s+\.\!\/<>“”,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", strs.strip())

def word_tokenize(sentence):
    sentence = rm_sign(sentence)
    return jieba.lcut(sentence)

def tokenize(sentence):
    return [s.strip() for s in word_tokenize(sentence) if s.strip()]


def parse_stories(lines):
    stories = []
    story = []
    for line in lines:
        line = line.strip()
        if not line:
            story = []
        else:
            line = line.split("[#]")[1]
            if line:
                if "[*]" in line:  # query line
                    q, answer, candidates = line.split("[*]", 2)
                    '''
                    print("*----------------------------*")
                    print(q,'<>', answer, '<>', candidates)
                    print("*----------------------------*")
                    '''
                    q = tokenize(q)
                    # use the first 10
                    candidates = [cand for cand in candidates.split('|')]
                    stories.append((story, q, answer, candidates))
                else:
                    story.append(tokenize(line))
    return stories


def get_stories(story_lines):
    stories = parse_stories(story_lines)
    flatten = lambda story: reduce(lambda x, y: x + y, story)
    #stories = [(flatten(story), q, a, candidates) for story, q, a, candidates in stories]
    stories = [(flatten(story), q, a, flatten(story)) for story, q, a, candidates in stories]
    '''
    print("##########################")
    for story, q, answer, candidates in stories:
        print("##########################")
        print("story > ", story)
        print("q> ", q)
        print("answer> ", answer)
        print("cand> ", candidates)
        print("##########################")
    print("##########################")
    '''
    return stories


def vectorize_stories(stories, vocab : Vocabulary):
    X = []
    Q = []
    C = []
    A = []

    for s, q, a, c in stories:
        x = vocab.convert2idx(s)
        xq = vocab.convert2idx(q)
        xc = vocab.convert2idx(c)
        X.append(x)
        Q.append(xq)
        C.append(xc)
        A.append(vocab.getIdx(a))

    X = X
    Q = Q
    C = C
    A = torch.LongTensor(A)
    return X, Q, A, C


def build_dict(stories):
    if os.path.isfile(vocab_file):
        with open(vocab_file, "r") as vf:
            word2idx = json.load(vf)
    else:

        vocab = sorted(set(itertools.chain(*(list(story) + q + [answer] + candidates for story, q, answer, candidates in stories))))
        vocab_size = len(vocab) + 2     # pad, unk
        print('词库大小:', vocab_size)
        word2idx = dict((w, i + 2) for i,w in enumerate(vocab))
        word2idx[PAD_WORD] = 0
        word2idx[UNK_WORD] = 1

        with open(vocab_file, "w") as vf:
            json.dump(word2idx, vf)

    return Vocabulary(word2idx)


def main():

    print('语料库处理中 ...')
    train_filename = os.path.join(data_path, data_filenames['train'])
    valid_filename = os.path.join(data_path, data_filenames['valid'])
    test_filename = os.path.join(data_path, data_filenames['test'])


    with open(train_filename, 'r') as tf, open(valid_filename, 'r') as vf, open(test_filename, 'r') as tef:
        tlines = tf.readlines()
        vlines = vf.readlines()
        telines = tef.readlines()
        train_stories, valid_stories, test_stories = Parallel(n_jobs=2)(delayed(get_stories)(story_lines)
                                                          for story_lines in [tlines, vlines, telines])
        print(len(train_stories), len(valid_stories), len(test_stories))
    print('创建字典 ...')
    vocab_dict = build_dict(train_stories + valid_stories + test_stories)

    print('创建训练集验证集 ...')
    train = {}
    valid = {}
    test = {}

    #train_data, valid_data, test_data = 
    train_data, valid_data, test_data = Parallel(n_jobs=2)(delayed(vectorize_stories)(stories, vocab_dict) for stories in [train_stories, valid_stories, test_stories])

    train['documents'], train['querys'], train['answers'], train['candidates'] = train_data
    valid['documents'], valid['querys'], valid['answers'], valid['candidates'] = valid_data
    test['documents'], test['querys'], test['answers'], test['candidates'] = test_data
    print('保存数据 \'' + data_path + '\'...')
    torch.save(vocab_dict, dict_file)
    torch.save(train, RUNTIME_DIR + '/train.pt')
    torch.save(valid, RUNTIME_DIR + '/valid.pt')
    torch.save(test, RUNTIME_DIR + '/test.pt')

if __name__ == '__main__':
    main()
